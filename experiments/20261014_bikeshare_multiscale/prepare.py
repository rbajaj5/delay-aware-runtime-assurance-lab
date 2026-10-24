"""Cached public inputs -> conservative, directed OSM cycling graphs.

No geometric intersection is inserted. Unknown pedestrian access, conditional
access, steps and bicycle-applicable restriction junctions are excluded rather
than silently granting access. This is a bounded research router, not navigation.
"""
from __future__ import annotations
import csv, hashlib, io, json, math, time, zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
import networkx as nx
import numpy as np
import requests
from pyproj import Transformer
from scipy.spatial import cKDTree
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union

ROOT=Path(__file__).resolve().parent
C=json.loads((ROOT/'config.json').read_text())
RAW=ROOT/'data'/'raw'; DATA=ROOT/'data'
RAW.mkdir(parents=True,exist_ok=True)

def dump(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,ensure_ascii=False,separators=(',',':'),allow_nan=False),encoding='utf-8')

def csvout(path,rows):
    if not rows:return
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def inside(lat,lon,b):return b[0]<=lat<=b[2] and b[1]<=lon<=b[3]

def fetch(name,url,query=None):
    p=RAW/name
    if p.exists():return p
    print('Downloading',name,flush=True)
    r=requests.get(url,params={'data':query} if query else None,timeout=(15,65),
                   headers={'User-Agent':'BikeshareMultiscaleResearch/1.0 (local educational prototype)'})
    r.raise_for_status()
    if query:
        obj=r.json()
        if obj.get('remark'):raise RuntimeError(obj['remark'])
    p.write_bytes(r.content)
    dump(RAW/(name+'.provenance.json'),{'url':url,'query':query,'retrieved_utc':datetime.now(timezone.utc).isoformat(),
         'sha256':sha(p),'bytes':p.stat().st_size,'last_modified':r.headers.get('Last-Modified')})
    return p

def osm(city):
    b=C['cities'][city]['buffer']; box=','.join(map(str,b))
    q=f'[out:json][timeout:25];(way["highway"]({box});way["waterway"="river"]({box});way["natural"="water"]({box});relation["type"="restriction"]({box}););out body;>;out skel qt;'
    for url in ['https://overpass.private.coffee/api/interpreter','https://overpass-api.de/api/interpreter','https://overpass.kumi.systems/api/interpreter']:
        try:return json.loads(fetch(city+'.osm.json',url,q).read_text(encoding='utf-8'))
        except Exception as e:print(type(e).__name__,str(e)[:180],flush=True)
    raise RuntimeError('Both OSM sources failed; no synthetic substitute permitted')

def barriers(city):
    box=','.join(map(str,C['cities'][city]['buffer']))
    q=f'[out:json][timeout:25];(node["barrier"]({box});node["bicycle"]({box});way["barrier"]({box}););out body;>;out skel qt;'
    for url in ['https://overpass-api.de/api/interpreter','https://overpass.private.coffee/api/interpreter']:
        try:return json.loads(fetch(city+'.barriers.json',url,q).read_text(encoding='utf-8'))
        except Exception as e:print(type(e).__name__,str(e)[:160],flush=True)
    raise RuntimeError('Barrier extract unavailable; cannot validate station access')

def access(tags):
    h=tags.get('highway',''); b=tags.get('bicycle')
    if any(k in tags for k in ['bicycle:conditional','access:conditional','oneway:conditional','oneway:bicycle:conditional']):return 'conditional'
    if b in ['no','private','use_sidepath','dismount']:return 'bicycle_restricted'
    if tags.get('access') in ['no','private','customers','destination','permit'] and b not in ['yes','designated','official','permissive']:return 'restricted_access'
    if tags.get('vehicle') in ['no','private'] and b not in ['yes','designated','official','permissive']:return 'vehicle_restricted'
    if h in ['steps','construction','proposed','motorway','motorway_link','raceway']:return 'excluded_highway'
    if h in ['footway','pedestrian','path','track','trunk','trunk_link'] and b not in ['yes','designated','official','permissive']:return 'unverified_bicycle_access'
    if h not in ['cycleway','residential','living_street','service','unclassified','tertiary','tertiary_link','secondary','secondary_link','primary','primary_link','footway','pedestrian','path','track','trunk','trunk_link']:return 'other_highway'
    return None

def directions(tags):
    o=tags.get('oneway:bicycle',tags.get('oneway','yes' if tags.get('junction')=='roundabout' else 'no'))
    if o in ['reversible','alternating']:return []
    if o=='-1':ds=[-1]
    elif o in ['yes','1','true']:ds=[1]
    else:ds=[1,-1]
    if 'oneway:bicycle' not in tags and any(str(tags.get(k,'')).startswith('opposite') for k in ['cycleway','cycleway:left','cycleway:right']):ds=[1,-1]
    if tags.get('bicycle:forward') in ['no','private','use_sidepath']:ds=[x for x in ds if x!=1]
    if tags.get('bicycle:backward') in ['no','private','use_sidepath']:ds=[x for x in ds if x!=-1]
    return ds

def build(city,raw,include_via_ways=False,save=True):
    tr=Transformer.from_crs(4326,C['cities'][city]['epsg'],always_xy=True)
    combined={}
    # Overpass recursion may repeat an element as an untagged skeleton.
    # Preserve its tagged body instead of erasing access/direction information.
    for e in raw['elements']+barriers(city)['elements']:
        key=(e['type'],e['id'])
        if key not in combined or e.get('tags'):combined[key]=e
    elements=list(combined.values())
    nodes={e['id']:e for e in elements if e['type']=='node'}
    xy={i:tr.transform(n['lon'],n['lat']) for i,n in nodes.items()}
    ways=[e for e in elements if e['type']=='way']
    rivers=[]; water=[];barrier_lines=[]
    for w in ways:
        t=w.get('tags',{});p=[xy[i] for i in w['nodes'] if i in xy]
        if len(p)<2:continue
        if t.get('waterway')=='river' and (city!='washington' or t.get('name') in ['Potomac River','Little River']):rivers.append(LineString(p))
        if t.get('natural')=='water' or t.get('waterway')=='river':water.append({'id':w['id'],'kind':t.get('waterway','polygon'),'name':t.get('name',''),'xy':p})
        if t.get('barrier') and t.get('bicycle') not in ['yes','designated']:barrier_lines.append({'id':w['id'],'xy':p,'tags':t})
        if t.get('highway') in ['motorway','motorway_link'] or t.get('bicycle') in ['no','private','use_sidepath']:
            barrier_lines.append({'id':w['id'],'xy':p,'tags':t,'kind':'excluded_road'})
    river=unary_union(rivers)
    blocked=set(); via_way=set(); exclusion=Counter(); restrictions=[]
    turn_rules=[]
    for e in elements:
        t=e.get('tags',{})
        if e['type']=='node' and (t.get('bicycle') in ['no','private','dismount'] or (t.get('barrier') in ['gate','lift_gate','stile','turnstile','block'] and t.get('bicycle') not in ['yes','designated'])):blocked.add(e['id'])
        if e['type']=='relation' and t.get('type')=='restriction':
            if 'bicycle' in t.get('except','').split(';'):continue
            if not any(k=='restriction' or k.startswith('restriction:bicycle') for k in t):continue
            restrictions.append(e)
            via=[m['ref'] for m in e['members'] if m['role']=='via' and m['type']=='node']
            frm=[m['ref'] for m in e['members'] if m['role']=='from' and m['type']=='way']
            to=[m['ref'] for m in e['members'] if m['role']=='to' and m['type']=='way']
            value=t.get('restriction:bicycle',t.get('restriction',''))
            if via and len(frm)==1 and to and (value.startswith('no_') or value.startswith('only_')):
                turn_rules.append({'id':e['id'],'via':via[0],'from':frm[0],'to':to,'value':value})
            elif via:blocked.update(via)
            via_way.update(m['ref'] for m in e['members'] if m['role']=='via' and m['type']=='way')
    G=nx.MultiDiGraph(); edges=[]; bridge_ways=[]
    for w in ways:
        tags=w.get('tags',{});reason=access(tags)
        if 'highway' not in tags:continue
        if reason or (w['id'] in via_way and not include_via_ways):
            exclusion[reason or 'restriction_via_way']+=1;continue
        bridge=tags.get('bridge','no') not in ['no','false','0']
        p=[xy[i] for i in w['nodes'] if i in xy]
        river_bridge=bridge and len(p)>1 and LineString(p).distance(river)<35
        if river_bridge and any(inside(nodes[i]['lat'],nodes[i]['lon'],C['cities'][city]['buffer']) for i in w['nodes'] if i in nodes):bridge_ways.append({'way':w['id'],'tags':tags,'xy':p})
        for j,(a,b) in enumerate(zip(w['nodes'],w['nodes'][1:])):
            if a not in xy or b not in xy or a in blocked or b in blocked:
                exclusion['blocked_or_missing_node_segments']+=1;continue
            if not all(inside(nodes[i]['lat'],nodes[i]['lon'],C['cities'][city]['buffer']) for i in [a,b]):
                exclusion['outside_buffer_segments']+=1;continue
            length=math.dist(xy[a],xy[b])
            if length<.01:continue
            for direction in directions(tags):
                u,v=(a,b) if direction==1 else (b,a)
                e={'id':f'{w["id"]}:{j}:{direction}','u':u,'v':v,'way':w['id'],'m':round(length,4),
                   'bridge':bridge,'river_bridge':river_bridge,'name':tags.get('name:en',tags.get('name','')),'tags':tags}
                edges.append(e);G.add_edge(u,v,key=e['id'],weight=e['m'])
    # Crossing groups are connected bridge geometries with nearby parallel lanes.
    BG=nx.Graph();BG.add_nodes_from(range(len(bridge_ways)))
    for i,a in enumerate(bridge_ways):
        for j,b in enumerate(bridge_ways[:i]):
            if LineString(a['xy']).distance(LineString(b['xy']))<80:BG.add_edge(i,j)
    crossings=[];wgroup={}
    for k,group in enumerate(sorted(nx.connected_components(BG),key=lambda c:min(bridge_ways[i]['way'] for i in c))):
        members=[bridge_ways[i] for i in group];names=sorted(set(m['tags'].get('name:en',m['tags'].get('name',m['tags'].get('wikipedia','').removeprefix('en:'))) for m in members)-{''})
        ident=f'B{k+1:02}';wayset=[m['way'] for m in members]
        crossings.append({'id':ident,'name':' / '.join(names) or 'OSM river crossing '+ident,'way_ids':wayset,
                          'evidence':'OSM bridge-tagged ways within 35 m of OSM river centerline; parallel components within 80 m',
                          'xy':np.mean([p for m in members for p in m['xy']],axis=0).tolist()})
        wgroup.update({w:ident for w in wayset})
    for e in edges:e['crossing']=wgroup.get(e['way'])
    components=sorted(nx.strongly_connected_components(G),key=len,reverse=True)
    # Retain entire network. Station selection uses the main SCC, recorded below.
    out={'city':city,'crs':f'EPSG:{C["cities"][city]["epsg"]}','source_crs':'EPSG:4326','osm_timestamp':raw.get('osm3s',{}).get('timestamp_osm_base'),
         'nodes':{str(i):{'xy':xy[i],'lat':nodes[i]['lat'],'lon':nodes[i]['lon']} for i in G},'edges':edges,'water':water,'barriers':barrier_lines,'crossings':crossings,'turn_rules':turn_rules,
         'main_scc':sorted(components[0]),'audit':{'nodes':len(G),'directed_edges':len(edges),'scc_sizes':list(map(len,components[:20])),
         'excluded_way_counts':dict(exclusion),'conservatively_blocked_junctions':len(blocked),'restriction_relations':len(restrictions),'node_via_turn_rules':len(turn_rules),'via_way_exclusions':len(via_way)},
         'core':C['cities'][city]['core'],'buffer':C['cities'][city]['buffer']}
    if save:
        dump(DATA/(city+'_network.json'),out);dump(DATA/(city+'_restrictions.json'),restrictions)
    print(city,out['audit'],'crossings',[(x['id'],x['name']) for x in crossings],flush=True)
    return out

def trips():
    p=fetch('202504-capitalbikeshare-tripdata.zip',C['trip_url'])
    audit=Counter();missing=Counter();coords=defaultdict(list);names={};rows=[];seen=set()
    with zipfile.ZipFile(p) as z:
        member=next(x for x in z.namelist() if x.endswith('.csv') and not x.startswith('__MACOSX'))
        reader=csv.DictReader(io.TextIOWrapper(z.open(member),encoding='utf-8-sig'))
        fields=reader.fieldnames
        for r in reader:
            audit['month_rows']+=1
            for k,v in r.items():
                if not v:missing[k]+=1
            if not C['week_start']<=r['started_at'][:10]<C['week_end_exclusive']:continue
            audit['week_rows']+=1
            if r['ride_id'] in seen:audit['duplicate_week_ride_ids']+=1;continue
            seen.add(r['ride_id'])
            if not r['start_station_id'] or not r['end_station_id']:audit['week_missing_station_id']+=1;continue
            try:
                pts=[tuple(float(r[f'{side}_{v}']) for v in ['lat','lng']) for side in ['start','end']]
                if not all(-90<=a<=90 and -180<=b<=180 for a,b in pts):raise ValueError()
            except (ValueError,TypeError):audit['week_bad_coordinates']+=1;continue
            if r['rideable_type']!='classic_bike':audit['week_non_classic']+=1;continue
            a,b=[inside(*pt,C['cities']['washington']['core']) for pt in pts]
            if a!=b:audit['week_core_boundary_trips']+=1
            if not a or not b:audit['week_outside_core']+=1;continue
            if r['start_station_id']==r['end_station_id']:audit['week_same_station']+=1;continue
            for side,pt in zip(['start','end'],pts):
                ident=r[f'{side}_station_id'];coords[ident].append(pt);names[ident]=r[f'{side}_station_name']
            rows.append(r)
    stations=[{'id':k,'name':names[k],'lat':float(np.median(np.array(v)[:,0])),'lon':float(np.median(np.array(v)[:,1])),'endpoint_observations':len(v)} for k,v in coords.items()]
    dump(DATA/'trip_schema_audit.json',{'counts':dict(audit),'fields':fields,'missing_month':dict(missing),
          'source_member':member,'types':{k:('datetime string' if k.endswith('_at') else 'float' if k.endswith(('_lat','_lng')) else 'string') for k in fields},
          'station_metadata':'Median coordinates/name from the selected historical week. No contemporary capacity or status substituted.',
          'map_date_mismatch':'April 2025 trips routed on October 2026 OSM snapshot, not a historical route reconstruction.'})
    return stations,rows

def stations_for(city,net,candidates=None):
    tr=Transformer.from_crs(4326,C['cities'][city]['epsg'],always_xy=True)
    restricted={r['via'] for r in net['turn_rules']}
    ids=sorted(int(i) for i in net['nodes'] if int(i) not in restricted);xy=np.array([net['nodes'][str(i)]['xy'] for i in ids]);tree=cKDTree(xy)
    if candidates is None:
        main=set(net['main_scc']);bridge_nodes={e[k] for e in net['edges'] if e['bridge'] for k in ['u','v']}
        core=[i for i in ids if i in main and i not in bridge_nodes and inside(net['nodes'][str(i)]['lat'],net['nodes'][str(i)]['lon'],C['cities'][city]['core'])]
        pts=np.array([net['nodes'][str(i)]['xy'] for i in core]);chosen=[int(np.argmin(np.linalg.norm(pts-pts.mean(0),axis=1)))]
        while len(chosen)<C['stations']:
            d=np.min(np.linalg.norm(pts[:,None,:]-pts[chosen][None,:,:],axis=2),axis=1);chosen.append(int(np.argmax(d)))
        candidates=[{'id':f'K{j+1:02}','name':f'Hypothetical station {j+1:02}',**{k:net['nodes'][str(core[i])][k] for k in ['lat','lon']}} for j,i in enumerate(chosen)]
    selected=[];snap_audit=[]
    river=unary_union([LineString(w['xy']) for w in net['water'] if w['kind']=='river'])
    water=unary_union([Polygon(w['xy']).buffer(0) for w in net['water'] if w['kind']=='polygon' and len(w['xy'])>=4 and w['xy'][0]==w['xy'][-1]])
    barrier_geom=unary_union([LineString(b['xy']) for b in net['barriers']])
    for s in sorted(candidates,key=lambda s:(-s.get('endpoint_observations',0),s['id'])):
        source=tr.transform(s['lon'],s['lat']);dist,index=tree.query(source)
        connector=LineString([source,xy[index]]);reasons=[]
        if dist>60:reasons.append('connector_over_60m')
        if ids[index] not in net['main_scc']:reasons.append('nearest_legal_node_outside_main_component')
        if dist>.1 and connector.intersects(river):reasons.append('crosses_river_centerline')
        if dist>.1 and connector.intersection(water).length>2:reasons.append('crosses_mapped_water')
        if dist>.1 and connector.intersects(barrier_geom):reasons.append('crosses_mapped_barrier')
        snap_audit.append({'station_id':s['id'],'nearest_node':ids[index],'connector_m':float(dist),'accepted':not reasons,'reasons':reasons,'source_xy':source,'target_xy':xy[index].tolist()})
        if reasons:continue
        selected.append({**s,'node':ids[index],'snap_m':float(dist),'xy':xy[index].tolist(),'capacity':C['capacity'],'initial':C['initial_bikes_per_station'],
                         'observed_location':city=='washington','capacity_inventory_observed':False})
        if len(selected)==C['stations']:break
    dump(DATA/(city+'_station_snap_audit.json'),snap_audit)
    assert len(selected)==C['stations'],(city,len(selected))
    dump(DATA/(city+'_stations.json'),selected);csvout(DATA/(city+'_stations.csv'),[{k:v for k,v in s.items() if k!='xy'} for s in selected])
    return selected

def main():
    stations,rows=trips()
    for city in C['cities']:
        n=build(city,osm(city));ss=stations_for(city,n,stations if city=='washington' else None)
        if city=='washington':
            chosen={s['id'] for s in ss};kept=[r for r in rows if r['start_station_id'] in chosen and r['end_station_id'] in chosen]
            csvout(DATA/'washington_week_requests.csv',kept)
            audit=json.loads((DATA/'trip_schema_audit.json').read_text());audit['counts'].update({'core_classic_valid_rows':len(rows),'selected_station_requests':len(kept),'excluded_not_selected_station':len(rows)-len(kept),'selected_stations':len(ss)})
            dump(DATA/'trip_schema_audit.json',audit)
    print('Prepared cached inputs.',flush=True)

if __name__=='__main__':main()
