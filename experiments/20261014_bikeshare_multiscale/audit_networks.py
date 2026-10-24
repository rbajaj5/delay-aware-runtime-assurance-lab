"""Quantify preparation assumptions before interpreting graph disconnections."""
import json,time,copy
from collections import Counter
import networkx as nx
import numpy as np
from prepare import C,DATA,ROOT,dump,csvout,build,osm,inside
from routing import network,stations,detailed_graph,Router,load_model,turn_allowed

def costs(net,ss,g):
    router=Router(net,g);rows={};start=time.perf_counter()
    for a in ss:
        for b in ss:
            if a['id']==b['id']:continue
            r=router.route(a['node'],b['node'])
            rows[(a['id'],b['id'])]=None if r is None else r['m']
    return rows,time.perf_counter()-start

def main():
    out=ROOT/'output';out.mkdir(exist_ok=True);allrows=[];summary=[];mapping_checks=[]
    for city in C['cities']:
        n=network(city);ss=stations(city);g=detailed_graph(n);base,wall=costs(n,ss,g)
        variants=[('turn_aware',n,g)]
        removed=copy.deepcopy(n);via={r['via'] for r in n['turn_rules']};removed['edges']=[e for e in n['edges'] if e['u'] not in via and e['v'] not in via]
        variants.append(('legacy_whole_junction_removal',removed,detailed_graph(removed)))
        relaxed=build(city,osm(city),include_via_ways=True,save=False)
        variants.append(('via_way_exclusion_relaxed_upper_bound',relaxed,detailed_graph(relaxed)))
        core=n['core'];outer=n['buffer'];inner=[(a+b)/2 for a,b in zip(core,outer)]
        keep={int(k) for k,v in n['nodes'].items() if inside(v['lat'],v['lon'],inner)}
        assert all(s['node'] in keep for s in ss)
        variants.append(('half_width_routing_buffer',n,detailed_graph(n,allowed=keep)))
        for name,nn,gg in variants:
            co,timing=(base,wall) if name=='turn_aware' else costs(nn,ss,gg)
            changes=[];lost=gained=0
            for pair,a in base.items():
                b=co[pair];lost+=a is not None and b is None;gained+=a is None and b is not None
                if a is not None and b is not None and a>0:changes.append(b/a-1)
                allrows.append({'city':city,'variant':name,'origin':pair[0],'destination':pair[1],'base_m':a,'variant_m':b})
            row={'city':city,'variant':name,'station_od_pairs':len(co),'unreachable':sum(v is None for v in co.values()),'lost_vs_turn_aware':lost,'gained_vs_turn_aware':gained,
                 'changed_finite_routes':sum(abs(v)>1e-7 for v in changes),'mean_finite_relative_change':float(np.mean(changes)),'max_finite_relative_change':max(changes),
                 'graph_nodes':len(gg),'graph_edges':gg.number_of_edges(),'cold_all_selected_station_od_s':timing}
            summary.append(row);print(row,flush=True)
        for scale in C['scales_m']:
            graph,info=load_model(city,scale);co,timing=costs(n,ss,graph)
            physical={e['id']:e for e in n['edges']};rules={}
            for r in n['turn_rules']:rules.setdefault(r['via'],[]).append(r)
            checked=0
            for u,v,k,d in graph.edges(keys=True,data=True):
                es=[physical[p] for p in d['path']]
                assert abs(sum(e['m'] for e in es)-d['weight'])<1e-6
                for a,b in zip(es,es[1:]):assert a['v']==b['u'] and turn_allowed(a,b,rules.get(a['v'],[]))
                crossings={e['crossing'] for e in es}-{None}
                if crossings:assert len(es)==1 and d['crossing'] in crossings
                checked+=len(es)
            vals=[]
            for pair,a in base.items():
                b=co[pair]
                if a is not None and b is not None:
                    assert b>=a-1e-5
                    if a>0:vals.append(b/a-1)
                allrows.append({'city':city,'variant':f'scale_{scale}','origin':pair[0],'destination':pair[1],'base_m':a,'variant_m':b})
            row={'city':city,'scale_m':scale,'station_od_pairs':len(base),'unreachable':sum(x is None for x in co.values()),
                 'newly_unreachable':sum(base[p] is not None and co[p] is None for p in base),'mean_route_error_pct':100*float(np.mean(vals)),
                 'p95_route_error_pct':100*float(np.quantile(vals,.95)),'max_route_error_pct':100*max(vals),
                 'nodes':len(graph),'edges':graph.number_of_edges(),'build_s':info['build_s'],'cold_routing_s':timing,'expanded_arcs_checked':checked}
            mapping_checks.append(row);print(row,flush=True)
    csvout(out/'network_assumption_sensitivity.csv',summary);csvout(out/'routing_scale_errors.csv',mapping_checks);csvout(out/'all_station_od_audit.csv',allrows)
    dump(out/'network_validation.json',{'passed':True,'variants':summary,'scales':mapping_checks,
         'note':'Via-way relaxed is an optimistic sensitivity bound, not legally validated routing. Half-width buffer is compared with the full downloaded buffer. Station connectors are map-screened, not field verified.'})

if __name__=='__main__':main()
