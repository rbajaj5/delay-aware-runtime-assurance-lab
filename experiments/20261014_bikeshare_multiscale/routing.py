"""Connected street boxes, directed hub skeletons and exact arc expansion."""
import json, math, time
from collections import defaultdict
import networkx as nx
from prepare import C,DATA,dump,csvout,sha

def network(city):return json.loads((DATA/(city+'_network.json')).read_text(encoding='utf-8'))
def stations(city):return json.loads((DATA/(city+'_stations.json')).read_text(encoding='utf-8'))

def turn_allowed(incoming,outgoing,rules):
    for r in rules:
        if incoming['way']!=r['from']:continue
        matches=outgoing['way'] in r['to']
        if r['value']=='no_u_turn':matches=matches and outgoing['v']==incoming['u']
        if r['value'].startswith('no_') and matches:return False
        if r['value'].startswith('only_') and not matches:return False
    return True

def detailed_graph(net,allowed=None,turns=True):
    g=nx.MultiDiGraph()
    rules=defaultdict(list)
    if turns:
        for r in net.get('turn_rules',[]):rules[r['via']].append(r)
    incoming=defaultdict(list)
    usable=[e for e in net['edges'] if allowed is None or (e['u'] in allowed and e['v'] in allowed)]
    for e in usable:incoming[e['v']].append(e)
    states={};owner={int(n):int(n) for n in net['nodes']}
    for node in sorted(rules):
        for e in sorted(incoming[node],key=lambda e:e['id']):
            state=-len(states)-1;states[e['id']]=state;owner[state]=node
    for e in usable:
        sources=[states[p['id']] for p in incoming[e['u']] if turn_allowed(p,e,rules[e['u']])] if e['u'] in rules else [e['u']]
        v=states[e['id']] if e['v'] in rules else e['v']
        for u in sources:g.add_edge(u,v,key=e['id'],weight=e['m'],path=[e['id']],crossing=e['crossing'])
    g.graph['physical_node']=owner
    return g

def partition(net,radius):
    g=nx.Graph();g.add_nodes_from(int(n) for n in net['nodes'])
    for e in net['edges']:
        if not e['bridge']:
            if not g.has_edge(e['u'],e['v']) or g[e['u']][e['v']]['weight']>e['m']:g.add_edge(e['u'],e['v'],weight=e['m'])
    unassigned=set(g);groups={};members=[]
    while unassigned:
        root=min(unassigned)
        found=nx.single_source_dijkstra_path_length(g.subgraph(unassigned),root,cutoff=radius,weight='weight')
        group=set(found);k=len(members);members.append(sorted(group))
        for n in group:groups[n]=k
        unassigned.difference_update(group)
    return groups,members

def construct(net,ss,radius):
    start=time.perf_counter();G=detailed_graph(net)
    if radius==0:return G,{}, {'scale_m':0,'nodes':len(G),'edges':G.number_of_edges(),'build_s':time.perf_counter()-start,'groups':len(G)}
    physical_labels,_=partition(net,radius)
    labels={n:physical_labels[G.graph['physical_node'][n]] for n in G}
    grouped=defaultdict(list)
    for n,k in labels.items():grouped[k].append(n)
    groups=[grouped[k] for k in sorted(grouped)];remap={old:new for new,old in enumerate(sorted(grouped))}
    labels={n:remap[k] for n,k in labels.items()};terminals={s['node'] for s in ss};keep=set()
    for u,v,k,d in G.edges(keys=True,data=True):
        e=next(iter(d['path']))
        if labels[u]!=labels[v] or d['crossing']:
            terminals.update([u,v]);keep.add((u,v,k))
    by_cluster=defaultdict(list)
    for u,v,k in G.edges(keys=True):
        if labels[u]==labels[v]:by_cluster[labels[u]].append((u,v,k))
    # Within each directed SCC retain paths to/from one hub. Reachability among
    # ports survives, but shortest paths need not: this is the approximation.
    for j,group in enumerate(groups):
        sub=G.subgraph(group);comp=list(nx.strongly_connected_components(sub));owner={n:i for i,c in enumerate(comp) for n in c}
        for u,v,k in by_cluster[j]:
            if owner[u]!=owner[v]:keep.add((u,v,k));terminals.update([u,v])
        for nodes in comp:
            sub2=sub.subgraph(nodes);ports=terminals.intersection(nodes)
            if not ports:continue
            hub=min(nodes);fwd=nx.single_source_dijkstra_path(sub2,hub,weight='weight')
            rev=nx.single_source_dijkstra_path(sub2.reverse(copy=False),hub,weight='weight')
            for t in ports:
                for p in [fwd[t],list(reversed(rev[t]))]:
                    for a,b in zip(p,p[1:]):keep.add((a,b,min(sub2[a][b],key=lambda k:(sub2[a][b][k]['weight'],k))))
    H=nx.MultiDiGraph();H.add_nodes_from(terminals)
    for u,v,k,d in G.edges(keys=True,data=True):
        if (u,v,k) in keep:H.add_edge(u,v,key=k,**d)
    # Contract only non-port degree-two chains. All macro-arcs expand into legal
    # directed original arcs, retaining nonzero costs and explicit crossing IDs.
    candidates=list(H);counter=0
    for n in candidates:
        if n not in H or n in terminals:continue
        neighbors=set(H.predecessors(n))|set(H.successors(n))
        if len(neighbors)!=2:continue
        incoming=list(H.in_edges(n,keys=True,data=True));outgoing=list(H.out_edges(n,keys=True,data=True))
        new=[]
        for u,_,k,a in incoming:
            for _,v,l,b in outgoing:
                if u!=v:new.append((u,v,{'weight':a['weight']+b['weight'],'path':a['path']+b['path'],'crossing':None}))
        H.remove_node(n)
        for u,v,d in new:
            counter+=1;H.add_edge(u,v,key=f'm{counter}',**d)
    info={'scale_m':radius,'nodes':len(H),'edges':H.number_of_edges(),'build_s':time.perf_counter()-start,'groups':len(groups),
          'original_arcs_retained':len(keep),'station_inventory_pooled':False}
    return H,labels,info

class Router:
    def __init__(self,net,graph):
        self.net=net;self.graph=graph;self.edges={e['id']:e for e in net['edges']};self.cache={};self.query_s=0.;self.queries=0
        self.views={};self.targets={s['node'] for s in stations(net['city'])};self.pairs={}
    def route(self,a,b,closed=None):
        pair=(a,b,closed)
        if pair in self.pairs:return self.pairs[pair]
        key=(a,closed)
        if key not in self.cache:
            start=time.perf_counter()
            if closed not in self.views:
                self.views[closed]=self.graph if not closed else nx.subgraph_view(self.graph,filter_edge=lambda u,v,k: self.graph[u][v][k]['crossing']!=closed).copy()
            g=self.views[closed]
            if a in g:
                pred,dist=nx.dijkstra_predecessor_and_distance(g,a,weight='weight');paths={}
                for target in self.targets:
                    if target not in dist:continue
                    reverse=[target]
                    while reverse[-1]!=a:reverse.append(min(pred[reverse[-1]]))
                    paths[target]=list(reversed(reverse))
                self.cache[key]=({n:dist[n] for n in paths},paths)
            else:self.cache[key]=({}, {})
            self.query_s+=time.perf_counter()-start;self.queries+=1
        lengths,paths=self.cache[key]
        if b not in paths:self.pairs[pair]=None;return None
        p=paths[b];out=[];g=self.views[closed]
        for u,v in zip(p,p[1:]):out.extend(g[u][v][min(g[u][v],key=lambda k:(g[u][v][k]['weight'],k))]['path'])
        rules=defaultdict(list)
        for rule in self.net.get('turn_rules',[]):rules[rule['via']].append(rule)
        for left,right in zip(out,out[1:]):
            p,q=self.edges[left],self.edges[right]
            assert p['v']==q['u'] and turn_allowed(p,q,rules[q['u']])
        self.pairs[pair]={'m':lengths[b],'edges':out};return self.pairs[pair]
    def itinerary(self,path,departure,speed,crossing,start,end):
        now=departure;legs=[];wait=0
        for k in path['edges']:
            e=self.edges[k]
            if e['crossing']==crossing and start<=now<end:
                wait+=end-now;now=end
            finish=now+e['m']/speed
            legs.append([k,now,finish]);now=finish
        return now,legs,wait

def save_model(city,scale,G,labels,info):
    info.update(model_version=3,network_sha256=sha(DATA/(city+'_network.json')),stations_sha256=sha(DATA/(city+'_stations.json')))
    dump(DATA/f'{city}_scale_{scale}.json',{'info':info,'node_to_group':labels,'arcs':[{'u':u,'v':v,'key':k,**d} for u,v,k,d in G.edges(keys=True,data=True)]})

def load_model(city,scale):
    p=DATA/f'{city}_scale_{scale}.json'
    d=json.loads(p.read_text()) if p.exists() else None
    if not d or d['info'].get('model_version')!=3 or d['info'].get('network_sha256')!=sha(DATA/(city+'_network.json')) or d['info'].get('stations_sha256')!=sha(DATA/(city+'_stations.json')):
        g,labels,info=construct(network(city),stations(city),scale);save_model(city,scale,g,labels,info)
        return g,info
    g=nx.MultiDiGraph()
    for e in d['arcs']:g.add_edge(e['u'],e['v'],key=e['key'],**{k:v for k,v in e.items() if k not in ['u','v','key']})
    return g,d['info']

if __name__=='__main__':
    for city in C['cities']:
        for scale in C['scales_m']:
            g,labels,info=construct(network(city),stations(city),scale);save_model(city,scale,g,labels,info);print(city,info,flush=True)
