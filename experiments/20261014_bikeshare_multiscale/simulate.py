"""Docked bikeshare event simulation. Bikes never drive themselves.

Closures forbid new entry; an already entered arc is cleared. Riders en route
wait at the next closed crossing until reopening. Full returns try nearby docks,
then wait and retry. Every unfinished bike remains in the conservation ledger.
"""
from __future__ import annotations
import argparse,csv,gzip,hashlib,heapq,json,math,time
from collections import Counter,defaultdict
from datetime import datetime
from pathlib import Path
import numpy as np
from prepare import ROOT,DATA,C,dump,csvout
from routing import Router,network,stations,load_model
OUT=ROOT/'output';OUT.mkdir(exist_ok=True)

def fingerprint(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def controlled_tape(city,seed,router,ss):
    rng=np.random.default_rng(seed);n=C['stations']*C['initial_bikes_per_station']
    count=round(n*C['controlled_requests_per_bike_hour']*C['controlled_demand_end_s']/3600)
    times=np.sort(rng.uniform(0,C['controlled_demand_end_s'],count));bins=rng.choice(3,count,p=[.35,.45,.20]);uniforms=rng.random(count)
    speed=rng.uniform(.90,1.10,count)
    # All 24 x 23 endpoint candidates, not an enumeration of paths. Paths are
    # single-source cached only as candidates are needed; retain bin support.
    pairs=defaultdict(list)
    for a in ss:
        for b in ss:
            if a['id']==b['id']:continue
            d=math.dist(a['xy'],b['xy'])
            if d<250 or d>6000:continue
            p=router.route(a['node'],b['node'])
            if not p:continue
            k=0 if p['m']<1500 else 1 if p['m']<3000 else 2
            if p['m']<=6000:pairs[k].append((a['id'],b['id'],p['m']))
    assert all(pairs[k] for k in range(3)),(city,{k:len(v) for k,v in pairs.items()})
    requests=[]
    for i,t in enumerate(times):
        candidates=pairs[int(bins[i])]
        # A common cyclic destination preference creates inventory pressure;
        # station identities and geography differ, so this is not local demand.
        target=ss[(int(t//3600)+seed%3)%len(ss)]['id']
        preferred=[p for p in candidates if p[1]==target]
        pool=preferred if preferred and uniforms[i]<.38 else candidates
        a,b,d=pool[min(int((uniforms[i]*2.631%1)*len(pool)),len(pool)-1)]
        requests.append({'id':i,'time':float(t),'origin':a,'destination':b,'speed_factor':float(speed[i]),'target_bin':int(bins[i]),'open_distance_m':d})
    tape={'city':city,'seed':seed,'workload':'controlled','horizon':C['controlled_horizon_s'],'closure_start':C['closure_start_s'],
          'closure_end':C['closure_end_s'],'initial':[s['initial'] for s in ss],'requests':requests,
          'shared_time_bin_speed_sha256':fingerprint([times.tolist(),bins.tolist(),speed.tolist()])}
    tape['sha256']=fingerprint(tape);return tape

def historical_tapes(ss):
    groups=defaultdict(list)
    with (DATA/'washington_week_requests.csv').open(encoding='utf-8') as f:
        for row in csv.DictReader(f):groups[row['started_at'][:10]].append(row)
    out=[]
    for j,(date,rows) in enumerate(sorted(groups.items())):
        midnight=datetime.fromisoformat(date);rng=np.random.default_rng(C['seed_start']+j)
        req=[]
        for r in sorted(rows,key=lambda r:(r['started_at'],r['ride_id'])):
            req.append({'id':len(req),'time':(datetime.fromisoformat(r['started_at'])-midnight).total_seconds(),
                        'origin':r['start_station_id'],'destination':r['end_station_id'],'speed_factor':float(rng.uniform(.9,1.1)),
                        'source_ride_id':r['ride_id']})
        tape={'city':'washington','seed':C['seed_start']+j,'workload':'observed_week','day':date,'horizon':86400,
              'closure_start':8*3600,'closure_end':12*3600,'initial':[s['initial'] for s in ss],'requests':req}
        tape['sha256']=fingerprint(tape);out.append(tape)
    return out

def simulate(net,ss,router,tape,crossing,policy,capture=False):
    start_clock=time.perf_counter();byid={s['id']:s for s in ss};inv={s['id']:v for s,v in zip(ss,tape['initial'])};total=sum(inv.values())
    pending=[];counter=0;events=[];frames=[];rides={};counts=Counter();crew={'station':ss[0]['id'],'load':0,'busy':False,'legs':[],'arrive':0}
    horizon=tape['horizon'];cstart=tape['closure_start'];cend=tape['closure_end'];moves=[];route_checks=0
    last_t=0.;empty_seconds=0.;hidden_shortage_seconds=0.;max_conservation_error=0
    station_groups=getattr(router,'station_groups',{s['id']:s['id'] for s in ss})
    def schedule(t,kind,payload):
        nonlocal counter
        counter+=1;heapq.heappush(pending,(t,counter,kind,payload))
    def log(t,event,station='',quantity=0,rid=''):
        events.append({'time_s':round(t,4),'event':event,'station':station,'quantity':quantity,'request_id':rid})
    def closed(t):return crossing if cstart<=t<cend else None
    def plan(a,b,t,speed):
        nonlocal route_checks
        p=router.route(byid[a]['node'],byid[b]['node'],closed(t))
        if p is None:return None
        arrive,legs,wait=router.itinerary(p,t,speed,crossing,cstart,cend)
        # Validate every expanded directed arc and closure-entry time, not only
        # the macro-edge endpoint or displayed line.
        expected=byid[a]['node']
        for key,depart,finish in legs:
            e=router.edges[key];assert e['u']==expected;expected=e['v']
            assert not (crossing and e['crossing']==crossing and cstart<=depart<cend)
            assert finish>depart;route_checks+=1
        assert expected==byid[b]['node']
        return {'arrive':arrive,'legs':legs,'m':p['m'],'wait':wait}
    def movement(t,rid,a,b,p,kind='rider'):
        if capture:moves.append({'id':rid,'kind':kind,'from':a,'to':b,'start':t,**p})
    def crew_dispatch(t):
        if policy!='cargo_crew' or crew['busy']:return
        if crew['load']:
            targets=sorted((s for s in ss if inv[s['id']]<C['initial_bikes_per_station']),key=lambda s:(inv[s['id']],s['id']))
            mode='unload'
        else:
            if not any(v<=3 for v in inv.values()):return
            targets=sorted((s for s in ss if inv[s['id']]>=11),key=lambda s:(-inv[s['id']],s['id']));mode='load'
        for s in targets:
            p=plan(crew['station'],s['id'],t,C['crew_speed_m_s'])
            if p:
                crew.update(busy=True,legs=p['legs'],arrive=p['arrive']);counts['crew_distance_m']+=p['m'];counts['crew_travel_s']+=p['arrive']-t
                movement(t,'crew',crew['station'],s['id'],p,'crew');schedule(p['arrive'],'crew_arrive',(s['id'],mode));return
    def return_bike(r,t):
        destination=r['current_destination']
        if inv[destination]<byid[destination]['capacity']:
            inv[destination]+=1;r['status']='completed';r['completed_s']=t;counts['completed']+=1
            log(t,'return',destination,1,r['id']);return
        counts['full_station_events']+=1;r['full_events']+=1;log(t,'full_return',destination,0,r['id'])
        candidates=[]
        for s in ss:
            if s['id']==destination or inv[s['id']]>=s['capacity']:continue
            p=router.route(byid[destination]['node'],s['node'],closed(t))
            if p and p['m']<=C['alternative_return_max_m']:candidates.append((p['m'],s['id']))
        if candidates:
            _,dest=min(candidates);p=plan(destination,dest,t,C['rider_speed_m_s']*r['speed_factor']);r['current_destination']=dest;r['distance_m']+=p['m'];r['closure_wait_s']+=p['wait']
            r['alternative_returns']+=1;counts['alternative_legs']+=1;r['status']='in_transit'
            movement(t,r['id'],destination,dest,p);schedule(p['arrive'],'return',r['id'])
        else:
            r['status']='waiting_for_dock';schedule(t+C['return_retry_s'],'return',r['id'])
    for r in tape['requests']:schedule(r['time'],'request',r)
    for t in range(0,horizon+1,300):schedule(t,'crew_tick',None)
    for t in range(0,horizon+1,C['snapshot_s']):schedule(t,'snapshot',None)
    while pending:
        t,seq,kind,payload=heapq.heappop(pending)
        if t>horizon:break
        group_inventory=Counter()
        for sid,bikes in inv.items():group_inventory[station_groups[sid]]+=bikes
        empty_seconds+=(t-last_t)*sum(v==0 for v in inv.values())
        hidden_shortage_seconds+=(t-last_t)*sum(v==0 and group_inventory[station_groups[sid]]>0 for sid,v in inv.items());last_t=t
        if kind=='request':
            counts['requests']+=1;r={**payload,'status':'requested','picked_up_s':None,'completed_s':None,'distance_m':0.,'closure_wait_s':0.,'full_events':0,'alternative_returns':0,'current_destination':payload['destination']};rides[r['id']]=r
            a,b=r['origin'],r['destination'];base=router.route(byid[a]['node'],byid[b]['node'])
            r['open_route_m']=base['m'] if base else None
            if inv[a]==0:
                r['status']='failed_pickup';counts['failed_pickups']+=1;log(t,'failed_pickup',a,0,r['id'])
            else:
                p=plan(a,b,t,C['rider_speed_m_s']*r['speed_factor'])
                if p is None:r['status']='unreachable';counts['unreachable']+=1;log(t,'unreachable',a,0,r['id'])
                else:
                    inv[a]-=1;r.update(status='in_transit',picked_up_s=t,distance_m=p['m'],closure_wait_s=p['wait']);counts['picked_up']+=1
                    log(t,'pickup',a,-1,r['id']);movement(t,r['id'],a,b,p);schedule(p['arrive'],'return',r['id'])
        elif kind=='return':return_bike(rides[payload],t)
        elif kind=='crew_tick':crew_dispatch(t)
        elif kind=='crew_arrive':
            dest,mode=payload;crew['station']=dest
            amount=min(C['crew_capacity']-crew['load'],max(0,inv[dest]-8)) if mode=='load' else min(crew['load'],max(0,8-inv[dest]))
            log(t,'crew_begin_'+mode,dest,amount)
            schedule(t+amount*C['handling_s_per_bike'],'crew_handle',(dest,mode,amount))
        elif kind=='crew_handle':
            dest,mode,planned_amount=payload
            amount=min(C['crew_capacity']-crew['load'],max(0,inv[dest]-8)) if mode=='load' else min(crew['load'],max(0,8-inv[dest]))
            amount=min(amount,planned_amount)
            sign=-1 if mode=='load' else 1;inv[dest]+=sign*amount;crew['load']-=sign*amount
            if mode=='unload':counts['redistributed_bikes']+=amount
            log(t,'crew_'+mode,dest,sign*amount);crew['busy']=False;crew_dispatch(t)
        onboard=sum(r['status'] in ['in_transit','waiting_for_dock'] for r in rides.values())
        assert all(0<=inv[s['id']]<=s['capacity'] for s in ss)
        assert sum(inv.values())+onboard+crew['load']==total
        if kind=='snapshot' and capture:
            frames.append({'t':t,'inventory':[inv[s['id']] for s in ss],'completed':counts['completed'],'requests':counts['requests'],
                'failed_pickups':counts['failed_pickups'],'unreachable':counts['unreachable'],'full_events':counts['full_station_events'],
                'onboard':onboard,'crew_load':crew['load'],'crew_station':crew['station'],'redistributed':counts['redistributed_bikes'],
                'closed':closed(t),'empty_stations':sum(v==0 for v in inv.values())})
    states=Counter(r['status'] for r in rides.values());assert sum(states.values())==len(tape['requests'])
    completed=[r for r in rides.values() if r['status']=='completed'];picked=[r for r in rides.values() if r['picked_up_s'] is not None]
    inflation=[r['distance_m']/r['open_route_m'] for r in picked if r['open_route_m'] and r['open_route_m']>0]
    elapsed=[(r['completed_s']-r['picked_up_s'])/(r['open_route_m']/(C['rider_speed_m_s']*r['speed_factor'])) for r in completed if r['open_route_m']]
    result={'city':net['city'],'workload':tape['workload'],'day':tape.get('day',''),'seed':tape['seed'],'scenario':crossing or 'all_open','policy':policy,'input_sha256':tape['sha256'],
            'requests':len(rides),'picked_up':len(picked),'completed':len(completed),'served_fraction':len(completed)/max(1,len(rides)),
            'failed_pickups':counts['failed_pickups'],'unreachable':counts['unreachable'],'full_station_events':counts['full_station_events'],
            'full_affected_requests':sum(r['full_events']>0 for r in rides.values()),'unfinished':sum(states[k] for k in ['in_transit','waiting_for_dock']),
            'alternative_legs':counts['alternative_legs'],'mean_completed_journey_s':float(np.mean([r['completed_s']-r['picked_up_s'] for r in completed])) if completed else None,
            'mean_finite_route_inflation':float(np.mean(inflation)) if inflation else None,
            'mean_completed_time_inflation':float(np.mean(elapsed)) if elapsed else None,'empty_station_hours':empty_seconds/3600,
            'empty_station_hours_hidden_by_group_totals':hidden_shortage_seconds/3600,
            'redistributed_bikes':counts['redistributed_bikes'],'crew_distance_km':counts['crew_distance_m']/1000,
            'closure_wait_rider_hours':sum(r['closure_wait_s'] for r in picked)/3600,'fleet':total,'capacity':sum(s['capacity'] for s in ss),
            'requests_per_bike_hour':len(rides)/total/(horizon/3600),'route_arcs_validated':route_checks,'wall_s':time.perf_counter()-start_clock}
    assert result['requests']==result['completed']+result['failed_pickups']+result['unreachable']+result['unfinished']
    return result,{'frames':frames,'movements':moves,'events':events,'requests':list(rides.values()),'result':result,'tape':{k:v for k,v in tape.items() if k!='requests'}}

def run(smoke=False):
    destination=ROOT/'smoke' if smoke else OUT;destination.mkdir(exist_ok=True)
    allrows=[];requestrows=[];mapping=[];validation=[]
    for city in C['cities']:
        net=network(city);ss=stations(city);g,_=load_model(city,0);base=Router(net,g)
        tapes=[controlled_tape(city,C['seed_start']+i,base,ss) for i in range(2 if smoke else C['seeds'])]
        usage=Counter()
        for r in tapes[0]['requests']:
            path=base.route(next(s['node'] for s in ss if s['id']==r['origin']),next(s['node'] for s in ss if s['id']==r['destination']))
            usage.update(set(base.edges[k]['crossing'] for k in path['edges'])-{None})
        chosen=[k for k,v in usage.most_common(3)];assert chosen,(city,'no river crossing demand')
        dump(DATA/(city+'_scenarios.json'),{'selected_crossings':chosen,'rule':'Three most used OSM river crossings in the first controlled baseline request tape, before examining closure effects.','baseline_usage':dict(usage)})
        if city=='washington':tapes+=historical_tapes(ss)[:1 if smoke else 7]
        for tape in tapes:dump(destination/'tapes'/f'{city}_{tape["workload"]}_{tape["seed"]}.json',tape)
        for scale in ([0,600] if smoke else C['scales_m']):
            graph,info=load_model(city,scale);router=Router(net,graph)
            labels=json.loads((DATA/f'{city}_scale_{scale}.json').read_text())['node_to_group']
            router.station_groups={s['id']:labels.get(str(s['node']),s['id']) for s in ss}
            for tape in tapes:
                for crossing in [None]+chosen:
                    for policy in C['policies']:
                        capture=tape['workload']=='controlled' and tape['seed']<C['seed_start']+2
                        result,replay=simulate(net,ss,router,tape,crossing,policy,capture)
                        result['scale_m']=scale;allrows.append(result)
                        ident=f'{city}_{tape["workload"]}_{tape["seed"]}_{crossing or "all_open"}_{policy}_{scale}'
                        for r in replay['requests']:requestrows.append({'episode':ident,**r})
                        # Event logs are retained for every episode, not only movies.
                        p=destination/'logs'/f'{ident}.json.gz';p.parent.mkdir(exist_ok=True)
                        with gzip.open(p,'wt',encoding='utf-8') as f:json.dump({'events':replay['events'],'result':result,'initial':tape['initial']},f)
                        if capture:
                            dump(destination/'replays'/f'{ident}.json',replay)
                        validation.append({'episode':ident,'conservation':True,'capacities':True,'closure_arc_checks':result['route_arcs_validated'],'request_accounting':True})
                print('done',city,scale,tape['workload'],tape['seed'],flush=True)
            info={**info,'city':city,'routing_cache_sources':router.queries,'routing_compute_s':router.query_s}
            mapping.append(info)
    csvout(destination/'episode_results.csv',allrows)
    # Union schema: controlled and historical requests have different metadata.
    fields=list(dict.fromkeys(k for r in requestrows for k in r))
    with gzip.open(destination/'request_results.csv.gz','wt',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(requestrows)
    csvout(destination/'model_costs.csv',mapping);dump(destination/'simulation_validation.json',validation)
    print('TOTAL',len(allrows),'episodes',len(requestrows),'request outcomes',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');p.add_argument('--pilot',action='store_true',help='The configured 12 seeds are exploratory pilot runs, never confirmatory.');run(p.parse_args().smoke)
