"""Bounded local matched-pair runner. Single interventions, no additive effects."""
import copy
import json
import math
import threading
from concurrent.futures import ThreadPoolExecutor
import networkx as nx
from prepare import ROOT, DATA, C, dump, sha
from routing import network, stations, load_model
from corrected_model import CorrectedRouter
from simulate import simulate, fingerprint
from infrastructure import modified
from replay_export import gzread, gzwrite, compact_replay, identity
from analyze import audit

RUNS=ROOT/'output'/'custom_runs'
POOL=ThreadPoolExecutor(max_workers=1)
JOBS={};LOCK=threading.Lock()


def validate(raw):
    allowed={'city','workload','seed','policy','scale_m','intervention','closure_start_s','closure_duration_s','initial_bikes'}
    if set(raw)-allowed:raise ValueError('Unsupported settings: '+','.join(sorted(set(raw)-allowed)))
    c={k:raw[k] for k in ['city','workload','policy','intervention']}
    if c['city'] not in C['cities'] or c['workload'] not in ['controlled','observed_week']:raise ValueError('Unknown city or workload')
    if c['city']=='kaliningrad' and c['workload']=='observed_week':raise ValueError('Observed demand exists only for Washington')
    if c['policy'] not in C['policies']:raise ValueError('Unknown policy')
    for k,default in [('seed',C['seed_start']),('scale_m',0),('closure_start_s',3600),('closure_duration_s',5400),('initial_bikes',8)]:
        v=raw.get(k,default)
        if isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or int(v)!=v:raise ValueError(k+' must be a finite integer')
        c[k]=int(v)
    if c['scale_m'] not in C['scales_m']:raise ValueError('Unknown routing resolution')
    if not 4<=c['initial_bikes']<=12:raise ValueError('Starting inventory must be 4 to 12 bicycles per station')
    file=ROOT/'output/tapes'/f"{c['city']}_{c['workload']}_{c['seed']}.json"
    if not file.is_file():raise ValueError('Select an existing matched tape')
    tape=json.loads(file.read_text())
    valid={'all_open','I1','I2','I3'}|{b['id'] for b in network(c['city'])['crossings']}
    if c['intervention'] not in valid:raise ValueError('Select exactly one predefined intervention or crossing')
    if not 0<=c['closure_start_s']<tape['horizon'] or not 1<=c['closure_duration_s']<=tape['horizon']-c['closure_start_s']:
        raise ValueError('Closure must have positive duration entirely inside the episode horizon')
    c['source_tape_sha256']=tape['sha256']
    c['engine']='corrected_v1';c['schema']=1
    return c,tape


def metric_pair(left,right):
    a,b=left['result'],right['result'];am={r['id']:r for r in left['requests'] if r['status']=='completed'};bm={r['id']:r for r in right['requests'] if r['status']=='completed'}
    ids=am.keys()&bm.keys();times=[(bm[i]['completed_s']-bm[i]['picked_up_s'])-(am[i]['completed_s']-am[i]['picked_up_s']) for i in ids]
    return {'additional_completed':b['completed']-a['completed'],'served_change_pp':100*(b['served_fraction']-a['served_fraction']),
            'common_completed_n':len(ids),'common_completed_time_change_s':sum(times)/len(times) if times else None,
            'baseline_unserved':a['failed_pickups']+a['unreachable'],'intervention_unserved':b['failed_pickups']+b['unreachable'],
            'baseline_unfinished':a['unfinished'],'intervention_unfinished':b['unfinished'],
            'empty_station_hours_change':b['empty_station_hours']-a['empty_station_hours'],
            'full_events_change':b['full_station_events']-a['full_station_events'],
            'crew_distance_km_change':b['crew_distance_km']-a['crew_distance_km']}


def saved_match(c,tape):
    if c['initial_bikes']!=8:return None
    intervention=c['intervention']
    if intervention.startswith('B') and (c['closure_start_s']!=tape['closure_start'] or c['closure_duration_s']!=tape['closure_end']-tape['closure_start']):return None
    cat=json.loads((ROOT/'app/data/catalog.json').read_text())
    rows=[]
    for scenario,version in [('all_open','corrected'),(intervention,'investment' if intervention.startswith('I') else 'corrected')]:
        matches=[r for r in cat if r['version']==version and r['scenario']==scenario and all(r[k]==c[k] for k in ['city','workload','seed','policy','scale_m'])]
        if not matches:return None
        rows.append(matches[0])
    data=[gzread(ROOT/r['file']) for r in rows]
    return {'status':'complete','origin':'exact_saved_match','config':c,'catalog':rows,'metrics':metric_pair(*data),
            'coverage':'One matched pair selected from the completed experiment; no new calculation.'}


def calculate(run_id,c,tape):
    try:
        folder=RUNS/run_id;folder.mkdir(parents=True,exist_ok=True)
        with LOCK:JOBS[run_id].update(status='running',message='Calculating matched baseline and intervention')
        net=network(c['city']);ss=stations(c['city'])
        tape=copy.deepcopy(tape);tape.pop('sha256');tape['closure_start']=c['closure_start_s'];tape['closure_end']=c['closure_start_s']+c['closure_duration_s'];tape['initial']=[c['initial_bikes']]*len(ss);tape['sha256']=fingerprint(tape)
        dump(folder/'config.json',c);dump(folder/'tape.json',tape)
        items=[];data=[]
        for arm,scenario in [('baseline','all_open'),('intervention',c['intervention'])]:
            use=net
            if scenario.startswith('I'):
                specs=json.loads((ROOT/'output/infrastructure/interventions.json').read_text())[c['city']]
                spec=next(s for s in specs if s['id']==scenario);use=modified(net,spec)
                model=gzread(ROOT/'output/infrastructure/models'/f"{c['city']}_{scenario}_{c['scale_m']}.json.gz")
                graph=nx.MultiDiGraph()
                for e in model['arcs']:graph.add_edge(e['u'],e['v'],key=e['key'],**{k:v for k,v in e.items() if k not in ['u','v','key']})
                labels=model['labels']
            else:
                graph,_=load_model(c['city'],c['scale_m']);labels=json.loads((DATA/f"{c['city']}_scale_{c['scale_m']}.json").read_text())['node_to_group']
            router=CorrectedRouter(use,graph);router.station_groups={s['id']:labels.get(str(s['node']),s['id']) for s in ss}
            result,replay=simulate(use,ss,router,tape,scenario if scenario.startswith('B') else None,c['policy'],True)
            result.update(scenario=scenario,scale_m=c['scale_m']);ident=run_id+'_'+arm
            replay=compact_replay(replay,{e['id']:i for i,e in enumerate(use['edges'])},router.edges,'custom',ident,'New bounded calculation; saved matched tape; independently audited event ledger')
            file=folder/(arm+'.json.gz');gzwrite(file,replay)
            gzwrite(folder/'logs'/(arm+'.json.gz'),{'events':replay['events'],'result':result,'initial':tape['initial']})
            items.append({'version':'custom',**result,'file':str(file.relative_to(ROOT)).replace('\\','/'),'network':c['city']+('_'+scenario if scenario.startswith('I') else '')});data.append(replay)
        validation=audit(folder)
        assert validation['episodes']==2 and data[0]['tape']['sha256']==data[1]['tape']['sha256']
        response={'id':run_id,'status':'complete','origin':'new_calculation','config':c,'catalog':items,'metrics':metric_pair(*data),
                  'validation':validation,'coverage':'One new matched seed pair only. No multi-seed uncertainty or multi-scale ranking inferred.',
                  'config_file':str((folder/'config.json').relative_to(ROOT)).replace('\\','/')}
        dump(folder/'result.json',response)
        with LOCK:JOBS[run_id]=response
    except Exception as e:
        with LOCK:JOBS[run_id]={'id':run_id,'status':'failed','message':str(e)}


def submit(raw):
    c,tape=validate(raw)
    c['source_hashes']={name:sha(ROOT/name) for name in ['config.json','simulate.py','routing.py','corrected_model.py','scenario_runner.py']}
    c['source_hashes']['network']=sha(DATA/f"{c['city']}_network.json")
    c['source_hashes']['stations']=sha(DATA/f"{c['city']}_stations.json")
    c['source_hashes']['model']=sha(DATA/f"{c['city']}_scale_{c['scale_m']}.json")
    c['source_hashes']['interventions']=sha(ROOT/'output/infrastructure/interventions.json')
    if c['intervention'].startswith('I'):
        c['source_hashes']['intervention_model']=sha(ROOT/'output/infrastructure/models'/f"{c['city']}_{c['intervention']}_{c['scale_m']}.json.gz")
    saved=saved_match(c,tape)
    if saved:return saved
    rid=fingerprint(c)[:24]
    existing=RUNS/rid/'result.json'
    if existing.exists():return json.loads(existing.read_text())|{'origin':'cached_custom_calculation'}
    with LOCK:
        if rid in JOBS:return JOBS[rid]
        if sum(j['status'] in ['queued','running'] for j in JOBS.values())>=3:raise ValueError('Three scenario jobs already pending; wait for completion')
        JOBS[rid]={'id':rid,'status':'queued','config':c,'message':'Waiting for the single calculation worker'}
        POOL.submit(calculate,rid,c,tape)
        return dict(JOBS[rid])


def status(rid):
    if len(rid)!=24 or any(c not in '0123456789abcdef' for c in rid):raise ValueError('Invalid job ID')
    with LOCK:
        if rid in JOBS:return dict(JOBS[rid])
    existing=RUNS/rid/'result.json'
    if existing.exists():return json.loads(existing.read_text())
    raise ValueError('Unknown job; a server restart may have interrupted it. Submit the same configuration to resume.')
