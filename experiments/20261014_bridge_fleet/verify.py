"""Independent checks of retained request/event tables, not renderer counters."""
import csv
import gzip
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from simulate import actual_closed, write_csv

HERE=Path(__file__).resolve().parent


def verify(out):
    config=json.loads((out/'config_used.json').read_text())
    scenarios={s['id']:s for s in config['scenarios']};edges={e['id']:e for e in config['edges']}
    rows=list(csv.DictReader((out/'episode_results.csv').open(newline='')))
    trips=defaultdict(list);event_counts=defaultdict(lambda:defaultdict(int));active={};observed_ids=set();edge_exits=0
    with gzip.open(out/'requests.csv.gz','rt',newline='',encoding='utf-8') as f:
        for r in csv.DictReader(f):trips[(r['scenario'],r['seed'],r['policy'])].append(r)
    with gzip.open(out/'events.csv.gz','rt',newline='',encoding='utf-8') as f:
        for r in csv.DictReader(f):
            key=r['scenario'],r['seed'],r['policy'];t=int(r['time_s']);ev=r['event'];event_counts[key][ev]+=1
            identity=key+(r['vehicle'],)
            if ev=='edge_entry':
                assert r['edge'] not in actual_closed(scenarios[r['scenario']],t)
                assert identity not in active
                a,b=r['detail'].split(';')[0].split('->');e=edges[r['edge']]
                assert {a,b}=={e['u'],e['v']}
                active[identity]=(t,r['edge'],int(r['detail'].split('onboard=')[1]))
                observed_ids.add(r['vehicle'])
            elif ev=='edge_exit':
                start,edge,onboard=active.pop(identity)
                assert edge==r['edge']
                moving=sum(edge not in actual_closed(scenarios[r['scenario']],s) for s in range(start,t))
                assert moving==edges[edge]['travel_s'],(identity,start,t,edge,moving)
                event_counts[key]['reconstructed_occupied_s' if onboard else 'reconstructed_empty_s']+=moving
                edge_exits+=1
            elif ev=='closed_entry_attempt':assert r['edge'] in actual_closed(scenarios[r['scenario']],t)
            elif ev=='forced_stop':assert r['edge'] in actual_closed(scenarios[r['scenario']],t)
    for identity,(start,edge,onboard) in active.items():
        key=identity[:3]
        moving=sum(edge not in actual_closed(scenarios[key[0]],s) for s in range(start,config['horizon_s']))
        assert moving<edges[edge]['travel_s']
        event_counts[key]['reconstructed_occupied_s' if onboard else 'reconstructed_empty_s']+=moving
    exposure=[]
    for r in rows:
        key=r['scenario'],r['seed'],r['policy'];group=trips[key];cnt=event_counts[key]
        assert len(group)==int(r['requests'])
        assert len({q['id'] for q in group})==len(group)
        complete=sum(q['state']=='completed' for q in group);waiting=sum(q['state']=='waiting' for q in group);onboard=sum(q['state']=='onboard' for q in group)
        assert complete==int(r['completed']) and waiting==int(r['waiting_at_end']) and onboard==int(r['onboard_at_end'])
        assert complete+waiting+onboard==len(group)
        waits=[float(q['pickup_wait_s']) for q in group if q['pickup_wait_s']!='']
        assert abs(np.mean(waits)-float(r['mean_pickup_wait_s']))<1e-9
        capped=[(float(q['pickup_s']) if q['pickup_s'] else config['horizon_s'])-int(q['arrival_s']) for q in group]
        assert abs(np.mean(capped)-float(r['capped_pickup_wait_s']))<1e-9
        assert cnt['closed_entry_attempt']==int(r['closed_entry_attempts'])
        assert cnt['dropoff']==complete and cnt['pickup']==len(waits)
        assert abs(cnt['reconstructed_empty_s']/60-float(r['empty_travel_vehicle_min']))<1e-9
        assert abs(cnt['reconstructed_occupied_s']/60-float(r['occupied_travel_vehicle_min']))<1e-9
        intents=cnt['edge_entry']+cnt['closed_entry_attempt']
        exposure.append({'scenario':key[0],'seed':key[1],'policy':key[2],'successful_edge_admissions':cnt['edge_entry'],
                         'entry_intents':intents,'closed_attempts_per_100_intents':100*cnt['closed_entry_attempt']/intents if intents else 0})
    write_csv(out/'movement_exposure.csv',exposure)
    for scenario in scenarios:
        for seed in {r['seed'] for r in rows}:
            paired=[r for r in rows if r['scenario']==scenario and r['seed']==seed]
            assert len(paired)==3 and len({r['input_sha256'] for r in paired})==1
    summary_checks=0
    with (out/'summary.csv').open(newline='') as f:
        for s in csv.DictReader(f):
            group=[float(r[s['metric']]) for r in rows if r['scenario']==s['scenario'] and r['policy']==s['policy'] and r[s['metric']]!='']
            expected={'mean':np.mean(group),'median':np.median(group),'sd':np.std(group,ddof=1),'p10':np.quantile(group,.1),'p90':np.quantile(group,.9)}
            for field,value in expected.items():assert abs(float(s[field])-value)<1e-9
            assert int(s['n_episodes'])==len(group);summary_checks+=1
    result={'all_passed':True,'episodes_reconciled':len(rows),'request_rows':sum(map(len,trips.values())),
            'edge_exits_reconstructed':edge_exits,'vehicle_ids_observed':sorted(observed_ids),
            'closure_entry_checks':True,'in_edge_motion_excludes_closed_seconds':True,
            'request_counts_waits_and_travel_match_summaries':True,'matched_tapes_verified':True,
            'summary_rows_independently_recomputed':summary_checks,
            'zero_closed_road_movement_is_a_common_environment_rule_not_a_policy_guarantee':True}
    (out/'independent_validation.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':verify(HERE/'output')
