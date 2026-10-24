"""Independent event-ledger checks and paired multiscale comparisons."""
import argparse,csv,gzip,json,math
from collections import Counter,defaultdict
import numpy as np
from scipy.stats import spearmanr
from prepare import ROOT,C,DATA,dump,csvout
from routing import stations

METRICS=['served_fraction','failed_pickups','unreachable','full_station_events','empty_station_hours','mean_completed_time_inflation','mean_completed_journey_s','crew_distance_km','empty_station_hours_hidden_by_group_totals','unfinished']
def read(path):
    with path.open(encoding='utf-8') as f:rows=list(csv.DictReader(f))
    for r in rows:
        for k,v in r.items():
            if k not in ['city','workload','day','scenario','policy','input_sha256']:
                try:r[k]=float(v) if v else None
                except ValueError:pass
    return rows

def audit(out):
    checked=0;event_count=0;paired=defaultdict(set);snapshots=0
    for file in sorted((out/'logs').glob('*.json.gz')):
        with gzip.open(file,'rt',encoding='utf-8') as f:d=json.load(f)
        res=d['result'];ss=stations(res['city']);inv={s['id']:v for s,v in zip(ss,d['initial'])};crew=active=0;counts=Counter();total=sum(inv.values());handling=None
        for e in d['events']:
            event_count+=1;kind=e['event'];sid=e['station'];q=e['quantity'];counts[kind]+=1
            if kind in ['pickup','return','crew_load','crew_unload']:inv[sid]+=q
            if kind=='pickup':active+=1
            if kind=='return':active-=1
            if kind in ['crew_load','crew_unload']:crew-=q
            if kind in ['crew_begin_load','crew_begin_unload']:handling=(e['time_s'],e['quantity'])
            if kind in ['crew_load','crew_unload'] and handling:
                assert abs(q)<=handling[1]
                assert e['time_s']-handling[0]>=abs(q)*C['handling_s_per_bike']-1e-3
            assert all(0<=inv[s['id']]<=s['capacity'] for s in ss),(file,e)
            assert 0<=crew<=C['crew_capacity'] and active>=0
            assert sum(inv.values())+crew+active==total
        assert counts['return']==res['completed'];assert counts['pickup']==res['picked_up']
        assert counts['failed_pickup']==res['failed_pickups'];assert counts['unreachable']==res['unreachable'];assert active==res['unfinished']
        assert counts['full_return']==res['full_station_events']
        assert res['completed']+res['failed_pickups']+res['unreachable']+res['unfinished']==res['requests']
        paired[(res['city'],res['workload'],res['seed'])].add(res['input_sha256']);checked+=1
    assert all(len(v)==1 for v in paired.values())
    for file in (out/'replays').glob('*.json'):
        d=json.loads(file.read_text(encoding='utf-8'));res=d['result'];events=d['events'];ss=stations(res['city']);inv={s['id']:v for s,v in zip(ss,d['tape']['initial'])};counts=Counter();i=0;crew=0
        for frame in d['frames']:
            while i<len(events) and events[i]['time_s']<=frame['t']+1e-5:
                e=events[i];kind=e['event'];counts[kind]+=1
                if kind in ['pickup','return','crew_load','crew_unload']:inv[e['station']]+=e['quantity']
                if kind in ['crew_load','crew_unload']:crew-=e['quantity']
                i+=1
            assert frame['inventory']==[inv[s['id']] for s in ss],(file,frame['t'])
            assert frame['completed']==counts['return'];assert frame['failed_pickups']==counts['failed_pickup']
            assert frame['onboard']==counts['pickup']-counts['return'];assert frame['crew_load']==crew;snapshots+=1
    validation={'passed':True,'episodes':checked,'ledger_events':event_count,'paired_inputs':len(paired),'replay_snapshots':snapshots,
                'checks':['station capacity after every event','bike conservation including riders and cargo','request outcome accounting','same tape across policies/closures/scales','saved frames match independent event reconstruction']}
    dump(out/'independent_validation.json',validation);print(validation,flush=True)
    return validation

def main(directory):
    out=ROOT/directory;validation=audit(out);rows=read(out/'episode_results.csv')
    assert len(rows)==validation['episodes']
    base={(r['city'],r['workload'],r['seed'],r['scenario'],r['policy']):r for r in rows if r['scale_m']==0}
    groups=defaultdict(list);errors=[];summary=[]
    for r in rows:
        key=(r['city'],r['workload'],r['scenario'],r['policy'],r['scale_m']);groups[key].append(r)
        b=base[(r['city'],r['workload'],r['seed'],r['scenario'],r['policy'])]
        for m in METRICS:
            if r.get(m) is not None and b.get(m) is not None:errors.append({**dict(zip(['city','workload','scenario','policy','scale_m'],key)),'seed':r['seed'],'metric':m,'detailed':b[m],'reduced':r[m],'error':r[m]-b[m]})
    for key,rr in groups.items():
        for m in METRICS:
            vals=[r[m] for r in rr if r.get(m) is not None]
            if not vals:continue
            summary.append({**dict(zip(['city','workload','scenario','policy','scale_m'],key)),'metric':m,'n':len(vals),'mean':float(np.mean(vals)),'sd':float(np.std(vals,ddof=1)) if len(vals)>1 else 0,'p10':float(np.quantile(vals,.1)),'p90':float(np.quantile(vals,.9))})
    errorgroups=defaultdict(list)
    for e in errors:errorgroups[(e['city'],e['workload'],e['scenario'],e['policy'],e['scale_m'],e['metric'])].append(e['error'])
    esummary=[]
    for key,vals in errorgroups.items():esummary.append({**dict(zip(['city','workload','scenario','policy','scale_m','metric'],key)),'n':len(vals),'mean_error':float(np.mean(vals)),'mae':float(np.mean(np.abs(vals))),'max_absolute_error':float(max(np.abs(vals)))})
    opens={(r['city'],r['workload'],r['seed'],r['policy'],r['scale_m']):r for r in rows if r['scenario']=='all_open'}
    rankgroups=defaultdict(lambda:defaultdict(list));ranking=[];rankcomparison=[]
    for r in rows:
        if r['scenario']=='all_open':continue
        key=(r['city'],r['workload'],r['policy'],r['scale_m']);b=opens[(r['city'],r['workload'],r['seed'],r['policy'],r['scale_m'])]
        rankgroups[key][r['scenario']].append(b['served_fraction']-r['served_fraction'])
    from scipy.stats import rankdata
    ranked={}
    for key,bridges in rankgroups.items():
        names=sorted(bridges);means=[float(np.mean(bridges[b])) for b in names];ranks=rankdata(-np.array(means),method='average')
        ranked[key]={b:(v,float(rank)) for b,v,rank in zip(names,means,ranks)}
        for b,v,rank in zip(names,means,ranks):ranking.append({**dict(zip(['city','workload','policy','scale_m'],key)),'crossing':b,'mean_completion_loss_pp':100*v,'rank':rank,'n':len(bridges[b])})
    for key,rank in ranked.items():
        ref=ranked[(*key[:3],0)];names=sorted(ref);a=[ref[n][1] for n in names];b=[rank[n][1] for n in names]
        rho=float(spearmanr(a,b).statistic) if len(set(a))>1 and len(set(b))>1 else None
        rankcomparison.append({**dict(zip(['city','workload','policy','scale_m'],key)),'crossings':len(names),'spearman_rank_correlation':rho,
                               'same_ranking':a==b,'detailed_top':','.join(n for n in names if ref[n][1]==min(a)), 'reduced_top':','.join(n for n in names if rank[n][1]==min(b))})
    csvout(out/'summary.csv',summary);csvout(out/'paired_scale_errors.csv',errors);csvout(out/'scale_error_summary.csv',esummary)
    csvout(out/'closure_rankings.csv',ranking);csvout(out/'closure_rank_agreement.csv',rankcomparison)
    print('Analyzed',len(rows),'episodes; rankings',rankcomparison,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--directory',default='output');main(p.parse_args().directory)
