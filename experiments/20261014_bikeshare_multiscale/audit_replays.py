"""Independently reconstruct compressed journeys and snapshot/event consistency."""
import csv
import gzip
import json
import math
from collections import Counter
from prepare import ROOT, dump, sha
from replay_export import gzread, compare, identity
from analyze import read


def main():
    summaries_checked=0
    for folder in [ROOT/'output',ROOT/'output/corrected',ROOT/'output/infrastructure']:
        for row in read(folder/'episode_results.csv'):
            saved=gzread(folder/'logs'/(identity(row)+'.json.gz'))['result']
            compare(row,saved);summaries_checked+=1
    catalog=json.loads((ROOT/'app/data/catalog.json').read_text())
    unique={r['file']:r for r in catalog}
    networks={}
    snapshots=movements=events_checked=ambiguous=0
    for index,(file,item) in enumerate(unique.items()):
        netkey=item.get('network',item['city'])
        if netkey not in networks:networks[netkey]=gzread(ROOT/'app/data'/f'{netkey}.json.gz')
        net=networks[netkey];d=gzread(ROOT/file);ss=net['stations'];sindex={s['id']:i for i,s in enumerate(ss)}
        inv=list(d['tape']['initial']);cargo=onboard=0;counts=Counter();ei=0;events=d['events']
        def apply(e):
            nonlocal cargo,onboard,events_checked
            kind=e['event'];counts[kind]+=1
            if kind in ['pickup','return','crew_load','crew_unload']:inv[sindex[e['station']]]+=e['quantity']
            if kind=='pickup':onboard+=1
            if kind=='return':onboard-=1
            if kind in ['crew_load','crew_unload']:cargo-=e['quantity']
            assert sum(inv)+onboard+cargo==d['result']['fleet']
            assert all(0<=v<=s['capacity'] for v,s in zip(inv,ss))
            events_checked+=1
        def matches(f):
            return inv==f['inventory'] and cargo==f['crew_load'] and onboard==f['onboard'] and counts['return']==f['completed'] and counts['failed_pickup']==f['failed_pickups']
        for f in d['frames']:
            while ei<len(events) and events[ei]['time_s']<f['t']-.00011:apply(events[ei]);ei+=1
            ok=matches(f);boundary=0
            while ei<len(events) and events[ei]['time_s']<=f['t']+.00011:
                apply(events[ei]);ei+=1;boundary+=1;ok=ok or matches(f)
            assert ok,(file,f['t'],'snapshot inconsistent with any timestamp-order prefix')
            if boundary:ambiguous+=1
            snapshots+=1
        while ei<len(events):apply(events[ei]);ei+=1
        assert counts['return']==d['result']['completed'] and onboard==d['result']['unfinished']
        assert len(d['requests'])==d['result']['requests']
        for m in d['movements']:
            path=d['paths'][m['path']];time=m['start'];waits=dict(m['waits']);distance=0
            if path:
                start=net['nodes'][net['edges'][path[0]][0]]
                finish=net['nodes'][net['edges'][path[-1]][1]]
                assert math.dist(start,ss[sindex[m['from']]]['xy'])<.001
                assert math.dist(finish,ss[sindex[m['to']]]['xy'])<.001
            for j,k in enumerate(path):
                e=net['edges'][k]
                if j:assert net['edges'][path[j-1]][1]==e[0]
                time+=waits.get(j,0)
                if d['result']['scenario'].startswith('B') and e[3]==d['result']['scenario']:
                    assert not d['tape']['closure_start']<=time<d['tape']['closure_end'] or abs(time-d['tape']['closure_end'])<1e-6, (file,time,m['id'])
                time+=e[2]/m['speed'];distance+=e[2]
            assert abs(time-m['arrive'])<1e-5
            assert math.isclose(distance,m['m'],abs_tol=1e-6)
            movements+=1
        if index%300==0:print('Journey audit',index,'/',len(unique),flush=True)
    manifest=json.loads((ROOT/'output/journeys/manifest.json').read_text())
    for file,expected in manifest['sources'].items():
        assert sha(ROOT/file)==expected,('retained source changed',file)
    result={'passed':True,'catalog_views':len(catalog),'unique_complete_replays':len(unique),'snapshot_checks':snapshots,
            'summary_csv_rows_checked_against_logs':summaries_checked,
            'movement_checks':movements,'events_checked':events_checked,'timestamp_boundary_snapshots':ambiguous,
            'boundary_rule':'Snapshots match an event-order prefix at timestamps rounded to 0.0001 s. UI uses right-continuous ledger state.',
            'retained_source_hashes_unchanged':True,'all_planned_legs_reconstructed':True}
    dump(ROOT/'output/planning/journey_validation.json',result)
    quality={}
    for name in ['episode_results','matched_benefits','ranked_benefits','benefit_scale_errors']:
        path=ROOT/'output/planning'/f'{name}.csv'
        with path.open(encoding='utf-8') as f:rows=list(csv.DictReader(f))
        fields={}
        for k in rows[0]:
            vals=[r[k] for r in rows];present=[v for v in vals if v!=''];numeric=True
            try:nums=[float(v) for v in present]
            except ValueError:numeric=False
            fields[k]={'type':'number' if numeric else 'string','missing':len(vals)-len(present)}
            if numeric and present:fields[k].update(min=min(nums),max=max(nums))
        quality[name]={'rows':len(rows),'fields':fields}
    dump(ROOT/'output/planning/data_quality.json',quality)
    print(result,flush=True)


if __name__=='__main__':main()
