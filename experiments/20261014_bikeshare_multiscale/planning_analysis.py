"""Matched investment benefits and reduction errors, separate from recovery."""
import csv
import gzip
import json
import math
from collections import defaultdict
import numpy as np
from scipy.stats import rankdata, spearmanr
from prepare import ROOT, C, dump, csvout
from replay_export import gzread, gzwrite
from analyze import audit, read

OUT = ROOT/'output'


def key(r):
    return (r['city'], r['workload'], int(r['seed']), r['policy'], int(r['scale_m']))


def main():
    baseline = read(OUT/'corrected'/'episode_results.csv')
    invest = read(OUT/'infrastructure'/'episode_results.csv')
    assert len(baseline) == 992 and len(invest) == 744
    validations = {'corrected': audit(OUT/'corrected'), 'infrastructure': audit(OUT/'infrastructure')}
    catalog = json.loads((ROOT/'app'/'data'/'catalog.json').read_text())
    catalog = [r for r in catalog if r['version'] != 'investment']
    investment_catalog = json.loads((OUT/'infrastructure'/'catalog.json').read_text())
    catalog += investment_catalog
    dump(ROOT/'app'/'data'/'catalog.json', catalog)
    # Visual station groups follow each rebuilt intervention model, not the old
    # network's partition. Inventories always remain station-specific.
    for city in C['cities']:
        for intervention in ['I1','I2','I3']:
            path=ROOT/'app'/'data'/f'{city}_{intervention}.json.gz'
            net=gzread(path)
            for scale in C['scales_m']:
                model=gzread(OUT/'infrastructure'/'models'/f'{city}_{intervention}_{scale}.json.gz')
                net['groups'][str(scale)]=[model['labels'].get(str(s['node']),s['id']) for s in net['stations']]
            gzwrite(path,net)
    rows = { (r['version'], *key(r), r['scenario']): r for r in catalog }
    opens = {key(r): r for r in baseline if r['scenario'] == 'all_open'}
    pairs = []
    for r in invest + [r for r in baseline if r['scenario'] != 'all_open']:
        recovery = r['scenario'].startswith('B')
        b = opens[key(r)]
        first, second = (r,b) if recovery else (b,r)
        version = 'corrected' if recovery else 'investment'
        br = gzread(ROOT/rows[('corrected', *key(b), 'all_open')]['file'])
        ir = gzread(ROOT/rows[(version, *key(r), r['scenario'])]['file'])
        assert br['tape']['sha256'] == ir['tape']['sha256']
        bm = {q['id']:q for q in br['requests'] if q['status'] == 'completed'}
        im = {q['id']:q for q in ir['requests'] if q['status'] == 'completed'}
        common = sorted(bm.keys() & im.keys())
        change = [(im[i]['completed_s']-im[i]['picked_up_s'])-(bm[i]['completed_s']-bm[i]['picked_up_s']) for i in common]
        row = {k:r[k] for k in ['city','workload','seed','policy','scale_m','input_sha256']}
        row.update(intervention=r['scenario'],comparison='recovery' if recovery else 'investment',
                   baseline_completed=first['completed'],intervention_completed=second['completed'],
                   additional_completed=second['completed']-first['completed'],
                   served_change_pp=100*(second['served_fraction']-first['served_fraction']),
                   requests=r['requests'], common_completed_n=len(common),
                   common_completed_journey_change_s=float(np.mean(change))*(-1 if recovery else 1) if change else None,
                   baseline_unserved=first['failed_pickups']+first['unreachable'],
                   intervention_unserved=second['failed_pickups']+second['unreachable'],
                   baseline_unfinished=first['unfinished'],intervention_unfinished=second['unfinished'],
                   unfinished_change=second['unfinished']-first['unfinished'],
                   failed_pickups_change=second['failed_pickups']-first['failed_pickups'],
                   full_events_change=second['full_station_events']-first['full_station_events'],
                   empty_station_hours_change=second['empty_station_hours']-first['empty_station_hours'],
                   crew_distance_km_change=second['crew_distance_km']-first['crew_distance_km'],
                   redistributed_bikes_change=second['redistributed_bikes']-first['redistributed_bikes'])
        pairs.append(row)
    fields = ['city','workload','comparison','policy','scale_m']
    groups=defaultdict(list)
    for row in pairs:
        groups[tuple(row[f] for f in fields)+(row['intervention'],)].append(row)
    summaries=[]
    metrics=['additional_completed','served_change_pp','common_completed_journey_change_s','baseline_completed',
             'intervention_completed','baseline_unserved','intervention_unserved','baseline_unfinished',
             'intervention_unfinished','failed_pickups_change','full_events_change','empty_station_hours_change',
             'crew_distance_km_change','redistributed_bikes_change','common_completed_n']
    for group, rr in groups.items():
        s=dict(zip(fields+['intervention'],group));s['n']=len(rr)
        for m in metrics:
            values=[r[m] for r in rr if r[m] is not None]
            s[m]=float(np.mean(values)) if values else None
            if m in ['additional_completed','served_change_pp','common_completed_journey_change_s']:
                for label,p in [('p10',.1),('p90',.9)]:
                    s[m+'_'+label]=float(np.quantile(values,p)) if values else None
        summaries.append(s)
    by_condition=defaultdict(list)
    for s in summaries:by_condition[tuple(s[k] for k in fields)].append(s)
    rank_comparisons=[]
    for group, ss in by_condition.items():
        ranks=rankdata(-np.array([s['additional_completed'] for s in ss]).round(10),method='min')
        for s,rank in zip(ss,ranks):s['rank']=int(rank)
    for group,ss in by_condition.items():
        detailed=by_condition[(*group[:4],0)]
        names=sorted(s['intervention'] for s in ss)
        dm={s['intervention']:s for s in detailed};rm={s['intervention']:s for s in ss}
        a=[dm[n]['rank'] for n in names];b=[rm[n]['rank'] for n in names]
        dtop=[n for n in names if dm[n]['rank']==1];rtop=[n for n in names if rm[n]['rank']==1]
        rank_comparisons.append(dict(zip(fields,group)) | {
            'spearman':float(spearmanr(a,b).statistic) if len(set(a))>1 and len(set(b))>1 else None,
            'same_ranking':a==b,'detailed_top':','.join(dtop),'reduced_top':','.join(rtop),
            'same_preferred_set':dtop==rtop,
            'max_mean_benefit_error_trips':max(abs(rm[n]['additional_completed']-dm[n]['additional_completed']) for n in names)})
    detailed_pairs={(r['city'],r['workload'],r['comparison'],r['seed'],r['policy'],r['intervention']):r for r in pairs if r['scale_m']==0}
    errors=[]
    for r in pairs:
        d=detailed_pairs[(r['city'],r['workload'],r['comparison'],r['seed'],r['policy'],r['intervention'])]
        errors.append({**{k:r[k] for k in ['city','workload','comparison','seed','policy','scale_m','intervention']},
                       'detailed_benefit':d['additional_completed'],'estimated_benefit':r['additional_completed'],
                       'benefit_error_trips':r['additional_completed']-d['additional_completed'],
                       'benefit_error_pp':r['served_change_pp']-d['served_change_pp']})
    export=OUT/'planning';export.mkdir(exist_ok=True)
    csvout(export/'matched_benefits.csv',pairs)
    csvout(export/'ranked_benefits.csv',summaries)
    csvout(export/'ranking_agreement.csv',rank_comparisons)
    csvout(export/'benefit_scale_errors.csv',errors)
    csvout(export/'episode_results.csv',[{'version':'corrected',**r} for r in baseline]+[{'version':'investment',**r} for r in invest])
    # Entire new-investment request table; retained/corrected full table is under journeys/.
    request_fields=['episode','city','workload','seed','policy','scale_m','scenario','id','time','origin','destination','status',
                    'picked_up_s','completed_s','distance_m','closure_wait_s','full_events','alternative_returns']
    with gzip.open(export/'investment_request_outcomes.csv.gz','wt',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=request_fields,extrasaction='ignore');w.writeheader()
        for item in investment_catalog:
            d=gzread(ROOT/item['file'])
            for r in d['requests']:w.writerow({**item,'episode':d['episode'],**r})
    report={'title':'Infrastructure interventions ranked by additional completed bike trips',
            'benefits':summaries,'rankings':rank_comparisons,'paired_benefits':pairs,'scale_errors':errors,
            'interventions':json.loads((OUT/'infrastructure'/'interventions.json').read_text()),
            'coverage':json.loads((OUT/'journeys'/'manifest.json').read_text()),'validations':validations,
            'network_audit':json.loads((OUT/'network_validation.json').read_text()),
            'costs':read(OUT/'infrastructure'/'model_costs.csv'),
            'parameters':C,
            'caveats':['April 1-7 2025 observed trips on October 2026 OSM geography; not historical route reconstruction.',
                       'Observed data are completed rentals, not total or denied demand. Station capacities and starting inventories are synthetic.',
                       'Kaliningrad stations and controlled demand are hypothetical. Controlled and observed-week workloads are separate.',
                       'Finite extracts and unresolved via-way restrictions remain; connectors are not field verified.',
                       'Hypothetical links assume unrestricted bicycle access and no capacity bottlenecks. No engineering feasibility or project-cost model.',
                       'Additional completions serve fixed requests, not newly generated demand. Whiskers are p10-p90 across matched runs, not confidence intervals.',
                       'Travel-time deltas use only identical request IDs completed in both arms, per episode. This excludes failures and unfinished trips.',
                       'Original all-open summaries had a no-closure timing bug. They remain auditable but are not used for investment ranking.']}
    dump(ROOT/'app'/'data'/'report.json',report)
    dump(export/'analysis_validation.json',{'passed':True,'investment_pairs':744,'recovery_pairs':744,
            'paired_input_hashes_match':True,'tie_tolerance':1e-10,'primary_metric':'additional_completed',
            'travel_population':'same request IDs completed in both compared arms', 'ledger_audits':validations})
    print('Planning analysis complete. Preferred-set disagreements:')
    for r in rank_comparisons:
        if not r['same_preferred_set']:print(r)


if __name__=='__main__':main()
