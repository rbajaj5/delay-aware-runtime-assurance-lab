"""Freeze a dashboard-native walkthrough with exact replay configurations."""
import json
from prepare import ROOT,dump
from replay_export import gzread
from make_movie import OUT


def main():
    OUT.mkdir(exist_ok=True)
    base=dict(city='washington',workload='controlled',seed='2026101400',policy='no_redistribution',evidence='corrected',comparison='investment',scenario='I1',scale='0')
    catalog=json.loads((ROOT/'app/data/catalog.json').read_text())
    item=next(r for r in catalog if r['version']=='investment' and r['city']=='washington' and r['workload']=='controlled' and r['seed']==2026101400 and r['scenario']=='I1' and r['policy']=='no_redistribution' and r['scale_m']==0)
    replay=gzread(ROOT/item['file']);net=gzread(ROOT/'app/data/washington_I1.json.gz')
    journey=next(m for m in replay['movements'] if m['kind']=='rider' and any((net['edges'][k][3] or '').startswith('H') for k in replay['paths'][m['path']]))
    origin=next(i for i,s in enumerate(net['stations']) if s['id']==journey['from'])
    scenes=[
      dict(start=0,end=15,settings=base|{'scenario':'all_open'},time_start=0,time_end=3600),
      dict(start=15,end=40,settings=base,time_start=0,time_end=14400),
      dict(start=40,end=60,settings=base,time_start=max(0,journey['start']-100),time_end=min(14400,journey['arrive']+180),route=journey['id'],station=origin),
      dict(start=60,end=85,settings=base|{'comparison':'scale','scale':'200'},time_start=0,time_end=14400),
      dict(start=85,end=110,settings=base|{'comparison':'scale','scale':'1400'},time_start=0,time_end=14400),
      dict(start=110,end=135,settings=base|{'city':'kaliningrad','policy':'cargo_crew','comparison':'scale','scale':'200'},time_start=0,time_end=14400),
      dict(start=135,end=150,settings=base|{'policy':'cargo_crew','comparison':'recovery','scenario':'B04'},time_start=3200,time_end=10000),
      dict(start=150,end=165,settings=base|{'workload':'observed_week','seed':'2026101403'},time_start=0,time_end=86400),
      dict(start=165,end=180,settings=base|{'scenario':'B04','scale':'200'},lab_mode='new',closure_start_min=70,closure_duration_min=40,initial_bikes=7,time_start=3500,time_end=8000)]
    texts=[
      (0,7,'Which infrastructure changes improve bike service, and does a simplified model make the same planning choice?'),
      (7,15,'We start with the existing network. The original seven-bridge graph introduces topology, but ordinary bike trips do not require an Euler tour.'),
      (15,27,'Now enable the fixed northern bicycle link. Purple shows proposed infrastructure. The baseline and intervention share exactly the same demand tape, bicycles, and operating policy.'),
      (27,40,'The ranking uses twelve matched seeds, while the maps show one saved episode. This link adds about zero point five eight completed trips on average, with substantial variation.'),
      (40,50,'Select a journey that actually uses the new connection. Blue marks its baseline path; purple marks its intervention path. The moving dots follow recorded route timing.'),
      (50,60,'The selected origin station has a linked inventory trace. A shorter journey can change return timing, then change which later riders find a bicycle or an empty dock.'),
      (60,72,'Change routing resolution to two hundred metres. The left map remains detailed; the right uses the reduced network. Both contain the same proposed infrastructure.'),
      (72,85,'For this candidate, the mean benefit changes sign, from plus zero point five eight to minus zero point five eight trips. The highest mean now belongs to another candidate.'),
      (85,98,'At fourteen hundred metres the distortion is much larger. The reduction preserves directed route connectivity but sends some trips through retained hubs, producing long detours.'),
      (98,110,'The maps and completion curves expose the consequences. Washington mean route-length error is about sixty-nine percent at this resolution. Poor simplifications remain in the comparison.'),
      (110,122,'In Kaliningrad, stations and demand are hypothetical. With a cargo crew, the western link adds about one point nine two trips with detailed routing, but zero point one seven at two hundred metres.'),
      (122,135,'The highest-mean intervention changes here too. Another fixed candidate is unused by these station routes and produces zero benefit. We do not substitute a more successful candidate.'),
      (135,150,'Reopening an existing crossing is shown separately, against its disrupted baseline. This is recovery, not investment. Extra completions mean serving fixed requests, not generating new real-world demand.'),
      (150,165,'The observed Washington workload covers seven April days, not months. Recorded rentals omit denied demand, and many benefits are tied. The whiskers describe variation across runs, not confidence intervals.'),
      (165,173,'New-scenario mode calculates a matched pair with changed closure timing and inventory. Pending settings never reuse old trajectories.'),
      (173,180,'Original results remain auditable. These small, variable effects are sensitivity evidence, not reliable construction recommendations.')]
    segments=[dict(start_s=a,end_s=b,text=s) for a,b,s in texts];dump(OUT/'narration_segments.json',segments)
    captions=[]
    for a,b,s in texts:
        words=s.split();start=a
        for i in range(0,len(words),15):
            chunk=words[i:i+15];end=start+(b-a)*len(chunk)/len(words);captions.append(dict(start_s=start,end_s=end,text=' '.join(chunk)));start=end
    dump(OUT/'captions.json',captions)
    def stamp(s):
        ms=round(s*1000);return f'{ms//3600000:02}:{ms//60000%60:02}:{ms//1000%60:02},{ms%1000:03}'
    (OUT/'captions.srt').write_text('\n\n'.join(f'{i+1}\n{stamp(c["start_s"])} --> {stamp(c["end_s"])}\n{c["text"]}' for i,c in enumerate(captions))+'\n',encoding='utf-8')
    dump(OUT/'walkthrough.json',{'duration_s':180,'output_fps':24,'dashboard_capture_hz':12,'resolution':[1920,1080],
         'scenes':scenes,'example_selection':'First controlled seed; first route using I1; April 4 observed day. Not best-outcome selection.',
         'source':'Actual local dashboard, compact demonstration layout. Each captured state appears for two output frames.',
         'saved_replay_coverage':'All 992 retained, 992 corrected-view, 744 investment-view episodes have complete journey exports.',
         'url':'http://127.0.0.1:8766/'})
    print('Dashboard walkthrough prepared; selected journey',journey['id'],flush=True)


if __name__=='__main__':main()
