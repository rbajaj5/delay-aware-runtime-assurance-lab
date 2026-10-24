"""Deterministic 1080p/24fps narrated demo from verified saved-run journeys.

Pillow draws the frames; NVENC encodes them. No generated imagery or simulated
outcomes are inserted by the renderer. Run --prepare, narrate.ps1, then this file.
"""
import argparse
import bisect
import csv
import hashlib
import json
import math
import subprocess
import textwrap
import time
import wave
from functools import lru_cache
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from prepare import ROOT, dump
from replay_export import gzread

OUT=ROOT/'output'/'movie'
W,H,FPS,DURATION=1920,1080,24,180
INK='#24373d';MUTED='#5c737b';TEAL='#0b7a70';BLUE='#3376bd';RED='#c44b3c';PURPLE='#9848a0'


@lru_cache(None)
def font(size,bold=False):
    return ImageFont.truetype('C:/Windows/Fonts/segoeuib.ttf' if bold else 'C:/Windows/Fonts/segoeui.ttf',size)


def text(d,xy,s,size=22,fill=INK,bold=False):d.text(xy,str(s),font=font(size,bold),fill=fill)


def wrap(d,s,width,size=22):
    lines=[];line=''
    for word in str(s).split():
        trial=(line+' '+word).strip()
        if d.textlength(trial,font=font(size))>width and line:lines.append(line);line=word
        else:line=trial
    if line:lines.append(line)
    return lines


def paragraph(d,xy,s,width,size=22,fill=MUTED,gap=6,bold=False):
    x,y=xy
    for line in wrap(d,s,width,size):text(d,(x,y),line,size,fill,bold);y+=size+gap
    return y


def clock(t):return f'{int(t)//3600:02}:{int(t)%3600//60:02}:{int(t)%60:02}'
def signed(v):return f'{v:+.2f}'


SCENES=[
 {'start':0,'end':18,'city':'washington','workload':'controlled','policy':'no_redistribution','scale':0,'scenario':'I1','mode':'investment','title':'From seven bridges to useful infrastructure'},
 {'start':18,'end':45,'city':'washington','workload':'controlled','policy':'no_redistribution','scale':0,'scenario':'I1','mode':'investment','title':'One new link. The same demand. A measured difference.'},
 {'start':45,'end':70,'city':'washington','workload':'controlled','policy':'no_redistribution','scale':200,'scenario':'I1','mode':'scale','title':'A small average route error can still change the ranking'},
 {'start':70,'end':100,'city':'washington','workload':'controlled','policy':'no_redistribution','scale':1400,'scenario':'I1','mode':'scale','title':'Aggressive reduction changes the service dynamics'},
 {'start':100,'end':130,'city':'kaliningrad','workload':'controlled','policy':'cargo_crew','scale':200,'scenario':'I1','mode':'scale','title':'The planning answer also depends on operations'},
 {'start':130,'end':150,'city':'washington','workload':'controlled','policy':'cargo_crew','scale':0,'scenario':'B04','mode':'recovery','title':'Reopening is recovery, not a new investment'},
 {'start':150,'end':165,'city':'washington','workload':'observed_week','policy':'no_redistribution','scale':0,'scenario':'I1','mode':'investment','title':'A real observed week is a different workload'},
 {'start':165,'end':180,'city':'washington','workload':'controlled','policy':'no_redistribution','scale':200,'scenario':'I1','mode':'scale','title':'Simplify only as far as the planning decision survives'}]


def prepare():
    OUT.mkdir(exist_ok=True)
    report=json.loads((ROOT/'app/data/report.json').read_text())
    def metric(city,policy,scale,ident):return next(r for r in report['benefits'] if r['city']==city and r['policy']==policy and r['scale_m']==scale and r['intervention']==ident and r['workload']=='controlled')
    wd=metric('washington','no_redistribution',0,'I1');wr=metric('washington','no_redistribution',200,'I1')
    kd=metric('kaliningrad','cargo_crew',0,'I1');kr=metric('kaliningrad','cargo_crew',200,'I1')
    segments=[
      (0,9,'Which river crossings would help a bikeshare system? And can a smaller network model identify the same useful investments?'),
      (9,18,'Our original seven-bridge graph introduces the question. Its four odd-degree regions prevent an Euler tour. Ordinary bike trips do not require that tour.'),
      (18,30,'Here is Washington. Purple marks a hypothetical new bicycle link, not a construction proposal. Both maps use identical requests, starting bicycles, station capacities, and rider speeds.'),
      (30,45,f'Blue dots are riders. Stations change color as bicycles leave and return. Across twelve matched seeds, this link adds an average of {wd["additional_completed"]:.2f} completed trips. The range across runs is wide.'),
      (45,57,'Now the left map keeps detailed routing. The right uses the two-hundred-metre reduction. Both include exactly the same hypothetical link. The moving cursor follows the current replay time.'),
      (57,70,f'The estimated benefit changes to {wr["additional_completed"]:.2f} trips. The highest-mean candidate also changes. Small average routing errors do not guarantee preservation of a delicate ranking.'),
      (70,84,'At fourteen hundred metres, some reduced paths detour through retained hubs. Longer journeys change the timing of returns, which changes who can find a bicycle or an empty dock.'),
      (84,100,'These counters distinguish the single replay from the twelve-run ranking. The mean routing error in Washington is about sixty-nine percent at this scale. We retain this poor simplification instead of hiding it.'),
      (100,115,f'Kaliningrad uses hypothetical stations and controlled demand. A cargo crew redistributes bicycles. With detailed routing, its western link adds {kd["additional_completed"]:.2f} completed trips on average.'),
      (115,130,f'At two hundred metres that estimate falls to {kr["additional_completed"]:.2f}. Another candidate has the highest mean. A different proposed crossing is unused by these station routes and gives no service improvement.'),
      (130,140,'This is a separate recovery comparison. The left network has a timed closure; the right keeps the existing crossing available. We do not mix reopening gains with investment gains.'),
      (140,150,'Completion is only one endpoint. The dashboard also exposes station shortages, full returns, unfinished trips, cargo effort, and travel times for requests completed in both arms.'),
      (150,165,'Washington also has seven observed days from April twenty twenty-five. These are recorded rentals, not denied demand. They are not monthly observations, and many intervention effects are tied.'),
      (165,173,'Every original episode was replayed and reconciled. A no-closure timing bug was corrected separately. Original results remain available as an audit-only view.'),
      (173,180,'This is a planning sensitivity prototype, not a validated construction recommendation. Use the linked controls to inspect both benefits and failures.')]
    dump(OUT/'narration_segments.json',[{'start_s':a,'end_s':b,'text':s} for a,b,s in segments])
    # Split long narration into readable captions without changing its words.
    captions=[]
    for a,b,s in segments:
        words=s.split();chunks=[words[i:i+14] for i in range(0,len(words),14)];cursor=a
        for chunk in chunks:
            end=cursor+(b-a)*len(chunk)/len(words)
            captions.append({'start_s':cursor,'end_s':end,'text':' '.join(chunk)});cursor=end
    dump(OUT/'captions.json',captions)
    def stamp(s):
        ms=round(s*1000);return f'{ms//3600000:02}:{ms//60000%60:02}:{ms//1000%60:02},{ms%1000:03}'
    (OUT/'captions.srt').write_text('\n\n'.join(f'{i+1}\n{stamp(c["start_s"])} --> {stamp(c["end_s"])}\n{c["text"]}' for i,c in enumerate(captions))+'\n',encoding='utf-8')
    dump(OUT/'scene_plan.json',{'scenes':SCENES,'example_rule':'First controlled seed, or April 4 observed day; not selected for maximum improvement.',
                               'fps':FPS,'seconds':DURATION,'resolution':[W,H],'source':'complete verified saved-run replay'})
    print('Narration and scene plan prepared.',flush=True)


def align_audio():
    segments=json.loads((OUT/'narration_segments.json').read_text());parts=OUT/'narration_parts';audit=[]
    assert json.loads((parts/'narration_source.json').read_text())==segments
    for i,seg in enumerate(segments):
        raw=parts/f'segment_{i:02}.wav';aligned=parts/f'aligned_{i:02}.wav'
        with wave.open(str(raw)) as f:duration=f.getnframes()/f.getframerate()
        slot=seg['end_s']-seg['start_s'];speed=max(1,duration/(slot-.2))
        assert speed<=1.35,(i,duration,slot,'Shorten narration rather than clip it')
        samples=round(slot*48000)
        subprocess.run(['ffmpeg','-y','-v','error','-i',str(raw),'-af',f'atempo={speed:.8f},loudnorm=I=-18:TP=-2:LRA=7,aresample=48000,asetpts=PTS-STARTPTS,apad=whole_len={samples},atrim=end_sample={samples}',
                        '-ac','1','-c:a','pcm_s16le',str(aligned)],check=True)
        audit.append({'segment':i,'source_duration_s':duration,'slot_s':slot,'tempo':speed})
    with wave.open(str(OUT/'narration.wav'),'wb') as f:
        f.setnchannels(1);f.setsampwidth(2);f.setframerate(48000)
        for i in range(len(segments)):
            with wave.open(str(parts/f'aligned_{i:02}.wav'),'rb') as source:f.writeframes(source.readframes(source.getnframes()))
    dump(OUT/'audio_validation.json',{'voice':'Microsoft Zira Desktop','local_automated_voice':True,'speech_truncated':False,'segments':audit})


class Replay:
    def __init__(self,item):
        self.item=item;self.d=gzread(ROOT/item['file']);self.net=gzread(ROOT/'app/data'/f"{item.get('network',item['city'])}.json.gz")
        self.ss=self.net['stations'];self.sid={s['id']:i for i,s in enumerate(self.ss)}
        self.trajectories=[]
        for m in self.d['movements']:
            points=[];starts=[];ends=[];cursor=m['start'];waits=dict(m['waits'])
            for j,k in enumerate(self.d['paths'][m['path']]):
                e=self.net['edges'][k];a=self.net['nodes'][e[0]];b=self.net['nodes'][e[1]]
                if j in waits:starts.append(cursor);ends.append(cursor+waits[j]);points.append((a,a));cursor+=waits[j]
                starts.append(cursor);cursor+=e[2]/m['speed'];ends.append(cursor);points.append((a,b))
            self.trajectories.append((m,starts,ends,points))
        self.times=[-1];inv=list(self.d['tape']['initial']);counts=CounterLike();self.states=[(list(inv),dict(counts))]
        for e in self.d['events']:
            kind=e['event'];counts[kind]=counts.get(kind,0)+1
            if kind in ['pickup','return','crew_load','crew_unload']:inv[self.sid[e['station']]]+=e['quantity']
            if kind in ['crew_load','crew_unload']:counts['cargo']=counts.get('cargo',0)-e['quantity']
            self.times.append(e['time_s']);self.states.append((list(inv),dict(counts)))
    def state(self,t):return self.states[bisect.bisect_right(self.times,t)-1]


class CounterLike(dict):
    def __missing__(self,key):return 0


class Renderer:
    def __init__(self):
        self.report=json.loads((ROOT/'app/data/report.json').read_text());self.catalog=json.loads((ROOT/'app/data/catalog.json').read_text())
        self.captions=json.loads((OUT/'captions.json').read_text());self.capstarts=[c['start_s'] for c in self.captions]
        self.oldgraph=json.loads((ROOT.parent/'20261014_bridge_fleet'/'config.json').read_text())
        self.scene_index=-1;self.cache={}
    def select(self,scene):
        seed=2026101403 if scene['workload']=='observed_week' else 2026101400
        def entry(version,scenario,scale):return next(r for r in self.catalog if r['version']==version and r['city']==scene['city'] and r['workload']==scene['workload'] and r['policy']==scene['policy'] and r['seed']==seed and r['scenario']==scenario and r['scale_m']==scale)
        if scene['mode']=='investment':items=[entry('corrected','all_open',scene['scale']),entry('investment',scene['scenario'],scene['scale'])];labels=['Existing network',f"Hypothetical {scene['scenario']}"]
        elif scene['mode']=='scale':items=[entry('investment',scene['scenario'],0),entry('investment',scene['scenario'],scene['scale'])];labels=[f"{scene['scenario']} / detailed",f"{scene['scenario']} / {scene['scale']} m reduction"]
        else:items=[entry('corrected',scene['scenario'],scene['scale']),entry('corrected','all_open',scene['scale'])];labels=[scene['scenario']+' / disrupted','Existing crossing / open']
        self.pair=[]
        for item in items:
            if item['file'] not in self.cache:self.cache[item['file']]=Replay(item)
            self.pair.append(self.cache[item['file']])
        self.labels=labels;ss=self.pair[0].ss;x,y=zip(*(s['xy'] for s in ss));self.bounds=(min(x)-230,min(y)-230,max(x)+230,max(y)+230)
        self.maps=[self.map_background(r) for r in self.pair]
        self.scene=scene
        self.static=self.make_static(scene)
    def project(self,p):
        x0,y0,x1,y1=self.bounds;s=min(595/(x1-x0),400/(y1-y0));return (315+(p[0]-(x0+x1)/2)*s,215-(p[1]-(y0+y1)/2)*s)
    def map_background(self,r):
        im=Image.new('RGB',(630,430),'#eef3f3');d=ImageDraw.Draw(im)
        for w in r.net['water']:
            pp=[self.project(p) for p in w['xy']]
            if len(pp)<2:continue
            if w['kind']=='polygon':d.polygon(pp,fill='#c9e1e9')
            else:d.line(pp,fill='#96becd',width=2)
        for e in r.net['edges']:
            if e[3] and e[3].startswith('H'):continue
            d.line([self.project(r.net['nodes'][e[0]]),self.project(r.net['nodes'][e[1]])],fill='#c2cfd0',width=1)
        for b in r.net['crossings']:
            x,y=self.project(b['xy'])
            if 0<x<600 and 0<y<410:text(d,(x,y),b['id'],14,PURPLE if b.get('hypothetical') else MUTED)
        text(d,(595,10),'N',14);d.line([(600,45),(600,28)],fill=INK,width=2)
        text(d,(10,407),'OSM / October 2026',13,MUTED)
        return im
    def benefits(self,scale=None):
        s=self.scene;kind='recovery' if s['mode']=='recovery' else 'investment'
        return sorted([r for r in self.report['benefits'] if r['city']==s['city'] and r['workload']==s['workload'] and r['policy']==s['policy'] and r['comparison']==kind and r['scale_m']==(s['scale'] if scale is None else scale)],key=lambda r:(-r['additional_completed'],r['intervention']))
    def make_static(self,s):
        im=Image.new('RGB',(W,H),'#f6f8f8');d=ImageDraw.Draw(im)
        d.rectangle((0,0,W,125),fill='white');text(d,(32,14),'RIVER LINKS / INFRASTRUCTURE PLANNING LAB',20,TEAL,True)
        text(d,(32,47),s['title'],38,INK,True)
        text(d,(34,96),'Synthetic service model on real map context  |  Matched saved-run replay  |  No construction recommendation',18,MUTED)
        name='Washington / Potomac' if s['city']=='washington' else 'Kaliningrad / Pregolya'
        policy='One cargo crew' if s['policy']=='cargo_crew' else 'No redistribution'
        workload='April 4, 2025 recorded rentals' if s['workload']=='observed_week' else 'Controlled demand / seed 2026101400'
        text(d,(32,141),name,24,INK,True);text(d,(400,143),policy+'  /  '+workload,21,MUTED)
        text(d,(1345,143),'AGGREGATE EVIDENCE',20,TEAL,True)
        for j,label in enumerate(self.labels):
            x=30+j*650;d.rectangle((x,183,x+630,681),fill='white');text(d,(x+10,190),label,24,BLUE if j==0 else TEAL,True)
        d.line((1323,180,1323,936),fill='#ccd8da',width=2)
        n=7 if s['workload']=='observed_week' else 12
        heading='Reopenings ranked by additional trips' if s['mode']=='recovery' else 'Infrastructure interventions ranked by additional completed bike trips'
        y=paragraph(d,(1350,185),heading,535,25,INK,bold=True)
        text(d,(1350,y+9),f'Mean over {n} matched '+('days' if n==7 else 'seeds'),18,MUTED)
        data=self.benefits();lo=min(0,min(r['additional_completed_p10'] for r in data));hi=max(1,max(r['additional_completed_p90'] for r in data));xx=lambda v:1485+300*(v-lo)/(hi-lo)
        y0=330
        for i,r in enumerate(data):
            yy=y0+i*70
            text(d,(1355,yy-14),r['intervention'],23,INK,True)
            d.line((xx(0),yy-20,xx(0),yy+20),fill='#bacbcc',width=1)
            d.rectangle((min(xx(0),xx(r['additional_completed'])),yy-11,max(xx(0),xx(r['additional_completed']))+1,yy+11),fill=TEAL if r['additional_completed']>=0 else RED)
            d.line((xx(r['additional_completed_p10']),yy,xx(r['additional_completed_p90']),yy),fill=INK,width=2)
            for v in [r['additional_completed_p10'],r['additional_completed_p90']]:d.line((xx(v),yy-6,xx(v),yy+6),fill=INK,width=2)
            text(d,(1805,yy-14),signed(r['additional_completed']),23,INK,True)
        text(d,(1350,520),'Whiskers: p10-p90 across runs; not a confidence interval.',17,MUTED)
        selected=next(r for r in data if r['intervention']==s['scenario'])
        rows=[('Served fraction',f"{selected['served_change_pp']:+.2f} pp"),
              ('Travel time, common completions',f"{selected['common_completed_journey_change_s']:+.1f} s"),
              ('Empty-station time',f"{selected['empty_station_hours_change']:+.2f} h"),
              ('Full-return events',f"{selected['full_events_change']:+.2f}"),
              ('Cargo distance',f"{selected['crew_distance_km_change']:+.2f} km")]
        for i,(a,b) in enumerate(rows):
            yy=558+i*37;text(d,(1350,yy),a,18,MUTED);text(d,(1780,yy),b,20,INK,True)
        detailed=self.benefits(0);dtop=','.join(r['intervention'] for r in detailed if r['rank']==1);rtop=','.join(r['intervention'] for r in data if r['rank']==1)
        d.rectangle((1345,770,1890,854),fill='#e7f1ed' if dtop==rtop else '#fff0df')
        text(d,(1360,780),'Highest mean benefit (ties retained)',19,INK,True)
        text(d,(1360,812),f'Detailed: {dtop}   |   Selected scale: {rtop}',22,INK)
        paragraph(d,(1350,868),'Small effects, only a few candidates. Ranking sensitivity is not proof that one construction project is superior.',530,18,MUTED)
        d.rectangle((0,947,W,1050),fill='#24373d')
        text(d,(32,1058),'Local automated narration / verified event replay / fixed demand, not newly generated ridership',15,MUTED)
        text(d,(1575,1058),'1920 x 1080 / 24 fps',15,MUTED)
        return im
    def draw_map(self,r,t):
        j=self.pair.index(r);im=self.maps[j].copy();d=ImageDraw.Draw(im);inv,counts=r.state(t)
        closed=r.item['scenario'].startswith('B') and r.d['tape']['closure_start']<=t<r.d['tape']['closure_end']
        for e in r.net['edges']:
            if e[3] and (e[3].startswith('H') or closed and e[3]==r.item['scenario']):
                d.line([self.project(r.net['nodes'][e[0]]),self.project(r.net['nodes'][e[1]])],fill=PURPLE if e[3].startswith('H') else RED,width=4)
        for m,starts,ends,points in r.trajectories:
            if not starts or t<m['start'] or t>m['arrive']:continue
            k=min(bisect.bisect_left(ends,t),len(ends)-1);f=max(0,min(1,(t-starts[k])/(ends[k]-starts[k])));a,b=points[k]
            x,y=self.project([a[0]+(b[0]-a[0])*f,a[1]+(b[1]-a[1])*f]);color='#a45b11' if m['kind']=='crew' else BLUE
            if m['kind']=='crew':d.rectangle((x-4,y-4,x+4,y+4),fill=color)
            else:d.ellipse((x-3,y-3,x+3,y+3),fill=color)
        for i,s in enumerate(r.ss):
            x,y=self.project(s['xy']);color=RED if inv[i]==0 else '#b67c16' if inv[i]==s['capacity'] else TEAL
            d.ellipse((x-5,y-5,x+5,y+5),fill=color,outline='white',width=1)
            text(d,(x+6,y-8),f'{i+1:02}',12,INK)
        return im,inv,counts
    def historical(self,d):
        d.rectangle((1340,180,1900,937),fill='#f6f8f8')
        text(d,(1350,190),'The original seven-bridge graph',26,INK,True)
        coords={k:(1350+v[0]*510,270+v[1]*300) for k,v in self.oldgraph['nodes'].items()}
        seen={}
        for e in self.oldgraph['edges']:
            key=(e['u'],e['v']);i=seen.get(key,0);seen[key]=i+1;a=coords[key[0]];b=coords[key[1]]
            offset=20 if i else -10
            points=[a,((a[0]+b[0])/2+offset,(a[1]+b[1])/2),b];d.line(points,fill=BLUE,width=4)
        for k,(x,y) in coords.items():d.ellipse((x-22,y-22,x+22,y+22),fill=TEAL);text(d,(x-10,y-17),k,25,'white',True)
        paragraph(d,(1350,610),'Degrees: 5, 3, 3, 3. No Eulerian trail. But transport service is not a requirement to traverse every edge exactly once.',520,25,INK)
        paragraph(d,(1350,765),'Each request needs a bicycle, a usable route, and a place to return it.',520,25,TEAL,bold=True)
    def render(self,movie_t):
        si=next(i for i,s in enumerate(SCENES) if s['start']<=movie_t<s['end'])
        if si!=self.scene_index:self.scene_index=si;self.select(SCENES[si])
        s=self.scene;frac=(movie_t-s['start'])/(s['end']-s['start']);simt=self.pair[0].d['tape']['horizon']*frac
        if movie_t>=165:simt=self.pair[0].d['tape']['horizon']*(.30+.65*frac)
        im=self.static.copy();d=ImageDraw.Draw(im);stats=[]
        for j,r in enumerate(self.pair):
            mapim,inv,c=self.draw_map(r,simt);x=30+j*650;im.paste(mapim,(x,236));stats.append(c)
            metrics=[('COMPLETED',c.get('return',0)),('ON BIKES',c.get('pickup',0)-c.get('return',0)),('FAILED PICKUP',c.get('failed_pickup',0)),('FULL RETURNS',c.get('full_return',0))]
            for k,(label,value) in enumerate(metrics):text(d,(x+k*155,692),label,15,MUTED);text(d,(x+k*155,715),value,32,BLUE if j==0 else TEAL,True)
            text(d,(x,757),f"Final episode: {r.item['completed']}/{r.item['requests']} completed; {r.item['unfinished']} unfinished",19,MUTED)
        text(d,(33,802),'COMPLETED TRIPS / MOVING CURSOR = MAP TIME',16,MUTED,True)
        ymax=max(r.item['completed'] for r in self.pair);x0,y0,ww,hh=42,842,1235,82
        for q in [0,.5,1]:d.line((x0,y0+hh*q,x0+ww,y0+hh*q),fill='#d6e0e2',width=1)
        for j,r in enumerate(self.pair):
            points=[(x0+f['t']/r.d['tape']['horizon']*ww,y0+hh*(1-f['completed']/max(1,ymax))) for f in r.d['frames']]
            d.line(points,fill=BLUE if j==0 else TEAL,width=3)
        cursor=x0+simt/self.pair[0].d['tape']['horizon']*ww;d.line((cursor,y0-5,cursor,y0+hh),fill=RED,width=2)
        text(d,(950,797),'REPLAY '+clock(simt),24,INK,True)
        if movie_t<18:self.historical(d)
        cap=self.captions[max(0,bisect.bisect_right(self.capstarts,movie_t)-1)]
        paragraph(d,(42,959),cap['text'],1836,29,'white',gap=5)
        d.rectangle((0,1046,W*movie_t/DURATION,1050),fill=TEAL)
        return im,{'frame':round(movie_t*FPS),'movie_time_s':movie_t,'simulation_time_s':simt,'scene':si,
                   'city':s['city'],'workload':s['workload'],'scenario':s['scenario'],'scale_m':s['scale'],
                   'left_episode':self.pair[0].d['episode'],'right_episode':self.pair[1].d['episode'],
                   'left_completed':stats[0].get('return',0),'right_completed':stats[1].get('return',0),
                   'input_sha256':self.pair[0].d['tape']['sha256']}


def render(preview=False):
    renderer=Renderer()
    if preview:
        for t in [5,35,62,88,120,143,158,176]:
            im,_=renderer.render(t);im.save(OUT/f'preview_{t:03}.png')
        print('Preview frames saved.',flush=True);return
    align_audio()
    silent=OUT/'river_links_silent.mp4';log=(OUT/'encoder.log').open('w')
    args=['ffmpeg','-y','-v','warning','-f','rawvideo','-pix_fmt','rgb24','-s',f'{W}x{H}','-r',str(FPS),'-i','pipe:0',
          '-an','-c:v','h264_nvenc','-preset','p5','-cq','22','-pix_fmt','yuv420p','-movflags','+faststart',str(silent)]
    proc=subprocess.Popen(args,stdin=subprocess.PIPE,stderr=log)
    started=time.perf_counter()
    with (OUT/'frame_index.csv').open('w',newline='',encoding='utf-8') as f:
        writer=None
        for n in range(DURATION*FPS):
            im,record=renderer.render(n/FPS)
            if writer is None:writer=csv.DictWriter(f,fieldnames=list(record));writer.writeheader()
            writer.writerow(record);proc.stdin.write(im.tobytes())
            if n%(FPS*15)==0:
                im.save(OUT/f'frame_{n:05}.png');print(f'Movie {n}/{DURATION*FPS} frames; elapsed {time.perf_counter()-started:.0f}s',flush=True)
    proc.stdin.close();assert proc.wait()==0;log.close()
    target=OUT/'river_links_planning_narrated.mp4'
    subprocess.run(['ffmpeg','-y','-v','error','-i',str(silent),'-i',str(OUT/'narration.wav'),'-i',str(OUT/'captions.srt'),
                    '-map','0:v:0','-map','1:a:0','-map','2:0','-c:v','copy','-c:a','aac','-b:a','160k','-c:s','mov_text',
                    '-metadata:s:s:0','language=eng','-movflags','+faststart',str(target)],check=True)
    probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-show_format','-of','json',str(target)],text=True))
    video=next(s for s in probe['streams'] if s['codec_type']=='video')
    assert (video['width'],video['height'],video['r_frame_rate'],int(video['nb_frames']))==(1920,1080,'24/1',4320)
    assert any(s['codec_type']=='audio' for s in probe['streams'])
    assert any(s['codec_type']=='subtitle' for s in probe['streams'])
    subprocess.run(['ffmpeg','-v','error','-i',str(target),'-f','null','-'],check=True)
    dump(OUT/'movie_validation.json',{'passed':True,'frames':4320,'fps':24,'resolution':[W,H],
            'duration_s':float(probe['format']['duration']),'full_decode':True,'encoder':'h264_nvenc',
            'rendering':'Pillow CPU, NVENC GPU video encoding; simulation uses CPU event routing, not GPU physics',
            'burned_captions':True,'selectable_subtitle_track':True,'audio':'local Microsoft Zira Desktop',
            'movie_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'elapsed_render_s':time.perf_counter()-started})
    print('COMPLETE',target,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--preview',action='store_true');a=p.parse_args()
    if a.prepare:prepare()
    else:render(a.preview)
