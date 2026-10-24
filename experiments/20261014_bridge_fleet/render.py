"""Render a three-minute explanatory movie from SAVED outputs only.

Pillow/FFmpeg rendering follows the repository's portfolio_rta_demo.py pattern.
There is no simulation call here. TTS is optional; burned-in captions are always
present, and the same narration text is exported as SRT and a transcript.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image,ImageDraw,ImageFont

HERE=Path(__file__).resolve().parent
BG='#f4f6f7';INK='#20323a';MUTED='#60727b';GRID='#d6e0e4';RED='#c43445';BLUE='#247b9e'
COLORS={'latest_received':'#a66b16','priority_updates':'#087f76','freshness_monitor':'#7953a4'}
SHORT={'latest_received':'Latest received','priority_updates':'Priority updates','freshness_monitor':'Freshness monitor'}


def load(out):
    config=json.loads((out/'config_used.json').read_text());replay=json.loads((out/'replay.json').read_text())
    summary={}
    with (out/'summary.csv').open(newline='') as f:
        for r in csv.DictReader(f):summary[r['scenario'],r['policy'],r['metric']]={k:float(v) if k in ['mean','median','p10','p90','sd'] else v for k,v in r.items()}
    return config,replay,summary


def narrative(config,replay,summary):
    def m(s,p,k):return summary[s,p,k]['mean']
    n=config['seeds_per_scenario'];p='priority_updates';r='freshness_monitor'
    return [
      (0,8,'Autonomous fleets on changing networks. This is a synthetic taxi experiment, inspired by Waymo, not a model of its software.'),
      (8,16,'Euler asked whether every one of seven bridges could be crossed exactly once. These four land areas all have odd degree.'),
      (16,25,'The answer is no. But ordinary passenger journeys remain possible. Taxi routing is a different problem from covering every bridge exactly once.'),
      (25,35,'We use the historical graph as a schematic, not a modern map of Kaliningrad. All demand, travel times, and closures are synthetic.'),
      (35,45,'Six taxis receive passenger requests. Road observations and vehicle reports travel through delayed, capacity-limited queues. Their knowledge can differ from actual conditions.'),
      (45,55,'We compare latest-received dispatch, priority message service, and a freshness monitor. The monitor waits or reroutes using received information, not hidden simulator knowledge.'),
      (55,65,'All three panels use the same clock, requests, closure schedule and delay tape. The replay was selected nearest the median outcomes, not for drama.'),
      (65,75,'Passenger requests are squares beside the nodes. Taxis keep their identities. Solid and hollow circles distinguish occupied and empty vehicles; diamonds indicate waiting.'),
      (75,85,'Links a and b now close. Red dashed lines show physical closures. Amber dots show where the dispatcher still believes a link is open.'),
      (85,95,'Every policy obeys the physical closure. Attempts are blocked. A taxi already on a closing link stops until reopening in this deliberately simplified model.'),
      (95,105,'A detour remains through links e and f. Freshness checks may reject stale information, but they cannot predict every newly closed link.'),
      (105,115,'The links have reopened. Old closure reports can also delay recovery. Compare completed journeys with waiting passengers, not just vehicle motion.'),
      (115,123,'Priority service can restore capacity that stale availability reports leave unused. The freshness monitor adds another constraint, which can cost passenger service.'),
      (123,130,'These are episode results. The next charts summarize repeated runs; a single replay is not the aggregate evidence.'),
      (130,140,f'Each scenario has {n} matched seeds. In the all-open case, priority service completes {100*m("all_open",p,"completion_fraction"):.1f} percent; the monitor completes {100*m("all_open",r,"completion_fraction"):.1f} percent.'),
      (140,150,f'With a detour, the monitor lowers mean blocked entry attempts from {m("detour",p,"closed_entry_attempts"):.2f} to {m("detour",r,"closed_entry_attempts"):.2f}, but completion and waiting performance worsen.'),
      (150,159,'When area B is isolated, topology limits everyone. Short waits among passengers who were picked up can conceal a large uncompleted backlog.'),
      (159,165,'Dots show means. Whiskers show run-to-run variability, not real-world confidence intervals.'),
      (165,173,'The lesson is a service-versus-information tradeoff, not universal monitor superiority. Count uncompleted requests and distinguish fresh information from true information.'),
      (173,180,'Next steps: verify modern roads and test more settings. This is not real taxi evidence or operational driving validation.')]


def save_text(out,segments):
    rows=[{'start_s':start,'end_s':end,'text':text} for start,end,text in segments]
    (out/'narration_segments.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    (out/'transcript.txt').write_text('\n\n'.join(f'{a:03d}-{b:03d} seconds\n{text}' for a,b,text in segments),encoding='utf-8')
    def stamp(t):return f'{int(t)//3600:02d}:{int(t)//60%60:02d}:{int(t)%60:02d},000'
    (out/'captions.srt').write_text('\n\n'.join(f'{i+1}\n{stamp(a)} --> {stamp(b)}\n{text}' for i,(a,b,text) in enumerate(segments))+'\n',encoding='utf-8')


class Movie:
    def __init__(self,out):
        self.out=out;self.c,self.replay,self.summary=load(out);self.segments=narrative(self.c,self.replay,self.summary)
        save_text(out,self.segments);self.fonts={};self.edges={e['id']:e for e in self.c['edges']}
        self.policies=self.c['policies'];self.frames={p:self.replay['policies'][p]['frames'] for p in self.policies}
        self.max_completed=max(f['completed'] for frames in self.frames.values() for f in frames)
        self.max_waiting=max(f['waiting'] for frames in self.frames.values() for f in frames)
        self.metric_specs=[('mean_pickup_wait_s','Pickup wait, among picked-up riders','seconds'),
                           ('completion_fraction','Completed by 15 minutes','% of requests'),
                           ('uncompleted','Uncompleted at 15 minutes','requests'),
                           ('empty_travel_vehicle_min','Empty motion','fleet-minutes'),
                           ('closed_entry_attempts','Blocked entry attempts','attempts / episode'),
                           ('fallback_vehicle_min','Freshness fallback duration','fleet-minutes')]
        self.maxima={metric:max(self.summary[s['id'],p,metric]['p90'] for s in self.c['scenarios'] for p in self.policies)*1.17 for metric,_,_ in self.metric_specs}
        self.maxima['completion_fraction']=1.0
        self.maxima={k:max(v,1.) for k,v in self.maxima.items()}
    def font(self,size,bold=False):
        key=size,bold
        if key not in self.fonts:
            candidates=[Path('C:/Windows/Fonts')/('segoeuib.ttf' if bold else 'segoeui.ttf'),Path('/usr/share/fonts/truetype/dejavu')/('DejaVuSans-Bold.ttf' if bold else 'DejaVuSans.ttf')]
            self.fonts[key]=ImageFont.truetype(str(next(p for p in candidates if p.exists())),size)
        return self.fonts[key]
    def text(self,d,xy,text,size=24,fill=INK,bold=False,anchor=None):d.text(xy,str(text),font=self.font(size,bold),fill=fill,anchor=anchor)
    def wrap(self,d,text,width,size):
        lines=['']
        for word in text.split():
            trial=(lines[-1]+' '+word).strip()
            if d.textlength(trial,font=self.font(size))>width:lines.append(word)
            else:lines[-1]=trial
        return lines
    def paragraph(self,d,xy,text,width,size=26,fill=INK,line=1.35):
        x,y=xy
        for row in self.wrap(d,text,width,size):self.text(d,(x,y),row,size,fill);y+=int(size*line)
        return y
    def marker(self,d,x,y,index,color,r=7):
        if index==0:d.rectangle((x-r,y-r,x+r,y+r),fill=color)
        elif index==1:d.ellipse((x-r,y-r,x+r,y+r),fill=color)
        else:d.polygon([(x,y-r-2),(x+r+2,y+r),(x-r-2,y+r)],fill=color)
    def base(self,t,title,kicker):
        im=Image.new('RGB',(1920,1080),BG);d=ImageDraw.Draw(im)
        self.text(d,(40,25),'JHU  /  DATA VISUALIZATION  /  SYNTHETIC MULTI-AGENT EXPERIMENT',18,BLUE,True)
        self.text(d,(40,56),title,43,INK,True);self.text(d,(40,112),kicker,23,MUTED)
        self.text(d,(1878,37),f'{int(t)//60}:{int(t)%60:02d} / 3:00',20,MUTED,anchor='ra')
        d.line((40,157,1880,157),fill=GRID,width=2)
        d.rectangle((0,945,1920,1080),fill='#e8eef1')
        caption=next(text for a,b,text in self.segments if a<=t<b)
        self.paragraph(d,(42,958),caption,1830,26,INK,1.25)
        self.text(d,(42,1051),'HISTORICAL SCHEMATIC, NOT A MODERN ROAD MAP  |  Synthetic demand, fleet and closures  |  No operational validation',17,MUTED)
        d.rectangle((0,1074,int(1920*t/180),1080),fill=BLUE)
        return im,d
    def point(self,node,box):
        x,y,w,h=box;u,v=self.c['nodes'][node];return x+w*u,y+h*v
    def edge_points(self,key,box):
        e=self.edges[key];a=np.array(self.point(e['u'],box));b=np.array(self.point(e['v'],box));mid=(a+b)/2
        delta=b-a;normal=np.array([-delta[1],delta[0]])/max(1,np.linalg.norm(delta))
        bend={'a':-.075,'b':.075,'c':-.075,'d':.075}.get(key,0)*box[2]
        control=mid+normal*bend
        return [(1-s)**2*a+2*(1-s)*s*control+s*s*b for s in np.linspace(0,1,81)]
    def edge_position(self,key,ratio,box,origin=None):
        if origin is not None and origin!=self.edges[key]['u']:ratio=1-ratio
        pts=self.edge_points(key,box);pos=np.clip(ratio,0,1)*80;i=min(79,int(pos));return pts[i]*(1-(pos-i))+pts[i+1]*(pos-i)
    def network(self,d,box,closed=(),belief=(),vehicles=None,waiting=None,show_weights=False,degrees=False):
        for key,e in self.edges.items():
            points=self.edge_points(key,box);xy=[tuple(p) for p in points]
            if key in closed:
                for i in range(0,76,8):d.line(xy[i:i+5],fill=RED,width=5)
            else:d.line(xy,fill='#9aadb5',width=4)
            if (key in closed)!=(key in belief):
                for i in range(9,72,10):
                    x,y=points[i];d.ellipse((x-3,y-3,x+3,y+3),fill='#a66b16')
            x,y=points[40];label=key+(f"  {e['travel_s']} s" if show_weights else '')
            font=self.font(21,True);bounds=d.textbbox((0,0),label,font=font);w=bounds[2]-bounds[0]+12
            d.rectangle((x-w/2,y-15,x+w/2,y+16),fill=BG)
            self.text(d,(x,y-1),label,21,RED if key in closed else MUTED,True,'mm')
        for node in self.c['nodes']:
            x,y=self.point(node,box);r=26
            d.ellipse((x-r,y-r,x+r,y+r),fill='#e4edf1',outline=BLUE,width=3);self.text(d,(x,y-1),node,27,INK,True,'mm')
            if degrees:self.text(d,(x+36,y-15),f"degree {5 if node=='A' else 3}",24,RED,True)
            if waiting is not None:
                count=waiting[node];d.rectangle((x+28,y+22,x+49,y+43),fill='#f1c354',outline=INK,width=1)
                self.text(d,(x+53,y+19),str(count),23,INK,True)
        if vehicles:
            placed=[]
            for v in vehicles:
                if v['edge'] is not None:
                    p=self.edge_position(v['edge'],v['progress_s']/self.edges[v['edge']]['travel_s'],box,v['node']);x,y=p
                else:
                    x,y=self.point(v['node'],box);angle=v['id']*2*math.pi/6;x+=39*math.cos(angle);y+=39*math.sin(angle)
                ox,oy=x,y
                if any(math.hypot(x-px,y-py)<29 for px,py in placed):
                    for radius in [29,40,52]:
                        found=False
                        for angle in np.linspace(0,2*math.pi,12,endpoint=False):
                            cx,cy=ox+radius*math.cos(angle),oy+radius*math.sin(angle)
                            if all(math.hypot(cx-px,cy-py)>=29 for px,py in placed):x,y=cx,cy;found=True;break
                        if found:break
                    d.line((ox,oy,x,y),fill=MUTED,width=1)
                placed.append((x,y))
                r=12;fill=BLUE if v['onboard'] else '#ffffff'
                if v['hold']:
                    d.polygon([(x,y-r-3),(x+r+3,y),(x,y+r+3),(x-r-3,y)],fill='#f1c354',outline=INK)
                else:d.ellipse((x-r,y-r,x+r,y+r),fill=fill,outline=INK,width=2)
                self.text(d,(x,y-1),str(v['id']+1),16,'#ffffff' if v['onboard'] and not v['hold'] else INK,True,'mm')
    def intro(self,t):
        im,d=self.base(t,'Autonomous fleets on changing networks','From K\u00f6nigsberg to Kaliningrad: a historical graph, a modern coordination question')
        self.network(d,(35,205,1000,640),degrees=t>=8)
        self.text(d,(1110,221),'ONE NETWORK. TWO QUESTIONS.',25,BLUE,True)
        y=self.paragraph(d,(1110,275),'Euler: can every bridge be crossed exactly once?',700,35,INK)
        self.text(d,(1110,y+18),'5, 3, 3, 3',56,RED,True)
        self.paragraph(d,(1110,y+95),'Four odd-degree vertices: no Euler trail.',700,29)
        self.text(d,(1110,585),'Taxi service: can each rider reach a destination?',33,INK,True)
        self.paragraph(d,(1110,665),'Yes, the initial network is connected. A shortest passenger route need not use every bridge.',700,28)
        self.text(d,(40,895),'Seven edges; parallel bridges remain separate. Diagram positions are schematic, not geographic.',21,MUTED)
        return im
    def model(self,t):
        im,d=self.base(t,'Model the information, not just the vehicles','Own dispatch policies for a hypothetical autonomous-taxi fleet inspired by Waymo')
        self.network(d,(20,220,840,580),show_weights=True)
        self.text(d,(40,852),'All edge times above are synthetic. One passenger per taxi.',23,MUTED)
        x=940
        assumptions=[('6 taxis / 15 simulated minutes','Demand arrives during the first 12 minutes; no request is silently discarded.'),
          ('Independent, matched input tapes','Demand, initial fleet, background traffic and delays are generated separately.'),
          ('Three physical scenarios','All open; links a+b closed with detour; links a+b+f closed, isolating B.'),
          ('Received information is not physical truth','Updates every 30 s (roads) and 10 s (vehicles); one serviced message per recipient per second.')]
        for i,(head,body) in enumerate(assumptions):
            yy=200+145*i;self.text(d,(x,yy),head,27,INK,True);self.paragraph(d,(x,yy+40),body,890,23,MUTED)
        names=['FIFO: use received state','Priority: road/status first','Monitor: freshness + route gate']
        for i,p in enumerate(self.policies):
            xx=55+i*620;d.line((xx,805,xx+570,805),fill=COLORS[p],width=5)
            self.text(d,(xx,819),names[i],25,COLORS[p],True)
        self.text(d,(40,901),'Monitor limits: road age <= 45 s; idle-status age <= 25 s. Fallback: wait at nodes or use a fresh known detour.',22,INK)
        return im
    def line_plot(self,d,box,key,title,maxy,now):
        x,y,w,h=box;left=x+55;right=x+w-20;top=y+38;bottom=y+h-28
        self.text(d,(x,y),title,23,INK,True)
        for k in range(4):
            yy=bottom-(bottom-top)*k/3;d.line((left,yy,right,yy),fill=GRID,width=1)
            self.text(d,(left-9,yy),f'{maxy*k/3:.0f}',18,MUTED,anchor='rm')
        for sec in [0,240,540,900]:
            xx=left+(right-left)*sec/900;self.text(d,(xx,bottom+9),str(sec),18,MUTED,anchor='mt')
        for j,p in enumerate(self.policies):
            pts=[(left+(right-left)*f['time_s']/900,bottom-(bottom-top)*f[key]/maxy) for f in self.frames[p] if f['time_s']<=now]
            if len(pts)>1:d.line(pts,fill=COLORS[p],width=3)
            if pts:self.marker(d,*pts[-1],j,COLORS[p],5)
        for event in [240,540]:
            xx=left+(right-left)*event/900
            for yy in range(int(top),int(bottom),10):d.line((xx,yy,xx,yy+5),fill='#adb9bf',width=1)
        xx=left+(right-left)*now/900;d.line((xx,top,xx,bottom),fill=INK,width=2)
        self.text(d,(right,bottom+24),'simulation seconds',17,MUTED,anchor='rt')
    def replay_frame(self,t):
        now=min(900,round((t-55)*12));index=now
        im,d=self.base(t,'One clock. Three policies. The same disruptions.',f"Representative detour replay | seed {self.replay['selection']['seed']} | {now:03d} / 900 simulation seconds | 12x playback")
        event='ALL LINKS OPEN' if now<240 else 'a + b CLOSED: detour A-D-B remains' if now<540 else 'a + b REOPENED: knowledge catches up'
        self.text(d,(1878,113),event,23,RED if 240<=now<540 else BLUE,True,'ra')
        for i,p in enumerate(self.policies):
            x=35+i*632;frame=self.frames[p][index]
            d.line((x,179,x+597,179),fill=COLORS[p],width=6);self.marker(d,x+9,211,i,COLORS[p],8);self.text(d,(x+29,192),SHORT[p],29,COLORS[p],True)
            mismatch=len(set(frame['actual_closed'])^set(frame['dispatcher_closed']))
            self.text(d,(x,231),f"Road-belief mismatch: {mismatch}/7   |   queued messages: {frame['queue_backlog']}",19,MUTED)
            self.network(d,(x-20,272,620,330),frame['actual_closed'],frame['dispatcher_closed'],frame['vehicles'],frame['waiting_by_node'])
            self.text(d,(x,642),f"Completed {frame['completed']}    Waiting {frame['waiting']}    Onboard {frame['onboard']}",23,INK,True)
            self.text(d,(x,678),f"Blocked entries {frame['closed_entry_attempts']}   |   RTA wait {frame['fallback_vehicle_s']/60:.1f} fleet-min",21,MUTED)
        self.line_plot(d,(40,738,900,185),'completed','Completed trips (count)',max(3,self.max_completed),now)
        self.line_plot(d,(985,738,895,185),'waiting','Waiting for pickup (count)',max(3,self.max_waiting),now)
        self.text(d,(40,710),'Taxi IDs 1-6: hollow = empty; filled = occupied; diamond = held. Yellow squares = waiting riders. Red dashes = closed; amber dots = stale road belief.',18,MUTED)
        return im
    def results(self,t):
        scenario='all_open' if t<140 else 'detour' if t<150 else 'district_isolation'
        label=next(s['label'] for s in self.c['scenarios'] if s['id']==scenario)
        episodes=self.c['seeds_per_scenario']*len(self.c['scenarios'])*len(self.policies)
        im,d=self.base(t,'Repeated-seed results: the monitor has costs',f"{label} | {self.c['seeds_per_scenario']} matched seeds per scenario | {episodes} total policy episodes")
        for i,p in enumerate(self.policies):
            x=40+i*620;self.marker(d,x+9,190,i,COLORS[p],8);self.text(d,(x+28,175),SHORT[p],25,COLORS[p],True)
        for i,(metric,title,unit) in enumerate(self.metric_specs):
            x=40+(i%3)*630;y=237+(i//3)*315;w=590;h=285
            d.line((x,y,x+w,y),fill=GRID,width=2);self.text(d,(x,y+15),title,25,INK,True);self.text(d,(x,y+48),unit,20,MUTED)
            lo=x+25;hi=x+w-95;scale=self.maxima[metric]
            for j,p in enumerate(self.policies):
                yy=y+98+j*49;r=self.summary[scenario,p,metric]
                a=lo+(hi-lo)*r['p10']/scale;b=lo+(hi-lo)*r['p90']/scale;m=lo+(hi-lo)*r['mean']/scale
                d.line((lo,yy,hi,yy),fill=GRID,width=1);d.line((a,yy,b,yy),fill=COLORS[p],width=4)
                d.line((a,yy-7,a,yy+7),fill=COLORS[p],width=2);d.line((b,yy-7,b,yy+7),fill=COLORS[p],width=2)
                self.marker(d,m,yy,j,COLORS[p],7)
                text=f"{100*r['mean']:.1f}%" if metric=='completion_fraction' else f"{r['mean']:.2f}" if metric=='closed_entry_attempts' else f"{r['mean']:.1f}"
                self.text(d,(x+w,yy),text,25,COLORS[p],True,'rm')
            self.text(d,(lo,y+243),'0',18,MUTED)
            cap=scale*100 if metric=='completion_fraction' else scale
            self.text(d,(hi,y+243),f'{cap:.0f}',18,MUTED,anchor='rt')
        self.text(d,(40,888),'DOT = mean episode result    WHISKER = empirical 10th-90th percentiles    Same metric scales across scenarios',22,INK,True)
        self.text(d,(40,921),'Pickup waits exclude riders not picked up. Uncompleted includes waiting + onboard. Counts must be interpreted with service exposure.',18,MUTED)
        return im
    def conclusion(self,t):
        im,d=self.base(t,'What transfers beyond this schematic?','A queueing and information-age experiment, not a self-driving-car validation')
        sections=[(45,'WHAT THE EXPERIMENT SHOWS',[
          'A connected network can still deliver poor service when availability reports arrive late.',
          'Priority service and freshness checks solve different problems.',
          'A monitor can reduce blocked entry attempts while delaying passengers.',
          'Always pair waiting-time summaries with uncompleted requests.']),
          (1000,'WHAT REMAINS TO BE TESTED',[
          'Modern Kaliningrad motor-vehicle roads: directions, access restrictions and verified crossings.',
          'Other queue loads, freshness thresholds, fleet sizes and demand patterns.',
          'Road physics, congestion, safe stopping and real autonomous-driving software.',
          'Fresh data can become wrong after a new closure. No unconditional safety guarantee.'])]
        for x,head,items in sections:
            self.text(d,(x,211),head,27,BLUE,True);y=270
            for i,text in enumerate(items):
                self.text(d,(x,y),f'{i+1:02d}',25,BLUE,True);y=self.paragraph(d,(x+55,y),text,790,28)+30
        d.line((40,810,1880,810),fill=GRID,width=2)
        self.text(d,(40,838),'Sources and reproducibility',25,INK,True)
        self.text(d,(40,878),'Euler context: Mathematical Association of America, Convergence. Topology: supplied seven-edge graph, checked with NetworkX.',21,MUTED)
        self.text(d,(40,911),'Saved inputs, event logs, matched-seed summaries, transcript and code accompany this movie. Source review: 14 October 2026.',20,MUTED)
        return im
    def frame(self,t):
        if t<25:return self.intro(t)
        if t<55:return self.model(t)
        if t<130:return self.replay_frame(t)
        if t<165:return self.results(t)
        return self.conclusion(t)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,default=HERE/'output');ap.add_argument('--encoder',default='h264_nvenc',choices=['h264_nvenc','libx264']);ap.add_argument('--stills-only',action='store_true')
    args=ap.parse_args();movie=Movie(args.out);fps=movie.c['fps'];duration=movie.c['movie_duration_s']
    with (args.out/'movie_frame_index.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=['frame','video_time_s','section','replay_time_s','scenario']);writer.writeheader()
        for i in range(fps*duration):
            t=i/fps
            section='Euler' if t<25 else 'Model' if t<55 else 'Replay' if t<130 else 'Results' if t<165 else 'Conclusion'
            scenario=movie.c['replay_scenario'] if section=='Replay' else ('all_open' if t<140 else 'detour' if t<150 else 'district_isolation') if section=='Results' else ''
            writer.writerow({'frame':i,'video_time_s':t,'section':section,'replay_time_s':min(900,round((t-55)*12)) if section=='Replay' else '', 'scenario':scenario})
    for sec,name in [(14,'01_euler'),(38,'02_model'),(88,'03_matched_replay'),(145,'04_repeated_results')]:movie.frame(sec).save(args.out/f'{name}.png')
    if args.stills_only:return
    target=args.out/'fleet_changing_networks_silent.mp4'
    cmd=['ffmpeg','-y','-hide_banner','-loglevel','warning','-f','rawvideo','-pix_fmt','rgb24','-s','1920x1080','-r',str(fps),'-i','-','-an','-c:v',args.encoder]
    cmd+=['-preset','p5','-cq','21','-b:v','0'] if args.encoder=='h264_nvenc' else ['-preset','fast','-crf','20']
    cmd+=['-pix_fmt','yuv420p','-movflags','+faststart',str(target)]
    with (args.out/'render.log').open('w') as log:
        process=subprocess.Popen(cmd,stdin=subprocess.PIPE,stderr=log)
        try:
            for i in range(fps*duration):
                process.stdin.write(movie.frame(i/fps).tobytes())
                if i%(fps*15)==0:print(f'Rendered {i/fps:.0f}/{duration} seconds',flush=True)
        finally:process.stdin.close()
        if process.wait()!=0:raise RuntimeError((args.out/'render.log').read_text())
    print(target)


if __name__=='__main__':main()
