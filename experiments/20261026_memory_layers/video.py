"""Data-driven 1080p bilingual teaching video with local synthetic narration."""
from pathlib import Path
from functools import lru_cache
import argparse
import json
import math
import shutil
import subprocess
import wave
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from simulate import CONFIG, OUT

HERE=Path(__file__).resolve().parent
B=HERE/"build"
D=HERE/"delivery"
W,H,FPS,LENGTH=1920,1080,24,128
FF=shutil.which("ffmpeg")
BG="#f3f5f6"; INK="#182d32"; MUTED="#536971"; LINE="#ced9dd"
TEAL="#007d86"; PURPLE="#6850a1"; RED="#bd4136"; GOLD="#997014"
PALE="#e3f0ed"; WHITE="#ffffff"
COLORS={"A":"#087e8b","B":"#6850a1","C":"#c86b21","D":"#3d6f36"}
POLICIES=CONFIG["policies"]
RUNS={(p,a):json.loads((OUT/f"{p}_{a}.json").read_text()) for p in CONFIG["profiles"] for a in POLICIES}
AGG=json.loads((OUT/"aggregates.json").read_text())
SEGMENTS=[
 (0,12,"Long-term memory stores knowledge we can reuse. Working memory maintains the information needed for the task in progress. A drone fleet makes this distinction visible."),
 (12,24,"Both controllers know the same map and procedure. Working memory tracks who has clearance. The message queue carries updates that have not yet been processed."),
 (24,34,"A and B accept their commands, but their receipts are lost. After one point two seconds, the timeout controller drops their reservations."),
 (34,44,"It then clears C and D. An empty pad may already be reserved. The queue-aware controller retains the first pair's unresolved commitments."),
 (44,54,"Two drones now share each pad. This is a resource conflict, not a simulated crash. Both controllers reject the old report."),
 (54,64,"Fresh telemetry confirms that the first drones have finished and their pads are clear. Only then does the queue-aware controller release the reservations and authorize the next pair."),
 (64,74,"The second pair now uses the pads. Waiting preserves one unfinished command per pad. Both controllers have identical stored knowledge."),
 (74,84,"This is an engineering analogy, not brain science. Reusable knowledge, active task state, and message transport serve different functions."),
 (84,94,"Now receipts are lost and telemetry stops after one second. The queue-aware controller keeps reservations because it cannot verify completion."),
 (94,104,"Two drones finish; two keep waiting. Caution prevents the modeled conflict but stalls progress. Recovery needs a separately justified procedure."),
 (104,116,"Thirty-two matched timing seeds reproduce this constructed example. Timely messages make the controllers identical. Lost receipts and lost observations reveal different tradeoffs."),
 (116,128,"Replace old observations. Retain unresolved commitments. Check before acting. This synthetic example comes with code, states, captions, and tests."),
]


@lru_cache(None)
def font(size,bold=False,cn=False):
    name="msyh.ttc" if cn else ("seguisb.ttf" if bold else "segoeui.ttf")
    return ImageFont.truetype(str(Path("C:/Windows/Fonts")/name),size)


def txt(draw,xy,text,size=28,color=INK,bold=False,anchor=None):
    text=str(text)
    cn=any("\u4e00"<=c<="\u9fff" for c in text)
    draw.text(xy,text,font=font(size,bold,cn),fill=color,anchor=anchor)


def wrap(draw,text,width,size=28):
    lines=[]; line=""
    for word in text.split():
        trial=(line+" "+word).strip()
        if draw.textlength(trial,font=font(size))>width and line:
            lines.append(line); line=word
        else: line=trial
    if line: lines.append(line)
    return lines


def para(draw,xy,text,width,size=28,color=INK,bold=False):
    for i,line in enumerate(wrap(draw,text,width,size)):
        txt(draw,(xy[0],xy[1]+i*(size+9)),line,size,color,bold)


def outline(draw,rect,fill=WHITE,color=LINE):
    draw.rounded_rectangle(rect,radius=7,fill=fill,outline=color,width=2)


def arrow(draw,a,b,color=TEAL):
    draw.line([a,b],fill=color,width=4)
    angle=math.atan2(b[1]-a[1],b[0]-a[0])
    draw.polygon([b,(b[0]-14*math.cos(angle-.5),b[1]-14*math.sin(angle-.5)),
                  (b[0]-14*math.cos(angle+.5),b[1]-14*math.sin(angle+.5))],fill=color)


def base(title,sub,section):
    im=Image.new("RGB",(W,H),BG); d=ImageDraw.Draw(im)
    txt(d,(56,26),"RAVI BAJAJ / QUEUE-AWARE AUTONOMY",23,TEAL,True)
    txt(d,(1864,26),section,23,MUTED,anchor="ra")
    txt(d,(56,68),title,49,INK,True)
    txt(d,(56,134),sub,27,MUTED)
    d.line([(56,177),(1864,177)],fill=LINE,width=2)
    return im,d


def caption(im,t):
    d=ImageDraw.Draw(im)
    text=next(s for a,b,s in SEGMENTS if a<=t<b)
    d.rectangle([0,934,W,H],fill="#14292c")
    for i,line in enumerate(wrap(d,text,1780,28)):
        txt(d,(960,949+i*37),line,28,WHITE,anchor="ma")
    d.rectangle([0,1073,W,1080],fill="#395356")
    d.rectangle([0,1073,int(W*t/LENGTH),1080],fill="#53d0bc")
    return im


def route_map(d,rect):
    x,y,x2,y2=rect
    for i in range(7):
        yy=y+i*(y2-y)/6
        d.line([(x,yy),(x2,yy)],fill=LINE,width=1)
    for px,label in [(x+100,"P1"),(x+300,"P2")]:
        d.ellipse([px-27,y2-70,px+27,y2-16],outline=TEAL,width=3)
        txt(d,(px,y2-43),label,23,TEAL,True,anchor="mm")
        d.line([(px,y+20),(px,y2-82)],fill=TEAL,width=3)


def introduction(t):
    im,d=base("Two kinds of memory. One unfinished task.",
              "An engineering illustration of long-term memory and working memory", "01 / MEMORY LAYERS")
    for x,col,title,english,subtitle in [(56,PURPLE,"长期记忆","LONG-TERM MEMORY","Reusable knowledge across tasks"),
                                      (990,TEAL,"工作记忆","WORKING MEMORY","Active state for this task")]:
        outline(d,(x,208,x+874,745))
        txt(d,(x+34,229),title,59,col,True)
        txt(d,(x+34,310),english,30,col,True)
        txt(d,(x+34,360),subtitle,29,MUTED)
    route_map(d,(92,423,540,653))
    para(d,(574,430),"Map v1\n",300,29,bold=True)
    para(d,(574,486),"One drone per pad.\nAcceptance is not completion.",290,29)
    txt(d,(92,681),"Same knowledge in both comparison arms",27,PURPLE,True)
    items=[("Task","Land A, B, C and D"),("Commitment","A owns P1; B owns P2"),
           ("Uncertainty","Acknowledgments missing"),("Release condition","Fresh completion evidence")]
    for j,(a,b) in enumerate(items):
        yy=418+j*68
        txt(d,(1024,yy),a,25,MUTED)
        txt(d,(1263,yy),b,27,INK,True)
    txt(d,(1024,699),"Example active record, not a replay frame",23,MUTED)
    d.rectangle([56,777,1864,896],fill="#e1e8ea")
    txt(d,(86,794),"消息队列",32,GOLD,True)
    txt(d,(86,842),"MESSAGE QUEUE",23,MUTED)
    for j,(a,b) in enumerate([("COMMAND","in transit"),("ACK","missing"),("TELEMETRY","awaiting processing")]):
        x=530+j*428
        txt(d,(x,799),a,27,INK,True)
        txt(d,(x,847),b,24,MUTED)
    return im


def replay_point(t):
    if 24<=t<84:
        profile="lost_ack"; sim_t=min(14.,(t-24)*14/60)
    elif 84<=t<104:
        profile="telemetry_outage"; sim_t=min(14.,(t-84)*14/20)
    else:
        return None
    index=min(140,int(sim_t/CONFIG["dt_s"]+1e-8))
    return profile,index


def drone(d,px,py,name,mode,t):
    color=COLORS[name]
    for dx,dy in [(-12,-10),(12,-10),(-12,10),(12,10)]:
        d.line([(px,py),(px+dx,py+dy)],fill=color,width=3)
        d.ellipse([px+dx-8,py+dy-8,px+dx+8,py+dy+8],outline=color,width=2)
        if mode not in ("WAITING","FINISHED","ON PAD"):
            angle=t*15+dx
            d.line([(px+dx-7*math.cos(angle),py+dy-7*math.sin(angle)),
                    (px+dx+7*math.cos(angle),py+dy+7*math.sin(angle))],fill=color,width=2)
    d.ellipse([px-7,py-7,px+7,py+7],fill=color)
    # Persistent offset labels remain legible when two logged positions coincide.
    side=-1 if name in "AB" else 1
    label_y=max(375,min(628,py-35 if name in "AB" else py+34))
    txt(d,(px+side*34,label_y),name,29,color,True,anchor="mm")


def replay(t):
    profile,index=replay_point(t)
    sim_t=RUNS[profile,POLICIES[0]]["frames"][index]["t"]
    outage=profile=="telemetry_outage"
    title="No fresh evidence: caution can stall progress" if outage else "A missing receipt is not a cancelled action"
    sub=f"{'ACK loss + telemetry stops at 1.0 s' if outage else 'ACK loss; telemetry remains available'}  |  simulation t = {sim_t:04.1f} s  |  matched seed {CONFIG['seed_start']}"
    im,d=base(title,sub,"03 / OUTAGE" if outage else "02 / MATCHED REPLAY")
    d.rectangle([56,195,1864,284],fill="#e9e4f1")
    txt(d,(80,209),"长期记忆",34,PURPLE,True)
    txt(d,(290,211),"SAME STORED KNOWLEDGE",25,PURPLE,True)
    txt(d,(80,256),"Map v1",21,MUTED)
    txt(d,(780,210),"2 pads / 4 drones",28,INK,True)
    txt(d,(1125,210),"One drone per pad; verify completion",27,INK)
    txt(d,(780,252),"Knowledge is unchanged throughout the replay.",22,MUTED)
    for arm,policy in enumerate(POLICIES):
        x=56+934*arm
        result=RUNS[profile,policy]; f=result["frames"][index]
        color=RED if arm==0 else TEAL
        txt(d,(x+4,296),"TIMEOUT-ONLY TASK STATE" if arm==0 else "QUEUE-AWARE WORKING STATE",28,color,True)
        outline(d,(x,339,x+874,665))
        ox,oy,scale=x+274,548,24
        for gx in range(-7,8):
            d.line([(ox+gx*scale,359),(ox+gx*scale,648)],fill="#e5ebed",width=1)
        for gy in range(-4,9):
            d.line([(x+82,oy-gy*scale),(x+467,oy-gy*scale)],fill="#e5ebed",width=1)
        for name,pos in CONFIG["knowledge"]["initial_xy"].items():
            pad=CONFIG["knowledge"]["pads"][CONFIG["knowledge"]["assignment"][name]]
            d.line([(ox+pos[0]*scale,oy-pos[1]*scale),(ox+pad[0]*scale,oy-pad[1]*scale)],fill="#b9cdd0",width=2)
        for pad,(px,py) in CONFIG["knowledge"]["pads"].items():
            cx,cy=ox+px*scale,oy-py*scale
            fill="#f9d8d2" if pad in f["conflict_pads"] else "#dfefe8"
            edge=RED if pad in f["conflict_pads"] else TEAL
            d.ellipse([cx-34,cy-34,cx+34,cy+34],fill=fill,outline=edge,width=3)
            label_x=cx+(-86 if pad=="P1" else 86)
            txt(d,(label_x,cy),f"{pad} ({len(f['occupancy'][pad])})",23,edge,True,anchor="mm")
        for name,v in f["drones"].items():
            if v["mode"]!="FINISHED":
                drone(d,ox+v["xy_m"][0]*scale,oy-v["xy_m"][1]*scale,name,v["mode"],sim_t)
        txt(d,(x+16,345),"PLAN VIEW",18,MUTED,True)
        txt(d,(x+470,642),"1 grid = 1 m",18,MUTED,anchor="ra")
        done=", ".join(name for name,v in f["drones"].items() if v["mode"]=="FINISHED")
        txt(d,(x+526,355),"FINISHED"+(": "+done if done else ""),20,MUTED,True)
        txt(d,(x+526,384),f"{f['finished']} / 4",49,INK,True)
        conflict_so_far=sum(bool(r["conflict_pads"]) for r in result["frames"][:index+1])*CONFIG["dt_s"]
        txt(d,(x+526,455),"PAD CONFLICT TIME",20,MUTED,True)
        txt(d,(x+526,487),f"{conflict_so_far:.1f} s",40,RED if conflict_so_far else TEAL,True)
        txt(d,(x+526,547),"OBSERVATION AGE",20,MUTED,True)
        age="not received" if f["observation_age_s"] is None else f"{f['observation_age_s']:.1f} s"
        txt(d,(x+526,579),age,32,TEAL if f["observation_fresh"] else RED,True)
        txt(d,(x+526,629),f"Old reports rejected: {f['rejected_old']}",20,MUTED)
        txt(d,(x+4,680),"工作记忆 / ACTIVE COMMITMENTS",25,color,True)
        for j,pad in enumerate(("P1","P2")):
            owner=f["owners"][pad]; y=725+j*44
            if owner:
                s=f"{pad}: {owner['command_id']} | ACK {'received' if owner['ack'] else 'missing'}"
            else:
                s=f"{pad}: no retained reservation"
            txt(d,(x+4,y),s,26,INK,True)
        txt(d,(x+4,820),f["state"],24,RED if not f["observation_fresh"] else MUTED)
        d.line([(x,860),(x+874,860)],fill=LINE,width=2)
        kinds={k:sum(p["kind"]==k for p in f["packets"]) for k in ("COMMAND","ACK","OBSERVATION")}
        txt(d,(x+4,873),"消息队列 / IN TRANSIT",23,GOLD,True)
        txt(d,(x+340,875),f"CMD {kinds['COMMAND']}   ACK {kinds['ACK']}   OBS {kinds['OBSERVATION']}",23,INK)
    txt(d,(56,906),"Map: active drones only / pad labels: occupant counts / synthetic CPU kinematics, not flight-physics validation",19,MUTED)
    return im


def results(t):
    im,d=base("Useful memory also has a cost", "32 matched timing seeds per arm and profile / 192 runs / synthetic mechanism demonstration", "04 / MEASURED RESULTS")
    txt(d,(70,218),"PROFILE",26,MUTED,True)
    txt(d,(565,218),"TIMEOUT-ONLY",28,RED,True)
    txt(d,(1200,218),"QUEUE-AWARE",28,TEAL,True)
    for j,profile in enumerate(CONFIG["profiles"]):
        y=295+j*166
        title={"timely":"Timely messages","lost_ack":"Lost acknowledgments","telemetry_outage":"Lost ACK + telemetry outage"}[profile]
        d.line([(56,y-21),(1864,y-21)],fill=LINE,width=2)
        txt(d,(70,y+12),title,29,INK,True)
        for arm,policy in enumerate(POLICIES):
            row=next(r for r in AGG if r["profile"]==profile and r["policy"]==policy)
            x=565+635*arm
            txt(d,(x,y),f"{row['conflict_episodes']} / {row['n']} conflict episodes",32,RED if row["conflict_episodes"] else TEAL,True)
            txt(d,(x,y+55),f"{row['mean_completed']:.0f} / 4 finished per episode",28,INK)
            txt(d,(x,y+99),f"{row['mean_conflict_duration_s']:.2f} s mean conflict time",24,MUTED)
    d.rectangle([56,812,1864,905],fill="#e3ecee")
    para(d,(80,826),"Same map. Same procedures. Same timing inputs. Different treatment of an unresolved clearance. Small timing jitter tests this constructed case, not general drone reliability.",1740,27)
    return im


def ending(t):
    im,d=base("Remember what still matters for the next action", "Long-term knowledge + active task state + checked messages", "05 / TAKEAWAY")
    labels=[("长期记忆","REUSE KNOWLEDGE","Maps, procedures, prior experience",PURPLE),
            ("工作记忆","MAINTAIN THE TASK","Owner, command ID, evidence age",TEAL),
            ("消息队列","PROCESS UPDATES","Transport is not task completion",GOLD)]
    for j,(cn,en,desc,color) in enumerate(labels):
        y=238+j*155
        txt(d,(70,y),cn,53,color,True)
        txt(d,(430,y+10),en,34,color,True)
        txt(d,(1050,y+18),desc,30,INK)
        d.line([(70,y+118),(1850,y+118)],fill=LINE,width=2)
    txt(d,(70,735),"Replace outdated observations. Retain unresolved commitments.",35,INK,True)
    txt(d,(70,799),"Cowan (2014): working memory  |  Yates et al. (2021): Age of Information",25,MUTED)
    txt(d,(70,843),"Code, states, captions, tests: rbajaj5/delay-aware-runtime-assurance-lab",26,TEAL)
    txt(d,(70,887),"Local synthetic English voice / Chinese labels / no claim of human cognition or general safety",22,MUTED)
    return im


def frame(t):
    if t<24: im=introduction(t)
    elif t<104: im=replay(t)
    elif t<116: im=results(t)
    else: im=ending(t)
    return caption(im,t)


def srt_time(t):
    ms=round(t*1000)
    return f"{ms//3600000:02}:{ms//60000%60:02}:{ms//1000%60:02},{ms%1000:03}"


def prepare():
    B.mkdir(exist_ok=True); D.mkdir(exist_ok=True)
    (B/"narration.json").write_text(json.dumps([dict(start=a,end=b,text=s) for a,b,s in SEGMENTS],indent=2)+"\n")
    (D/"memory_layers.srt").write_text("\n\n".join(f"{j+1}\n{srt_time(a)} --> {srt_time(b)}\n{s}" for j,(a,b,s) in enumerate(SEGMENTS))+"\n",encoding="utf-8")
    index=[]
    for i in range(FPS*LENGTH):
        t=i/FPS; point=replay_point(t)
        index.append(dict(video_frame=i,video_t_s=round(t,6),profile=point[0] if point else None,
                          state_index=point[1] if point else None,
                          simulation_t_s=round(point[1]*CONFIG["dt_s"],6) if point else None))
    (OUT/"video_frame_index.json").write_text(json.dumps(index,separators=(",",":"))+"\n")


def preview():
    for t in (5,17,29,31,39,44,48,55,65,79,88,94,102,109,122):
        frame(t).save(B/f"preview_{t:03}.jpg",quality=94)
    frame(44).save(D/"memory_layers_poster.jpg",quality=94)


def render():
    p=subprocess.Popen([FF,"-v","error","-y","-f","rawvideo","-pix_fmt","rgb24","-s",f"{W}x{H}",
        "-r",str(FPS),"-i","-","-an","-c:v","libx264","-preset","fast","-crf","20","-pix_fmt","yuv420p",
        "-movflags","+faststart",str(B/"silent.mp4")],stdin=subprocess.PIPE)
    try:
        for i in range(FPS*LENGTH):
            p.stdin.write(frame(i/FPS).tobytes())
            if i%(FPS*8)==0: print(f"Rendered {i//FPS}/{LENGTH} s",flush=True)
    finally:
        p.stdin.close()
    if p.wait(): raise RuntimeError("ffmpeg render failed")


def mux():
    audio=np.zeros(48000*LENGTH,dtype=np.float64); report=[]
    for i,(start,end,text) in enumerate(SEGMENTS):
        raw=B/f"voice_{i:02}.wav"
        with wave.open(str(raw),"rb") as w: duration=w.getnframes()/w.getframerate()
        speed=max(1.,duration/(end-start-.6))
        if speed>1.25: raise ValueError(f"Narration too dense: {i}, speed={speed}")
        fit=B/f"fit_{i:02}.wav"
        subprocess.run([FF,"-v","error","-y","-i",str(raw),"-af",f"atempo={speed:.6f}","-ar","48000",
                        "-ac","1","-c:a","pcm_s16le",str(fit)],check=True)
        with wave.open(str(fit),"rb") as w: values=np.frombuffer(w.readframes(w.getnframes()),np.int16).astype(float)
        offset=round((start+.2)*48000)
        assert offset+len(values)<end*48000
        audio[offset:offset+len(values)]=values
        report.append(dict(segment=i,raw_s=duration,speed=speed,audio_s=len(values)/48000,start_s=start+.2,end_s=(offset+len(values))/48000))
    audio*=min(1.8,29000/max(1,np.max(np.abs(audio))))
    with wave.open(str(B/"narration.wav"),"wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(48000); w.writeframes(audio.astype(np.int16).tobytes())
    subprocess.run([FF,"-v","error","-y","-i",str(B/"silent.mp4"),"-i",str(B/"narration.wav"),
        "-i",str(D/"memory_layers.srt"),"-map","0:v","-map","1:a","-map","2:s","-c:v","copy",
        "-c:a","aac","-b:a","160k","-c:s","mov_text","-metadata:s:a:0","language=eng",
        "-metadata:s:s:0","language=eng","-metadata","title=Long-term Memory vs Working Memory: Queue-aware Drone Coordination",
        "-movflags","+faststart",str(D/"memory_layers_drone_fleet.mp4")],check=True)
    (OUT/"audio_validation.json").write_text(json.dumps(dict(segments=report,peak_pcm16=float(np.max(np.abs(audio))),
         rms_pcm16=float(np.sqrt(np.mean(audio**2)))),indent=2)+"\n")


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    for name in ("prepare","preview","render","mux"): parser.add_argument("--"+name,action="store_true")
    args=parser.parse_args()
    for name in ("prepare","preview","render","mux"):
        if getattr(args,name): globals()[name]()
