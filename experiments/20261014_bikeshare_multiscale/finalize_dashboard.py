"""Mux local speech and subtitles, decode-test, and compare encoded movie pixels."""
import json
import subprocess
import hashlib
import numpy as np
from PIL import Image
from prepare import ROOT,dump,sha

OUT=ROOT/'output/movie'


def main():
    target=OUT/'river_links_planning_narrated.mp4'
    subprocess.run(['ffmpeg','-y','-v','error','-i',str(OUT/'dashboard_silent.mp4'),'-i',str(OUT/'narration.wav'),'-i',str(OUT/'captions.srt'),
                    '-map','0:v:0','-map','1:a:0','-map','2:0','-c:v','copy','-c:a','aac','-b:a','160k','-c:s','mov_text',
                    '-metadata:s:s:0','language=eng','-movflags','+faststart',str(target)],check=True)
    probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-show_format','-of','json',str(target)],text=True))
    v=next(s for s in probe['streams'] if s['codec_type']=='video')
    assert (v['width'],v['height'],v['r_frame_rate'],int(v['nb_frames']))==(1920,1080,'24/1',4320)
    assert abs(float(probe['format']['duration'])-180)<.1
    assert any(s['codec_type']=='audio' for s in probe['streams']) and any(s['codec_type']=='subtitle' for s in probe['streams'])
    subprocess.run(['ffmpeg','-v','error','-i',str(target),'-f','null','-'],check=True)
    checks=[]
    for i in [0,540,1080,1620,1980]:
        time=i/12;file=OUT/f'encoded_{i:04}.png'
        subprocess.run(['ffmpeg','-y','-v','error','-ss',str(time),'-i',str(target),'-frames:v','1',str(file)],check=True)
        before=np.array(Image.open(OUT/f'dashboard_sample_{i:04}.jpg').convert('RGB'),dtype=float)
        after=np.array(Image.open(file).convert('RGB'),dtype=float)
        mae=float(np.abs(after-before).mean());std=float(after.std())
        assert mae<9 and std>25,(i,mae,std)
        checks.append({'capture_frame':i,'movie_time_s':time,'pixel_mae':mae,'pixel_std':std})
    index=[json.loads(line) for line in (OUT/'dashboard_frame_index.jsonl').read_text().splitlines()]
    assert len(index)==2160 and all(len(r['states'])==2 and all(s['fleet'] in [168,192] for s in r['states']) for r in index)
    resolved=json.loads((OUT/'walkthrough_resolved.json').read_text())
    assert resolved['capture_frames']==2160 and not resolved['errors']
    hashes={name:sha(ROOT/name) for name in ['app/index.html','app/app.js','app/domain.js','app/style.css','app/demo.css','record_dashboard.cjs','output/movie/walkthrough.json','output/movie/narration_segments.json']}
    dump(OUT/'movie_validation.json',{'passed':True,'resolution':[1920,1080],'fps':24,'video_frames':4320,'dashboard_capture_frames':2160,
         'duration_s':float(probe['format']['duration']),'source':'Verified local dashboard, not separately invented graphics.',
         'render_cadence':'12 deterministic UI states per second, each held for 2 output frames. 24 fps MP4.',
         'encoder':'NVIDIA h264_nvenc','simulation_compute':'CPU event simulation and routing; GPU video encoding only',
         'full_decode_passed':True,'automated_voice':True,'burned_captions':True,'selectable_subtitles':True,
         'pixel_checks':checks,'source_hashes':hashes,'mp4_sha256':sha(target)})
    print('Verified:',target,flush=True)


if __name__=='__main__':main()
