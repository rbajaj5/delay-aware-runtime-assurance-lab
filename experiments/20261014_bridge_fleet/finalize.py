"""Mux, verify, and package locally. This script never publishes or uploads."""
import csv
import hashlib
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import zipfile

import networkx
import numpy as np
import PIL
from PIL import Image

from render import Movie
from make_docs import main as make_docs

HERE=Path(__file__).resolve().parent


def main():
    out=HERE/'output';target=out/'fleet_changing_networks.mp4'
    narration=out/'narration.wav';silent=out/'fleet_changing_networks_silent.mp4';captions=out/'captions.srt'
    subprocess.run(['ffmpeg','-y','-v','error','-i',str(silent),'-i',str(narration),'-i',str(captions),
        '-map','0:v:0','-map','1:a:0','-map','2:s:0','-c:v','copy','-c:a','aac','-b:a','160k',
        '-c:s','mov_text','-metadata:s:s:0','language=eng','-t','180','-movflags','+faststart',str(target)],check=True)
    probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-show_format','-of','json',str(target)],text=True))
    video=next(s for s in probe['streams'] if s['codec_type']=='video');audio=next(s for s in probe['streams'] if s['codec_type']=='audio')
    assert video['width']==1920 and video['height']==1080 and video['avg_frame_rate']=='24/1' and int(video['nb_frames'])==4320
    assert abs(float(probe['format']['duration'])-180)<.1
    assert abs(float(audio['duration'])-180)<.1
    assert any(s['codec_type']=='subtitle' for s in probe['streams'])
    decode=subprocess.run(['ffmpeg','-v','error','-i',str(target),'-f','null','-'],capture_output=True,text=True)
    assert decode.returncode==0 and not decode.stderr.strip(),decode.stderr
    movie=Movie(out);qa=out/'qa';qa.mkdir(exist_ok=True);checks=[]
    for second in [14,38,75,88,100,129.958333333,145,155,171,179]:
        image=qa/f'encoded_{second:06.2f}.png'
        subprocess.run(['ffmpeg','-y','-v','error','-ss',str(second),'-i',str(target),'-frames:v','1',str(image)],check=True)
        actual=np.asarray(Image.open(image).convert('RGB'),dtype=float)
        expected=np.asarray(movie.frame(round(second*24)/24),dtype=float)
        mae=float(np.mean(abs(actual-expected)));spread=float(actual.std())
        assert mae<8 and spread>20,(second,mae,spread)
        checks.append({'second':second,'mean_absolute_pixel_error_vs_saved_data_render':mae,'pixel_std':spread})
    frame_rows=list(csv.DictReader((out/'movie_frame_index.csv').open(newline='')))
    assert len(frame_rows)==4320
    for r in frame_rows:
        if r['section']=='Replay':assert int(r['replay_time_s'])==min(900,round((float(r['video_time_s'])-55)*12))
    n=json.loads((out/'narration_validation.json').read_text())
    assert n['narration_source_sha256']==hashlib.sha256((out/'narration_segments.json').read_bytes()).hexdigest()
    media={'all_passed':True,'duration_s':float(probe['format']['duration']),'width':video['width'],'height':video['height'],
           'frames':int(video['nb_frames']),'fps':video['avg_frame_rate'],'audio_codec':audio['codec_name'],'subtitle_track':True,
           'full_decode_error_count':0,'sampled_encoded_frames':checks,'frame_index_rows':len(frame_rows),
           'captions_match_narration_source':True,'numerical_device':'CPU','video_encoder':video.get('tags',{}).get('encoder','see render command'),
           'python':platform.python_version(),'numpy':np.__version__,'networkx':networkx.__version__,'Pillow':PIL.__version__,
           'source_hashes_at_verification':{name:hashlib.sha256((out/name).read_bytes()).hexdigest() for name in ['config_used.json','replay.json','summary.csv']},
           'scope':'Playback, pixel and numerical checks are not validation of a real taxi system.'}
    (out/'media_validation.json').write_text(json.dumps(media,indent=2),encoding='utf-8')
    make_docs()
    files=[]
    for p in HERE.rglob('*'):
        if not p.is_file():continue
        rel=p.relative_to(HERE)
        if any(part in ['pilot','__pycache__','narration_parts','qa'] for part in rel.parts):continue
        if p.name in ['manifest.json','fleet_changing_networks_silent.mp4'] or p.suffix=='.log':continue
        files.append(p)
    manifest={p.relative_to(HERE).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}
    (HERE/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8');files.append(HERE/'manifest.json')
    destination=Path.home()/'Downloads';video_copy=destination/'Autonomous_Fleets_Changing_Networks_20261014.mp4';shutil.copy2(target,video_copy)
    package=destination/'Autonomous_Fleets_Changing_Networks_20261014.zip'
    with zipfile.ZipFile(package,'w',zipfile.ZIP_DEFLATED) as archive:
        for p in files:archive.write(p,'Autonomous_Fleets_Changing_Networks/'+p.relative_to(HERE).as_posix())
    with zipfile.ZipFile(package) as archive:assert archive.testzip() is None
    print(json.dumps({'movie':str(video_copy),'package':str(package),'package_bytes':package.stat().st_size,'media_validation':media},indent=2))


if __name__=='__main__':main()
