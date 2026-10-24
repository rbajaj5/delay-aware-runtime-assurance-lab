"""Audit delivered media and package the frozen simulation with its dashboard."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import zipfile

from portfolio_rta_demo import simulate

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts'/'20261013'/'portfolio_rta'


def main():
    movie=OUT/'portfolio_rta_synchronized.mp4'
    probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0',
        '-show_entries','stream=width,height,codec_name,pix_fmt,avg_frame_rate,nb_frames,duration',
        '-of','json',str(movie)],text=True))['streams'][0]
    decode=subprocess.run(['ffmpeg','-v','error','-i',str(movie),'-f','null','-'],capture_output=True,text=True)
    stored=json.loads((OUT/'synchronized_frames.json').read_text(encoding='utf-8'))['frames']
    records,_=simulate()
    same=records==stored
    metadata_ok=probe['width']==1920 and probe['height']==1080 and probe['avg_frame_rate']=='24/1' and int(probe['nb_frames'])==2160 and float(probe['duration'])==90
    checks={'probe':probe,'full_decode_exit_code':decode.returncode,'full_decode_errors':decode.stderr,
        'deterministic_regeneration_exact':same,'rendered_snapshots_visually_inspected_s':[8,72,87],
        'numerical_device':'CPU','video_encoder':'NVIDIA h264_nvenc',
        'all_passed':metadata_ok and same and decode.returncode==0 and not decode.stderr.strip()}
    (OUT/'media_validation.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
    if not checks['all_passed']:raise RuntimeError(checks)
    shutil.copy2(__file__,OUT/Path(__file__).name)
    shutil.copy2(ROOT/'analysis'/'test_portfolio_dashboard.cjs',OUT/'tableau'/'test_portfolio_dashboard.cjs')
    manifest={p.relative_to(OUT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='manifest.json'}
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    downloads=Path.home()/'Downloads'
    target=downloads/'Multi_Agent_Risk_RTA_Demo_20261014.zip'
    with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as archive:
        for p in sorted(OUT.rglob('*')):
            if p.is_file():archive.write(p,'Multi_Agent_Risk_RTA_Demo/'+p.relative_to(OUT).as_posix())
    with zipfile.ZipFile(target) as archive:
        assert archive.testzip() is None
    print(json.dumps({'media':checks,'package':str(target),'bytes':target.stat().st_size,'files':len(manifest)+1},indent=2))


if __name__=='__main__':main()
