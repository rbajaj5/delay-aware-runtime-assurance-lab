"""Verify the final artifact as well as its data and source provenance."""
from pathlib import Path
import csv, hashlib, json, math, shutil, subprocess
from PIL import Image, ImageDraw
from domain_examples import run

HERE=Path(__file__).resolve().parent
E=HERE/'evidence';D=HERE/'delivery';B=HERE/'build'
video=D/'Working_Memory_Assured_Autonomy.mp4'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert video.is_file()
validation=json.loads((E/'validation.json').read_text())
for p,h in validation['source_sha256'].items():assert sha(HERE.parents[2]/p)==h
rows=list(csv.DictReader((E/'matched_runs.csv').open()))
summary=json.loads((E/'queue_summary.json').read_text())
assert len(rows)==30
for s in summary:
    r=[x for x in rows if x['policy']==s['policy']]
    assert sorted(int(x['seed']) for x in r)==list(range(20261026,20261036))
    for field in ('stale_decision_rate','background_processed','useful_processed'):
        assert math.isclose(sum(float(x[field]) for x in r)/10,s['mean_'+field],rel_tol=1e-12)
assert run()==json.loads((E/'domain_examples.json').read_text())
assert validation['protocol_tests_passed'] and validation['protocol_demo_passed']

probe=json.loads(subprocess.check_output([shutil.which('ffprobe'),'-v','error','-show_streams','-show_format','-of','json',str(video)]))
vs=next(s for s in probe['streams'] if s['codec_type']=='video')
au=next(s for s in probe['streams'] if s['codec_type']=='audio')
sub=next(s for s in probe['streams'] if s['codec_type']=='subtitle')
assert (vs['width'],vs['height'],vs['r_frame_rate'])==(1920,1080,'24/1')
assert vs['codec_name']=='h264' and au['codec_name']=='aac' and sub['codec_name']=='mov_text'
assert abs(float(probe['format']['duration'])-160)<.1
subprocess.run([shutil.which('ffmpeg'),'-v','error','-i',str(video),'-map','0:v','-map','0:a','-f','null','-'],check=True)
audio=json.loads((E/'audio_validation.json').read_text())
assert audio['rms_pcm16']>100 and audio['peak_pcm16']<32767
assert len(audio['segments'])==15
for r in audio['segments']:assert r['rendered_duration']+.2<r['window']

# Inspect encoded rather than only pre-encode frames.
times=[4,17,29,43,59,74,86,98,118,130,143,154]
qa=B/'encoded_qa';qa.mkdir(exist_ok=True)
for t in times:
    subprocess.run([shutil.which('ffmpeg'),'-v','error','-y','-ss',str(t),'-i',str(video),'-frames:v','1',str(qa/f'{t:03d}.png')],check=True)
sheet=Image.new('RGB',(1280,1080),'#0b1723')
for i,t in enumerate(times):
    im=Image.open(qa/f'{t:03d}.png');im.thumbnail((426,240));x=(i%3)*426;y=(i//3)*270
    sheet.paste(im,(x,y));ImageDraw.Draw(sheet).text((x+8,y+242),f'{t:03d}s',fill='white')
sheet.save(D/'Contact_Sheet.jpg',quality=93)

media_source=HERE.parents[2]/'humanoid-laundry-queues/laundry_task/output/laundry_task_silent.mp4'
report=dict(passed=True,matched_runs=30,protocol_tests=16,domain_examples=3,
    video_duration_s=float(probe['format']['duration']),width=1920,height=1080,fps=24,
    codecs=['h264','aac','mov_text'],full_video_audio_decode='passed',
    audio_peak_pcm16=audio['peak_pcm16'],audio_rms_pcm16=audio['rms_pcm16'],
    retained_clip_sha256=sha(media_source),source_hashes_unchanged=True,
    encoded_frames_checked=times,delivery_sha256={p.name:sha(p) for p in D.iterdir() if p.is_file()},
    code_sha256={p.name:sha(p) for p in HERE.glob('*.py')},
    scope='Matched queue execution, centralized synthetic examples and existing protocol tests; no new physics run.')
(E/'final_validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in report.items() if 'sha256' not in k},indent=2))
