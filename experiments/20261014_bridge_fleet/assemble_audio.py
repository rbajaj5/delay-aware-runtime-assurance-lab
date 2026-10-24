"""Fit local TTS segments into caption intervals; never silently truncate speech."""
import json
import hashlib
import wave
from pathlib import Path
import subprocess

HERE=Path(__file__).resolve().parent


def main():
    out=HERE/'output';segments=json.loads((out/'narration_segments.json').read_text());parts=out/'narration_parts';audit=[]
    assert json.loads((parts/'narration_source.json').read_text())==segments,'Narration text changed; regenerate local TTS first.'
    for i,seg in enumerate(segments):
        raw=parts/f'segment_{i:02d}.wav';target=parts/f'aligned_{i:02d}.wav'
        duration=float(subprocess.check_output(['ffprobe','-v','error','-show_entries','format=duration','-of','default=nw=1:nk=1',str(raw)],text=True))
        slot=seg['end_s']-seg['start_s'];speed=max(1.,duration/(slot-.25))
        if speed>1.35:raise RuntimeError(f'Segment {i} is too long ({duration:.1f}s for {slot}s); shorten narration, not clip it.')
        samples=slot*48000
        subprocess.run(['ffmpeg','-y','-v','error','-i',str(raw),'-af',f'atempo={speed:.8f},loudnorm=I=-18:TP=-2:LRA=7,aresample=48000,asetpts=PTS-STARTPTS,apad=whole_len={samples},atrim=end_sample={samples}',
                        '-ac','1','-c:a','pcm_s16le',str(target)],check=True)
        with wave.open(str(target),'rb') as handle:assert handle.getnframes()==samples
        audit.append({'segment':i,'start_s':seg['start_s'],'end_s':seg['end_s'],'original_duration_s':duration,'tempo_factor':speed})
    with wave.open(str(out/'narration.wav'),'wb') as joined:
        joined.setnchannels(1);joined.setsampwidth(2);joined.setframerate(48000)
        for i in range(len(segments)):
            with wave.open(str(parts/f'aligned_{i:02d}.wav'),'rb') as segment:joined.writeframes(segment.readframes(segment.getnframes()))
    (out/'narration_validation.json').write_text(json.dumps({'voice':'Microsoft Zira Desktop','local_tts':True,'segments':audit,'speech_truncated':False,
        'narration_source_sha256':hashlib.sha256((out/'narration_segments.json').read_bytes()).hexdigest()},indent=2))
    print('Narration aligned to 180 seconds without cutting speech.')


if __name__=='__main__':main()
