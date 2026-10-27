"""Neural bilingual narration of public research explanations; no private inputs."""
import asyncio
import hashlib
import json
from pathlib import Path
import subprocess
import wave
import edge_tts

HERE = Path(__file__).resolve().parent


async def main():
    script = json.loads((HERE/"lesson.json").read_text(encoding="utf-8"))
    build, out = HERE/"build", HERE/"delivery"
    build.mkdir(exist_ok=True); out.mkdir(exist_ok=True)
    timeline, clock = [], 0.
    for i, chapter in enumerate(script["chapters"]):
        cues = [("zh", chapter["term"]+"。"), ("en", chapter["text"])]
        if i == len(script["chapters"])-1:
            cues = cues[1:]
        for lang, text in cues:
            voice = script["chinese_voice" if lang == "zh" else "english_voice"]
            fingerprint = hashlib.sha256(json.dumps([text, voice, script["rate"]], ensure_ascii=False).encode()).hexdigest()
            name = f"cue_{i:02}_{lang}"
            mp3, wav, meta = [build/(name+suffix) for suffix in (".mp3", ".wav", ".json")]
            if not (mp3.exists() and meta.exists() and json.loads(meta.read_text())["request_sha256"] == fingerprint):
                for attempt in range(3):
                    try:
                        await edge_tts.Communicate(text, voice, rate=script["rate"]).save(str(mp3))
                        break
                    except edge_tts.exceptions.NoAudioReceived:
                        if attempt == 2:
                            raise
                        await asyncio.sleep(5*(attempt+1))
                meta.write_text(json.dumps(dict(request_sha256=fingerprint)))
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(mp3), "-ar", "48000", "-ac", "1", str(wav)], check=True)
            with wave.open(str(wav), "rb") as f:
                duration = f.getnframes()/f.getframerate()
            cue = dict(chapter=i, lang=lang, text=text, voice=voice,
                caption=(f"{chapter['term']} | {chapter['pinyin']} | {chapter['english']}" if lang=="zh" else text),
                start=clock, end=clock+duration+.65, audio_start=clock+.15,
                audio_duration=duration, file=wav.name, sha256=hashlib.sha256(wav.read_bytes()).hexdigest(),
                profile=chapter["profile"], from_s=chapter["from_s"],
                to_s=chapter["to_s"] if lang=="en" else chapter["from_s"])
            timeline.append(cue); clock = cue["end"]
            print(name, round(duration, 2), flush=True)
    (out/"timeline.json").write_text(json.dumps(dict(duration_s=clock, cues=timeline,
        script_sha256=hashlib.sha256((HERE/"lesson.json").read_bytes()).hexdigest(), rate=script["rate"],
        method="Microsoft online neural TTS via edge-tts; no post-synthesis acceleration"), ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print("Total", clock, flush=True)


if __name__ == "__main__":
    asyncio.run(main())
