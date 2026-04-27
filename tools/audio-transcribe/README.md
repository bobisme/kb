# kb-audio-transcribe

Speaker-diarized transcripts from audio (`.m4a`, `.mp3`, `.wav`, `.mp4`) as kb-ready markdown sources.

This is the **prototype phase** for kb's audio ingestion (bone bn-3hfw). The output format is locked here so the Rust reimpl (bn-2lp2) can match it.

## Prereqs (one-time)

1. Accept pyannote ToS: <https://hf.co/pyannote/speaker-diarization-3.1>
2. `huggingface-cli login` (or set `HF_TOKEN`).
3. `ffmpeg` available on `PATH`.
4. CUDA-capable GPU recommended (CPU works but slow).

## Install

```bash
cd tools/audio-transcribe
uv venv --python 3.11
uv pip install -e .
```

## Run

```bash
uv run kb-audio-transcribe \
    --audio /path/to/recording.m4a \
    --title "2026-04-07 LiveRamp USB Team Intro Transcript" \
    --recording-date 2026-04-07 \
    --out /path/to/kb/raw/inbox/transcript.md
```

Or to stdout:

```bash
uv run kb-audio-transcribe --audio recording.m4a --title "..." --recording-date 2026-04-07
```

## Output format

```markdown
---
type: source
title: 2026-04-07 LiveRamp USB Team Intro Transcript
recording_date: 2026-04-07
audio_file: GMT20260407-143343_Recording.m4a
duration_seconds: 3612
speakers_detected: 3
asr_model: whisper-large-v3
diarization_model: pyannote/speaker-diarization-3.1
language: en
---

# Transcript

[00:00:00 → 00:00:25] **SPEAKER_00:** Hi everyone, can you hear me?

[00:00:26 → 00:01:10] **SPEAKER_01:** Thanks, all good. So today we wanted to walk through the new architecture.
```

## Cache

Transcripts are cached at `~/.kb/cache/transcripts/<sha256-of-audio>.md` — re-running on the same audio is a no-op and prints the cached path.

## Speaker aliases (optional)

```bash
uv run kb-audio-transcribe \
    --audio recording.m4a \
    --aliases speakers.toml \
    ...
```

Where `speakers.toml`:

```toml
SPEAKER_00 = "Bob"
SPEAKER_01 = "Xiaodong"
SPEAKER_02 = "Sean"
```
