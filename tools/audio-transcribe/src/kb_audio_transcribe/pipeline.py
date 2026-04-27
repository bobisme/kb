"""WhisperX pipeline: transcribe → align → diarize → speaker-merge turns."""
from __future__ import annotations

import gc
import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class PipelineConfig:
    audio_path: Path
    asr_model: str = "large-v3"
    device: str = "cuda"
    compute_type: str = "float16"  # for cuda; auto-overridden for cpu
    batch_size: int = 16
    language: Optional[str] = None
    min_speakers: Optional[int] = None
    max_speakers: Optional[int] = None
    hf_token: Optional[str] = None
    diar_model_id: str = "pyannote/speaker-diarization-community-1"


@dataclass
class Turn:
    start: float
    end: float
    speaker: str
    text: str


@dataclass
class TranscriptArtifact:
    turns: list[Turn]
    duration_seconds: float
    language: str
    asr_model: str
    diarization_model: str
    raw_segments: list[dict] = field(default_factory=list)


def _ensure_wav(audio_path: Path, tmpdir: Path) -> Path:
    """Convert any audio container to mono 16kHz wav for whisperx.

    Whisperx accepts most formats but its `load_audio` shells out to ffmpeg
    anyway and the explicit step makes failures show up earlier with clear
    messages.
    """
    if audio_path.suffix.lower() == ".wav":
        return audio_path
    out = tmpdir / "audio.wav"
    cmd = [
        "ffmpeg", "-nostdin", "-loglevel", "error",
        "-i", str(audio_path),
        "-ac", "1",
        "-ar", "16000",
        "-c:a", "pcm_s16le",
        str(out),
    ]
    subprocess.run(cmd, check=True)
    return out


def run_pipeline(cfg: PipelineConfig) -> TranscriptArtifact:
    # Defer heavy imports so --help is fast.
    import torch
    import whisperx

    if cfg.device == "cpu":
        compute_type = "int8"  # faster-whisper default for CPU
    else:
        compute_type = cfg.compute_type

    with tempfile.TemporaryDirectory() as tmpstr:
        tmpdir = Path(tmpstr)
        wav_path = _ensure_wav(cfg.audio_path, tmpdir)
        audio = whisperx.load_audio(str(wav_path))
        duration = float(len(audio)) / 16000.0

        # --- ASR ---
        asr_model = whisperx.load_model(
            cfg.asr_model,
            cfg.device,
            compute_type=compute_type,
            language=cfg.language,
        )
        result = asr_model.transcribe(audio, batch_size=cfg.batch_size)
        language = result["language"]

        # Free ASR before loading alignment model.
        del asr_model
        gc.collect()
        if cfg.device == "cuda":
            torch.cuda.empty_cache()

        # --- alignment ---
        align_model, align_metadata = whisperx.load_align_model(
            language_code=language,
            device=cfg.device,
        )
        result = whisperx.align(
            result["segments"],
            align_model,
            align_metadata,
            audio,
            cfg.device,
            return_char_alignments=False,
        )
        del align_model
        gc.collect()
        if cfg.device == "cuda":
            torch.cuda.empty_cache()

        # --- diarization ---
        token = cfg.hf_token or os.environ.get("HF_TOKEN")
        diarize_model = whisperx.diarize.DiarizationPipeline(
            model_name=cfg.diar_model_id,
            token=token,
            device=cfg.device,
        )
        diar_kwargs: dict = {}
        if cfg.min_speakers is not None:
            diar_kwargs["min_speakers"] = cfg.min_speakers
        if cfg.max_speakers is not None:
            diar_kwargs["max_speakers"] = cfg.max_speakers
        diarize_segments = diarize_model(audio, **diar_kwargs)

        result = whisperx.assign_word_speakers(diarize_segments, result)

    turns = merge_turns(result.get("segments", []))
    return TranscriptArtifact(
        turns=turns,
        duration_seconds=duration,
        language=language,
        asr_model=cfg.asr_model,
        diarization_model=cfg.diar_model_id,
        raw_segments=result.get("segments", []),
    )


INTRA_TURN_GAP_SECONDS = 10.0
MAX_TURN_DURATION_SECONDS = 300.0  # 5 minutes — split monologues into readable chunks


def merge_turns(segments: list[dict]) -> list[Turn]:
    """Whisperx returns sentence-level segments tagged with speaker; merge consecutive
    same-speaker segments into single turns when the gap is small enough that no
    other speaker plausibly intervened. Caps a single turn's duration so a single
    speaker monologue still gets visual breaks."""
    turns: list[Turn] = []
    for seg in segments:
        speaker = seg.get("speaker") or "SPEAKER_UNKNOWN"
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", start))
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        can_merge = (
            turns
            and turns[-1].speaker == speaker
            and (start - turns[-1].end) < INTRA_TURN_GAP_SECONDS
            and (end - turns[-1].start) < MAX_TURN_DURATION_SECONDS
        )
        if can_merge:
            turns[-1].end = end
            turns[-1].text = (turns[-1].text + " " + text).strip()
        else:
            turns.append(Turn(start=start, end=end, speaker=speaker, text=text))
    return turns
