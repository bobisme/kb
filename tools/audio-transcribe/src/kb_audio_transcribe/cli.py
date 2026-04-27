"""CLI entry point — see README.md for usage."""
from __future__ import annotations

import hashlib
import json
import sys
import tomllib
from pathlib import Path
from typing import Optional

import click

from .pipeline import (
    PipelineConfig,
    TranscriptArtifact,
    merge_turns,
    run_pipeline,
)


CACHE_ROOT = Path.home() / ".kb" / "cache" / "transcripts"


def _audio_hash(path: Path) -> str:
    """SHA-256 of the audio file bytes."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_aliases(path: Optional[Path]) -> dict[str, str]:
    if path is None:
        return {}
    with path.open("rb") as f:
        return tomllib.load(f)


@click.command()
@click.option("--audio", "audio_path", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True)
@click.option("--title", required=True, help="Source title (used as markdown frontmatter title)")
@click.option("--recording-date", required=True, help="Recording date in YYYY-MM-DD")
@click.option("--out", "out_path", type=click.Path(dir_okay=False, path_type=Path), default=None,
              help="Write markdown here. Default: stdout.")
@click.option("--aliases", type=click.Path(exists=True, dir_okay=False, path_type=Path), default=None,
              help="TOML file mapping SPEAKER_NN → real name.")
@click.option("--model", default="large-v3", show_default=True,
              help="Whisper model size (tiny/base/small/medium/large-v3).")
@click.option("--device", default="auto", show_default=True,
              help="Device for ASR + diarization (cuda/cpu/auto).")
@click.option("--language", default=None, help="Force language code (e.g. en). Default: auto-detect.")
@click.option("--min-speakers", type=int, default=None, help="Hint: minimum speakers.")
@click.option("--max-speakers", type=int, default=None, help="Hint: maximum speakers.")
@click.option("--no-cache", is_flag=True, help="Bypass transcript cache.")
@click.option("--hf-token", envvar="HF_TOKEN", default=None,
              help="HuggingFace token (or use HF_TOKEN env / huggingface-cli login).")
def main(
    audio_path: Path,
    title: str,
    recording_date: str,
    out_path: Optional[Path],
    aliases: Optional[Path],
    model: str,
    device: str,
    language: Optional[str],
    min_speakers: Optional[int],
    max_speakers: Optional[int],
    no_cache: bool,
    hf_token: Optional[str],
) -> None:
    """Transcribe + diarize audio into kb-ready markdown."""
    audio_path = audio_path.resolve()
    cache_key = _audio_hash(audio_path)
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    artifact_cache = CACHE_ROOT / f"{cache_key}.json"

    if artifact_cache.exists() and not no_cache:
        click.echo(f"[cache hit] {artifact_cache}", err=True)
        artifact = _load_artifact(artifact_cache)
    else:
        if device == "auto":
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"

        cfg = PipelineConfig(
            audio_path=audio_path,
            asr_model=model,
            device=device,
            language=language,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            hf_token=hf_token,
        )
        click.echo(f"[transcribe] {audio_path.name} on {device} (model={model})...", err=True)
        artifact = run_pipeline(cfg)
        _save_artifact(artifact, artifact_cache)
        click.echo(f"[cached] {artifact_cache}", err=True)

    markdown = render_markdown(
        artifact,
        title=title,
        recording_date=recording_date,
        audio_filename=audio_path.name,
        aliases=_load_aliases(aliases),
    )

    if out_path is None:
        sys.stdout.write(markdown)
    else:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(markdown)
        click.echo(f"[wrote] {out_path}", err=True)


def render_markdown(
    artifact: TranscriptArtifact,
    *,
    title: str,
    recording_date: str,
    audio_filename: str,
    aliases: dict[str, str],
) -> str:
    """Render the locked markdown source format."""
    speakers_seen: list[str] = []
    for turn in artifact.turns:
        if turn.speaker not in speakers_seen:
            speakers_seen.append(turn.speaker)

    def label(s: str) -> str:
        return aliases.get(s, s)

    fm = [
        "---",
        "type: source",
        f"title: {title}",
        f"recording_date: {recording_date}",
        f"audio_file: {audio_filename}",
        f"duration_seconds: {artifact.duration_seconds:.1f}",
        f"speakers_detected: {len(speakers_seen)}",
        f"asr_model: {artifact.asr_model}",
        f"diarization_model: {artifact.diarization_model}",
        f"language: {artifact.language}",
        "---",
        "",
        "# Transcript",
        "",
    ]

    body = []
    for turn in artifact.turns:
        body.append(
            f"[{_format_ts(turn.start)} → {_format_ts(turn.end)}] "
            f"**{label(turn.speaker)}:** {turn.text.strip()}"
        )
        body.append("")

    return "\n".join(fm) + "\n" + "\n".join(body)


def _format_ts(seconds: float) -> str:
    """Format seconds as HH:MM:SS."""
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{sec:02d}"


def _save_artifact(artifact: TranscriptArtifact, path: Path) -> None:
    payload = {
        "duration_seconds": artifact.duration_seconds,
        "language": artifact.language,
        "asr_model": artifact.asr_model,
        "diarization_model": artifact.diarization_model,
        "raw_segments": artifact.raw_segments,
    }
    path.write_text(json.dumps(payload))


def _load_artifact(path: Path) -> TranscriptArtifact:
    """Reload from JSON cache. Re-runs `merge_turns` so format-config changes
    take effect without re-running ASR/diarization."""
    payload = json.loads(path.read_text())
    raw_segments = payload.get("raw_segments", [])
    turns = merge_turns(raw_segments)
    return TranscriptArtifact(
        turns=turns,
        duration_seconds=float(payload["duration_seconds"]),
        language=payload["language"],
        asr_model=payload["asr_model"],
        diarization_model=payload["diarization_model"],
        raw_segments=raw_segments,
    )


if __name__ == "__main__":
    main()
