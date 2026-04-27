"""kbtx render/parse — kb transcript markup convention.

See notes/transcript-spec.md for the full specification.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, Optional

from .pipeline import TranscriptArtifact


# ---------------------------------------------------------------------------
# Speaker IDs
# ---------------------------------------------------------------------------

_SPEAKER_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


def normalize_speaker_id(raw: str) -> str:
    """Convert a diarization label like 'SPEAKER_03' or an alias 'Xiaodong Ma'
    to a kbtx-legal `@<id>` form (without the leading '@').

    - Lowercase.
    - Spaces and dots become hyphens.
    - Drop characters that aren't `[a-z0-9_-]`.
    - Collapse runs of '-' to a single '-'.
    - Strip leading/trailing '-'.
    - Empty result falls back to 'speaker'.
    """
    s = raw.strip().lower()
    s = re.sub(r"[\s.]+", "-", s)
    s = re.sub(r"[^a-z0-9_-]", "", s)
    s = re.sub(r"-+", "-", s).strip("-")
    if not s:
        s = "speaker"
    if not _SPEAKER_ID_RE.match(s):
        s = "speaker"
    return s


# ---------------------------------------------------------------------------
# AST
# ---------------------------------------------------------------------------


@dataclass
class RosterEntry:
    speaker_id: str
    display_name: Optional[str] = None
    role: Optional[str] = None


@dataclass
class TurnNode:
    speaker_id: str
    start: float
    end: float
    text: str


@dataclass
class ActionLine:
    text: str  # the bracketed body, without the surrounding []


@dataclass
class TranscriptDoc:
    frontmatter: dict
    roster: list[RosterEntry] = field(default_factory=list)
    body: list = field(default_factory=list)  # mix of TurnNode and ActionLine


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------


def format_timestamp(seconds: float) -> str:
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{sec:02d}"


def render(
    artifact: TranscriptArtifact,
    *,
    title: str,
    recording_date: str,
    audio_filename: str,
    aliases: Optional[dict[str, str]] = None,
    pause_threshold_seconds: float = 5.0,
    extra_roster: Optional[Iterable[RosterEntry]] = None,
) -> str:
    """Render an artifact as a kbtx document.

    `aliases` maps diarization labels (e.g. `SPEAKER_03`) to roster IDs (e.g.
    `xiaodong`). Unmapped diarization labels are normalized in place
    (`SPEAKER_03` → `speaker_03`).
    """
    aliases = aliases or {}

    # Resolve each turn's speaker to a roster ID.
    resolved_turns: list[TurnNode] = []
    for t in artifact.turns:
        roster_id = aliases.get(t.speaker, t.speaker)
        roster_id = normalize_speaker_id(roster_id)
        resolved_turns.append(
            TurnNode(speaker_id=roster_id, start=t.start, end=t.end, text=t.text)
        )

    # Build roster (preserve first-seen order).
    seen: dict[str, None] = {}
    for tn in resolved_turns:
        seen.setdefault(tn.speaker_id, None)
    roster: list[RosterEntry] = [RosterEntry(speaker_id=sid) for sid in seen]

    # Augment roster with display-name info from aliases (TOML may carry richer
    # entries) and any caller-supplied entries.
    alias_to_display: dict[str, str] = {}
    for alias in aliases.values():
        rid = normalize_speaker_id(alias)
        if rid not in alias_to_display:
            alias_to_display[rid] = alias
    for entry in roster:
        if entry.speaker_id in alias_to_display:
            entry.display_name = alias_to_display[entry.speaker_id]
    if extra_roster is not None:
        extra_by_id = {e.speaker_id: e for e in extra_roster}
        for entry in roster:
            override = extra_by_id.get(entry.speaker_id)
            if override is not None:
                entry.display_name = override.display_name or entry.display_name
                entry.role = override.role or entry.role

    # ---- emit frontmatter ----
    out: list[str] = []
    out.append("---")
    out.append("type: source")
    out.append(f"title: {title}")
    out.append(f"recording_date: {recording_date}")
    out.append(f"audio_file: {audio_filename}")
    out.append(f"duration_seconds: {artifact.duration_seconds:.1f}")
    out.append(f"speakers_detected: {len(roster)}")
    out.append(f"asr_model: {artifact.asr_model}")
    out.append(f"diarization_model: {artifact.diarization_model}")
    out.append(f"language: {artifact.language}")
    out.append("---")
    out.append("")

    # ---- roster ----
    out.append("## Speakers")
    out.append("")
    for entry in roster:
        line = f"- @{entry.speaker_id}:"
        if entry.display_name:
            line += f" {entry.display_name}"
        else:
            line += " unknown"
        if entry.role:
            line += f" ({entry.role})"
        out.append(line)
    out.append("")

    # ---- body ----
    out.append("# Transcript")
    out.append("")

    prev_end: Optional[float] = None
    for tn in resolved_turns:
        if prev_end is not None and pause_threshold_seconds > 0:
            gap = tn.start - prev_end
            if gap >= pause_threshold_seconds:
                out.append(f"> [pause: {int(round(gap))}s]")
                out.append("")
        out.append(
            f"## @{tn.speaker_id} [{format_timestamp(tn.start)} → "
            f"{format_timestamp(tn.end)}]"
        )
        out.append("")
        out.append(tn.text.strip())
        out.append("")
        prev_end = tn.end

    return "\n".join(out)
