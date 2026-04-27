# kbtx — kb Transcript Format Specification

**Status:** draft v0.1
**Date:** 2026-04-27
**Bone:** bn-3h4h

A lightweight, plain-text transcript format for kb. Designed in the spirit of [Fountain](https://fountain.io) (a screenplay format) — minimal markup, reads cleanly without a renderer, parses with a small grammar, round-trips losslessly.

Files use the `.kbtx.md` double-extension and are valid Markdown. The double extension lets tooling recognize the format from the filename without reading content, while still rendering everywhere a regular `.md` does. The format is a *convention* layered on Markdown, not a fork.

## Goals

1. **Human-readable raw.** A transcript should read well in a terminal and in any Markdown viewer with no special support.
2. **Stable structure.** Parser sees clear boundaries for: speaker turns, speaker roster, action lines, source metadata.
3. **Citation-friendly.** Each speaker turn is independently addressable so citations can land on a specific utterance.
4. **Embedding-friendly.** Speaker turns are natural chunk boundaries for retrieval.
5. **Lossless round-trip.** Parse → AST → render → byte-identical output (modulo whitespace normalization).

## Non-goals

- Linguistic annotation (overlap, prosody, intonation). Use CHAT or Jefferson for that.
- Subtitle precision (per-word timestamps). Use VTT/SRT for that — kbtx can be derived from a VTT but doesn't replace it.
- Authoring beyond what a diarized transcript needs. Long-form essay structure is what regular Markdown is for.

## File Anatomy

A kbtx file has up to three parts in this order:

```
+-----------------------+
| 1. Frontmatter (YAML) |   --- ... ---
+-----------------------+
| 2. Speaker Roster     |   ## Speakers
+-----------------------+
| 3. Body               |   # Transcript  → ## @speaker-id [start → end] ...
+-----------------------+
```

All three are optional in the grammar but the body is the one that matters; tooling treats a file with only frontmatter as malformed.

### 1. Frontmatter

YAML frontmatter delimited by `---` lines. Required fields are part of kb's broader source contract; transcript-specific fields are added.

```yaml
---
type: source
title: 2026-04-07 LiveRamp USB Team Intro Transcript
recording_date: 2026-04-07
audio_file: GMT20260407-143343_Recording.m4a
duration_seconds: 4493.1
speakers_detected: 6
asr_model: whisper-large-v3
diarization_model: pyannote/speaker-diarization-community-1
language: en
---
```

| Field | Required | Notes |
|---|---|---|
| `type: source` | yes | Identifies this as a kb source. |
| `title` | yes | Source title used in wiki. |
| `recording_date` | yes | ISO date `YYYY-MM-DD`. |
| `duration_seconds` | yes | Float. |
| `language` | yes | ISO 639-1 code. |
| `audio_file` | recommended | Original audio filename for traceability. |
| `asr_model` | recommended | Model identifier (e.g. `whisper-large-v3`). |
| `diarization_model` | recommended | Pipeline ID. |
| `speakers_detected` | optional | Integer; informational. |

### 2. Speaker Roster

A `## Speakers` section listing every speaker referenced in the body, with optional display name and role.

```markdown
## Speakers

- @joshua: Joshua Cox (LiveRamp engineering manager)
- @xiaodong: Xiaodong Ma (USB team lead)
- @bob: Bob (consultant)
- @speaker_03: unknown
```

**Grammar.** Each entry is a single Markdown list item:

```
- @<id>: <display_name> [(<role>)]
```

- `@<id>` — speaker identifier; `[a-z0-9][a-z0-9_-]*`. Lowercase, kebab-or-snake. Must match exactly (case-sensitive) the speaker references in the body.
- `<display_name>` — free text up to an open-paren or end-of-line.
- `(<role>)` — optional role/affiliation in parentheses; one set per entry.
- For un-aliased anonymous diarization labels, use `@speaker_NN` matching the diarizer output, with display name `unknown` (or omitted).

The roster is **the source of truth** for speaker IDs. Aliases (mapping `SPEAKER_03` → `@xiaodong`) are applied at render time from a sidecar TOML; the canonical kbtx file always uses roster IDs.

The roster section is optional. If absent, the parser treats every `@id` referenced in the body as an implicit unknown-speaker roster entry.

### 3. Body

A `# Transcript` heading marks the body. The body is a sequence of:

- **Turns** — what a single speaker said, identified by a `## @<id> [<start> → <end>]` heading and a free-prose body.
- **Action lines** — non-speech events between turns, formatted as a blockquote whose content is bracketed: `> [<text>]`.

#### Turn

```markdown
## @xiaodong [00:01:35 → 00:02:20]

So USB is the new segment builder, replacing CSB. We own the API, backend,
job service, segment engine, and FedSQL translation.

The catalog metadata side has its own service that we depend on but don't
own.
```

**Grammar.**

```
## @<id> [<HH:MM:SS> → <HH:MM:SS>]
<blank line>
<paragraph>+
```

- `<id>` references a roster entry.
- `[<start> → <end>]` — start and end timestamps, U+2192 RIGHTWARDS ARROW (`→`). ASCII fallback `->` is also accepted by the parser; the renderer always emits `→`.
- Timestamps are zero-padded `HH:MM:SS`. Sub-second precision is dropped at render time but preserved in the cached artifact JSON.
- Body is one or more Markdown paragraphs separated by blank lines. Inline annotations like `[brief pause]` are allowed and pass through as plain text.

A turn's chunk anchor (used for citations and embeddings) is `<id>-<HH-MM-SS>` derived from the heading — e.g. `xiaodong-00-01-35`. Renderers must surface this as an HTML anchor.

#### Action line

A standalone blockquote whose content is fully bracketed:

```markdown
> [laughter]

> [screen share: USB architecture diagram]

> [overlap: SPEAKER_03 and SPEAKER_04 talking simultaneously]
```

These belong **between** turns, not inside them. For brief side-events that interrupt a single turn without breaking it, use an inline `[bracketed]` annotation in the turn body instead.

#### Inline annotation

Inside a turn body:

```markdown
## @bob [00:05:12 → 00:05:34]

So the Matricule service is the part that... [phone rings, brief pause]
...handles the actual materialization.
```

Plain Markdown text, ignored structurally by the parser; useful for human readability.

## Formal Grammar (PEG)

Whitespace and blank-line rules are GFM-standard. Below is the kbtx-specific structure:

```peg
File           ← Frontmatter? Roster? Body?

Frontmatter    ← "---" Newline YamlContent "---" Newline

Roster         ← "## Speakers" Newline RosterEntry+
RosterEntry    ← "- " SpeakerRef ":" Whitespace DisplayName Role? Newline
SpeakerRef     ← "@" [a-z0-9] [a-z0-9_-]*
DisplayName    ← (! "(" ! Newline .)+
Role           ← "(" (! ")" ! Newline .)+ ")"

Body           ← "# Transcript" Newline (Turn / ActionLine / Newline)*
Turn           ← TurnHeading Newline Newline Paragraph (Newline Newline Paragraph)*
TurnHeading    ← "## " SpeakerRef " [" Timestamp " " ("→" / "->") " " Timestamp "]"
ActionLine     ← "> [" BracketContent "]" Newline

Timestamp      ← Digit Digit ":" Digit Digit ":" Digit Digit
BracketContent ← (! "]" ! Newline .)+

Paragraph      ← Line (Newline Line)*
Line           ← (! Newline .)+
```

The parser is permissive about leading/trailing whitespace and accepts both `→` and `->` for the timestamp arrow. The renderer always emits the canonical form.

## Examples

### Minimal

```markdown
---
type: source
title: Standup 2026-04-15
recording_date: 2026-04-15
duration_seconds: 612.0
language: en
---

# Transcript

## @speaker_00 [00:00:00 → 00:00:25]

Yeah let's get started. I think we wanted to walk through the deploy plan.
```

### Full

```markdown
---
type: source
title: 2026-04-07 LiveRamp USB Team Intro Transcript
recording_date: 2026-04-07
audio_file: GMT20260407-143343_Recording.m4a
duration_seconds: 4493.1
speakers_detected: 3
asr_model: whisper-large-v3
diarization_model: pyannote/speaker-diarization-community-1
language: en
---

## Speakers

- @joshua: Joshua Cox (LiveRamp engineering manager)
- @xiaodong: Xiaodong Ma (USB team lead)
- @bob: Bob (consultant)

# Transcript

## @joshua [00:00:01 → 00:00:53]

proposal doc. But Bob's going to be coming on to help us kind of build out
some POCs on this local stack digital twin type service, mock service type
system. So this meeting is really for Bob to understand.

> [screen share: agenda doc]

## @xiaodong [00:01:35 → 00:02:20]

So USB is the new segment builder, replacing CSB. We own the API, backend,
job service, segment engine, and FedSQL translation. [phone rings] We
depend on Catalog and DSM but don't own them.

> [laughter]

## @bob [00:02:25 → 00:02:48]

Got it. Can you walk me through how a segment gets materialized today?
```

## Anchors and Citations

The renderer (kb-web, etc.) MUST emit each `## @<id> [<start> → <end>]` heading with a stable HTML anchor derived from the speaker id and the start timestamp:

```
<id>-<HH>-<MM>-<SS>
```

So `## @xiaodong [00:01:35 → 00:02:20]` becomes `#xiaodong-00-01-35`.

A kb citation referencing a specific turn:

```
[src-ujs#xiaodong-00-01-35]
```

resolves to that turn in the transcript. Citation lints and the resolution layer (`kb resolve`) understand this anchor form.

## Aliases (sidecar TOML)

Diarization produces anonymous labels (`SPEAKER_00`, `SPEAKER_01`, …). To rename them to roster IDs without rewriting the transcript every time, an alias file lives next to the audio:

```toml
# ~/chief/consulting/.../usb-team-intro.aliases.toml
SPEAKER_00 = "speaker_00"
SPEAKER_01 = "joshua"
SPEAKER_03 = "xiaodong"
SPEAKER_04 = "bob"
```

The transcribe tool consumes this at render time and substitutes both the body references and the roster section. The cached artifact JSON keeps the original `SPEAKER_NN` labels so re-renders with updated aliases are free.

## Migration from current format

The bn-3hfw prototype emits this format today (a subset):

```markdown
[00:00:01 → 00:00:53] **SPEAKER_03:** ...text...
```

This is **superseded** by:

```markdown
## @speaker_03 [00:00:01 → 00:00:53]

...text...
```

Migration is a one-shot text transform plus a re-render from cached artifact JSON. No re-running of ASR/diarization is needed thanks to the artifact cache.

## Design decisions

**Why `## @<id>` and not `[<id>]:` or `<id>:` ?**
H2 headings give us natural Markdown anchors out of the box, render with visible separation in any viewer, and signal clearly that each turn is an addressable unit. Inline `Speaker:` syntax (used by Otter, podcast transcripts) reads more like dialogue but loses the anchor and chunk boundary.

**Why `→` instead of `--`, `..`, or `to` ?**
`→` is unambiguous and visually distinctive. `--` collides with Markdown emphasis edge cases. `..` is too quiet. `to` is too verbose. The parser accepts `->` as a fallback for keyboards/editors that struggle with U+2192.

**Why `> [bracketed]` for action lines ?**
A standalone blockquote is already visually distinct in any Markdown viewer; bracketing the content disambiguates from regular quoted text. This is the clearest structural signal for "non-speech event between turns" without inventing new syntax.

**Why a separate Speakers section instead of inline declarations ?**
Centralizes the roster (debugging, alias management, audit). Avoids redeclaration on every turn. Mirrors Fountain's title page / dramatis personae split. Optional, so simple transcripts don't pay the cost.

**Why no per-paragraph timestamps inside a turn ?**
Speaker turns are the citation unit; intra-turn navigation hasn't been a real need in practice. The renderer can compute approximate paragraph timestamps from the cached artifact if a future feature wants them.

**Why no overlap markup like Jefferson `[ ` and `]` ?**
Diarization tools don't reliably mark overlaps and the markup harms readability. For overlapping speech we just split the turn or mark `> [overlap]`. If a corpus needs Jefferson-style precision, that's a different format.

## Future extensions (not v1)

- `## Topics` section with `[hh:mm:ss] topic name` lines for explicit chapter markers.
- `## Decisions` / `## Action items` sections lifted by an LLM post-pass.
- Word-level timestamps cached alongside the artifact JSON, surfaced in the renderer for "click any word to jump to that audio position" UX.
- Multi-language transcripts (one block per language with `language` annotation per turn).
- Speaker-confidence annotations.

## Synthetic pause action lines

When the cached artifact shows a silence between turns longer than a threshold, the renderer SHOULD emit a synthetic action line:

```markdown
> [pause: 23s]
```

**Default threshold: 5 seconds** between turn end and the next turn start. Configurable via the renderer (CLI flag `--pause-threshold-seconds`). Set to `0` to disable, or a high number to suppress most pauses.

Pauses are emitted **only between turns**, never inside a single turn's body — within-turn gaps below the merge threshold (`INTRA_TURN_GAP_SECONDS = 10s`) are absorbed into the turn body without annotation.

Pauses count toward neither speaker's turn duration; they are pure structural annotation.

## Open questions

1. **Should we standardize a `> [silence]` vs `> [non-speech audio]` distinction?** Defer to v2 once we have audio classification in the pipeline. For v1, only the explicit `> [pause: Ns]` synthesis above is supported.

## Round-trip test

A reference test for the parser/renderer:

```
parse(s) -> AST
render(AST) -> s'
assert normalize(s) == normalize(s')
```

Where `normalize` collapses runs of blank lines and strips trailing whitespace per line. The two LiveRamp transcripts (USB, Delta) are the canonical fixtures.
