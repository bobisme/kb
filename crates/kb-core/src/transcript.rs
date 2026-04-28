//! kbtx — kb transcript markup convention parser + renderer.
//!
//! See `notes/transcript-spec.md` for the format definition. In short:
//!
//! ```text
//! ---
//! type: source
//! title: ...
//! ---
//!
//! ## Speakers
//!
//! - @joshua: Joshua Cox (LiveRamp eng manager)
//!
//! # Transcript
//!
//! ## @joshua [00:00:01 → 00:00:53]
//!
//! turn body...
//!
//! > [pause: 6s]
//!
//! ## @xiaodong [00:01:00 → 00:01:35]
//!
//! turn body...
//! ```
//!
//! The parser is deliberately permissive: ASCII `->` is accepted in place of
//! the canonical `→` arrow, and roster entries may omit display name and
//! role. The renderer always emits the canonical form.

use std::fmt::Write;

use serde::{Deserialize, Serialize};
use thiserror::Error;

const HEADING_TRANSCRIPT: &str = "# Transcript";
const HEADING_SPEAKERS: &str = "## Speakers";
const ARROW_CANONICAL: &str = "→";
const ARROW_FALLBACK: &str = "->";

/// AST root.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct TranscriptDoc {
    /// YAML frontmatter, kept as raw text so we don't need a YAML lib here.
    pub frontmatter: String,
    pub roster: Vec<RosterEntry>,
    pub body: Vec<BodyNode>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RosterEntry {
    pub speaker_id: String,
    pub display_name: Option<String>,
    pub role: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum BodyNode {
    Turn(Turn),
    ActionLine(ActionLine),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Turn {
    pub speaker_id: String,
    pub start_seconds: u32,
    pub end_seconds: u32,
    /// Body of the turn; one or more paragraphs separated by blank lines.
    pub text: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ActionLine {
    pub text: String,
}

#[derive(Debug, Error)]
pub enum ParseError {
    #[error("missing closing --- for frontmatter")]
    UnterminatedFrontmatter,
    #[error("invalid roster entry on line {line}: {detail}")]
    InvalidRosterEntry { line: usize, detail: String },
    #[error("invalid turn heading on line {line}: {detail}")]
    InvalidTurnHeading { line: usize, detail: String },
    #[error("invalid action line on line {line}: {detail}")]
    InvalidActionLine { line: usize, detail: String },
    #[error("invalid timestamp: {0}")]
    InvalidTimestamp(String),
}

// ---------------------------------------------------------------------------
// Parser
// ---------------------------------------------------------------------------

/// Parse a kbtx-format transcript into a structured `TranscriptDoc`.
///
/// # Errors
///
/// Returns a `ParseError` for malformed frontmatter, malformed roster
/// entries, malformed turn headings, or invalid timestamps.
pub fn parse(input: &str) -> Result<TranscriptDoc, ParseError> {
    let mut doc = TranscriptDoc::default();
    let mut lines = input.lines().enumerate().peekable();

    // Frontmatter: optional, must be the very first non-empty content.
    while let Some(&(_, line)) = lines.peek() {
        if line.is_empty() {
            lines.next();
            continue;
        }
        if line.trim_end() == "---" {
            lines.next();
            doc.frontmatter = parse_frontmatter(&mut lines)?;
        }
        break;
    }

    // Optional roster.
    while let Some(&(_, line)) = lines.peek() {
        if line.trim().is_empty() {
            lines.next();
            continue;
        }
        if line.trim_end() == HEADING_SPEAKERS {
            lines.next();
            doc.roster = parse_roster(&mut lines)?;
        }
        break;
    }

    // Optional body.
    while let Some(&(_, line)) = lines.peek() {
        if line.trim().is_empty() {
            lines.next();
            continue;
        }
        if line.trim_end() == HEADING_TRANSCRIPT {
            lines.next();
            doc.body = parse_body(&mut lines)?;
        }
        break;
    }

    Ok(doc)
}

type Iter<'a> = std::iter::Peekable<std::iter::Enumerate<std::str::Lines<'a>>>;

fn parse_frontmatter(lines: &mut Iter) -> Result<String, ParseError> {
    let mut buf = String::new();
    for (_, line) in lines.by_ref() {
        if line.trim_end() == "---" {
            return Ok(buf.trim_end_matches('\n').to_string());
        }
        buf.push_str(line);
        buf.push('\n');
    }
    Err(ParseError::UnterminatedFrontmatter)
}

fn parse_roster(lines: &mut Iter) -> Result<Vec<RosterEntry>, ParseError> {
    let mut roster = Vec::new();
    while let Some(&(line_no, line)) = lines.peek() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            lines.next();
            continue;
        }
        if trimmed == HEADING_TRANSCRIPT || trimmed == HEADING_SPEAKERS {
            // End of roster section.
            break;
        }
        if !trimmed.starts_with("- @") {
            // Not a roster entry — body starts here.
            break;
        }
        roster.push(parse_roster_entry(line_no + 1, trimmed)?);
        lines.next();
    }
    Ok(roster)
}

fn parse_roster_entry(line_no: usize, line: &str) -> Result<RosterEntry, ParseError> {
    // Format: - @id: Display Name (role)
    let after_dash = line.strip_prefix("- ").ok_or_else(|| ParseError::InvalidRosterEntry {
        line: line_no,
        detail: "expected '- ' prefix".to_string(),
    })?;
    let after_at = after_dash.strip_prefix('@').ok_or_else(|| ParseError::InvalidRosterEntry {
        line: line_no,
        detail: "expected '@' before speaker id".to_string(),
    })?;
    let colon = after_at.find(':').ok_or_else(|| ParseError::InvalidRosterEntry {
        line: line_no,
        detail: "expected ':' after speaker id".to_string(),
    })?;
    let speaker_id = after_at[..colon].trim();
    if speaker_id.is_empty() {
        return Err(ParseError::InvalidRosterEntry {
            line: line_no,
            detail: "empty speaker id".to_string(),
        });
    }
    let rest = after_at[colon + 1..].trim();

    // Split off optional `(role)` at the end.
    let (display_name, role) = rest.rfind('(').map_or_else(
        || {
            if rest.is_empty() {
                (None, None)
            } else {
                (Some(rest.to_string()), None)
            }
        },
        |open| {
            if rest.ends_with(')') {
                let display = rest[..open].trim().to_string();
                let role = rest[open + 1..rest.len() - 1].trim().to_string();
                let display = if display.is_empty() { None } else { Some(display) };
                let role = if role.is_empty() { None } else { Some(role) };
                (display, role)
            } else {
                (Some(rest.to_string()), None)
            }
        },
    );

    let display_name = match display_name.as_deref() {
        Some("unknown") => None, // canonicalize "unknown" to no display name
        _ => display_name,
    };

    Ok(RosterEntry {
        speaker_id: speaker_id.to_string(),
        display_name,
        role,
    })
}

fn parse_body(lines: &mut Iter) -> Result<Vec<BodyNode>, ParseError> {
    let mut body = Vec::new();
    while let Some(&(line_no, line)) = lines.peek() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            lines.next();
            continue;
        }
        if let Some(heading) = trimmed.strip_prefix("## @") {
            lines.next();
            let turn = parse_turn(line_no + 1, heading, lines)?;
            body.push(BodyNode::Turn(turn));
        } else if let Some(action_text) = strip_action_line(trimmed) {
            lines.next();
            body.push(BodyNode::ActionLine(ActionLine {
                text: action_text.to_string(),
            }));
        } else {
            return Err(ParseError::InvalidTurnHeading {
                line: line_no + 1,
                detail: format!("unexpected content: {trimmed}"),
            });
        }
    }
    Ok(body)
}

fn strip_action_line(trimmed: &str) -> Option<&str> {
    let rest = trimmed.strip_prefix("> [")?;
    rest.strip_suffix(']')
}

fn parse_turn(line_no: usize, heading: &str, lines: &mut Iter) -> Result<Turn, ParseError> {
    // heading is the content after `## @`. e.g. "joshua [00:00:01 → 00:00:53]"
    let bracket_open = heading.find(" [").ok_or_else(|| ParseError::InvalidTurnHeading {
        line: line_no,
        detail: "expected ' [' before timestamp".to_string(),
    })?;
    let speaker_id = heading[..bracket_open].trim();
    if speaker_id.is_empty() {
        return Err(ParseError::InvalidTurnHeading {
            line: line_no,
            detail: "empty speaker id".to_string(),
        });
    }
    let after_open = &heading[bracket_open + 2..];
    let bracket_close = after_open.rfind(']').ok_or_else(|| ParseError::InvalidTurnHeading {
        line: line_no,
        detail: "expected closing ']'".to_string(),
    })?;
    let timespec = &after_open[..bracket_close];
    let (start, end) = parse_timespec(timespec)?;

    // Body: until next H2 heading (turn) or action line or EOF.
    let mut body_lines: Vec<&str> = Vec::new();
    while let Some(&(_, line)) = lines.peek() {
        let trimmed = line.trim();
        if trimmed.starts_with("## @") {
            break;
        }
        if strip_action_line(trimmed).is_some() {
            break;
        }
        body_lines.push(line);
        lines.next();
    }
    // Trim trailing blank lines.
    while body_lines.last().is_some_and(|s| s.trim().is_empty()) {
        body_lines.pop();
    }
    // Trim leading blank lines.
    while body_lines.first().is_some_and(|s| s.trim().is_empty()) {
        body_lines.remove(0);
    }

    Ok(Turn {
        speaker_id: speaker_id.to_string(),
        start_seconds: start,
        end_seconds: end,
        text: body_lines.join("\n"),
    })
}

fn parse_timespec(spec: &str) -> Result<(u32, u32), ParseError> {
    // "HH:MM:SS → HH:MM:SS" or "HH:MM:SS -> HH:MM:SS"
    let arrow = if spec.contains(ARROW_CANONICAL) {
        ARROW_CANONICAL
    } else if spec.contains(ARROW_FALLBACK) {
        ARROW_FALLBACK
    } else {
        return Err(ParseError::InvalidTimestamp(spec.to_string()));
    };
    let mut parts = spec.split(arrow).map(str::trim);
    let start = parts.next().ok_or_else(|| ParseError::InvalidTimestamp(spec.to_string()))?;
    let end = parts.next().ok_or_else(|| ParseError::InvalidTimestamp(spec.to_string()))?;
    Ok((parse_timestamp(start)?, parse_timestamp(end)?))
}

fn parse_timestamp(ts: &str) -> Result<u32, ParseError> {
    let mut parts = ts.split(':');
    let h: u32 = parts.next().and_then(|p| p.parse().ok()).ok_or_else(|| ParseError::InvalidTimestamp(ts.to_string()))?;
    let m: u32 = parts.next().and_then(|p| p.parse().ok()).ok_or_else(|| ParseError::InvalidTimestamp(ts.to_string()))?;
    let s: u32 = parts.next().and_then(|p| p.parse().ok()).ok_or_else(|| ParseError::InvalidTimestamp(ts.to_string()))?;
    if parts.next().is_some() {
        return Err(ParseError::InvalidTimestamp(ts.to_string()));
    }
    Ok(h * 3600 + m * 60 + s)
}

// ---------------------------------------------------------------------------
// Renderer
// ---------------------------------------------------------------------------

#[must_use]
pub fn render(doc: &TranscriptDoc) -> String {
    let mut out = String::new();

    if !doc.frontmatter.is_empty() {
        out.push_str("---\n");
        out.push_str(&doc.frontmatter);
        if !doc.frontmatter.ends_with('\n') {
            out.push('\n');
        }
        out.push_str("---\n\n");
    }

    if !doc.roster.is_empty() {
        out.push_str(HEADING_SPEAKERS);
        out.push_str("\n\n");
        for entry in &doc.roster {
            let _ = write!(out, "- @{}", entry.speaker_id);
            out.push(':');
            match (&entry.display_name, &entry.role) {
                (Some(name), Some(role)) => {
                    let _ = write!(out, " {name} ({role})");
                }
                (Some(name), None) => {
                    let _ = write!(out, " {name}");
                }
                (None, Some(role)) => {
                    let _ = write!(out, " unknown ({role})");
                }
                (None, None) => {
                    out.push_str(" unknown");
                }
            }
            out.push('\n');
        }
        out.push('\n');
    }

    if !doc.body.is_empty() {
        out.push_str(HEADING_TRANSCRIPT);
        out.push_str("\n\n");
        for node in &doc.body {
            match node {
                BodyNode::Turn(t) => {
                    let _ = write!(
                        out,
                        "## @{} [{} {} {}]\n\n{}\n\n",
                        t.speaker_id,
                        format_timestamp(t.start_seconds),
                        ARROW_CANONICAL,
                        format_timestamp(t.end_seconds),
                        t.text.trim_end(),
                    );
                }
                BodyNode::ActionLine(a) => {
                    let _ = write!(out, "> [{}]\n\n", a.text);
                }
            }
        }
    }

    // Strip a single trailing newline if present (we always want a single \n at EOF).
    if out.ends_with("\n\n") {
        out.pop();
    }
    out
}

#[must_use]
pub fn format_timestamp(seconds: u32) -> String {
    let h = seconds / 3600;
    let m = (seconds % 3600) / 60;
    let s = seconds % 60;
    format!("{h:02}:{m:02}:{s:02}")
}

/// Anchor slug for a turn — used by renderers (kb-web etc.) to give each turn
/// a stable HTML anchor: `<id>-<HH>-<MM>-<SS>`.
#[must_use]
pub fn turn_anchor(speaker_id: &str, start_seconds: u32) -> String {
    let h = start_seconds / 3600;
    let m = (start_seconds % 3600) / 60;
    let s = start_seconds % 60;
    format!("{speaker_id}-{h:02}-{m:02}-{s:02}")
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = "---\ntype: source\ntitle: Test\nrecording_date: 2026-04-27\nduration_seconds: 120.0\nlanguage: en\n---\n\n## Speakers\n\n- @joshua: Joshua Cox (eng manager)\n- @xiaodong: Xiaodong Ma\n- @speaker_03: unknown\n\n# Transcript\n\n## @joshua [00:00:01 → 00:00:25]\n\nFirst turn body. Goes here.\n\n> [pause: 6s]\n\n## @xiaodong [00:00:31 → 00:01:00]\n\nSecond turn body.\n";

    #[test]
    fn parses_full_doc() {
        let doc = parse(SAMPLE).expect("parse should succeed");
        assert!(doc.frontmatter.contains("title: Test"));
        assert_eq!(doc.roster.len(), 3);
        assert_eq!(doc.roster[0].speaker_id, "joshua");
        assert_eq!(doc.roster[0].display_name.as_deref(), Some("Joshua Cox"));
        assert_eq!(doc.roster[0].role.as_deref(), Some("eng manager"));
        assert_eq!(doc.roster[2].speaker_id, "speaker_03");
        assert_eq!(doc.roster[2].display_name, None);

        assert_eq!(doc.body.len(), 3);
        match &doc.body[0] {
            BodyNode::Turn(t) => {
                assert_eq!(t.speaker_id, "joshua");
                assert_eq!(t.start_seconds, 1);
                assert_eq!(t.end_seconds, 25);
                assert_eq!(t.text, "First turn body. Goes here.");
            }
            BodyNode::ActionLine(a) => panic!("expected Turn, got ActionLine {a:?}"),
        }
        match &doc.body[1] {
            BodyNode::ActionLine(a) => assert_eq!(a.text, "pause: 6s"),
            BodyNode::Turn(t) => panic!("expected ActionLine, got Turn {t:?}"),
        }
    }

    #[test]
    fn round_trip_byte_identical() {
        let doc = parse(SAMPLE).expect("sample parses");
        let rendered = render(&doc);
        assert_eq!(rendered, SAMPLE);
    }

    #[test]
    fn ascii_arrow_accepted() {
        let with_ascii = SAMPLE.replace(" → ", " -> ");
        let doc = parse(&with_ascii).expect("ascii arrow variant parses");
        // First turn parses fine.
        match &doc.body[0] {
            BodyNode::Turn(t) => assert_eq!(t.start_seconds, 1),
            BodyNode::ActionLine(_) => panic!("expected Turn at body[0]"),
        }
        // Renderer always uses canonical arrow, so round-trip is to canonical form.
        let rendered = render(&doc);
        assert!(rendered.contains(" → "));
        assert!(!rendered.contains(" -> "));
    }

    #[test]
    fn turn_anchor_matches_spec() {
        assert_eq!(turn_anchor("xiaodong", 95), "xiaodong-00-01-35");
        assert_eq!(turn_anchor("joshua", 0), "joshua-00-00-00");
        assert_eq!(turn_anchor("speaker_03", 4493), "speaker_03-01-14-53");
    }

    #[test]
    fn unterminated_frontmatter_errors() {
        let bad = "---\ntype: source\n";
        assert!(matches!(parse(bad), Err(ParseError::UnterminatedFrontmatter)));
    }

    #[test]
    fn parses_minimal_doc() {
        let s = "# Transcript\n\n## @speaker_00 [00:00:00 → 00:00:25]\n\nBody.\n";
        let doc = parse(s).expect("minimal doc parses");
        assert!(doc.frontmatter.is_empty());
        assert!(doc.roster.is_empty());
        assert_eq!(doc.body.len(), 1);
    }
}
