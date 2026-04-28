//! Markdown → HTML via `pulldown_cmark`.
//!
//! We enable tables, footnotes, strikethrough, and task lists — enough to
//! render a kb wiki faithfully without needing a full markdown runtime.
//! The output is assumed to be rendered inside the wiki shell template, so
//! this function returns only the rendered `<article>`-style body.

use pulldown_cmark::{CowStr, Event, HeadingLevel, Options, Parser, Tag, html};

use kb_core::transcript;

/// Render a markdown string into an HTML fragment.
///
/// Any leading YAML frontmatter block (delimited by `---` on its own line)
/// is stripped before rendering so it doesn't leak into the page body. See
/// [`strip_frontmatter`] for the precise rules.
///
/// kbtx turn headings (`## @<id> [HH:MM:SS → HH:MM:SS]`) get a stable
/// `id="<id>-<HH>-<MM>-<SS>"` attribute so citations of the form
/// `[src-foo#xiaodong-00-01-35]` resolve cleanly.
#[must_use]
pub fn render(source: &str) -> String {
    let source = strip_frontmatter(source);

    let mut options = Options::empty();
    options.insert(Options::ENABLE_TABLES);
    options.insert(Options::ENABLE_FOOTNOTES);
    options.insert(Options::ENABLE_STRIKETHROUGH);
    options.insert(Options::ENABLE_TASKLISTS);

    let events: Vec<Event> = Parser::new_ext(source, options).collect();
    let events = inject_turn_anchors(events);

    let mut out = String::with_capacity(source.len() * 2);
    html::push_html(&mut out, events.into_iter());
    out
}

/// Walk the event stream and, for each H2 whose text matches a kbtx turn
/// heading, replace the heading tag with one that has `id="<anchor>"` set.
/// Non-matching headings are passed through unchanged.
fn inject_turn_anchors(mut events: Vec<Event<'_>>) -> Vec<Event<'_>> {
    let len = events.len();
    let mut i = 0;
    while i < len {
        let is_h2_start = matches!(
            &events[i],
            Event::Start(Tag::Heading { level: HeadingLevel::H2, id: None, .. })
        );
        if !is_h2_start {
            i += 1;
            continue;
        }
        // Find the matching end event and collect heading text.
        let mut j = i + 1;
        let mut text = String::new();
        while j < len {
            match &events[j] {
                Event::End(pulldown_cmark::TagEnd::Heading(HeadingLevel::H2)) => break,
                Event::Text(t) | Event::Code(t) => text.push_str(t),
                _ => {}
            }
            j += 1;
        }
        if let Some(anchor) = parse_turn_heading_anchor(&text) {
            // Replace the start tag with one that carries the id.
            if let Event::Start(Tag::Heading {
                level,
                classes,
                attrs,
                ..
            }) = std::mem::replace(&mut events[i], Event::SoftBreak)
            {
                events[i] = Event::Start(Tag::Heading {
                    level,
                    id: Some(CowStr::from(anchor)),
                    classes,
                    attrs,
                });
            }
        }
        i = j + 1;
    }
    events
}

/// If `text` matches `@<id> [HH:MM:SS → HH:MM:SS]` (or the ASCII `->` fallback),
/// return the canonical anchor.
fn parse_turn_heading_anchor(text: &str) -> Option<String> {
    let trimmed = text.trim();
    let after_at = trimmed.strip_prefix('@')?;
    let bracket_open = after_at.find(" [")?;
    let speaker_id = after_at[..bracket_open].trim();
    if speaker_id.is_empty() {
        return None;
    }
    let after_open = &after_at[bracket_open + 2..];
    let bracket_close = after_open.rfind(']')?;
    let timespec = &after_open[..bracket_close];
    let arrow_pos = timespec
        .find(" → ")
        .map(|i| (i, " → ".len()))
        .or_else(|| timespec.find(" -> ").map(|i| (i, " -> ".len())))?;
    let start_str = timespec[..arrow_pos.0].trim();
    let mut parts = start_str.split(':');
    let h: u32 = parts.next()?.parse().ok()?;
    let m: u32 = parts.next()?.parse().ok()?;
    let s: u32 = parts.next()?.parse().ok()?;
    if parts.next().is_some() {
        return None;
    }
    let total = h * 3600 + m * 60 + s;
    Some(transcript::turn_anchor(speaker_id, total))
}

/// Strip a leading YAML frontmatter block, if present.
///
/// The frontmatter must start at byte 0 with a line that is exactly `---`
/// (optional `\r` before the newline); it is terminated by the next line
/// that is exactly `---` (optional trailing whitespace / `\r`). The
/// returned slice begins just after the closing delimiter's newline.
///
/// If the input does not start with `---` on its own line, or if no closing
/// `---` line is found, the input is returned unchanged. This keeps a stray
/// `---` horizontal rule at the top of a page from being mangled.
#[must_use]
pub fn strip_frontmatter(source: &str) -> &str {
    // Must open with `---` on the very first line.
    let Some(rest) = source
        .strip_prefix("---\n")
        .or_else(|| source.strip_prefix("---\r\n"))
    else {
        return source;
    };

    // Scan line by line for the closing `---`. Track byte offsets so we can
    // return a slice into `source` that starts after the closing newline.
    let base = source.len() - rest.len();
    let mut cursor = 0usize;
    while cursor <= rest.len() {
        let line_end = rest[cursor..]
            .find('\n')
            .map_or(rest.len(), |i| cursor + i);
        let line = &rest[cursor..line_end];
        let trimmed = line.trim_end_matches('\r').trim_end();
        if trimmed == "---" {
            // Found the closer — return the slice after its newline. If the
            // closing `---` is the final line with no trailing newline, the
            // remainder is empty.
            let after = if line_end < rest.len() {
                line_end + 1
            } else {
                line_end
            };
            return &source[base + after..];
        }
        if line_end == rest.len() {
            break;
        }
        cursor = line_end + 1;
    }

    // No closer — treat as not-frontmatter and leave the input alone.
    source
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn renders_headings_and_emphasis() {
        let html = render("# Hello\n\n**world**\n");
        assert!(html.contains("<h1>Hello</h1>"));
        assert!(html.contains("<strong>world</strong>"));
    }

    #[test]
    fn renders_tables() {
        let html = render("| a | b |\n|---|---|\n| 1 | 2 |\n");
        assert!(html.contains("<table>"));
        assert!(html.contains("<td>1</td>"));
    }

    #[test]
    fn renders_code_blocks() {
        let html = render("```rust\nfn main() {}\n```\n");
        assert!(html.contains("<pre><code"));
        assert!(html.contains("fn main()"));
    }

    #[test]
    fn strips_frontmatter_before_render() {
        let src = "---\nid: concept:aries\nname: ARIES\naliases:\n  - a\n  - b\n---\n# Body\n\ntext\n";
        let html = render(src);
        assert!(!html.contains("id: concept:aries"));
        assert!(!html.contains("name: ARIES"));
        assert!(!html.contains("aliases"));
        assert!(html.contains("<h1>Body</h1>"));
        assert!(html.contains("text"));
    }

    #[test]
    fn no_frontmatter_is_unchanged() {
        let src = "# Title\n\nbody text\n";
        assert_eq!(strip_frontmatter(src), src);
    }

    #[test]
    fn top_level_hr_without_closing_triple_dash_is_preserved() {
        // A page that opens with `---` as a horizontal rule (not a YAML
        // block — no closing `---` line follows) should be left alone so
        // pulldown-cmark can render it as an `<hr>` / setext heading.
        let src = "---\n\n# Heading\n\nbody\n";
        assert_eq!(strip_frontmatter(src), src);
    }

    #[test]
    fn empty_frontmatter_is_stripped() {
        let src = "---\n---\n# Body\n";
        assert_eq!(strip_frontmatter(src), "# Body\n");
        let html = render(src);
        assert!(html.contains("<h1>Body</h1>"));
        assert!(!html.contains("---"));
    }

    #[test]
    fn frontmatter_with_crlf_line_endings() {
        let src = "---\r\nid: x\r\n---\r\n# Body\r\n";
        let stripped = strip_frontmatter(src);
        assert_eq!(stripped, "# Body\r\n");
    }

    #[test]
    fn frontmatter_closer_with_trailing_whitespace() {
        // A closing `---` with trailing spaces should still be recognized.
        let src = "---\nid: x\n---   \nbody\n";
        assert_eq!(strip_frontmatter(src), "body\n");
    }

    #[test]
    fn frontmatter_at_end_of_file_without_trailing_newline() {
        // `---\nid: x\n---` with no content after and no trailing newline
        // should strip to empty.
        let src = "---\nid: x\n---";
        assert_eq!(strip_frontmatter(src), "");
    }

    #[test]
    fn kbtx_turn_heading_gets_anchor_id() {
        let src = "## @xiaodong [00:01:35 → 00:02:20]\n\nbody.\n";
        let html = render(src);
        // pulldown-cmark renders heading IDs as `id="..."`.
        assert!(
            html.contains(r#"id="xiaodong-00-01-35""#),
            "expected anchor id, got: {html}"
        );
    }

    #[test]
    fn kbtx_turn_heading_ascii_arrow_works() {
        let src = "## @joshua [00:00:01 -> 00:00:25]\n\nbody.\n";
        let html = render(src);
        assert!(
            html.contains(r#"id="joshua-00-00-01""#),
            "expected anchor id, got: {html}"
        );
    }

    #[test]
    fn non_transcript_h2_unchanged() {
        let src = "## Summary\n\nbody.\n";
        let html = render(src);
        assert!(html.contains("<h2>Summary</h2>"), "got: {html}");
        assert!(!html.contains(r#"id=""#), "should not have an id");
    }
}
