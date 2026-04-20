//! Markdown → HTML via `pulldown_cmark`.
//!
//! We enable tables, footnotes, strikethrough, and task lists — enough to
//! render a kb wiki faithfully without needing a full markdown runtime.
//! The output is assumed to be rendered inside the wiki shell template, so
//! this function returns only the rendered `<article>`-style body.

use pulldown_cmark::{Options, Parser, html};

/// Render a markdown string into an HTML fragment.
///
/// Any leading YAML frontmatter block (delimited by `---` on its own line)
/// is stripped before rendering so it doesn't leak into the page body. See
/// [`strip_frontmatter`] for the precise rules.
#[must_use]
pub fn render(source: &str) -> String {
    let source = strip_frontmatter(source);

    let mut options = Options::empty();
    options.insert(Options::ENABLE_TABLES);
    options.insert(Options::ENABLE_FOOTNOTES);
    options.insert(Options::ENABLE_STRIKETHROUGH);
    options.insert(Options::ENABLE_TASKLISTS);

    let parser = Parser::new_ext(source, options);
    let mut out = String::with_capacity(source.len() * 2);
    html::push_html(&mut out, parser);
    out
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
}
