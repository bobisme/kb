//! Markdown heading extraction for normalized-document heading ids.
//!
//! We only care about ATX-style H1/H2/H3 headings (`#`, `##`, `###`) at the
//! start of a line, which matches the canonical markdown we ingest. Headings
//! inside fenced code blocks are ignored so that indented `#` lines in code
//! examples don't masquerade as section anchors.

use kb_core::slug_from_title;

/// Extract slugs for H1/H2/H3 ATX headings from `markdown`.
///
/// Returns slugs in document order. Duplicates are preserved — callers that
/// need uniqueness should dedupe with their own disambiguation rules.
#[must_use]
pub fn extract_heading_ids(markdown: &str) -> Vec<String> {
    let mut ids = Vec::new();
    let mut in_fence = false;

    for line in markdown.lines() {
        let trimmed = line.trim_start();
        if trimmed.starts_with("```") || trimmed.starts_with("~~~") {
            in_fence = !in_fence;
            continue;
        }
        if in_fence {
            continue;
        }

        let Some(rest) = trimmed.strip_prefix('#') else {
            continue;
        };
        // Allow up to two more '#' for H2/H3; then require whitespace.
        let (level, rest) = parse_heading_prefix(rest);
        if level == 0 {
            continue;
        }
        let title = rest.trim().trim_end_matches('#').trim();
        if title.is_empty() {
            continue;
        }
        let slug = slug_from_title(title);
        if !slug.is_empty() {
            ids.push(slug);
        }
    }

    ids
}

/// Returns the number of additional `#` consumed (0, 1, or 2) and the
/// remainder after the heading prefix. Returns `(0, rest)` when the prefix
/// isn't a valid H1/H2/H3 heading.
fn parse_heading_prefix(rest: &str) -> (u8, &str) {
    let (extra, tail) = rest.strip_prefix("##").map_or_else(
        || {
            rest.strip_prefix('#')
                .map_or((0u8, rest), |t| (1u8, t))
        },
        |t| (2u8, t),
    );

    // After the optional extra #'s, H4+ is another '#', which we reject.
    if tail.starts_with('#') {
        return (0, rest);
    }
    // A valid heading requires whitespace after the hash run.
    match tail.chars().next() {
        Some(c) if c.is_whitespace() => (extra + 1, tail),
        _ => (0, rest),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_h1_h2_h3_in_order() {
        let md = "# Top\n\nBody\n\n## Middle section\n\n### Deep-Dive\n\n#### H4 should be skipped\n";
        assert_eq!(
            extract_heading_ids(md),
            vec![
                "top".to_string(),
                "middle-section".to_string(),
                "deep-dive".to_string(),
            ]
        );
    }

    #[test]
    fn ignores_headings_in_code_fences() {
        let md = "# Real\n\n```\n# Not a heading\n```\n\n## Also real\n";
        assert_eq!(
            extract_heading_ids(md),
            vec!["real".to_string(), "also-real".to_string()]
        );
    }

    #[test]
    fn strips_trailing_atx_closers() {
        let md = "# Title ##\n";
        assert_eq!(extract_heading_ids(md), vec!["title".to_string()]);
    }

    #[test]
    fn no_headings_yields_empty_list() {
        let md = "plain text\nwith no headings\n";
        assert!(extract_heading_ids(md).is_empty());
    }

    #[test]
    fn requires_space_after_hash() {
        let md = "#notaheading\n#tagline\n# real\n";
        assert_eq!(extract_heading_ids(md), vec!["real".to_string()]);
    }
}
