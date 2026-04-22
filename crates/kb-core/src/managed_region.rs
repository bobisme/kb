#[derive(Debug, PartialEq, Eq, Clone)]
pub struct ManagedRegion<'a> {
    pub id: &'a str,
    pub content_start: usize,
    pub content_end: usize,
    pub full_start: usize,
    pub full_end: usize,
}

impl<'a> ManagedRegion<'a> {
    #[must_use]
    pub fn body(&self, text: &'a str) -> &'a str {
        &text[self.content_start..self.content_end]
    }
}

/// Extracts all managed regions from the given text.
///
/// Fences format:
/// `<!-- kb:begin id=slug -->`
/// `<!-- kb:end id=slug -->`
#[must_use]
pub fn extract_managed_regions(text: &str) -> Vec<ManagedRegion<'_>> {
    let mut regions = Vec::new();
    let mut search_idx = 0;

    let begin_prefix = "<!-- kb:begin id=";
    let tag_suffix = " -->";

    while let Some(start_pos) = text[search_idx..].find(begin_prefix) {
        let abs_start = search_idx + start_pos;
        let id_start = abs_start + begin_prefix.len();

        let Some(suffix_pos) = text[id_start..].find(tag_suffix) else {
            search_idx = id_start;
            continue;
        };

        let abs_suffix_start = id_start + suffix_pos;
        let id = &text[id_start..abs_suffix_start];
        let content_start = abs_suffix_start + tag_suffix.len();

        let end_tag = format!("<!-- kb:end id={id} -->");

        if let Some(end_pos) = text[content_start..].find(&end_tag) {
            let abs_end = content_start + end_pos;
            let full_end = abs_end + end_tag.len();

            regions.push(ManagedRegion {
                id,
                content_start,
                content_end: abs_end,
                full_start: abs_start,
                full_end,
            });

            search_idx = full_end;
        } else {
            search_idx = content_start;
        }
    }
    regions
}

/// Rewrites the content inside a managed region with the target ID.
///
/// Keeps the surrounding fences intact. Returns `Some(new_string)` if
/// the region was found and replaced, `None` otherwise.
#[must_use]
pub fn rewrite_managed_region(text: &str, target_id: &str, new_content: &str) -> Option<String> {
    let regions = extract_managed_regions(text);
    regions
        .into_iter()
        .find(|r| r.id == target_id)
        .map(|region| {
            let mut new_text = String::with_capacity(text.len() + new_content.len());
            new_text.push_str(&text[..region.content_start]);
            new_text.push_str(new_content);
            new_text.push_str(&text[region.content_end..]);
            new_text
        })
}

/// Derives a stable slug from a section title on first assignment.
///
/// Lowercases, replaces non-alphanumeric runs with hyphens, and trims edge hyphens.
/// Once written into a marker, the slug is never re-derived from the title —
/// renaming the section heading does not change the ID.
#[must_use]
pub fn slug_from_title(title: &str) -> String {
    let mut slug = String::new();
    let mut last_was_hyphen = true;
    for ch in title.chars() {
        if ch.is_alphanumeric() {
            slug.push(ch.to_ascii_lowercase());
            last_was_hyphen = false;
        } else if !last_was_hyphen {
            slug.push('-');
            last_was_hyphen = true;
        }
    }
    if slug.ends_with('-') {
        slug.pop();
    }
    slug
}

/// Default max length for filename slugs.
///
/// Applies to both wiki source pages and question output directories.
/// Chosen to keep browseable filenames under the ~255-char POSIX
/// `NAME_MAX` even when combined with the id prefix and extension, while
/// still leaving enough characters for a human-readable snippet of the
/// title.
pub const DEFAULT_FILENAME_SLUG_MAX_CHARS: usize = 60;

/// Derive a filename-safe slug from a human-facing title.
///
/// Unlike [`slug_from_title`], this variant is purpose-built for browseable
/// artifact filenames (e.g. `wiki/sources/src-1wz-<slug>.md` and
/// `outputs/questions/q-1xk-<slug>/`). It:
///
/// - lowercases ASCII,
/// - replaces any run of characters outside `[a-z0-9]` with a single `-`,
/// - trims leading and trailing `-`,
/// - truncates at `max_chars` on a word boundary (backing up to the last `-`
///   if the cap lands mid-word), and
/// - returns an empty string when the title collapses to nothing so callers
///   can fall back to an id-only filename.
///
/// Non-ASCII letters are dropped (not transliterated) so the result is always
/// ASCII-safe for the filesystem. Callers who want full Unicode support should
/// reach for [`slug_from_title`] instead — which keeps Unicode alphanumerics —
/// but that slug can break path handling on filesystems that normalize
/// differently from the ingester.
///
/// When `max_chars == 0` the function treats it as "no truncation" so tests
/// can exercise the unbounded form; production callers should pass
/// [`DEFAULT_FILENAME_SLUG_MAX_CHARS`].
#[must_use]
pub fn slug_for_filename(title: &str, max_chars: usize) -> String {
    let mut slug = String::new();
    let mut last_was_hyphen = true;
    for ch in title.chars() {
        let lower = ch.to_ascii_lowercase();
        if lower.is_ascii_alphanumeric() {
            slug.push(lower);
            last_was_hyphen = false;
        } else if !last_was_hyphen {
            slug.push('-');
            last_was_hyphen = true;
        }
    }
    while slug.ends_with('-') {
        slug.pop();
    }
    while slug.starts_with('-') {
        slug.remove(0);
    }

    if max_chars == 0 || slug.chars().count() <= max_chars {
        return slug;
    }

    // Truncate at max_chars; if that lands mid-word, back up to the last `-`.
    // Using `chars().count()` above guards against truncating multi-byte
    // codepoints mid-sequence — but since our slug is ASCII-only by
    // construction, byte offsets are equivalent to char offsets here.
    let mut truncated = slug[..max_chars].to_string();
    if let Some(next_char) = slug[max_chars..].chars().next() {
        if next_char != '-' {
            // Mid-word: back up to the last '-'.
            if let Some(last_dash) = truncated.rfind('-') {
                truncated.truncate(last_dash);
            } else {
                // No dash to fall back on — the whole truncated chunk is one
                // word. Drop the slug entirely rather than produce a mangled
                // prefix that misleads readers. The caller will fall back to
                // the id-only filename.
                return String::new();
            }
        }
    }
    while truncated.ends_with('-') {
        truncated.pop();
    }
    truncated
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_managed_regions() {
        let text = "\
Some prefix text.
<!-- kb:begin id=summary -->
This is the summary.
It has multiple lines.
<!-- kb:end id=summary -->
Middle text.
<!-- kb:begin id=details -->
Details go here.
<!-- kb:end id=details -->
Suffix text.";

        let regions = extract_managed_regions(text);
        assert_eq!(regions.len(), 2);

        assert_eq!(regions[0].id, "summary");
        assert_eq!(
            regions[0].body(text),
            "\nThis is the summary.\nIt has multiple lines.\n"
        );

        assert_eq!(regions[1].id, "details");
        assert_eq!(regions[1].body(text), "\nDetails go here.\n");
    }

    #[test]
    fn test_rewrite_managed_region() {
        let text = "\
Some prefix text.
<!-- kb:begin id=summary -->
Old summary.
<!-- kb:end id=summary -->
Middle text.";

        let new_text =
            rewrite_managed_region(text, "summary", "\nNew summary.\n").expect("region found");
        assert_eq!(
            new_text,
            "\
Some prefix text.
<!-- kb:begin id=summary -->
New summary.
<!-- kb:end id=summary -->
Middle text."
        );
    }

    #[test]
    fn test_rewrite_non_existent_region() {
        let text = "No regions here.";
        assert_eq!(rewrite_managed_region(text, "summary", "New"), None);
    }

    #[test]
    fn test_slug_from_title() {
        assert_eq!(slug_from_title("Executive Summary"), "executive-summary");
        assert_eq!(
            slug_from_title("Background & Context"),
            "background-context"
        );
        assert_eq!(slug_from_title("  Trimmed  "), "trimmed");
        assert_eq!(slug_from_title("Multi---Dash"), "multi-dash");
        assert_eq!(slug_from_title("v1.0 Release Notes"), "v1-0-release-notes");
    }

    #[test]
    fn slug_for_filename_basic() {
        assert_eq!(
            slug_for_filename("2026-04-07 LiveRamp USB team intro", 60),
            "2026-04-07-liveramp-usb-team-intro"
        );
        assert_eq!(
            slug_for_filename("Produce a mermaid graph of USB team", 60),
            "produce-a-mermaid-graph-of-usb-team"
        );
    }

    #[test]
    fn slug_for_filename_empty_and_nonalpha() {
        assert_eq!(slug_for_filename("", 60), "");
        assert_eq!(slug_for_filename("   ", 60), "");
        assert_eq!(slug_for_filename("!!!@#$%", 60), "");
        assert_eq!(slug_for_filename("-", 60), "");
        assert_eq!(slug_for_filename("---", 60), "");
    }

    #[test]
    fn slug_for_filename_strips_symbols() {
        assert_eq!(
            slug_for_filename("Background & Context / v2", 60),
            "background-context-v2"
        );
        assert_eq!(slug_for_filename("Multi---Dash!!!", 60), "multi-dash");
        assert_eq!(slug_for_filename("  Trimmed  ", 60), "trimmed");
    }

    #[test]
    fn slug_for_filename_truncates_on_word_boundary() {
        // 60-char cap on a long title: must not split mid-word.
        let title = "how does the observability pipeline handle retries and backoffs in practice";
        let slug = slug_for_filename(title, 60);
        assert!(slug.len() <= 60, "slug too long: {slug}");
        assert!(!slug.ends_with('-'), "trailing dash not stripped: {slug}");
        // Must end on a full word. The 60-char mark lands inside "backoffs";
        // truncation should back up to the previous `-`, dropping "backoffs".
        assert_eq!(
            slug,
            "how-does-the-observability-pipeline-handle-retries-and"
        );
    }

    #[test]
    fn slug_for_filename_empty_when_single_word_exceeds_cap() {
        // A title that's one enormous word has no dash to back up to; we'd
        // rather return empty (caller falls back to id-only) than emit a
        // misleading prefix.
        let slug = slug_for_filename("supercalifragilisticexpialidocious", 10);
        assert_eq!(slug, "");
    }

    #[test]
    fn slug_for_filename_no_truncation_when_within_cap() {
        assert_eq!(slug_for_filename("short title", 60), "short-title");
    }

    #[test]
    fn slug_for_filename_unicode_is_dropped() {
        // Non-ASCII chars are dropped, not transliterated — keep filesystem
        // behavior portable. If every alphanumeric is non-ASCII, the slug
        // collapses to empty.
        assert_eq!(slug_for_filename("café", 60), "caf");
        assert_eq!(slug_for_filename("日本語", 60), "");
        assert_eq!(slug_for_filename("emoji 🎉 party", 60), "emoji-party");
    }

    #[test]
    fn slug_for_filename_zero_max_disables_truncation() {
        let title = "a ".to_string() + &"word ".repeat(50);
        let slug = slug_for_filename(&title, 0);
        assert!(
            slug.len() > 60,
            "max_chars=0 must skip truncation: got {} chars",
            slug.len()
        );
    }

    #[test]
    fn slug_for_filename_handles_mid_boundary_at_dash() {
        // If the char exactly at `max_chars` is a dash, we're on a boundary
        // already — truncate without backing up.
        let title = "aaaa bbbb cccc dddd eeee ffff gggg hhhh iiii";
        let slug = slug_for_filename(title, 9);
        // Position 9 is the '-' after "aaaa-bbbb"; keep "aaaa-bbbb".
        assert_eq!(slug, "aaaa-bbbb");
    }
}
