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
    regions.into_iter().find(|r| r.id == target_id).map(|region| {
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
        assert_eq!(regions[0].body(text), "\nThis is the summary.\nIt has multiple lines.\n");

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
        assert_eq!(slug_from_title("Background & Context"), "background-context");
        assert_eq!(slug_from_title("  Trimmed  "), "trimmed");
        assert_eq!(slug_from_title("Multi---Dash"), "multi-dash");
        assert_eq!(slug_from_title("v1.0 Release Notes"), "v1-0-release-notes");
    }
}
