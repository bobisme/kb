use std::path::{Path, PathBuf};
use std::sync::LazyLock;

use anyhow::Result;
use kb_core::{
    DEFAULT_FILENAME_SLUG_MAX_CHARS, rewrite_managed_region, slug_for_filename,
    slug_redundant_with_id,
};
use regex::Regex;
use serde_yaml::{Mapping, Number, Value};

const TITLE_REGION_ID: &str = "title";
const SUMMARY_REGION_ID: &str = "summary";
const KEY_TOPICS_REGION_ID: &str = "key_topics";
const CITATIONS_REGION_ID: &str = "citations";
/// bn-3ij3: per-page anchor list. Only emitted when the source carries
/// `<!-- kb:page N -->` markers (page-aware PDF ingest); for legacy or
/// non-PDF sources the section is omitted entirely.
const PAGES_REGION_ID: &str = "pages";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourcePageInput<'a> {
    pub page_id: &'a str,
    pub title: &'a str,
    pub source_document_id: &'a str,
    pub source_revision_id: &'a str,
    pub generated_at: u64,
    pub build_record_id: &'a str,
    pub summary: &'a str,
    pub key_topics: &'a [String],
    pub citations: &'a [String],
    /// Total page count of the source, when the source was ingested with
    /// page-aware extraction (bn-3ij3). When `None` or `Some(0)`, the
    /// `## Pages` managed region is omitted entirely. When `Some(N)` the
    /// region renders inline `<a name="page-K"></a>` anchors for K in
    /// 1..=N so citations of the form `[src-id p.K]` can deep-link to the
    /// right place via `wiki/sources/<src>.md#page-K`.
    pub pages: Option<u32>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourcePageArtifact {
    pub path: PathBuf,
    pub frontmatter: Mapping,
    pub body: String,
    pub markdown: String,
}

/// Compute the id-only source page path: `wiki/sources/<src-id>.md`.
///
/// Used when the caller does not have a title to derive a slug from — e.g.
/// the build graph rebuilder matching wiki-page nodes back to their owning
/// source document. For lookup of the on-disk file (which may carry a
/// slug suffix), callers must use [`resolve_source_page_path`] instead.
///
/// For *writing* a source page, use [`source_page_path_for`] which includes
/// the slugified title when one is available. Writes that use this function
/// would overwrite the id-only form and orphan any existing slugged file;
/// callers that produce files on disk should prefer the title-aware variant.
#[must_use]
pub fn source_page_path_for_id(source_id: &str) -> PathBuf {
    PathBuf::from("wiki/sources").join(format!("{}.md", slug_for_path(source_id)))
}

/// Compute the destination path for a source wiki page.
///
/// Either `wiki/sources/<src-id>-<slug>.md` (when the title slug is
/// non-empty and doesn't already begin with the id) or
/// `wiki/sources/<src-id>.md` (when the title collapses to an empty slug
/// OR the slug is itself the id / starts with `<id>-`).
///
/// `src_id` is used as the stable prefix (e.g. `src-1wz`); the slug is
/// appended after `-` and derived from `title` via [`slug_for_filename`] with
/// [`DEFAULT_FILENAME_SLUG_MAX_CHARS`]. Prefix-match lookups (`kb inspect
/// src-1wz`) still resolve because the id is always the first segment.
///
/// bn-6puc guard: a title that slugifies to `src-<id>` (or `src-<id>-...`)
/// would produce an ugly `wiki/sources/src-<id>-src-<id>.md` double.
/// Detect that case and emit just `<src-id>.md` instead — the user can
/// always write in a real title later, and a slug-equals-id outcome carries
/// no additional human-readable information beyond the id itself.
#[must_use]
pub fn source_page_path_for(source_id: &str, title: &str) -> PathBuf {
    let id_slug = slug_for_path(source_id);
    let title_slug = slug_for_filename(title, DEFAULT_FILENAME_SLUG_MAX_CHARS);
    let file_name = if title_slug.is_empty() || slug_redundant_with_id(&title_slug, &id_slug) {
        format!("{id_slug}.md")
    } else {
        format!("{id_slug}-{title_slug}.md")
    };
    PathBuf::from("wiki/sources").join(file_name)
}

/// Resolve the on-disk source wiki page path for `src_id`, returning `None`
/// when no file matches.
///
/// Accepts either the legacy `wiki/sources/<src-id>.md` form or the new
/// `wiki/sources/<src-id>-<slug>.md` form (introduced in bn-nlw9). When both
/// happen to coexist — a write-in-progress, for example — the id-slug form
/// is preferred. All callers that previously used
/// `format!("wiki/sources/{id}.md")` or [`source_page_path_for_id`] for
/// *lookups* must call this helper instead to stay forward-compatible.
///
/// The returned path is relative to `_root`; callers `join` it with the root
/// to get an absolute path. (The root argument is accepted for parity with
/// other resolvers even though the lookup is already rooted.)
#[must_use]
pub fn resolve_source_page_path(root: &Path, src_id: &str) -> Option<PathBuf> {
    let dir = root.join("wiki/sources");
    if !dir.exists() {
        return None;
    }
    let id_slug = slug_for_path(src_id);
    let id_only = dir.join(format!("{id_slug}.md"));
    let id_prefix = format!("{id_slug}-");

    let mut id_slug_match: Option<PathBuf> = None;
    if let Ok(entries) = std::fs::read_dir(&dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) != Some("md") {
                continue;
            }
            let Some(stem) = path.file_stem().and_then(|s| s.to_str()) else {
                continue;
            };
            if stem.starts_with(&id_prefix) {
                id_slug_match = Some(path);
                break;
            }
        }
    }
    if let Some(p) = id_slug_match {
        return Some(p);
    }
    if id_only.exists() {
        return Some(id_only);
    }
    None
}

/// Regex matching `![alt](path)` image references in markdown.
///
/// Mirrors the scanner in `kb_ingest::image_refs` but is redeclared here to
/// avoid pulling `kb-ingest` into kb-compile's dependency graph just for one
/// regex.
///
/// Capture groups:
/// - group 1: alt text
/// - group 2: angle-bracket-wrapped path (`<path with spaces.png>`)
/// - group 3: plain path (no whitespace)
///
/// An optional `CommonMark` title after the path is tolerated and discarded.
/// See bn-18qs: ingest now rewrites spaces-in-basename refs to the angle-
/// bracket form, so this mirror has to match them to re-anchor refs when
/// building the wiki source page.
static IMAGE_REF_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r#"!\[([^\]]*)\]\(\s*(?:<([^>]+)>|([^)\s"]+))(?:\s+"[^"]*")?\s*\)"#)
        .expect("hard-coded image regex is valid")
});

/// True if `path` contains any ASCII whitespace — the signal for emitting
/// the angle-bracket-wrapped form on rewrite.
fn path_needs_angle_wrap(path: &str) -> bool {
    path.chars().any(char::is_whitespace)
}

/// Rewrite `![alt](assets/<basename>)` references in `text` so they resolve
/// from the wiki source page's location.
///
/// Normalized sources store image refs as `assets/<basename>` — relative to
/// `.kb/normalized/<src>/source.md`. The wiki source page lives at
/// `wiki/sources/<src>.md`, so going up two components lands at the KB root;
/// from there we hop into `.kb/normalized/<src>/assets/` to reach the image.
/// The correct relative path from a wiki source page to its source's assets
/// is therefore `../../.kb/normalized/<src>/assets/<basename>`.
///
/// Only refs whose path starts with `assets/` are rewritten — we don't touch
/// external URLs or refs that have already been re-anchored.
#[must_use]
pub fn rewrite_summary_image_refs(text: &str, source_id: &str) -> String {
    let new_prefix = format!(
        "../../{}/{}/{source_id}/assets",
        kb_core::KB_DIR,
        kb_core::NORMALIZED_SUBDIR
    );
    rewrite_asset_refs_with_prefix(text, &new_prefix)
}

fn rewrite_asset_refs_with_prefix(text: &str, new_prefix: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut cursor = 0usize;

    for capture in IMAGE_REF_RE.captures_iter(text) {
        let whole = capture.get(0).expect("regex always has group 0");
        out.push_str(&text[cursor..whole.start()]);
        cursor = whole.end();

        let alt = capture.get(1).map_or("", |m| m.as_str());
        // Path lives in either capture group 2 (angle-bracket form) or
        // group 3 (plain form); callers should treat them uniformly.
        let path = capture
            .get(2)
            .or_else(|| capture.get(3))
            .map_or("", |m| m.as_str().trim());

        if let Some(rest) = path.strip_prefix("assets/") {
            // Preserve the angle-bracket form on output when the basename
            // contains whitespace so the rewritten ref stays valid
            // CommonMark (bn-18qs).
            let wrap = path_needs_angle_wrap(rest);
            out.push_str("![");
            out.push_str(alt);
            if wrap {
                out.push_str("](<");
            } else {
                out.push_str("](");
            }
            out.push_str(new_prefix);
            if !new_prefix.ends_with('/') {
                out.push('/');
            }
            out.push_str(rest);
            if wrap {
                out.push_str(">)");
            } else {
                out.push(')');
            }
        } else {
            out.push_str(whole.as_str());
        }
    }

    out.push_str(&text[cursor..]);
    out
}

/// Render or update a source wiki page with managed regions and source metadata.
///
/// # Errors
///
/// Returns an error if the existing markdown contains an unterminated frontmatter block
/// or if the generated frontmatter cannot be serialized to YAML.
pub fn render_source_page(
    input: &SourcePageInput<'_>,
    existing_markdown: Option<&str>,
) -> Result<SourcePageArtifact> {
    let frontmatter = build_frontmatter(input);
    let existing_body = existing_markdown
        .map(split_frontmatter)
        .transpose()?
        .flatten()
        .map_or_else(String::new, |(_, body)| body);
    let body = upsert_source_page_body(&existing_body, input);
    let markdown = serialize_frontmatter(&frontmatter, &body)?;

    Ok(SourcePageArtifact {
        path: source_page_path_for(input.source_document_id, input.title),
        frontmatter,
        body,
        markdown,
    })
}

/// Delete stale source wiki pages for `src_id` that don't match `keep_path`.
///
/// Used by the compile pipeline so that when a source's title changes,
/// the old `wiki/sources/<src-id>-<old-slug>.md` (or the legacy id-only
/// file) is cleaned up before writing the new page.
///
/// Matches both the id-only form (`<src-id>.md`) and any id-slug form
/// (`<src-id>-*.md`) under `wiki/sources/`. `keep_path` is exempt even if it
/// matches; everything else is removed.
///
/// # Errors
///
/// Returns an error when a matching stale file cannot be removed.
pub fn clean_stale_source_pages(root: &Path, src_id: &str, keep_path: &Path) -> Result<Vec<PathBuf>> {
    let dir = root.join("wiki/sources");
    if !dir.exists() {
        return Ok(Vec::new());
    }
    let id_slug = slug_for_path(src_id);
    let id_prefix = format!("{id_slug}-");
    let id_only_stem = id_slug;

    let keep_abs = if keep_path.is_absolute() {
        keep_path.to_path_buf()
    } else {
        root.join(keep_path)
    };

    let mut removed = Vec::new();
    for entry in std::fs::read_dir(&dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) != Some("md") {
            continue;
        }
        let Some(stem) = path.file_stem().and_then(|s| s.to_str()) else {
            continue;
        };
        let matches_id = stem == id_only_stem || stem.starts_with(&id_prefix);
        if !matches_id {
            continue;
        }
        if path == keep_abs {
            continue;
        }
        std::fs::remove_file(&path)?;
        removed.push(path);
    }
    Ok(removed)
}

fn build_frontmatter(input: &SourcePageInput<'_>) -> Mapping {
    let mut frontmatter = Mapping::new();
    frontmatter.insert(
        Value::String("id".into()),
        Value::String(input.page_id.into()),
    );
    frontmatter.insert(Value::String("type".into()), Value::String("source".into()));
    frontmatter.insert(
        Value::String("title".into()),
        Value::String(input.title.into()),
    );
    frontmatter.insert(
        Value::String("source_document_id".into()),
        Value::String(input.source_document_id.into()),
    );
    frontmatter.insert(
        Value::String("source_revision_id".into()),
        Value::String(input.source_revision_id.into()),
    );
    frontmatter.insert(
        Value::String("generated_at".into()),
        Value::Number(Number::from(input.generated_at)),
    );
    frontmatter.insert(
        Value::String("build_record_id".into()),
        Value::String(input.build_record_id.into()),
    );
    frontmatter
}

fn upsert_source_page_body(existing_body: &str, input: &SourcePageInput<'_>) -> String {
    let mut body = existing_body.to_string();
    body = upsert_section(
        &body,
        "# Source",
        TITLE_REGION_ID,
        &format!("\n{}\n", input.title.trim()),
    );
    body = upsert_section(
        &body,
        "## Summary",
        SUMMARY_REGION_ID,
        &format!("\n{}\n", input.summary.trim()),
    );
    body = upsert_section(
        &body,
        "## Key topics",
        KEY_TOPICS_REGION_ID,
        &render_bullet_list(input.key_topics),
    );
    body = upsert_section(
        &body,
        "## Citations",
        CITATIONS_REGION_ID,
        &render_bullet_list(input.citations),
    );
    // bn-3ij3: per-page anchor block. Only emitted when the source went
    // through page-aware ingest. Existing region (from a previous compile
    // when pages was Some) is preserved on stale-but-still-paged sources;
    // a None pages count clears the region's content but leaves the fence
    // in place so downstream tools see "we knew this was a paged source
    // but the count is missing now".
    if let Some(pages) = input.pages
        && pages > 0
    {
        body = upsert_section(
            &body,
            "## Pages",
            PAGES_REGION_ID,
            &render_page_anchors(pages),
        );
    }
    body
}

/// Render the per-page anchor block for the `## Pages` managed region.
///
/// Each page emits one `<a name="page-N"></a>` HTML anchor on its own
/// line so a citation link `wiki/sources/<src>.md#page-N` resolves
/// directly without relying on heading-id slugification (which differs
/// across Markdown renderers). Markdown viewers that support inline HTML
/// honor the anchor; viewers that don't see the human-readable label
/// `Page N` next to it.
///
/// Page numbering follows the citation convention: 1-based, inclusive.
fn render_page_anchors(pages: u32) -> String {
    use std::fmt::Write as _;
    let mut out = String::from("\n");
    for n in 1..=pages {
        writeln!(
            out,
            "- <a name=\"{}\"></a>Page {n}",
            kb_core::page_anchor_name(n),
        )
        .expect("write to String");
    }
    out
}

fn upsert_section(existing_body: &str, heading: &str, region_id: &str, content: &str) -> String {
    if let Some(updated) = rewrite_managed_region(existing_body, region_id, content) {
        return updated;
    }

    let mut body = existing_body.trim_end().to_string();
    if !body.is_empty() {
        body.push_str("\n\n");
    }
    body.push_str(heading);
    body.push('\n');
    body.push_str(&managed_region(region_id, content));
    body.push('\n');
    body
}

fn managed_region(region_id: &str, content: &str) -> String {
    let mut rendered = String::new();
    rendered.push_str("<!-- kb:begin id=");
    rendered.push_str(region_id);
    rendered.push_str(" -->");
    rendered.push_str(content);
    if !content.ends_with('\n') {
        rendered.push('\n');
    }
    rendered.push_str("<!-- kb:end id=");
    rendered.push_str(region_id);
    rendered.push_str(" -->");
    rendered
}

fn render_bullet_list(items: &[String]) -> String {
    let mut rendered = String::from("\n");
    if items.is_empty() {
        rendered.push_str("- _None yet._\n");
        return rendered;
    }

    for item in items {
        rendered.push_str("- ");
        rendered.push_str(item.trim());
        rendered.push('\n');
    }
    rendered
}

fn serialize_frontmatter(frontmatter: &Mapping, body: &str) -> Result<String> {
    let yaml = serde_yaml::to_string(frontmatter)?;
    Ok(format!("---\n{yaml}---\n{body}"))
}

fn split_frontmatter(markdown: &str) -> Result<Option<(String, String)>> {
    let mut lines = markdown.split_inclusive('\n');
    let Some(first_line) = lines.next() else {
        return Ok(None);
    };

    if first_line != "---\n" && first_line != "---\r\n" && first_line != "---" {
        return Ok(None);
    }

    let mut frontmatter_text = String::new();
    let mut offset = first_line.len();

    for line in lines {
        if line == "---\n" || line == "---\r\n" || line == "---" {
            let body = markdown[offset + line.len()..].to_string();
            return Ok(Some((frontmatter_text, body)));
        }

        frontmatter_text.push_str(line);
        offset += line.len();
    }

    anyhow::bail!("frontmatter block was not terminated with a closing --- line")
}

fn slug_for_path(value: &str) -> String {
    let mut slug = String::new();
    let mut last_was_hyphen = false;

    for ch in value.chars() {
        if ch.is_ascii_alphanumeric() {
            slug.push(ch.to_ascii_lowercase());
            last_was_hyphen = false;
        } else if !last_was_hyphen {
            slug.push('-');
            last_was_hyphen = true;
        }
    }

    let trimmed = slug.trim_matches('-').to_string();
    if trimmed.is_empty() {
        "source".to_string()
    } else {
        trimmed
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn input<'a>(key_topics: &'a [String], citations: &'a [String]) -> SourcePageInput<'a> {
        SourcePageInput {
            page_id: "wiki-source-1",
            title: "Example Source",
            source_document_id: "Source Document/1",
            source_revision_id: "source-revision-1",
            generated_at: 1_700_000_000,
            build_record_id: "build-1",
            summary: "A concise summary.",
            key_topics,
            citations,
            pages: None,
        }
    }

    fn input_with_pages<'a>(
        key_topics: &'a [String],
        citations: &'a [String],
        pages: u32,
    ) -> SourcePageInput<'a> {
        SourcePageInput {
            pages: Some(pages),
            ..input(key_topics, citations)
        }
    }

    #[test]
    fn render_source_page_emits_frontmatter_and_managed_regions() {
        let key_topics = vec!["topic one".to_string(), "topic two".to_string()];
        let citations = vec!["[[wiki/concepts/rust]]".to_string()];
        let artifact = render_source_page(&input(&key_topics, &citations), None).expect("render");

        // Path includes the src-id (slugified) plus a slug derived from the
        // title — the new browseable layout.
        assert_eq!(
            artifact.path,
            PathBuf::from("wiki/sources/source-document-1-example-source.md")
        );
        assert_eq!(
            artifact.frontmatter.get("id"),
            Some(&Value::String("wiki-source-1".into()))
        );
        assert_eq!(
            artifact.frontmatter.get("type"),
            Some(&Value::String("source".into()))
        );
        assert_eq!(
            artifact.frontmatter.get("source_document_id"),
            Some(&Value::String("Source Document/1".into()))
        );
        assert_eq!(
            artifact.frontmatter.get("source_revision_id"),
            Some(&Value::String("source-revision-1".into()))
        );
        assert!(artifact.markdown.contains("<!-- kb:begin id=title -->"));
        assert!(artifact.markdown.contains("<!-- kb:begin id=summary -->"));
        assert!(
            artifact
                .markdown
                .contains("<!-- kb:begin id=key_topics -->")
        );
        assert!(artifact.markdown.contains("<!-- kb:begin id=citations -->"));
        assert!(artifact.markdown.contains("A concise summary."));
        assert!(artifact.markdown.contains("- topic one"));
        assert!(artifact.markdown.contains("- [[wiki/concepts/rust]]"));
    }

    #[test]
    fn render_source_page_rewrites_regions_in_place_without_clobbering_manual_content() {
        let existing = "---\nid: old\n---\n# Source\n<!-- kb:begin id=title -->\nOld Title\n<!-- kb:end id=title -->\n\n## Summary\n<!-- kb:begin id=summary -->\nOld summary.\n<!-- kb:end id=summary -->\n\n## Notes\nKeep me.\n\n## Key topics\n<!-- kb:begin id=key_topics -->\n- old topic\n<!-- kb:end id=key_topics -->\n\n## Citations\n<!-- kb:begin id=citations -->\n- old citation\n<!-- kb:end id=citations -->\n";
        let key_topics = vec!["fresh topic".to_string()];
        let citations = vec!["[[wiki/concepts/new]]".to_string()];
        let artifact =
            render_source_page(&input(&key_topics, &citations), Some(existing)).expect("render");

        assert!(artifact.body.contains("Example Source"));
        assert!(artifact.body.contains("A concise summary."));
        assert!(artifact.body.contains("- fresh topic"));
        assert!(artifact.body.contains("- [[wiki/concepts/new]]"));
        assert!(artifact.body.contains("## Notes\nKeep me."));
        assert!(!artifact.body.contains("Old Title"));
        assert!(!artifact.body.contains("Old summary."));
        assert!(!artifact.body.contains("old topic"));
        assert!(!artifact.body.contains("old citation"));
    }

    #[test]
    fn render_source_page_uses_empty_placeholder_lists() {
        let artifact = render_source_page(&input(&[], &[]), None).expect("render");
        assert!(artifact.body.contains("- _None yet._"));
    }

    /// bn-3ij3: when the source has no page metadata, the `## Pages`
    /// region must NOT appear so legacy/non-PDF sources stay clean.
    #[test]
    fn render_source_page_omits_pages_region_when_pages_absent() {
        let artifact = render_source_page(&input(&[], &[]), None).expect("render");
        assert!(!artifact.body.contains("<!-- kb:begin id=pages -->"));
        assert!(!artifact.body.contains("## Pages"));
    }

    /// bn-3ij3: when the source carries page metadata, the `## Pages`
    /// region renders an inline `<a name="page-N"></a>` for every page so
    /// citation links of the form `wiki/sources/<src>.md#page-N` resolve.
    #[test]
    fn render_source_page_emits_pages_region_when_pages_known() {
        let artifact =
            render_source_page(&input_with_pages(&[], &[], 3), None).expect("render");
        assert!(
            artifact.body.contains("<!-- kb:begin id=pages -->"),
            "pages region missing; body: {}",
            artifact.body
        );
        assert!(artifact.body.contains("<a name=\"page-1\">"));
        assert!(artifact.body.contains("<a name=\"page-2\">"));
        assert!(artifact.body.contains("<a name=\"page-3\">"));
        assert!(artifact.body.contains("Page 1"));
    }

    /// bn-1geb Part B acceptance: `rewrite_summary_image_refs` re-anchors
    /// normalized `assets/foo.png` references so they resolve from
    /// `wiki/sources/<src>.md`'s location — which is two directories away
    /// from KB root and two dirs away from `normalized/<src>/assets/`.
    #[test]
    fn compile_rewrites_image_paths_in_source_page() {
        let summary = "See ![fig](assets/diagram.png) and ![url](https://x/y.png).";
        let out = rewrite_summary_image_refs(summary, "src-abc");
        assert!(
            out.contains("![fig](../../.kb/normalized/src-abc/assets/diagram.png)"),
            "rewritten summary must re-anchor assets/ refs: {out}"
        );
        // External URLs untouched.
        assert!(out.contains("![url](https://x/y.png)"));
    }

    #[test]
    fn rewrite_summary_image_refs_is_noop_when_no_image_refs() {
        let summary = "plain prose with [a link](other.md)\n";
        let out = rewrite_summary_image_refs(summary, "src-xyz");
        assert_eq!(out, summary);
    }

    /// Regression guard: references that have already been re-anchored
    /// (somehow — e.g. an LLM quoting the normalized form verbatim) aren't
    /// rewritten twice. Only `assets/...` matches.
    #[test]
    fn rewrite_summary_image_refs_skips_non_assets_paths() {
        let summary = "![a](../../.kb/normalized/src-xyz/assets/already.png)\n![b](other/thing.png)\n";
        let out = rewrite_summary_image_refs(summary, "src-xyz");
        assert_eq!(out, summary);
    }

    /// bn-18qs: when ingest rewrites a spaces-in-basename ref to the
    /// angle-bracket form (`![](<assets/foo bar.png>)`), the compile-time
    /// re-anchor must match it AND preserve the angle-bracket form so the
    /// wiki page's rewritten ref stays valid `CommonMark`.
    #[test]
    fn rewrite_summary_image_refs_handles_angle_bracket_form_for_paths_with_spaces() {
        let summary = "![](<assets/Screenshot from 2026-04-07.png>)\n";
        let out = rewrite_summary_image_refs(summary, "src-abc");
        assert!(
            out.contains("![](<../../.kb/normalized/src-abc/assets/Screenshot from 2026-04-07.png>)"),
            "angle-bracket form must survive re-anchoring: {out}",
        );
    }

    /// And titles after the path still work — discarded, but the ref
    /// rewrites cleanly.
    #[test]
    fn rewrite_summary_image_refs_discards_title_attribute() {
        let summary = r#"![x](assets/foo.png "caption")"#;
        let out = rewrite_summary_image_refs(summary, "src-abc");
        assert_eq!(
            out,
            "![x](../../.kb/normalized/src-abc/assets/foo.png)",
            "title must be discarded on rewrite",
        );
    }

    #[test]
    fn source_page_path_for_includes_title_slug() {
        let path = source_page_path_for("src-1wz", "2026-04-07 LiveRamp USB team intro");
        assert_eq!(
            path,
            PathBuf::from("wiki/sources/src-1wz-2026-04-07-liveramp-usb-team-intro.md")
        );
    }

    #[test]
    fn source_page_path_for_falls_back_to_id_when_title_empty() {
        let path = source_page_path_for("src-1wz", "");
        assert_eq!(path, PathBuf::from("wiki/sources/src-1wz.md"));
    }

    #[test]
    fn source_page_path_for_falls_back_to_id_when_title_all_symbols() {
        let path = source_page_path_for("src-1wz", "!!!");
        assert_eq!(path, PathBuf::from("wiki/sources/src-1wz.md"));
    }

    // bn-6puc Part 2 / Part 4: the slug writer must guard against titles
    // that slugify to the src id itself (or start with it). Emit the
    // id-only form instead of a double like `src-1wz-src-1wz.md`.

    #[test]
    fn source_page_path_for_guards_against_title_equal_to_src_id() {
        let path = source_page_path_for("src-1wz", "src-1wz");
        assert_eq!(path, PathBuf::from("wiki/sources/src-1wz.md"));
    }

    #[test]
    fn source_page_path_for_guards_against_title_starting_with_src_id() {
        let path = source_page_path_for("src-1wz", "src-1wz extra notes");
        assert_eq!(path, PathBuf::from("wiki/sources/src-1wz.md"));
    }

    #[test]
    fn source_page_path_for_uses_stem_slug_when_title_is_real_filename() {
        let path =
            source_page_path_for("src-1wz", "2026-04-07-LiveRamp-USB-team-intro");
        assert_eq!(
            path,
            PathBuf::from(
                "wiki/sources/src-1wz-2026-04-07-liveramp-usb-team-intro.md",
            )
        );
    }

    /// Case-insensitivity: a title with mixed-case `SRC-1WZ` must also
    /// collapse, because the slugifier lowercases before comparison.
    #[test]
    fn source_page_path_for_guards_against_uppercase_id_title() {
        let path = source_page_path_for("src-1wz", "SRC-1WZ");
        assert_eq!(path, PathBuf::from("wiki/sources/src-1wz.md"));
    }

    #[test]
    fn resolve_source_page_path_prefers_id_slug_form() {
        let dir = tempfile::tempdir().expect("tempdir");
        let root = dir.path();
        std::fs::create_dir_all(root.join("wiki/sources")).expect("mkdir");
        std::fs::write(
            root.join("wiki/sources/src-abc-hello-world.md"),
            "slugged",
        )
        .expect("write slugged");

        let resolved = resolve_source_page_path(root, "src-abc").expect("resolved");
        assert_eq!(resolved, root.join("wiki/sources/src-abc-hello-world.md"));
    }

    #[test]
    fn resolve_source_page_path_falls_back_to_id_only() {
        let dir = tempfile::tempdir().expect("tempdir");
        let root = dir.path();
        std::fs::create_dir_all(root.join("wiki/sources")).expect("mkdir");
        std::fs::write(root.join("wiki/sources/src-abc.md"), "plain").expect("write plain");

        let resolved = resolve_source_page_path(root, "src-abc").expect("resolved");
        assert_eq!(resolved, root.join("wiki/sources/src-abc.md"));
    }

    #[test]
    fn resolve_source_page_path_returns_none_when_missing() {
        let dir = tempfile::tempdir().expect("tempdir");
        let root = dir.path();
        assert!(resolve_source_page_path(root, "src-missing").is_none());
    }

    #[test]
    fn resolve_source_page_path_ignores_other_src_ids() {
        let dir = tempfile::tempdir().expect("tempdir");
        let root = dir.path();
        std::fs::create_dir_all(root.join("wiki/sources")).expect("mkdir");
        std::fs::write(root.join("wiki/sources/src-other-hello.md"), "other")
            .expect("write other");
        assert!(resolve_source_page_path(root, "src-abc").is_none());
    }

    #[test]
    fn clean_stale_source_pages_removes_mismatched_files() {
        let dir = tempfile::tempdir().expect("tempdir");
        let root = dir.path();
        std::fs::create_dir_all(root.join("wiki/sources")).expect("mkdir");
        std::fs::write(root.join("wiki/sources/src-abc.md"), "id-only").expect("write");
        std::fs::write(root.join("wiki/sources/src-abc-old-title.md"), "old").expect("write");
        std::fs::write(root.join("wiki/sources/src-abc-new-title.md"), "new").expect("write");

        let keep = PathBuf::from("wiki/sources/src-abc-new-title.md");
        let removed = clean_stale_source_pages(root, "src-abc", &keep).expect("clean");
        assert_eq!(removed.len(), 2);
        assert!(root.join("wiki/sources/src-abc-new-title.md").exists());
        assert!(!root.join("wiki/sources/src-abc.md").exists());
        assert!(!root.join("wiki/sources/src-abc-old-title.md").exists());
    }

    #[test]
    fn clean_stale_source_pages_keeps_unrelated_files() {
        let dir = tempfile::tempdir().expect("tempdir");
        let root = dir.path();
        std::fs::create_dir_all(root.join("wiki/sources")).expect("mkdir");
        std::fs::write(root.join("wiki/sources/src-other.md"), "other").expect("write");
        std::fs::write(root.join("wiki/sources/index.md"), "index").expect("write");

        let keep = PathBuf::from("wiki/sources/src-abc-new-title.md");
        let removed = clean_stale_source_pages(root, "src-abc", &keep).expect("clean");
        assert!(removed.is_empty());
        assert!(root.join("wiki/sources/src-other.md").exists());
        assert!(root.join("wiki/sources/index.md").exists());
    }
}
