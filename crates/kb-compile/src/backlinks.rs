use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use ignore::WalkBuilder;
use kb_core::fs::atomic_write;
use kb_core::rewrite_managed_region;
use regex::Regex;
use serde_yaml::Value;

const WIKI_DIR: &str = "wiki";
const CONCEPT_DIR: &str = "wiki/concepts";
const SOURCE_DIR: &str = "wiki/sources";
const NORMALIZED_DIR: &str = "normalized";

pub const BACKLINKS_REGION_ID: &str = "backlinks";
pub const REFERENCED_BY_CONCEPTS_REGION_ID: &str = "referenced_by_concepts";

const BACKLINKS_HEADING: &str = "## Backlinks";
const REFERENCED_BY_CONCEPTS_HEADING: &str = "## Referenced by concepts";

/// Updated markdown for one wiki page touched by the backlinks pass.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BacklinksArtifact {
    /// Absolute path to the markdown file.
    pub path: PathBuf,
    /// Existing markdown from disk before update.
    pub existing_markdown: String,
    /// Markdown after managed region rewrite/upsert.
    pub updated_markdown: String,
}

impl BacklinksArtifact {
    #[must_use]
    pub fn needs_update(&self) -> bool {
        self.existing_markdown != self.updated_markdown
    }
}

/// Regenerate managed backlink regions across concept and source pages.
///
/// The pass performs two scans and then one rewrite phase:
///
/// 1. Scan every wiki page for Obsidian-style `[[...]]` wiki-links to concept pages
///    (legacy behavior — captures manual references and any auto-emitted wiki-links).
/// 2. Scan every concept page's YAML frontmatter for `sources: [{ heading_anchor, quote }]`
///    entries. Each `heading_anchor` is resolved against `normalized/<src>/metadata.json`
///    to recover the contributing `source_document_id`, which maps to a `wiki/sources/<src>.md`
///    page slug. Additionally, the concept's `source_document_ids: [...]` frontmatter
///    field is consulted as a no-anchor fallback — `concept_merge` can emit
///    `sources: [{quote: "..."}]` entries without a `heading_anchor`, and this field
///    preserves the source attribution those entries would otherwise lose.
///
/// Concept pages receive a `backlinks` region listing referring wiki pages (both
/// wiki-link and frontmatter-source-backed references). Source pages receive a
/// `referenced_by_concepts` region listing concept pages that cited them.
///
/// # Errors
///
/// Returns an error if any target wiki file cannot be walked, read, or parsed.
pub fn run_backlinks_pass(root: &Path) -> Result<Vec<BacklinksArtifact>> {
    let concept_pages = discover_wiki_pages(root, CONCEPT_DIR)?;
    let source_pages = discover_wiki_pages(root, SOURCE_DIR)?;

    // anchor -> list of source_document_ids that own that heading (via normalized metadata)
    let anchor_to_source_docs = build_anchor_to_source_docs(root)?;

    let mut concept_backlinks: BTreeMap<String, BTreeSet<String>> = concept_pages
        .keys()
        .cloned()
        .map(|id| (id, BTreeSet::new()))
        .collect();
    let mut source_referenced_by: BTreeMap<String, BTreeSet<String>> = source_pages
        .keys()
        .cloned()
        .map(|id| (id, BTreeSet::new()))
        .collect();

    collect_wiki_link_backlinks(root, &mut concept_backlinks)?;
    collect_frontmatter_source_backlinks(
        &concept_pages,
        &source_pages,
        &anchor_to_source_docs,
        &mut concept_backlinks,
        &mut source_referenced_by,
    )?;
    collect_mention_backlinks(&concept_pages, &source_pages, &mut concept_backlinks)?;

    let mut artifacts = Vec::with_capacity(concept_pages.len() + source_pages.len());

    for (concept_id, path) in concept_pages {
        let existing_markdown = std::fs::read_to_string(&path)
            .with_context(|| format!("read concept page {}", path.display()))?;
        let links = concept_backlinks.remove(&concept_id).unwrap_or_default();
        let updated_markdown = upsert_section(
            &existing_markdown,
            BACKLINKS_HEADING,
            BACKLINKS_REGION_ID,
            &render_link_list(&links),
        );
        artifacts.push(BacklinksArtifact {
            path,
            existing_markdown,
            updated_markdown,
        });
    }

    for (source_id, path) in source_pages {
        let existing_markdown = std::fs::read_to_string(&path)
            .with_context(|| format!("read source page {}", path.display()))?;
        let links = source_referenced_by.remove(&source_id).unwrap_or_default();
        let updated_markdown = upsert_section(
            &existing_markdown,
            REFERENCED_BY_CONCEPTS_HEADING,
            REFERENCED_BY_CONCEPTS_REGION_ID,
            &render_link_list(&links),
        );
        artifacts.push(BacklinksArtifact {
            path,
            existing_markdown,
            updated_markdown,
        });
    }

    Ok(artifacts)
}

/// Persist backlink artifacts to disk using atomic writes.
///
/// # Errors
/// Returns an error if any atomic write fails.
pub fn persist_backlinks_artifacts(artifacts: &[BacklinksArtifact]) -> Result<()> {
    for artifact in artifacts {
        atomic_write(&artifact.path, artifact.updated_markdown.as_bytes())
            .with_context(|| format!("write {}", artifact.path.display()))?;
    }
    Ok(())
}

fn discover_wiki_pages(root: &Path, relative_dir: &str) -> Result<BTreeMap<String, PathBuf>> {
    let mut pages = BTreeMap::new();
    for path in list_markdown_files(root, relative_dir)? {
        let page_id = page_id_from_path(root, &path)?;
        pages.insert(page_id, path);
    }
    Ok(pages)
}

/// Walk `normalized/<source_document_id>/metadata.json` to build a reverse index
/// from every `heading_id` to the set of `source_document_id`s that declared it.
///
/// We use a Vec rather than a single value because two independent sources may
/// share a heading like "summary"; the caller picks the intersection with known
/// source pages.
fn build_anchor_to_source_docs(root: &Path) -> Result<BTreeMap<String, Vec<String>>> {
    let mut map: BTreeMap<String, Vec<String>> = BTreeMap::new();
    let normalized_root = root.join(NORMALIZED_DIR);
    if !normalized_root.exists() {
        return Ok(map);
    }

    let entries = std::fs::read_dir(&normalized_root)
        .with_context(|| format!("read normalized dir {}", normalized_root.display()))?;
    for entry in entries {
        let entry = entry.with_context(|| format!("walk {}", normalized_root.display()))?;
        if !entry.file_type().map(|t| t.is_dir()).unwrap_or(false) {
            continue;
        }
        let source_doc_id = entry.file_name().to_string_lossy().into_owned();
        let metadata_path = entry.path().join("metadata.json");
        if !metadata_path.is_file() {
            continue;
        }
        let raw = std::fs::read_to_string(&metadata_path)
            .with_context(|| format!("read {}", metadata_path.display()))?;
        let parsed: serde_json::Value = serde_json::from_str(&raw)
            .with_context(|| format!("parse {}", metadata_path.display()))?;
        let Some(headings) = parsed.get("heading_ids").and_then(|v| v.as_array()) else {
            continue;
        };
        for heading in headings {
            if let Some(h) = heading.as_str() {
                map.entry(h.to_string())
                    .or_default()
                    .push(source_doc_id.clone());
            }
        }
    }

    for docs in map.values_mut() {
        docs.sort();
        docs.dedup();
    }
    Ok(map)
}

fn collect_wiki_link_backlinks(
    root: &Path,
    concept_backlinks: &mut BTreeMap<String, BTreeSet<String>>,
) -> Result<()> {
    let link_re = Regex::new(r"\[\[([^\]\r\n]+)\]\]").context("compile wikilink regex")?;

    for page in list_markdown_files(root, WIKI_DIR)? {
        let source_id = page_id_from_path(root, &page)?;
        let markdown = std::fs::read_to_string(&page)
            .with_context(|| format!("read wiki page {}", page.display()))?;

        for capture in link_re.captures_iter(&markdown) {
            let Some(raw_link) = capture.get(1).map(|c| c.as_str()) else {
                continue;
            };
            let Some(target_id) = normalize_wiki_link(raw_link) else {
                continue;
            };
            if let Some(referers) = concept_backlinks.get_mut(&target_id) {
                referers.insert(source_id.clone());
            }
        }
    }

    Ok(())
}

/// For each concept page, parse its YAML frontmatter `sources:` block, resolve
/// each `heading_anchor` to contributing source pages via normalized metadata,
/// and credit both sides of the relation.
fn collect_frontmatter_source_backlinks(
    concept_pages: &BTreeMap<String, PathBuf>,
    source_pages: &BTreeMap<String, PathBuf>,
    anchor_to_source_docs: &BTreeMap<String, Vec<String>>,
    concept_backlinks: &mut BTreeMap<String, BTreeSet<String>>,
    source_referenced_by: &mut BTreeMap<String, BTreeSet<String>>,
) -> Result<()> {
    // Build a source_document_id -> wiki/sources/<slug> map by reading source page frontmatter.
    let mut source_doc_to_page: BTreeMap<String, String> = BTreeMap::new();
    for (page_id, path) in source_pages {
        let markdown = std::fs::read_to_string(path)
            .with_context(|| format!("read source page {}", path.display()))?;
        let Some((fm, _body)) = split_frontmatter(&markdown) else {
            continue;
        };
        let Ok(parsed) = serde_yaml::from_str::<Value>(&fm) else {
            continue;
        };
        let Some(doc_id) = parsed
            .get("source_document_id")
            .and_then(|v| v.as_str())
            .map(str::to_string)
        else {
            continue;
        };
        source_doc_to_page.insert(doc_id, page_id.clone());
    }

    for (concept_id, path) in concept_pages {
        let markdown = std::fs::read_to_string(path)
            .with_context(|| format!("read concept page {}", path.display()))?;
        let Some((fm, _body)) = split_frontmatter(&markdown) else {
            continue;
        };
        let Ok(parsed) = serde_yaml::from_str::<Value>(&fm) else {
            continue;
        };
        // The concept's own `source_document_ids:` frontmatter field is the
        // authoritative set of contributing source documents. Gate anchor
        // resolution by this set so unrelated sources that happen to share a
        // heading name (e.g. `## Pitfalls`) don't get spuriously credited.
        // Fallback: if the field is missing (older concept pages), allow any
        // anchor match — preserves legacy behavior.
        let concept_source_docs: BTreeSet<String> = parsed
            .get("source_document_ids")
            .and_then(|v| v.as_sequence())
            .map(|seq| {
                seq.iter()
                    .filter_map(|v| v.as_str().map(str::to_string))
                    .collect()
            })
            .unwrap_or_default();

        if let Some(sources) = parsed.get("sources").and_then(|v| v.as_sequence()) {
            for entry in sources {
                let Some(anchor) = entry.get("heading_anchor").and_then(|v| v.as_str()) else {
                    continue;
                };
                let Some(candidate_docs) = anchor_to_source_docs.get(anchor) else {
                    continue;
                };
                for doc_id in candidate_docs {
                    if !concept_source_docs.is_empty() && !concept_source_docs.contains(doc_id) {
                        continue;
                    }
                    let Some(source_page_id) = source_doc_to_page.get(doc_id) else {
                        continue;
                    };
                    if let Some(referers) = concept_backlinks.get_mut(concept_id) {
                        referers.insert(source_page_id.clone());
                    }
                    if let Some(referers) = source_referenced_by.get_mut(source_page_id) {
                        referers.insert(concept_id.clone());
                    }
                }
            }
        }

        // No-anchor fallback: concept's `source_document_ids:` frontmatter field
        // lists every source document that contributed to this concept, regardless
        // of whether the merged `sources:` entries carried heading anchors.
        // This ensures concepts with `sources: [{quote: "..."}]` (no heading_anchor)
        // still cross-link with their source pages.
        if let Some(doc_ids) = parsed
            .get("source_document_ids")
            .and_then(|v| v.as_sequence())
        {
            for doc_id_val in doc_ids {
                let Some(doc_id) = doc_id_val.as_str() else {
                    continue;
                };
                let Some(source_page_id) = source_doc_to_page.get(doc_id) else {
                    continue;
                };
                if let Some(referers) = concept_backlinks.get_mut(concept_id) {
                    referers.insert(source_page_id.clone());
                }
                if let Some(referers) = source_referenced_by.get_mut(source_page_id) {
                    referers.insert(concept_id.clone());
                }
            }
        }
    }

    Ok(())
}

/// Minimum token length for mention-based backlinks. Tokens shorter than this
/// (e.g. "a", "of", "it") would cause extreme false-positive rates, so they're
/// skipped entirely when derived from concept names or aliases.
const MIN_MENTION_TOKEN_LEN: usize = 3;

/// Aliases equal to one of these (case-insensitive, after normalization) are
/// skipped when building mention tokens. These are generic English words that
/// concepts occasionally list as aliases but would flood every source page.
const MENTION_STOPWORDS: &[&str] = &[
    "the", "and", "for", "with", "that", "this", "from", "are", "was", "has",
    "have", "not", "but", "all", "any", "can", "its", "one", "two", "new",
    "use", "via",
];

/// For each concept page, collect its `name` + `aliases` as mention tokens and
/// scan every source page's body. A source page that mentions a concept token
/// as a standalone word (case-insensitive, Unicode word boundaries) OR that
/// contains a `[[<slug>]]`-style wiki-link whose normalized target matches the
/// concept's slug or an alias slug is added to the concept's backlinks set.
///
/// This is additive on top of the source_document_ids-gated primary path, and
/// is deduped via the `concept_backlinks` `BTreeSet`.
fn collect_mention_backlinks(
    concept_pages: &BTreeMap<String, PathBuf>,
    source_pages: &BTreeMap<String, PathBuf>,
    concept_backlinks: &mut BTreeMap<String, BTreeSet<String>>,
) -> Result<()> {
    // Per-concept tokens + slug matchers, precomputed once.
    struct ConceptMatcher {
        concept_id: String,
        word_regex: Option<Regex>,
        /// Wiki-link target slugs (full `wiki/concepts/<slug>`) that should
        /// count as a mention of this concept.
        slug_targets: BTreeSet<String>,
    }

    let mut matchers: Vec<ConceptMatcher> = Vec::with_capacity(concept_pages.len());
    for (concept_id, path) in concept_pages {
        let markdown = std::fs::read_to_string(path)
            .with_context(|| format!("read concept page {}", path.display()))?;
        let (name_opt, alias_tokens, alias_slugs) = match split_frontmatter(&markdown) {
            Some((fm, _body)) => parse_concept_name_and_aliases(&fm),
            None => (None, Vec::new(), Vec::new()),
        };

        // Build the set of word tokens to match: concept name + aliases, each
        // filtered by length and stopwords. Names/aliases map to tokens via
        // `normalize_mention_token`.
        let mut tokens: BTreeSet<String> = BTreeSet::new();
        if let Some(name) = name_opt.as_ref() {
            if let Some(t) = normalize_mention_token(name) {
                tokens.insert(t);
            }
        }
        for alias in &alias_tokens {
            if let Some(t) = normalize_mention_token(alias) {
                tokens.insert(t);
            }
        }

        // Build wiki-link slug targets: the concept's own page id, plus any
        // slug-normalized aliases under wiki/concepts/. This lets an alias
        // written as `[[aliases-slug]]` route to its concept.
        let mut slug_targets: BTreeSet<String> = BTreeSet::new();
        slug_targets.insert(concept_id.clone());
        for slug in alias_slugs {
            if !slug.is_empty() {
                slug_targets.insert(format!("wiki/concepts/{slug}"));
            }
        }

        let word_regex = if tokens.is_empty() {
            None
        } else {
            // Case-insensitive, Unicode-aware word boundaries. `regex::Regex`
            // uses Unicode word boundaries by default.
            let pattern = tokens
                .iter()
                .map(|t| regex::escape(t))
                .collect::<Vec<_>>()
                .join("|");
            let full = format!(r"(?i)\b(?:{pattern})\b");
            Some(Regex::new(&full).with_context(|| format!("compile mention regex for {concept_id}"))?)
        };

        matchers.push(ConceptMatcher {
            concept_id: concept_id.clone(),
            word_regex,
            slug_targets,
        });
    }

    let wiki_link_re = Regex::new(r"\[\[([^\]\r\n]+)\]\]").context("compile wikilink regex")?;

    // Scan each source page body once and test every concept matcher against it.
    for (source_id, path) in source_pages {
        let markdown = std::fs::read_to_string(path)
            .with_context(|| format!("read source page {}", path.display()))?;
        let body = match split_frontmatter(&markdown) {
            Some((_fm, body)) => body,
            None => markdown.clone(),
        };

        // Precollect the normalized wiki-link targets present in this body.
        let mut wiki_targets: BTreeSet<String> = BTreeSet::new();
        // Also collect raw bare forms (e.g. `[[raft]]` -> "raft") so we can
        // match alias-slug style links that don't carry a `wiki/concepts/` prefix.
        let mut bare_targets: BTreeSet<String> = BTreeSet::new();
        for capture in wiki_link_re.captures_iter(&body) {
            let Some(raw) = capture.get(1).map(|c| c.as_str()) else {
                continue;
            };
            if let Some(target) = normalize_wiki_link(raw) {
                wiki_targets.insert(target);
            }
            // Bare form: strip alias/anchor but don't require `wiki/concepts/` prefix.
            if let Some(bare) = bare_wiki_link_slug(raw) {
                bare_targets.insert(bare);
            }
        }

        for matcher in &matchers {
            let mut hit = false;

            // 1) wiki-link match against any known slug target for this concept.
            for target in &matcher.slug_targets {
                if wiki_targets.contains(target) {
                    hit = true;
                    break;
                }
                // Bare-form match: compare the final path segment.
                if let Some(slug) = target.rsplit('/').next() {
                    if bare_targets.contains(slug) {
                        hit = true;
                        break;
                    }
                }
            }

            // 2) word-boundary match on concept name/aliases.
            if !hit {
                if let Some(re) = matcher.word_regex.as_ref() {
                    if re.is_match(&body) {
                        hit = true;
                    }
                }
            }

            if hit {
                if let Some(referers) = concept_backlinks.get_mut(&matcher.concept_id) {
                    referers.insert(source_id.clone());
                }
            }
        }
    }

    Ok(())
}

/// Extract the `name` string and `aliases` sequence from a concept's YAML
/// frontmatter. Returns `(name, alias_strings, alias_slugs)`. Alias slugs are
/// the slug-form aliases useful for `[[<slug>]]`-style wiki-link matching
/// (dashed, lowercase), distinct from the raw alias strings which are used as
/// word-boundary tokens.
fn parse_concept_name_and_aliases(fm: &str) -> (Option<String>, Vec<String>, Vec<String>) {
    let Ok(parsed) = serde_yaml::from_str::<Value>(fm) else {
        return (None, Vec::new(), Vec::new());
    };
    let name = parsed
        .get("name")
        .and_then(|v| v.as_str())
        .map(str::to_string);

    let mut aliases: Vec<String> = Vec::new();
    let mut alias_slugs: Vec<String> = Vec::new();
    if let Some(seq) = parsed.get("aliases").and_then(|v| v.as_sequence()) {
        for entry in seq {
            if let Some(s) = entry.as_str() {
                aliases.push(s.to_string());
                alias_slugs.push(slugify_for_wiki_link(s));
            }
        }
    }
    (name, aliases, alias_slugs)
}

/// Normalize an alias/name string into a word-boundary-matched mention token,
/// or return `None` if the alias should be skipped (too short, stopword, or
/// contains no word characters).
fn normalize_mention_token(raw: &str) -> Option<String> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return None;
    }
    // Require the token to contain at least one alphanumeric character.
    if !trimmed.chars().any(char::is_alphanumeric) {
        return None;
    }
    if trimmed.chars().count() < MIN_MENTION_TOKEN_LEN {
        return None;
    }
    let lower = trimmed.to_lowercase();
    if MENTION_STOPWORDS.contains(&lower.as_str()) {
        return None;
    }
    Some(trimmed.to_string())
}

/// Slugify an alias for wiki-link target matching: lowercase, ASCII, spaces
/// become dashes. This is a lightweight slug that aligns with how concept
/// page ids are produced.
fn slugify_for_wiki_link(raw: &str) -> String {
    let mut out = String::with_capacity(raw.len());
    for ch in raw.trim().chars() {
        if ch.is_alphanumeric() {
            for c in ch.to_lowercase() {
                out.push(c);
            }
        } else if (ch.is_whitespace() || ch == '-' || ch == '_')
            && !out.ends_with('-')
        {
            out.push('-');
        }
    }
    out.trim_matches('-').to_string()
}

/// Strip alias, anchor, leading paths, and `.md` suffix from a raw wiki-link
/// target, returning just the final path segment (bare slug). Unlike
/// `normalize_wiki_link`, this does not require a `wiki/concepts/` prefix.
fn bare_wiki_link_slug(raw: &str) -> Option<String> {
    let without_alias = raw.split('|').next()?.trim();
    if without_alias.is_empty() {
        return None;
    }
    let without_anchor = without_alias
        .split('#')
        .next()
        .map(str::trim)
        .filter(|v| !v.is_empty())?;
    if without_anchor.starts_with("http://")
        || without_anchor.starts_with("https://")
        || without_anchor.starts_with("mailto:")
        || without_anchor.starts_with('#')
    {
        return None;
    }
    let mut target = without_anchor.trim_start_matches("./");
    target = target.trim_start_matches('/');
    if std::path::Path::new(target)
        .extension()
        .is_some_and(|ext| ext.eq_ignore_ascii_case("md"))
    {
        target = target.trim_end_matches(".md");
    }
    target = target.trim_end_matches('/');
    let last = target.rsplit('/').next().unwrap_or(target).trim();
    if last.is_empty() {
        None
    } else {
        Some(last.to_string())
    }
}

/// Split YAML frontmatter from a markdown document. Returns `(frontmatter_yaml, body)`.
fn split_frontmatter(markdown: &str) -> Option<(String, String)> {
    let mut lines = markdown.split_inclusive('\n');
    let first = lines.next()?;
    if first != "---\n" && first != "---\r\n" && first != "---" {
        return None;
    }
    let mut fm = String::new();
    let mut offset = first.len();
    for line in lines {
        if line == "---\n" || line == "---\r\n" || line == "---" {
            let body = markdown[offset + line.len()..].to_string();
            return Some((fm, body));
        }
        fm.push_str(line);
        offset += line.len();
    }
    None
}

fn render_link_list(referers: &BTreeSet<String>) -> String {
    let mut content = String::from("\n");
    if referers.is_empty() {
        content.push_str("- _None yet._\n");
        return content;
    }
    for referer in referers {
        content.push_str("- [[");
        content.push_str(referer);
        content.push_str("]]\n");
    }
    content
}

fn upsert_section(
    existing_body: &str,
    heading: &str,
    region_id: &str,
    content: &str,
) -> String {
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

fn normalize_wiki_link(raw: &str) -> Option<String> {
    let without_alias = raw.split('|').next()?.trim();
    if without_alias.is_empty() {
        return None;
    }

    let without_anchor = without_alias
        .split('#')
        .next()
        .map(str::trim)
        .filter(|value| !value.is_empty())?;

    if without_anchor.starts_with("http://")
        || without_anchor.starts_with("https://")
        || without_anchor.starts_with("mailto:")
        || without_anchor.starts_with('#')
    {
        return None;
    }

    let mut target = without_anchor.trim_start_matches("./");
    target = target.trim_start_matches('/');
    if std::path::Path::new(target)
        .extension()
        .is_some_and(|ext| ext.eq_ignore_ascii_case("md"))
    {
        target = target.trim_end_matches(".md");
    }
    if target.ends_with('/') {
        target = target.trim_end_matches('/');
    }

    if target.starts_with("wiki/concepts/") {
        Some(target.to_string())
    } else {
        None
    }
}

fn page_id_from_path(root: &Path, page: &Path) -> Result<String> {
    let relative = page
        .strip_prefix(root)
        .with_context(|| format!("{} is not under root {}", page.display(), root.display()))?;
    Ok(relative
        .to_string_lossy()
        .replace('\\', "/")
        .trim_end_matches(".md")
        .to_string())
}

fn list_markdown_files(root: &Path, relative_dir: &str) -> Result<Vec<PathBuf>> {
    let root_dir = root.join(relative_dir);
    if !root_dir.exists() {
        return Ok(Vec::new());
    }

    let mut files = Vec::new();
    for entry in WalkBuilder::new(&root_dir)
        .standard_filters(true)
        .require_git(false)
        .build()
    {
        let entry = entry.with_context(|| format!("walk wiki files in {}", root_dir.display()))?;

        let is_markdown = entry
            .path()
            .extension()
            .is_some_and(|ext| ext == std::ffi::OsStr::new("md"));

        if entry.file_type().is_some_and(|kind| kind.is_file()) && is_markdown {
            files.push(entry.into_path());
        }
    }

    files.sort_unstable();
    Ok(files)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::tempdir;

    fn write(path: &Path, markdown: &str) {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).expect("create parent directory");
        }
        fs::write(path, markdown).expect("write markdown fixture");
    }

    #[test]
    fn normalize_wiki_link_removes_aliases_and_anchors() {
        assert_eq!(
            normalize_wiki_link("wiki/concepts/rust"),
            Some("wiki/concepts/rust".to_string())
        );
        assert_eq!(
            normalize_wiki_link("wiki/concepts/rust.md#section"),
            Some("wiki/concepts/rust".to_string())
        );
        assert_eq!(
            normalize_wiki_link("wiki/concepts/rust|Rust language"),
            Some("wiki/concepts/rust".to_string())
        );
        assert_eq!(
            normalize_wiki_link("./wiki/concepts/rust.md|Rust"),
            Some("wiki/concepts/rust".to_string())
        );
        assert_eq!(normalize_wiki_link("wiki/sources/page"), None);
        assert_eq!(normalize_wiki_link("https://example.com"), None);
    }

    #[test]
    fn run_backlinks_pass_adds_and_updates_backlinks_region() {
        let root = tempdir().expect("tempdir");

        write(
            &root.path().join("wiki/concepts/rust.md"),
            "# Rust\n\nOld intro for rust.\n",
        );
        write(
            &root.path().join("wiki/concepts/borrow-checker.md"),
            "# Borrow checker\n\nUses [[wiki/concepts/rust#intro|Rust]].\n",
        );
        write(
            &root.path().join("wiki/sources/page-a.md"),
            "# Page A\n\nReferences rust: [[wiki/concepts/rust|Rust]] and [[wiki/sources/other]].\n",
        );
        write(
            &root.path().join("wiki/sources/page-b.md"),
            "# Page B\n\nMentions [[wiki/concepts/rust#summary]].\n",
        );

        let mut artifacts = run_backlinks_pass(root.path()).expect("run backlink pass");
        artifacts.sort_by(|a, b| a.path.cmp(&b.path));

        let rust = artifacts
            .iter()
            .find(|artifact| artifact.path.ends_with("wiki/concepts/rust.md"))
            .expect("rust concept exists");

        assert!(rust.updated_markdown.contains("## Backlinks"));
        assert!(
            rust.updated_markdown
                .contains("- [[wiki/concepts/borrow-checker]]")
        );
        assert!(rust.updated_markdown.contains("- [[wiki/sources/page-a]]"));
        assert!(rust.updated_markdown.contains("- [[wiki/sources/page-b]]"));
        assert!(!rust.updated_markdown.contains("- _None yet._"));
        assert!(rust.needs_update());

        let borrow = artifacts
            .iter()
            .find(|artifact| artifact.path.ends_with("wiki/concepts/borrow-checker.md"))
            .expect("borrow concept exists");
        assert!(borrow.updated_markdown.contains("- _None yet._"));
    }

    #[test]
    fn run_backlinks_pass_preserves_manual_content() {
        let root = tempdir().expect("tempdir");

        write(
            &root.path().join("wiki/concepts/rust.md"),
            "# Rust\n\n\nNotes section\n\n## Backlinks\n<!-- kb:begin id=backlinks -->\n- old-link\n<!-- kb:end id=backlinks -->\n",
        );
        write(
            &root.path().join("wiki/sources/page-a.md"),
            "# Page A\n\n[[wiki/concepts/rust]].\n",
        );

        let artifacts = run_backlinks_pass(root.path()).expect("run backlink pass");
        let rust = artifacts
            .iter()
            .find(|artifact| artifact.path.ends_with("wiki/concepts/rust.md"))
            .expect("rust concept exists");

        assert!(!rust.updated_markdown.contains("- old-link"));
        assert!(rust.updated_markdown.contains("- [[wiki/sources/page-a]]"));
    }

    #[test]
    fn no_references_renders_placeholder() {
        let root = tempdir().expect("tempdir");

        write(
            &root.path().join("wiki/concepts/isolated.md"),
            "# Isolated\n",
        );

        let artifacts = run_backlinks_pass(root.path()).expect("run backlink pass");
        let isolated = artifacts
            .iter()
            .find(|artifact| artifact.path.ends_with("wiki/concepts/isolated.md"))
            .expect("isolated concept exists");

        assert!(isolated.updated_markdown.contains("- _None yet._"));
    }

    #[test]
    fn frontmatter_sources_drive_cross_references_between_concepts_and_sources() {
        let root = tempdir().expect("tempdir");

        // normalized metadata: anchor "python-gil" belongs to source document "src-abc"
        write(
            &root.path().join("normalized/src-abc/metadata.json"),
            r#"{
                "metadata": {"id": "src-abc"},
                "source_revision_id": "rev-123",
                "heading_ids": ["python-gil", "alternatives"]
            }"#,
        );

        // source page pointing at src-abc / rev-123 (no wiki-links anywhere)
        write(
            &root.path().join("wiki/sources/src-abc.md"),
            "---\nid: wiki-source-src-abc\ntype: source\ntitle: Python GIL\nsource_document_id: src-abc\nsource_revision_id: rev-123\n---\n# Source\n\n## Citations\n- rev-123#python-gil\n",
        );

        // concept page whose frontmatter sources: references the anchor above
        write(
            &root.path().join("wiki/concepts/global-interpreter-lock.md"),
            "---\nid: concept:global-interpreter-lock\nname: Global Interpreter Lock\nsources:\n- heading_anchor: python-gil\n  quote: The Python GIL is a mutex\n- heading_anchor: alternatives\n  quote: async/await\n---\n\n# Global Interpreter Lock\n",
        );

        let artifacts = run_backlinks_pass(root.path()).expect("run backlink pass");

        let concept = artifacts
            .iter()
            .find(|a| a.path.ends_with("wiki/concepts/global-interpreter-lock.md"))
            .expect("concept artifact");
        assert!(concept.updated_markdown.contains("## Backlinks"));
        assert!(
            concept
                .updated_markdown
                .contains("- [[wiki/sources/src-abc]]"),
            "concept should backlink to source page via frontmatter sources: resolution, got:\n{}",
            concept.updated_markdown
        );
        assert!(!concept.updated_markdown.contains("- _None yet._"));

        let source = artifacts
            .iter()
            .find(|a| a.path.ends_with("wiki/sources/src-abc.md"))
            .expect("source artifact");
        assert!(
            source
                .updated_markdown
                .contains("## Referenced by concepts")
        );
        assert!(source.updated_markdown.contains(
            "<!-- kb:begin id=referenced_by_concepts -->"
        ));
        assert!(
            source
                .updated_markdown
                .contains("- [[wiki/concepts/global-interpreter-lock]]"),
            "source should list referencing concept, got:\n{}",
            source.updated_markdown
        );
    }

    #[test]
    fn source_document_ids_backfill_when_heading_anchor_absent() {
        let root = tempdir().expect("tempdir");

        // Source page carrying a source_document_id in frontmatter.
        // No normalized/ dir at all — exercising the no-anchor fallback path.
        write(
            &root.path().join("wiki/sources/src-4846dd29.md"),
            "---\nid: wiki-source-src-4846dd29\ntype: source\ntitle: Rust repo\nsource_document_id: src-4846dd29\nsource_revision_id: rev-9\n---\n# Source\n\nSome content.\n",
        );

        // Concept page with sources: entries lacking heading_anchor but with
        // source_document_ids: fallback.
        write(
            &root.path().join("wiki/concepts/rust.md"),
            "---\nid: concept:rust\nname: Rust\nsources:\n- quote: This is the main source code repository for Rust.\n- quote: It contains the compiler, standard library, and documentation.\nsource_document_ids:\n- src-4846dd29\n---\n\n# Rust\n",
        );

        let artifacts = run_backlinks_pass(root.path()).expect("run backlink pass");

        let concept = artifacts
            .iter()
            .find(|a| a.path.ends_with("wiki/concepts/rust.md"))
            .expect("concept artifact");
        assert!(
            concept
                .updated_markdown
                .contains("- [[wiki/sources/src-4846dd29]]"),
            "concept should backlink to source via source_document_ids fallback, got:\n{}",
            concept.updated_markdown
        );
        assert!(
            !concept.updated_markdown.contains("- _None yet._"),
            "concept Backlinks should not be empty, got:\n{}",
            concept.updated_markdown
        );

        let source = artifacts
            .iter()
            .find(|a| a.path.ends_with("wiki/sources/src-4846dd29.md"))
            .expect("source artifact");
        assert!(
            source
                .updated_markdown
                .contains("- [[wiki/concepts/rust]]"),
            "source should list referencing concept, got:\n{}",
            source.updated_markdown
        );
    }

    #[test]
    fn heading_anchor_and_source_document_ids_paths_combine_without_duplicates() {
        let root = tempdir().expect("tempdir");

        // Normalized metadata so heading_anchor path resolves.
        write(
            &root.path().join("normalized/src-abc/metadata.json"),
            r#"{
                "metadata": {"id": "src-abc"},
                "source_revision_id": "rev-123",
                "heading_ids": ["gil"]
            }"#,
        );
        write(
            &root.path().join("wiki/sources/src-abc.md"),
            "---\nid: wiki-source-src-abc\ntype: source\ntitle: GIL\nsource_document_id: src-abc\nsource_revision_id: rev-123\n---\n# Source\n",
        );
        // Second source document only reachable via source_document_ids.
        write(
            &root.path().join("wiki/sources/src-def.md"),
            "---\nid: wiki-source-src-def\ntype: source\ntitle: Extra\nsource_document_id: src-def\nsource_revision_id: rev-456\n---\n# Source\n",
        );

        write(
            &root.path().join("wiki/concepts/gil.md"),
            "---\nid: concept:gil\nname: GIL\nsources:\n- heading_anchor: gil\n  quote: GIL locks things.\n- quote: Only anchor-less quote.\nsource_document_ids:\n- src-abc\n- src-def\n---\n\n# GIL\n",
        );

        let artifacts = run_backlinks_pass(root.path()).expect("run backlink pass");
        let concept = artifacts
            .iter()
            .find(|a| a.path.ends_with("wiki/concepts/gil.md"))
            .expect("concept artifact");
        // Both sources present; each listed once (BTreeSet dedupes).
        assert!(
            concept.updated_markdown.contains("- [[wiki/sources/src-abc]]"),
            "missing src-abc backlink:\n{}",
            concept.updated_markdown
        );
        assert!(
            concept.updated_markdown.contains("- [[wiki/sources/src-def]]"),
            "missing src-def backlink:\n{}",
            concept.updated_markdown
        );
        let abc_count = concept.updated_markdown.matches("- [[wiki/sources/src-abc]]").count();
        assert_eq!(abc_count, 1, "src-abc should appear exactly once");
    }

    #[test]
    fn heading_anchor_resolution_respects_concept_source_document_ids() {
        let root = tempdir().expect("tempdir");

        // Two unrelated sources that happen to share a heading named "summary".
        write(
            &root.path().join("normalized/src-a/metadata.json"),
            r#"{
                "metadata": {"id": "src-a"},
                "source_revision_id": "rev-a",
                "heading_ids": ["summary"]
            }"#,
        );
        write(
            &root.path().join("normalized/src-b/metadata.json"),
            r#"{
                "metadata": {"id": "src-b"},
                "source_revision_id": "rev-b",
                "heading_ids": ["summary"]
            }"#,
        );
        write(
            &root.path().join("wiki/sources/src-a.md"),
            "---\nid: wiki-source-src-a\ntype: source\ntitle: A\nsource_document_id: src-a\nsource_revision_id: rev-a\n---\n# Source A\n",
        );
        write(
            &root.path().join("wiki/sources/src-b.md"),
            "---\nid: wiki-source-src-b\ntype: source\ntitle: B\nsource_document_id: src-b\nsource_revision_id: rev-b\n---\n# Source B\n",
        );

        // Concept sourced only from src-a. Its sources[].heading_anchor matches
        // "summary" — which globally resolves to both src-a and src-b — but the
        // concept's source_document_ids restricts the candidates to src-a only.
        write(
            &root.path().join("wiki/concepts/topic.md"),
            "---\nid: concept:topic\nname: Topic\nsources:\n- heading_anchor: summary\n  quote: A summary from src-a.\nsource_document_ids:\n- src-a\n---\n\n# Topic\n",
        );

        let artifacts = run_backlinks_pass(root.path()).expect("run backlink pass");

        let concept = artifacts
            .iter()
            .find(|a| a.path.ends_with("wiki/concepts/topic.md"))
            .expect("concept artifact");
        assert!(
            concept.updated_markdown.contains("- [[wiki/sources/src-a]]"),
            "concept should backlink to src-a:\n{}",
            concept.updated_markdown
        );
        assert!(
            !concept.updated_markdown.contains("- [[wiki/sources/src-b]]"),
            "concept must NOT backlink to unrelated src-b via shared anchor:\n{}",
            concept.updated_markdown
        );

        let src_b = artifacts
            .iter()
            .find(|a| a.path.ends_with("wiki/sources/src-b.md"))
            .expect("src-b artifact");
        assert!(
            !src_b.updated_markdown.contains("- [[wiki/concepts/topic]]"),
            "src-b must NOT list topic under Referenced by concepts:\n{}",
            src_b.updated_markdown
        );

        let src_a = artifacts
            .iter()
            .find(|a| a.path.ends_with("wiki/sources/src-a.md"))
            .expect("src-a artifact");
        assert!(
            src_a.updated_markdown.contains("- [[wiki/concepts/topic]]"),
            "src-a should list topic under Referenced by concepts:\n{}",
            src_a.updated_markdown
        );
    }

    #[test]
    fn mention_backlinks_credit_sources_that_mention_concept_name_or_aliases() {
        let root = tempdir().expect("tempdir");

        // Authoritative source for the Raft concept.
        write(
            &root.path().join("wiki/sources/src-a.md"),
            "---\nid: wiki-source-src-a\ntype: source\ntitle: Raft overview\nsource_document_id: src-a\nsource_revision_id: rev-a\n---\n# Raft overview\n\nRaft is a consensus algorithm.\n",
        );
        // Source that only mentions the concept's name as a standalone word.
        // Also contains the word "aircraft" to exercise word-boundary correctness.
        write(
            &root.path().join("wiki/sources/src-b.md"),
            "---\nid: wiki-source-src-b\ntype: source\ntitle: Read paths\nsource_document_id: src-b\nsource_revision_id: rev-b\n---\n# Read paths\n\nThe Raft leader serves linearizable reads. An aircraft reference should NOT match.\n",
        );
        // Source that references the concept via an alias-only mention.
        write(
            &root.path().join("wiki/sources/src-c.md"),
            "---\nid: wiki-source-src-c\ntype: source\ntitle: Alias mentions\nsource_document_id: src-c\nsource_revision_id: rev-c\n---\n# Alias\n\nSome systems use Raft-Consensus heavily in the design.\n",
        );
        // Unrelated source that should NOT be credited.
        write(
            &root.path().join("wiki/sources/src-d.md"),
            "---\nid: wiki-source-src-d\ntype: source\ntitle: Unrelated\nsource_document_id: src-d\nsource_revision_id: rev-d\n---\n# Unrelated\n\nThe word craft appears here and also aircraft, but no consensus algorithm.\n",
        );

        // Concept page sourced only from src-a; name "Raft", one alias.
        write(
            &root.path().join("wiki/concepts/raft.md"),
            "---\nid: concept:raft\nname: Raft\naliases:\n- Raft-Consensus\nsource_document_ids:\n- src-a\n---\n\n# Raft\n",
        );

        let artifacts = run_backlinks_pass(root.path()).expect("run backlink pass");
        let concept = artifacts
            .iter()
            .find(|a| a.path.ends_with("wiki/concepts/raft.md"))
            .expect("concept artifact");

        // Authoritative (source_document_ids) path must still work.
        assert!(
            concept.updated_markdown.contains("- [[wiki/sources/src-a]]"),
            "concept should backlink to authoritative src-a:\n{}",
            concept.updated_markdown
        );
        // Mention scan picks up the standalone "Raft" word in src-b.
        assert!(
            concept.updated_markdown.contains("- [[wiki/sources/src-b]]"),
            "concept should backlink to src-b via name mention:\n{}",
            concept.updated_markdown
        );
        // Mention scan picks up the alias "Raft-Consensus" in src-c.
        assert!(
            concept.updated_markdown.contains("- [[wiki/sources/src-c]]"),
            "concept should backlink to src-c via alias mention:\n{}",
            concept.updated_markdown
        );
        // Word-boundary correctness: "aircraft" / "craft" must NOT credit src-d.
        assert!(
            !concept.updated_markdown.contains("- [[wiki/sources/src-d]]"),
            "concept must NOT backlink to src-d (substring-only match):\n{}",
            concept.updated_markdown
        );

        // Dedup: src-a should appear only once despite being listed both as an
        // authoritative source and (trivially) matching its own mention regex.
        let a_count = concept
            .updated_markdown
            .matches("- [[wiki/sources/src-a]]")
            .count();
        assert_eq!(a_count, 1, "src-a should appear exactly once");
    }

    #[test]
    fn mention_backlinks_skip_short_aliases_and_stopwords() {
        let root = tempdir().expect("tempdir");

        write(
            &root.path().join("wiki/sources/src-a.md"),
            "---\nid: wiki-source-src-a\ntype: source\ntitle: Generic\nsource_document_id: src-a\nsource_revision_id: rev-a\n---\n# Source\n\nThe quick brown fox jumps and AI rules the day.\n",
        );

        // Concept with name "ArtificialIntelligence" but aliases "AI" (too short)
        // and "the" (stopword). Only the long name should produce a token.
        write(
            &root.path().join("wiki/concepts/ai.md"),
            "---\nid: concept:ai\nname: ArtificialIntelligence\naliases:\n- AI\n- the\n---\n\n# AI\n",
        );

        let artifacts = run_backlinks_pass(root.path()).expect("run backlink pass");
        let concept = artifacts
            .iter()
            .find(|a| a.path.ends_with("wiki/concepts/ai.md"))
            .expect("concept artifact");

        // "AI" and "the" appear in src-a but must NOT credit it (short/stopword).
        // The long name "ArtificialIntelligence" does not appear in src-a.
        assert!(
            !concept.updated_markdown.contains("- [[wiki/sources/src-a]]"),
            "short alias 'AI' and stopword 'the' must not trigger a mention backlink:\n{}",
            concept.updated_markdown
        );
    }

    #[test]
    fn mention_backlinks_word_boundary_does_not_match_substring() {
        let root = tempdir().expect("tempdir");

        // Source body mentions "TSGD" and "SGDM" (should not match "SGD") and
        // "SGD" as a standalone word (should match).
        write(
            &root.path().join("wiki/sources/src-with-sgd.md"),
            "---\nid: wiki-source-src-with-sgd\ntype: source\ntitle: Optimizers\nsource_document_id: src-x\nsource_revision_id: rev-x\n---\n# Optimizers\n\nSGD is classic. TSGD and SGDM are variants.\n",
        );
        write(
            &root.path().join("wiki/sources/src-substring-only.md"),
            "---\nid: wiki-source-src-substring-only\ntype: source\ntitle: Substrings only\nsource_document_id: src-y\nsource_revision_id: rev-y\n---\n# Substrings\n\nOnly TSGD and SGDM appear here, never the standalone form.\n",
        );

        write(
            &root.path().join("wiki/concepts/sgd.md"),
            "---\nid: concept:sgd\nname: SGD\n---\n\n# SGD\n",
        );

        let artifacts = run_backlinks_pass(root.path()).expect("run backlink pass");
        let concept = artifacts
            .iter()
            .find(|a| a.path.ends_with("wiki/concepts/sgd.md"))
            .expect("concept artifact");

        assert!(
            concept
                .updated_markdown
                .contains("- [[wiki/sources/src-with-sgd]]"),
            "concept should backlink to source with standalone SGD:\n{}",
            concept.updated_markdown
        );
        assert!(
            !concept
                .updated_markdown
                .contains("- [[wiki/sources/src-substring-only]]"),
            "concept must NOT backlink via substring-only matches (TSGD, SGDM):\n{}",
            concept.updated_markdown
        );
    }
}
