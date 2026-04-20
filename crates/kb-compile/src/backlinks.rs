use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use ignore::WalkBuilder;
use kb_core::fs::atomic_write;
use kb_core::rewrite_managed_region;
use regex::Regex;
use serde_yaml::Value;

use crate::source_page::source_page_path_for_id;

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
    collect_mention_backlinks(&concept_pages, &source_pages, &mut concept_backlinks, root)?;

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
        if is_index_page(&path) {
            continue;
        }
        let page_id = page_id_from_path(root, &path)?;
        pages.insert(page_id, path);
    }
    Ok(pages)
}

/// Auto-generated index pages (`wiki/index.md`, `wiki/sources/index.md`,
/// `wiki/concepts/index.md`) enumerate every source/concept and therefore
/// would inflate backlink sets with false positives (both as wiki-link
/// sources citing every concept and — in the mention-scanner — as source
/// pages whose body contains every concept's name). Skip them globally.
fn is_index_page(path: &Path) -> bool {
    path.file_name()
        .and_then(|n| n.to_str())
        .is_some_and(|name| name == "index.md")
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
        if is_index_page(&page) {
            continue;
        }
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
/// scan every `normalized/<src>/source.md` body (the full original text after
/// frontmatter strip). A source whose normalized body mentions a concept token
/// as a standalone word (case-insensitive, Unicode word boundaries) OR that
/// contains a `[[<slug>]]`-style wiki-link whose normalized target matches the
/// concept's slug or an alias slug is added to the concept's backlinks set,
/// keyed by the source's `wiki/sources/<src>` page id.
///
/// Walking the normalized source text (not the wiki source summary body)
/// ensures we don't miss mentions that the 100-word LLM summary elided.
///
/// This is additive on top of the source_document_ids-gated primary path, and
/// is deduped via the `concept_backlinks` `BTreeSet`.
/// Per-concept mention matcher: union regex over concept name + alias tokens
/// (plus plural/singular inflections) and the set of wiki-link slug targets
/// that count as a mention of this concept.
struct ConceptMatcher {
    concept_id: String,
    word_regex: Option<Regex>,
    /// Wiki-link target slugs (full `wiki/concepts/<slug>`) that should
    /// count as a mention of this concept.
    slug_targets: BTreeSet<String>,
}

/// Build a `ConceptMatcher` for a single concept page by parsing its
/// frontmatter name/aliases, expanding tokens with simple plural/singular
/// inflections, and compiling the union word-boundary regex.
fn build_concept_matcher(concept_id: &str, path: &Path) -> Result<ConceptMatcher> {
    let markdown = std::fs::read_to_string(path)
        .with_context(|| format!("read concept page {}", path.display()))?;
    let (name_opt, alias_tokens, alias_slugs) = match split_frontmatter(&markdown) {
        Some((fm, _body)) => parse_concept_name_and_aliases(&fm),
        None => (None, Vec::new(), Vec::new()),
    };

    // Build the set of word tokens to match: concept name + aliases, each
    // filtered by length and stopwords. Names/aliases map to tokens via
    // `normalize_mention_token`, then expanded to cover simple plural and
    // singular inflections so a source that says "quorums" still credits
    // an alias "quorum".
    let mut tokens: BTreeSet<String> = BTreeSet::new();
    if let Some(name) = name_opt.as_ref() {
        if let Some(t) = normalize_mention_token(name) {
            for form in inflect_mention_forms(&t) {
                tokens.insert(form);
            }
        }
    }
    for alias in &alias_tokens {
        if let Some(t) = normalize_mention_token(alias) {
            for form in inflect_mention_forms(&t) {
                tokens.insert(form);
            }
        }
    }

    // Build wiki-link slug targets: the concept's own page id, plus any
    // slug-normalized aliases under wiki/concepts/. This lets an alias
    // written as `[[aliases-slug]]` route to its concept.
    let mut slug_targets: BTreeSet<String> = BTreeSet::new();
    slug_targets.insert(concept_id.to_string());
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
        Some(
            Regex::new(&full)
                .with_context(|| format!("compile mention regex for {concept_id}"))?,
        )
    };

    Ok(ConceptMatcher {
        concept_id: concept_id.to_string(),
        word_regex,
        slug_targets,
    })
}

fn collect_mention_backlinks(
    concept_pages: &BTreeMap<String, PathBuf>,
    source_pages: &BTreeMap<String, PathBuf>,
    concept_backlinks: &mut BTreeMap<String, BTreeSet<String>>,
    root: &Path,
) -> Result<()> {
    let mut matchers: Vec<ConceptMatcher> = Vec::with_capacity(concept_pages.len());
    for (concept_id, path) in concept_pages {
        matchers.push(build_concept_matcher(concept_id, path)?);
    }

    let wiki_link_re = Regex::new(r"\[\[([^\]\r\n]+)\]\]").context("compile wikilink regex")?;

    // Scan each normalized source's full original text (after frontmatter and
    // code-block/span stripping) and credit concepts to the corresponding
    // `wiki/sources/<src>.md` page. Walking `normalized/<src>/source.md`
    // (not `wiki/sources/<src>.md`) avoids missing mentions that the 100-word
    // LLM summary elided from the wiki source page body.
    let normalized_root = root.join(NORMALIZED_DIR);
    if !normalized_root.exists() {
        return Ok(());
    }

    let entries = std::fs::read_dir(&normalized_root)
        .with_context(|| format!("read normalized dir {}", normalized_root.display()))?;
    for entry in entries {
        let entry = entry.with_context(|| format!("walk {}", normalized_root.display()))?;
        if !entry.file_type().map(|t| t.is_dir()).unwrap_or(false) {
            continue;
        }
        let src_id = entry.file_name().to_string_lossy().into_owned();
        let source_md_path = entry.path().join("source.md");
        if !source_md_path.is_file() {
            continue;
        }

        // Map `normalized/<src_id>/` to the corresponding `wiki/sources/<slug>`
        // page id by reusing the same slug rule the source-page generator uses.
        // We only credit this src_id if it has a matching entry in `source_pages`;
        // otherwise the source has no wiki page to link to.
        let source_page_rel = source_page_path_for_id(&src_id);
        let source_page_id = source_page_rel
            .to_string_lossy()
            .replace('\\', "/")
            .trim_end_matches(".md")
            .to_string();
        if !source_pages.contains_key(&source_page_id) {
            continue;
        }

        let markdown = std::fs::read_to_string(&source_md_path)
            .with_context(|| format!("read normalized source {}", source_md_path.display()))?;
        let body = match split_frontmatter(&markdown) {
            Some((_fm, body)) => body,
            None => markdown,
        };
        // Strip fenced code blocks and inline code spans so that tokens
        // appearing only inside `code` / ```fences``` don't trigger
        // backlinks. This mirrors bn-bup's existing skip rules used elsewhere.
        let scanned = strip_code_regions(&body);

        // Precollect the normalized wiki-link targets present in this body.
        let mut wiki_targets: BTreeSet<String> = BTreeSet::new();
        let mut bare_targets: BTreeSet<String> = BTreeSet::new();
        for capture in wiki_link_re.captures_iter(&scanned) {
            let Some(raw) = capture.get(1).map(|c| c.as_str()) else {
                continue;
            };
            if let Some(target) = normalize_wiki_link(raw) {
                wiki_targets.insert(target);
            }
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
                    if re.is_match(&scanned) {
                        hit = true;
                    }
                }
            }

            if hit {
                if let Some(referers) = concept_backlinks.get_mut(&matcher.concept_id) {
                    referers.insert(source_page_id.clone());
                }
            }
        }
    }

    Ok(())
}

/// Remove fenced code blocks (```...```) and inline code spans (`...`) from a
/// markdown body. Replaces the removed content with spaces so byte offsets are
/// not needed but line structure is preserved enough for word-boundary regex
/// matching to behave correctly on surrounding text.
fn strip_code_regions(body: &str) -> String {
    let mut out = String::with_capacity(body.len());
    let mut in_fence = false;
    for line in body.split_inclusive('\n') {
        let trimmed_start = line.trim_start();
        if trimmed_start.starts_with("```") || trimmed_start.starts_with("~~~") {
            in_fence = !in_fence;
            // Replace the fence line itself with a blank so the opening/closing
            // fence tokens don't survive into the scanned text.
            for _ in 0..line.len() {
                out.push(' ');
            }
            if line.ends_with('\n') {
                // Last byte was the newline we already spaced over; restore it.
                out.pop();
                out.push('\n');
            }
            continue;
        }
        if in_fence {
            for _ in 0..line.len() {
                out.push(' ');
            }
            if line.ends_with('\n') {
                out.pop();
                out.push('\n');
            }
            continue;
        }
        // Strip inline backtick spans.
        let bytes = line.as_bytes();
        let mut i = 0;
        while i < bytes.len() {
            if bytes[i] == b'`' {
                if let Some(close_offset) = bytes[i + 1..].iter().position(|&b| b == b'`') {
                    let close = i + 1 + close_offset;
                    for _ in i..=close {
                        out.push(' ');
                    }
                    i = close + 1;
                    continue;
                }
                out.push('`');
                i += 1;
                while i < bytes.len() {
                    out.push(bytes[i] as char);
                    i += 1;
                }
                break;
            }
            out.push(bytes[i] as char);
            i += 1;
        }
    }
    out
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

/// Minimum token length for pluralizing / singularizing. Below this threshold
/// (e.g. "SQL", "BFT") morphological tweaks almost always produce false
/// positives, so we keep the token literal.
const MIN_INFLECTION_LEN: usize = 4;

/// Expand an alias/name token into the set of word-boundary forms the
/// mention-scanner should match. Always includes the original. For single-word
/// tokens of sufficient length we additionally emit basic English plural and
/// singular inflections:
///
/// * `<tok>` + "s"  (quorum → quorums) when the token ends in a consonant.
/// * `<tok>` + "es" (box → boxes)      when the token ends in s/sh/ch/x/z.
/// * singular by stripping a trailing "s" (requirements → requirement) when
///   the token has length >= 5.
///
/// The rules intentionally stay small — no Porter stemmer — and only apply to
/// single-word tokens so we don't mangle multi-word aliases like "Byzantine
/// fault tolerance" (which get matched as-is).
fn inflect_mention_forms(token: &str) -> Vec<String> {
    let mut forms: Vec<String> = Vec::new();
    forms.push(token.to_string());

    let is_single_word = !token.chars().any(char::is_whitespace);
    if !is_single_word {
        return forms;
    }

    let char_count = token.chars().count();
    if char_count < MIN_INFLECTION_LEN {
        return forms;
    }

    let lower = token.to_lowercase();
    let last_char = lower.chars().last();
    let ends_with_vowel = matches!(last_char, Some('a' | 'e' | 'i' | 'o' | 'u'));

    // -es plural for sibilant endings (box → boxes, dish → dishes).
    let es_plural = lower.ends_with('s')
        || lower.ends_with("sh")
        || lower.ends_with("ch")
        || lower.ends_with('x')
        || lower.ends_with('z');

    if es_plural {
        forms.push(format!("{token}es"));
    } else if !ends_with_vowel {
        // Simple -s plural, but only when the token ends in a consonant to
        // avoid nonsense like "data" → "datas".
        forms.push(format!("{token}s"));
    }

    // Singularize: "requirements" → "requirement". Keep >= 5 total chars so
    // stripping yields a token still >= MIN_MENTION_TOKEN_LEN long.
    if char_count >= 5 && lower.ends_with('s') && !lower.ends_with("ss") {
        // Drop the trailing ASCII 's'. All mention tokens are lowercase-normalized
        // at match time via `(?i)`, so stripping a trailing byte is safe for
        // ASCII-ending tokens; for non-ASCII tails we leave the token alone.
        if token.is_char_boundary(token.len() - 1) && token.ends_with('s') {
            let singular = &token[..token.len() - 1];
            if singular.chars().count() >= MIN_MENTION_TOKEN_LEN {
                forms.push(singular.to_string());
            }
        }
    }

    forms
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

        // Authoritative source for the Raft concept. Wiki page is a short
        // summary; full original text lives under normalized/<src>/source.md.
        write(
            &root.path().join("wiki/sources/src-a.md"),
            "---\nid: wiki-source-src-a\ntype: source\ntitle: Raft overview\nsource_document_id: src-a\nsource_revision_id: rev-a\n---\n# Raft overview\n\nSummary only.\n",
        );
        write(
            &root.path().join("normalized/src-a/source.md"),
            "# Raft overview\n\nRaft is a consensus algorithm.\n",
        );
        // Source that only mentions the concept's name as a standalone word.
        // Also contains the word "aircraft" to exercise word-boundary correctness.
        write(
            &root.path().join("wiki/sources/src-b.md"),
            "---\nid: wiki-source-src-b\ntype: source\ntitle: Read paths\nsource_document_id: src-b\nsource_revision_id: rev-b\n---\n# Read paths\n\nSummary.\n",
        );
        write(
            &root.path().join("normalized/src-b/source.md"),
            "# Read paths\n\nThe Raft leader serves linearizable reads. An aircraft reference should NOT match.\n",
        );
        // Source that references the concept via an alias-only mention.
        write(
            &root.path().join("wiki/sources/src-c.md"),
            "---\nid: wiki-source-src-c\ntype: source\ntitle: Alias mentions\nsource_document_id: src-c\nsource_revision_id: rev-c\n---\n# Alias\n\nSummary.\n",
        );
        write(
            &root.path().join("normalized/src-c/source.md"),
            "# Alias\n\nSome systems use Raft-Consensus heavily in the design.\n",
        );
        // Unrelated source that should NOT be credited.
        write(
            &root.path().join("wiki/sources/src-d.md"),
            "---\nid: wiki-source-src-d\ntype: source\ntitle: Unrelated\nsource_document_id: src-d\nsource_revision_id: rev-d\n---\n# Unrelated\n\nSummary.\n",
        );
        write(
            &root.path().join("normalized/src-d/source.md"),
            "# Unrelated\n\nThe word craft appears here and also aircraft, but no consensus algorithm.\n",
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
            "---\nid: wiki-source-src-a\ntype: source\ntitle: Generic\nsource_document_id: src-a\nsource_revision_id: rev-a\n---\n# Source\n\nSummary only.\n",
        );
        write(
            &root.path().join("normalized/src-a/source.md"),
            "# Source\n\nThe quick brown fox jumps and AI rules the day.\n",
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
    fn inflect_mention_forms_adds_plural_for_consonant_ending() {
        let forms = inflect_mention_forms("Quorum");
        assert!(forms.iter().any(|f| f == "Quorum"));
        assert!(
            forms.iter().any(|f| f == "Quorums"),
            "expected plural 'Quorums', got {forms:?}"
        );
    }

    #[test]
    fn inflect_mention_forms_adds_es_plural_for_sibilant_ending() {
        // "index" -> "indexes" (ends in x, length 5 >= MIN_INFLECTION_LEN).
        let forms = inflect_mention_forms("index");
        assert!(
            forms.iter().any(|f| f == "indexes"),
            "expected -es plural 'indexes', got {forms:?}"
        );

        // "class" -> "classes" (ends in s). Also tests we don't singularize
        // past an -ss ending (should NOT emit "clas").
        let forms_class = inflect_mention_forms("class");
        assert!(
            forms_class.iter().any(|f| f == "classes"),
            "expected -es plural 'classes', got {forms_class:?}"
        );
        assert!(
            !forms_class.iter().any(|f| f == "clas"),
            "must not strip 's' from '-ss' ending: {forms_class:?}"
        );
    }

    #[test]
    fn inflect_mention_forms_singularizes_trailing_s() {
        let forms = inflect_mention_forms("requirements");
        assert!(
            forms.iter().any(|f| f == "requirement"),
            "expected singular 'requirement', got {forms:?}"
        );
    }

    #[test]
    fn inflect_mention_forms_skips_short_tokens() {
        let forms = inflect_mention_forms("SQL");
        // Only the literal; no pluralization / singularization on 3-char tokens.
        assert_eq!(forms, vec!["SQL".to_string()]);
    }

    #[test]
    fn inflect_mention_forms_skips_multi_word_tokens() {
        let forms = inflect_mention_forms("Byzantine fault tolerance");
        // Multi-word tokens are matched as-is; we don't pluralize them.
        assert_eq!(forms, vec!["Byzantine fault tolerance".to_string()]);
    }

    #[test]
    fn inflect_mention_forms_does_not_double_pluralize_ss() {
        // "business" ends in "ss" — singularizing would drop to "busines" which
        // is nonsense. The -es rule DOES still apply and yields "businesses".
        let forms = inflect_mention_forms("business");
        assert!(forms.iter().any(|f| f == "business"));
        assert!(
            !forms.iter().any(|f| f == "busines"),
            "must not strip 's' from '-ss' ending: {forms:?}"
        );
        assert!(
            forms.iter().any(|f| f == "businesses"),
            "expected -es plural 'businesses', got {forms:?}"
        );
    }

    #[test]
    fn mention_backlinks_match_plural_forms_of_alias() {
        let root = tempdir().expect("tempdir");

        // Source page whose body mentions only the plural "quorums".
        write(
            &root.path().join("wiki/sources/src-plural.md"),
            "---\nid: wiki-source-src-plural\ntype: source\ntitle: Plural form\nsource_document_id: src-plural\nsource_revision_id: rev-1\n---\n# Plural\n\nSummary.\n",
        );
        write(
            &root.path().join("normalized/src-plural/source.md"),
            "# Plural\n\nMajority quorums are required to commit.\n",
        );
        // Source that uses a compound mention; since we try the alias as-is and
        // plural, "majority quorum" as a two-word phrase still matches the plain
        // "quorum" token via word-boundary.
        write(
            &root.path().join("wiki/sources/src-compound.md"),
            "---\nid: wiki-source-src-compound\ntype: source\ntitle: Compound\nsource_document_id: src-compound\nsource_revision_id: rev-2\n---\n# Compound\n\nSummary.\n",
        );
        write(
            &root.path().join("normalized/src-compound/source.md"),
            "# Compound\n\nA majority quorum is the usual protocol requirement.\n",
        );
        // Source that should NOT be credited — "craft" must not match "quorum"
        // substring tricks nor should the word "requires" trigger "requirement".
        write(
            &root.path().join("wiki/sources/src-nomatch.md"),
            "---\nid: wiki-source-src-nomatch\ntype: source\ntitle: Nope\nsource_document_id: src-nomatch\nsource_revision_id: rev-3\n---\n# Nope\n\nSummary.\n",
        );
        write(
            &root.path().join("normalized/src-nomatch/source.md"),
            "# Nope\n\nThis page is about aircraft and draft notes only.\n",
        );

        // Concept with alias "quorum" (triggers plural expansion) and
        // "requirement" (tests -s plural matching "requirements" in body).
        write(
            &root.path().join("wiki/concepts/quorum.md"),
            "---\nid: concept:quorum\nname: Quorum\naliases:\n- quorum\n- requirement\n---\n\n# Quorum\n",
        );

        let artifacts = run_backlinks_pass(root.path()).expect("run backlink pass");
        let concept = artifacts
            .iter()
            .find(|a| a.path.ends_with("wiki/concepts/quorum.md"))
            .expect("concept artifact");

        assert!(
            concept
                .updated_markdown
                .contains("- [[wiki/sources/src-plural]]"),
            "plural 'quorums' should match alias 'quorum':\n{}",
            concept.updated_markdown
        );
        assert!(
            concept
                .updated_markdown
                .contains("- [[wiki/sources/src-compound]]"),
            "compound 'majority quorum' should match alias 'quorum' by word boundary:\n{}",
            concept.updated_markdown
        );
        assert!(
            !concept
                .updated_markdown
                .contains("- [[wiki/sources/src-nomatch]]"),
            "aircraft/draft must NOT match quorum or requirement aliases:\n{}",
            concept.updated_markdown
        );
    }

    #[test]
    fn mention_backlinks_singularize_source_uses_alias() {
        let root = tempdir().expect("tempdir");

        // Source body says "requirements" (plural); concept has alias "requirement".
        write(
            &root.path().join("wiki/sources/src-r.md"),
            "---\nid: wiki-source-src-r\ntype: source\ntitle: Reqs\nsource_document_id: src-r\nsource_revision_id: rev-r\n---\n# Reqs\n\nSummary.\n",
        );
        write(
            &root.path().join("normalized/src-r/source.md"),
            "# Reqs\n\nFunctional requirements and non-functional requirements both matter.\n",
        );

        write(
            &root.path().join("wiki/concepts/requirement.md"),
            "---\nid: concept:requirement\nname: Requirement\naliases:\n- requirement\n---\n\n# Requirement\n",
        );

        let artifacts = run_backlinks_pass(root.path()).expect("run backlink pass");
        let concept = artifacts
            .iter()
            .find(|a| a.path.ends_with("wiki/concepts/requirement.md"))
            .expect("concept artifact");

        assert!(
            concept.updated_markdown.contains("- [[wiki/sources/src-r]]"),
            "concept alias 'requirement' should match plural 'requirements':\n{}",
            concept.updated_markdown
        );
    }

    #[test]
    fn mention_backlinks_short_alias_is_not_pluralized() {
        let root = tempdir().expect("tempdir");

        // "SQLs" appears in body but no standalone "SQL" word. The alias "SQL"
        // has 3 chars, which is below MIN_INFLECTION_LEN (4) — so we do NOT
        // emit "SQLs" as a pluralized form. The literal `\bSQL\b` also doesn't
        // match inside "SQLs" due to the trailing 's'. Net effect: no backlink.
        write(
            &root.path().join("wiki/sources/src-sql.md"),
            "---\nid: wiki-source-src-sql\ntype: source\ntitle: Databases\nsource_document_id: src-sql\nsource_revision_id: rev-sql\n---\n# Databases\n\nSummary.\n",
        );
        write(
            &root.path().join("normalized/src-sql/source.md"),
            "# Databases\n\nSome SQLs are bad, and SQLite is fine.\n",
        );

        write(
            &root.path().join("wiki/concepts/sql.md"),
            "---\nid: concept:sql\nname: SQL\naliases:\n- SQL\n---\n\n# Structured Query Language\n",
        );

        let artifacts = run_backlinks_pass(root.path()).expect("run backlink pass");
        let concept = artifacts
            .iter()
            .find(|a| a.path.ends_with("wiki/concepts/sql.md"))
            .expect("concept artifact");
        assert!(
            !concept.updated_markdown.contains("- [[wiki/sources/src-sql]]"),
            "short alias 'SQL' (< MIN_MENTION_TOKEN_LEN) must not produce mention backlinks:\n{}",
            concept.updated_markdown
        );
    }

    #[test]
    fn mention_backlinks_scan_normalized_source_not_wiki_summary() {
        // Regression for bn-gdd: the wiki/sources/<src>.md summary is a
        // ~100-word LLM compression that may omit specific technical terms
        // present in the original. The mention-scanner must walk
        // `normalized/<src>/source.md` (the full original) instead.
        let root = tempdir().expect("tempdir");

        // Summary says nothing about "mutex"; the full normalized body does.
        write(
            &root.path().join("wiki/sources/src-kern.md"),
            "---\nid: wiki-source-src-kern\ntype: source\ntitle: Kernel notes\nsource_document_id: src-kern\nsource_revision_id: rev-k\n---\n# Kernel notes\n\nDiscussion of synchronization primitives.\n",
        );
        write(
            &root.path().join("normalized/src-kern/source.md"),
            "# Kernel notes\n\nA mutex is held while entering the critical section; mutexes ensure exclusive access.\n",
        );

        write(
            &root.path().join("wiki/concepts/mutex.md"),
            "---\nid: concept:mutex\nname: Mutex\n---\n\n# Mutex\n",
        );

        let artifacts = run_backlinks_pass(root.path()).expect("run backlink pass");
        let concept = artifacts
            .iter()
            .find(|a| a.path.ends_with("wiki/concepts/mutex.md"))
            .expect("concept artifact");

        assert!(
            concept
                .updated_markdown
                .contains("- [[wiki/sources/src-kern]]"),
            "mutex concept should backlink to src-kern via normalized source text mention, got:\n{}",
            concept.updated_markdown
        );
    }

    #[test]
    fn mention_backlinks_exclude_auto_generated_index_pages() {
        // Regression for bn-gdd: `wiki/sources/index.md` (auto-generated)
        // lists every source's title; a concept named "Interrupt" would
        // otherwise spuriously gain the index as a backlink. Index pages
        // must be filtered out of both the scanned candidate set and the
        // wiki-link referer set.
        let root = tempdir().expect("tempdir");

        // A real source whose normalized body legitimately mentions "interrupt".
        write(
            &root.path().join("wiki/sources/src-real.md"),
            "---\nid: wiki-source-src-real\ntype: source\ntitle: Interrupts\nsource_document_id: src-real\nsource_revision_id: rev-r\n---\n# Interrupts\n\nSummary.\n",
        );
        write(
            &root.path().join("normalized/src-real/source.md"),
            "# Interrupts\n\nHardware interrupts are delivered via IRQ lines.\n",
        );

        // Auto-generated index listing the source title and a wiki-link to the
        // interrupt concept. Must NOT appear in any concept's backlinks.
        write(
            &root.path().join("wiki/sources/index.md"),
            "# Sources index\n\n- [[wiki/sources/src-real]] — Interrupts\n- [[wiki/concepts/interrupt]]\n",
        );
        write(
            &root.path().join("wiki/concepts/index.md"),
            "# Concepts index\n\n- [[wiki/concepts/interrupt]]\n",
        );

        write(
            &root.path().join("wiki/concepts/interrupt.md"),
            "---\nid: concept:interrupt\nname: Interrupt\n---\n\n# Interrupt\n",
        );

        let artifacts = run_backlinks_pass(root.path()).expect("run backlink pass");
        let concept = artifacts
            .iter()
            .find(|a| a.path.ends_with("wiki/concepts/interrupt.md"))
            .expect("concept artifact");

        assert!(
            concept
                .updated_markdown
                .contains("- [[wiki/sources/src-real]]"),
            "real source with 'interrupt' in normalized text should be a backlink, got:\n{}",
            concept.updated_markdown
        );
        assert!(
            !concept
                .updated_markdown
                .contains("- [[wiki/sources/index]]"),
            "auto-generated sources index must NOT appear in backlinks:\n{}",
            concept.updated_markdown
        );
        assert!(
            !concept
                .updated_markdown
                .contains("- [[wiki/concepts/index]]"),
            "auto-generated concepts index must NOT appear in backlinks:\n{}",
            concept.updated_markdown
        );
    }

    #[test]
    fn mention_backlinks_skip_fenced_code_and_inline_code() {
        // bn-bup invariant: mentions inside fenced code blocks and inline
        // code spans must not trigger backlinks.
        let root = tempdir().expect("tempdir");

        write(
            &root.path().join("wiki/sources/src-code.md"),
            "---\nid: wiki-source-src-code\ntype: source\ntitle: Code only\nsource_document_id: src-code\nsource_revision_id: rev-c\n---\n# Code\n\nSummary.\n",
        );
        write(
            &root.path().join("normalized/src-code/source.md"),
            "# Code only\n\nSee the API.\n\n```\npthread_mutex_lock(&mutex);\n```\n\nAlso `mutex_init()` is called.\n",
        );

        write(
            &root.path().join("wiki/concepts/mutex.md"),
            "---\nid: concept:mutex\nname: Mutex\n---\n\n# Mutex\n",
        );

        let artifacts = run_backlinks_pass(root.path()).expect("run backlink pass");
        let concept = artifacts
            .iter()
            .find(|a| a.path.ends_with("wiki/concepts/mutex.md"))
            .expect("concept artifact");

        assert!(
            !concept
                .updated_markdown
                .contains("- [[wiki/sources/src-code]]"),
            "mentions only inside code regions must not credit the source:\n{}",
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
            "---\nid: wiki-source-src-with-sgd\ntype: source\ntitle: Optimizers\nsource_document_id: src-with-sgd\nsource_revision_id: rev-x\n---\n# Optimizers\n\nSummary.\n",
        );
        write(
            &root.path().join("normalized/src-with-sgd/source.md"),
            "# Optimizers\n\nSGD is classic. TSGD and SGDM are variants.\n",
        );
        write(
            &root.path().join("wiki/sources/src-substring-only.md"),
            "---\nid: wiki-source-src-substring-only\ntype: source\ntitle: Substrings only\nsource_document_id: src-substring-only\nsource_revision_id: rev-y\n---\n# Substrings\n\nSummary.\n",
        );
        write(
            &root.path().join("normalized/src-substring-only/source.md"),
            "# Substrings\n\nOnly TSGD and SGDM appear here, never the standalone form.\n",
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
