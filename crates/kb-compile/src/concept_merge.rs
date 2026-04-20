use std::collections::{BTreeSet, HashMap};
use std::fmt::Write as FmtWrite;
use std::path::{Path, PathBuf};
use std::sync::LazyLock;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, bail};
use kb_core::fs::atomic_write;
use kb_core::frontmatter::{read_frontmatter, write_frontmatter};
use kb_core::{
    BuildRecord, EntityMetadata, ReviewItem, ReviewKind, ReviewStatus, Status, hash_many,
    save_review_item, slug_from_title,
};
use kb_llm::{
    ConceptCandidate, LlmAdapter, LlmAdapterError, MergeConceptCandidatesRequest,
    MergeConceptCandidatesResponse, MergeGroup, ProvenanceRecord, SourceAnchor,
};
use regex::Regex;
use serde_yaml::{Mapping, Value};

pub const WIKI_CONCEPTS_DIR: &str = "wiki/concepts";
pub const MERGE_REVIEWS_DIR: &str = "reviews/merges";

/// A canonical concept page produced by the merge pass.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConceptPage {
    /// URL-safe slug derived from the canonical name.
    pub slug: String,
    /// Destination path under `wiki/concepts/`.
    pub path: PathBuf,
    /// Full markdown content including YAML frontmatter.
    pub content: String,
    pub canonical_name: String,
    pub aliases: Vec<String>,
}

/// An uncertain merge queued for human review.
///
/// Wraps a [`ReviewItem`] so the record shares the canonical review-queue schema
/// with every other reviewer-facing item (e.g. the duplicate-concepts check in
/// `kb-lint`). `kb review list/show/approve/reject` consume this through
/// `list_review_items` / `load_review_item` without special-casing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MergeReviewRecord {
    pub item: ReviewItem,
}

impl MergeReviewRecord {
    #[must_use]
    pub fn merge_id(&self) -> &str {
        &self.item.metadata.id
    }
}

/// Output emitted by the concept merge pass.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConceptMergeArtifact {
    pub concept_pages: Vec<ConceptPage>,
    pub review_records: Vec<MergeReviewRecord>,
    pub build_record: BuildRecord,
}

/// Errors produced by the concept merge pass.
#[derive(Debug, thiserror::Error)]
pub enum ConceptMergeError {
    #[error("llm merge_concept_candidates failed: {0}")]
    Llm(#[from] LlmAdapterError),
    #[error("failed to serialize: {0}")]
    Serialize(String),
    #[error("failed to read system time: {0}")]
    Time(String),
}

/// Persist a concept page produced by [`run_concept_merge_pass`].
///
/// # Errors
///
/// Returns an error when the target file cannot be written atomically.
pub fn persist_concept_page(page: &ConceptPage) -> Result<()> {
    atomic_write(&page.path, page.content.as_bytes())
        .with_context(|| format!("write concept page {}", page.path.display()))
}

/// Persist a merge review record produced by [`run_concept_merge_pass`].
///
/// Delegates to `kb_core::save_review_item` so the on-disk format matches other
/// review-queue producers and so the built-in rejected-item de-dup (same source
/// hashes → skip rewrite) prevents re-queueing work the user already declined.
///
/// # Errors
///
/// Returns an error when the target file cannot be written atomically.
pub fn persist_merge_review(root: &Path, record: &MergeReviewRecord) -> Result<()> {
    save_review_item(root, &record.item)
        .with_context(|| format!("write merge review {}", record.item.metadata.id))
}

/// Cluster concept candidates into canonical concept pages and review items.
///
/// Candidates are sorted by name before being sent to the LLM so that identical
/// inputs always produce identical outputs across runs.
///
/// Confident merge groups become `wiki/concepts/<slug>.md`; uncertain groups go to
/// `reviews/merges/<merge-id>.json` for human approval.
///
/// # Errors
///
/// Returns [`ConceptMergeError`] when the LLM call fails, the response cannot be
/// serialized, or the system clock cannot be read for build-record timestamps.
pub fn run_concept_merge_pass<A: LlmAdapter + ?Sized>(
    adapter: &A,
    candidates: Vec<ConceptCandidate>,
    root: &Path,
) -> Result<ConceptMergeArtifact, ConceptMergeError> {
    run_concept_merge_pass_with_origins(adapter, candidates, root, &HashMap::new())
}

/// Same as [`run_concept_merge_pass`] but threads an origin side-map.
///
/// The map `candidate_name -> source_document_ids` lets confident concept pages
/// emit a `source_document_ids:` frontmatter field. Missing entries are
/// silently ignored — tests and callers without provenance fall back to the
/// empty map.
///
/// # Errors
///
/// Returns [`ConceptMergeError`] when the LLM call fails, the response cannot be
/// serialized, or the system clock cannot be read for build-record timestamps.
pub fn run_concept_merge_pass_with_origins<A, S>(
    adapter: &A,
    mut candidates: Vec<ConceptCandidate>,
    root: &Path,
    candidate_origins: &HashMap<String, Vec<String>, S>,
) -> Result<ConceptMergeArtifact, ConceptMergeError>
where
    A: LlmAdapter + ?Sized,
    S: std::hash::BuildHasher,
{
    candidates.sort_unstable_by(|a, b| a.name.cmp(&b.name));

    let request = MergeConceptCandidatesRequest {
        candidates: candidates.clone(),
    };
    let (response, provenance) = adapter.merge_concept_candidates(request)?;

    let concept_pages = build_concept_pages(&response, root, candidate_origins)?;
    let review_records =
        build_review_records(&response, root, &provenance).map_err(ConceptMergeError::Serialize)?;
    let build_record = build_record_for_merge(
        &candidates,
        &response,
        &concept_pages,
        &review_records,
        &provenance,
    )?;

    Ok(ConceptMergeArtifact {
        concept_pages,
        review_records,
        build_record,
    })
}

fn build_concept_pages<S: std::hash::BuildHasher>(
    response: &MergeConceptCandidatesResponse,
    root: &Path,
    candidate_origins: &HashMap<String, Vec<String>, S>,
) -> Result<Vec<ConceptPage>, ConceptMergeError> {
    response
        .groups
        .iter()
        .filter(|g| g.confident)
        .map(|group| render_concept_page(group, root, candidate_origins))
        .collect()
}

fn build_review_records(
    response: &MergeConceptCandidatesResponse,
    root: &Path,
    provenance: &ProvenanceRecord,
) -> Result<Vec<MergeReviewRecord>, String> {
    response
        .groups
        .iter()
        .filter(|g| !g.confident)
        .map(|group| render_review_record(group, root, provenance))
        .collect()
}

/// Narrow-variant phrasings that signal the merged body describes one alias
/// rather than the canonical concept.
///
/// Each entry is a pre-compiled regex applied to the trimmed start of the body.
/// Patterns are intentionally coarse — we want to flag suspicious prose, not
/// classify every definition. The first match wins and is surfaced in the
/// warning log line for triage.
static NARROW_VARIANT_BODY_PATTERNS: LazyLock<Vec<(&'static str, Regex)>> = LazyLock::new(|| {
    vec![
        // "A Paxos variant that …" / "A Cheap Paxos variant that …" — any
        // 1-to-~30-char noun followed by the word "variant". The body we saw
        // on the Paxos page was exactly this shape.
        (
            "narrow_variant_prefix",
            Regex::new(r"(?i)^\s*an?\s+\S.{0,30}?\s+variant\b").expect("narrow_variant_prefix"),
        ),
        // "A specialization of X" / "An instance of X" / "A special case of X"
        // — body frames itself as a sub-case rather than the general idea.
        (
            "specialization_prefix",
            Regex::new(r"(?i)^\s*an?\s+(specialization|special\s+case|instance|sub-?case|sub-?class|subtype|kind)\s+of\b")
                .expect("specialization_prefix"),
        ),
    ]
});

/// Returns the name of the first matching narrow-variant pattern for `body`,
/// or `None` if the body does not look like a narrow-variant definition.
///
/// Skips the check when the canonical name itself is literally the first
/// "variant" noun in the body (e.g. a canonical called "Paxos variant" would
/// otherwise always match), so we only fire on bodies that lead with an
/// *alias* or unrelated specialization, not the canonical itself.
fn narrow_variant_body_match(
    body: &str,
    canonical_name: &str,
    aliases: &[String],
) -> Option<&'static str> {
    let trimmed = body.trim_start();
    // Only inspect the first sentence — the regression we are guarding
    // against is the opening clause, not deeper discussion later on.
    let first_sentence = trimmed
        .split_once(['.', '\n'])
        .map_or(trimmed, |(head, _)| head);

    for (name, pattern) in NARROW_VARIANT_BODY_PATTERNS.iter() {
        if pattern.is_match(first_sentence) {
            // Suppress when the canonical name itself is the one appearing in
            // the "X variant" slot — that case is not the bug we are catching.
            if *name == "narrow_variant_prefix"
                && body_narrow_noun_is_canonical(first_sentence, canonical_name, aliases)
            {
                continue;
            }
            return Some(name);
        }
    }
    None
}

/// True when the noun filling the "A <X> variant" slot is the canonical name
/// itself rather than one of the aliases. We only want to warn when the body
/// names a specific alias (e.g. "Cheap Paxos") in place of the canonical.
fn body_narrow_noun_is_canonical(
    first_sentence: &str,
    canonical_name: &str,
    aliases: &[String],
) -> bool {
    // Pull the noun that sits between the leading article and the word "variant".
    static NOUN_CAPTURE: LazyLock<Regex> = LazyLock::new(|| {
        Regex::new(r"(?i)^\s*an?\s+(?P<noun>\S.{0,30}?)\s+variant\b")
            .expect("noun capture regex")
    });
    let Some(caps) = NOUN_CAPTURE.captures(first_sentence) else {
        return false;
    };
    let noun = caps.name("noun").map_or("", |m| m.as_str()).trim();
    let matches_canonical = noun.eq_ignore_ascii_case(canonical_name);
    // If it matches an alias, it's the bug — return false so the warn fires.
    let matches_alias = aliases.iter().any(|a| noun.eq_ignore_ascii_case(a));
    matches_canonical && !matches_alias
}

/// Extract the first sentence of a concept body for inspection.
///
/// Stops at the first period, question mark, exclamation point, or newline.
/// The canonical/alias checks below are intentionally scoped to the first
/// sentence because that is where variant-scoping leaks in practice: the
/// subject of the opening clause tells the reader what concept the page is
/// actually about.
fn first_sentence(body: &str) -> &str {
    let trimmed = body.trim_start();
    trimmed
        .split_once(['.', '?', '!', '\n'])
        .map_or(trimmed, |(head, _)| head)
}

/// Post-check 1 — canonical-prefix check.
///
/// Returns `true` when the first sentence's opening tokens do NOT start with
/// the canonical concept name (case-insensitive, word-boundary aware). This
/// flags bodies like "A Raft safety property stating …" where the canonical
/// is "Raft" but the sentence is actually about Leader Completeness.
///
/// Leading articles ("A", "An", "The") are tolerated so that "The borrow
/// checker validates references" still counts as canonical-prefixed when the
/// canonical is "Borrow checker".
///
/// The canonical name must also be followed by a non-word byte (or the end of
/// the sentence) so that "Raft" does NOT match "Raft safety property …" —
/// the byte after "raft" there is a space followed by "safety property",
/// which is the variant-scoping bug. Contrast "Raft is a consensus
/// algorithm" where the byte after "raft" is " is ..." and the subject is
/// genuinely "Raft". We differentiate these by looking at whether the token
/// immediately after the canonical name is a linking verb ("is", "are",
/// "refers", "means", "represents", "denotes") or a punctuation/end-of-
/// sentence marker.
fn body_missing_canonical_prefix(body: &str, canonical_name: &str) -> bool {
    let sentence = first_sentence(body);
    let lower = sentence.to_ascii_lowercase();
    let canonical_lower = canonical_name.trim().to_ascii_lowercase();
    if canonical_lower.is_empty() {
        return false;
    }

    // Strip a single leading article so "The borrow checker ..." and
    // "A borrow checker ..." both count as canonical-prefixed.
    let stripped = ["the ", "a ", "an "]
        .iter()
        .find_map(|article| lower.strip_prefix(article))
        .unwrap_or(lower.as_str())
        .trim_start();

    if !stripped.starts_with(&canonical_lower) {
        return true;
    }

    // Canonical name is a prefix — but is the subject actually the canonical,
    // or is the canonical just part of a longer noun phrase like
    // "raft safety property"? Look at the next token.
    let after = stripped[canonical_lower.len()..].trim_start();
    if after.is_empty() {
        return false;
    }
    // Accept standard linking/declarative verbs that mean the sentence's
    // subject is the canonical itself. If the canonical is followed by one
    // of LINKING_VERBS, the first sentence is about the canonical.
    // Otherwise, the canonical appears as an adjective/modifier inside a
    // longer noun phrase (e.g. "Raft safety property …") and the subject
    // is NOT the canonical.
    if LINKING_VERBS.iter().any(|v| after.starts_with(v)) {
        return false;
    }
    // Explicit compound-noun markers: if the canonical is followed by one
    // of these, we know for sure it's a variant-scoping compound.
    if COMPOUND_NOUN_MARKERS.iter().any(|n| after.starts_with(n)) {
        return true;
    }
    // Otherwise be conservative: if the next token doesn't look like a
    // verb (via a simple inflection heuristic), flag it — the sentence's
    // subject is probably a longer noun phrase.
    !next_token_looks_like_verb(after)
}

/// Verbs/phrases that, when following the canonical name, indicate the
/// sentence's subject is the canonical itself (not a longer noun phrase).
const LINKING_VERBS: &[&str] = &[
    "is ",
    "are ",
    "was ",
    "were ",
    "refers ",
    "means ",
    "represents ",
    "denotes ",
    "describes ",
    "covers ",
    "provides ",
    "defines ",
    "is,",
    "refers,",
    "stands ",
    "validates ",
    "holds ",
    "ensures ",
    "enforces ",
    "prevents ",
    "returns ",
    "yields ",
    "checks ",
    "runs ",
    "performs ",
    "operates ",
    "executes ",
    "serves ",
    "handles ",
    "manages ",
    "maintains ",
    "applies ",
    "works ",
    "supports ",
    "requires ",
];

/// Common noun markers that, when they follow the canonical name, mean the
/// canonical is being used as an adjective inside a compound noun phrase —
/// i.e. the subject is not the canonical itself but a variant/sub-concept.
const COMPOUND_NOUN_MARKERS: &[&str] = &[
    "safety property",
    "liveness property",
    "variant",
    "instance",
    "procedure",
    "protocol",
    "algorithm",
    "component",
    "subcomponent",
    "subtype",
    "subclass",
    "phase",
    "step",
    "rule",
];

/// Keywords that, when present in a multi-alias concept body, signal the
/// body acknowledges its umbrella scope rather than scoping to one variant.
const VARIANT_SCOPING_KEYWORDS: &[&str] = &[
    "variant",
    "variants",
    "family",
    "include",
    "includes",
    "including",
];

/// Post-check 2 — variant-scoping coverage check.
///
/// For concepts with three or more aliases, the body must either mention one
/// of the family/variants keywords ("variant", "variants", "family",
/// "include", "includes", "including") OR name at least two of the aliases
/// verbatim. Returns `true` when neither condition holds — i.e. a multi-alias
/// concept's body failed to signal its umbrella scope.
fn body_missing_variant_scoping(body: &str, aliases: &[String]) -> bool {
    if aliases.len() < 3 {
        return false;
    }
    let lower = body.to_ascii_lowercase();
    if VARIANT_SCOPING_KEYWORDS
        .iter()
        .any(|k| contains_word(&lower, k))
    {
        return false;
    }
    let mentioned = aliases
        .iter()
        .filter(|a| contains_word(&lower, &a.to_ascii_lowercase()))
        .count();
    mentioned < 2
}

/// Post-check 3 — alias-in-first-sentence check.
///
/// Returns `Some(alias)` when the first sentence contains the name of a
/// non-canonical alias as a standalone word (case-insensitive, word
/// boundary). That strongly signals the body is about a specific variant
/// rather than the canonical concept — e.g. "Byzantine quorums ..." on a
/// page whose canonical is "Quorum" with aliases [Byzantine quorum, …].
///
/// Aliases that are substrings of the canonical name itself (e.g. canonical
/// "Paxos" with alias "Paxos" — shouldn't happen, but defensive) are
/// ignored so we don't double-count the canonical.
fn first_sentence_names_alias<'a>(
    body: &str,
    canonical_name: &str,
    aliases: &'a [String],
) -> Option<&'a str> {
    let sentence = first_sentence(body).to_ascii_lowercase();
    let canonical_lower = canonical_name.to_ascii_lowercase();
    for alias in aliases {
        let alias_lower = alias.to_ascii_lowercase();
        if alias_lower == canonical_lower {
            continue;
        }
        // Match the alias, or a trivial English plural of it ("quorum" →
        // "quorums"). Without the plural fallback we'd miss the Quorum
        // regression whose body reads "Byzantine quorums ..." against an
        // alias of "Byzantine quorum".
        let alias_plural = format!("{alias_lower}s");
        if contains_word(&sentence, &alias_lower)
            || contains_word(&sentence, &alias_plural)
        {
            return Some(alias.as_str());
        }
    }
    None
}

/// True when `haystack` contains `needle` delimited by non-word characters
/// (or string boundaries). Case-sensitive — callers lowercase both sides
/// before calling. Matching is simple and allocation-free: we scan byte
/// positions and inspect the char before/after for ASCII word-ness.
fn contains_word(haystack: &str, needle: &str) -> bool {
    if needle.is_empty() {
        return false;
    }
    let bytes = haystack.as_bytes();
    let nlen = needle.len();
    let mut start = 0usize;
    while let Some(rel) = haystack[start..].find(needle) {
        let pos = start + rel;
        let left_ok = pos == 0
            || !is_word_byte(bytes[pos - 1]);
        let right_end = pos + nlen;
        let right_ok = right_end == bytes.len()
            || !is_word_byte(bytes[right_end]);
        if left_ok && right_ok {
            return true;
        }
        start = pos + 1;
    }
    false
}

const fn is_word_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}

/// Heuristic: does the next whitespace-delimited token in `after` look like
/// a verb? We accept tokens ending in common verbal inflections ("s", "es",
/// "ed", "ing") when they're short enough to plausibly be a verb rather
/// than a multi-word noun phrase. Conservative — prefer false positives
/// here since the post-check is warn-only and the cost of a missed warning
/// is a rendering with no post-check review.
fn next_token_looks_like_verb(after: &str) -> bool {
    let token = after.split_whitespace().next().unwrap_or("").trim_end_matches(',');
    if token.is_empty() || token.len() > 14 {
        return false;
    }
    let lower = token.to_ascii_lowercase();
    lower.ends_with('s') || lower.ends_with("ed") || lower.ends_with("ing")
}

#[allow(clippy::too_many_lines)]
fn render_concept_page<S: std::hash::BuildHasher>(
    group: &MergeGroup,
    root: &Path,
    candidate_origins: &HashMap<String, Vec<String>, S>,
) -> Result<ConceptPage, ConceptMergeError> {
    use serde_yaml::{Mapping, Value};

    let slug = slug_from_title(&group.canonical_name);
    let concept_id = format!("concept:{slug}");
    let path = root.join(WIKI_CONCEPTS_DIR).join(format!("{slug}.md"));

    let source_anchors: Vec<&SourceAnchor> = group
        .members
        .iter()
        .flat_map(|m| m.source_anchors.iter())
        .collect();

    // Resolve contributing source_document_ids by looking each member's name up
    // in the origin side-map. Missing members (tests, or candidates without
    // recorded provenance) simply contribute nothing. Sorted + de-duplicated so
    // the frontmatter is deterministic across runs.
    let source_document_ids: Vec<String> = group
        .members
        .iter()
        .flat_map(|m| {
            candidate_origins
                .get(&m.name)
                .map_or(&[][..], Vec::as_slice)
        })
        .cloned()
        .collect::<BTreeSet<String>>()
        .into_iter()
        .collect();

    let definition_hint: Option<&str> = group
        .members
        .iter()
        .find_map(|m| m.definition_hint.as_deref());

    // Post-checks: if the merged body looks like it copied a narrow variant's
    // definition (e.g. "A Paxos variant ...") instead of synthesizing across
    // aliases, warn. Do NOT block the write — a wrong body is better than a
    // missing concept page, and a human can fix it via `kb review`.
    //
    // Four stacked checks; any of them can trip independently because the
    // observed pass-10 regressions slipped bn-2tw's single regex by using
    // different phrasings (safety-property-prefix, alias-subject, variant-
    // specific without the word "variant", etc.).
    if let Some(hint) = definition_hint {
        // 1. bn-2tw's existing narrow-variant regex (e.g. "A Cheap Paxos variant …").
        if let Some(pattern) =
            narrow_variant_body_match(hint, &group.canonical_name, &group.aliases)
        {
            tracing::warn!(
                concept = %group.canonical_name,
                matched_pattern = pattern,
                body_start = %hint.chars().take(80).collect::<String>(),
                "merged concept body looks like it describes a narrow variant, not the \
                 canonical concept; review via `kb review` if wrong"
            );
        }

        // 2. Canonical-prefix check. Catches "A Raft safety property stating …"
        //    on a page whose canonical is "Raft" — the subject is a folded
        //    alias (Leader Completeness), not the canonical itself.
        if body_missing_canonical_prefix(hint, &group.canonical_name) {
            tracing::warn!(
                concept = %group.canonical_name,
                matched_pattern = "canonical_prefix_missing",
                body_start = %hint.chars().take(80).collect::<String>(),
                "merged concept body's first sentence does not start with the canonical \
                 concept name; review via `kb review` if the scope is wrong"
            );
        }

        // 3. Variant-scoping coverage check. For concepts with ≥3 aliases, the
        //    body must mention "variant"/"family"/"includes" or name ≥2
        //    aliases. Catches a Paxos body that silently describes only Basic
        //    Paxos even though aliases span Basic/Multi/Cheap Paxos.
        if body_missing_variant_scoping(hint, &group.aliases) {
            tracing::warn!(
                concept = %group.canonical_name,
                matched_pattern = "variant_scoping_missing",
                alias_count = group.aliases.len(),
                body_start = %hint.chars().take(80).collect::<String>(),
                "merged concept body covers ≥3 aliases but does not mention \
                 'variant'/'family'/'includes' or name at least two aliases; \
                 review via `kb review` if the scope is wrong"
            );
        }

        // 4. Alias-in-first-sentence check. Catches "Byzantine quorums are sized
        //    to tolerate f Byzantine failures" on a page whose canonical is
        //    "Quorum" — the first sentence's subject is a non-canonical alias.
        if let Some(alias) =
            first_sentence_names_alias(hint, &group.canonical_name, &group.aliases)
        {
            tracing::warn!(
                concept = %group.canonical_name,
                matched_pattern = "first_sentence_names_alias",
                alias = alias,
                body_start = %hint.chars().take(80).collect::<String>(),
                "merged concept body's first sentence names a non-canonical alias; \
                 probable variant-scoping — review via `kb review` if the scope is wrong"
            );
        }
    }

    let mut fm = Mapping::new();
    fm.insert(Value::String("id".into()), Value::String(concept_id));
    fm.insert(
        Value::String("name".into()),
        Value::String(group.canonical_name.clone()),
    );

    if !group.aliases.is_empty() {
        let aliases: Vec<Value> = group
            .aliases
            .iter()
            .map(|a| Value::String(a.clone()))
            .collect();
        fm.insert(Value::String("aliases".into()), Value::Sequence(aliases));
    }

    if !source_document_ids.is_empty() {
        let ids: Vec<Value> = source_document_ids
            .iter()
            .map(|id| Value::String(id.clone()))
            .collect();
        fm.insert(
            Value::String("source_document_ids".into()),
            Value::Sequence(ids),
        );
    }

    if !source_anchors.is_empty() {
        let sources: Vec<Value> = source_anchors
            .iter()
            .map(|a| {
                let mut map = Mapping::new();
                if let Some(h) = &a.heading_anchor {
                    map.insert(
                        Value::String("heading_anchor".into()),
                        Value::String(h.clone()),
                    );
                }
                if let Some(q) = &a.quote {
                    map.insert(Value::String("quote".into()), Value::String(q.clone()));
                }
                Value::Mapping(map)
            })
            .collect();
        fm.insert(Value::String("sources".into()), Value::Sequence(sources));
    }

    let frontmatter_yaml = serde_yaml::to_string(&fm)
        .map_err(|e| ConceptMergeError::Serialize(e.to_string()))?;

    let mut content = format!(
        "---\n{frontmatter_yaml}---\n\n# {}\n",
        group.canonical_name
    );

    if let Some(hint) = definition_hint {
        content.push('\n');
        content.push_str(hint);
        content.push('\n');
    }

    Ok(ConceptPage {
        slug,
        path,
        content,
        canonical_name: group.canonical_name.clone(),
        aliases: group.aliases.clone(),
    })
}

fn render_review_record(
    group: &MergeGroup,
    _root: &Path,
    provenance: &ProvenanceRecord,
) -> Result<MergeReviewRecord, String> {
    let slug = slug_from_title(&group.canonical_name);
    let merge_id = format!("merge:{slug}");
    let destination_rel = PathBuf::from(WIKI_CONCEPTS_DIR).join(format!("{slug}.md"));

    // Hash the candidate members so save_review_item's dedup correctly skips
    // re-queuing the same merge after the user has rejected it.
    let member_fingerprint = hash_many(
        &group
            .members
            .iter()
            .map(|m| m.name.as_bytes())
            .collect::<Vec<_>>(),
    )
    .to_hex();

    let now = unix_time_ms().map_err(|err| format!("read system time: {err}"))?;

    let mut comment = String::new();
    writeln!(comment, "Proposed canonical: {}", group.canonical_name)
        .map_err(|err| format!("render review comment: {err}"))?;
    if !group.aliases.is_empty() {
        writeln!(comment, "Aliases: {}", group.aliases.join(", "))
            .map_err(|err| format!("render review comment: {err}"))?;
    }
    writeln!(
        comment,
        "Members: {}",
        group
            .members
            .iter()
            .map(|m| m.name.as_str())
            .collect::<Vec<_>>()
            .join(", ")
    )
    .map_err(|err| format!("render review comment: {err}"))?;
    if let Some(rationale) = &group.rationale {
        writeln!(comment, "Rationale: {rationale}")
            .map_err(|err| format!("render review comment: {err}"))?;
    }
    writeln!(
        comment,
        "Produced by {} via {}",
        provenance.model, provenance.prompt_template_name
    )
    .map_err(|err| format!("render review comment: {err}"))?;

    let item = ReviewItem {
        metadata: EntityMetadata {
            id: merge_id.clone(),
            created_at_millis: now,
            updated_at_millis: now,
            source_hashes: vec![member_fingerprint],
            model_version: Some(provenance.model.clone()),
            tool_version: Some(format!(
                "{}/{}",
                env!("CARGO_PKG_NAME"),
                env!("CARGO_PKG_VERSION")
            )),
            prompt_template_hash: Some(provenance.prompt_template_hash.to_hex()),
            dependencies: group.members.iter().map(|m| m.name.clone()).collect(),
            output_paths: vec![destination_rel.clone()],
            status: Status::NeedsReview,
        },
        kind: ReviewKind::ConceptMerge,
        target_entity_id: merge_id,
        proposed_destination: Some(destination_rel.clone()),
        citations: Vec::new(),
        affected_pages: vec![destination_rel],
        created_at_millis: now,
        status: ReviewStatus::Pending,
        comment: comment.trim_end().to_string(),
    };

    Ok(MergeReviewRecord { item })
}

fn build_record_for_merge(
    candidates: &[ConceptCandidate],
    response: &MergeConceptCandidatesResponse,
    concept_pages: &[ConceptPage],
    review_records: &[MergeReviewRecord],
    provenance: &ProvenanceRecord,
) -> Result<BuildRecord, ConceptMergeError> {
    let now = unix_time_ms()?;
    let prompt_template_hash = provenance.prompt_template_hash.to_hex();
    let prompt_render_hash = provenance.prompt_render_hash.to_hex();

    let candidate_names_joined: String = candidates
        .iter()
        .map(|c| c.name.as_str())
        .collect::<Vec<_>>()
        .join("\n");

    let group_count = response.groups.len().to_string();
    let confident_count = response
        .groups
        .iter()
        .filter(|g| g.confident)
        .count()
        .to_string();

    let manifest_hash = hash_many(&[
        candidate_names_joined.as_bytes(),
        b"\0",
        group_count.as_bytes(),
        b"\0",
        confident_count.as_bytes(),
        b"\0",
        prompt_render_hash.as_bytes(),
    ])
    .to_hex();

    let input_ids: Vec<String> = candidates.iter().map(|c| c.name.clone()).collect();

    // Review records are addressed by their review id (`merge:<slug>`) rather than
    // a filesystem path since `save_review_item` picks the concrete directory.
    let output_ids: Vec<String> = concept_pages
        .iter()
        .map(|p| p.path.display().to_string())
        .chain(review_records.iter().map(|r| r.item.metadata.id.clone()))
        .collect();

    let output_paths: Vec<PathBuf> = concept_pages
        .iter()
        .map(|p| p.path.clone())
        .chain(
            review_records
                .iter()
                .filter_map(|r| r.item.proposed_destination.clone()),
        )
        .collect();

    Ok(BuildRecord {
        pass_name: "merge_concept_candidates".to_string(),
        metadata: EntityMetadata {
            id: "build:merge-concept-candidates".to_string(),
            created_at_millis: now,
            updated_at_millis: now,
            source_hashes: vec![candidate_names_joined, prompt_render_hash],
            model_version: Some(provenance.model.clone()),
            tool_version: Some(format!(
                "{}/{}",
                env!("CARGO_PKG_NAME"),
                env!("CARGO_PKG_VERSION")
            )),
            prompt_template_hash: Some(prompt_template_hash),
            dependencies: input_ids.clone(),
            output_paths,
            status: Status::Fresh,
        },
        input_ids,
        output_ids,
        manifest_hash,
    })
}

/// Summary of an applied concept merge.
///
/// Returned by [`apply_concept_merge`] so callers (the CLI `kb review approve`
/// path) can render the same summary on first apply and on re-apply of an
/// already-approved item.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AppliedMerge {
    /// Canonical concept page path (relative to the kb root).
    pub canonical_path: PathBuf,
    /// Canonical page existed before apply (true on first apply, true on re-apply).
    pub canonical_existed: bool,
    /// Paths of subsumed member concept files that were (or already were) removed.
    pub removed_members: Vec<PathBuf>,
    /// Aliases merged into the canonical page on this call.
    pub added_aliases: Vec<String>,
    /// Source document IDs merged into the canonical page on this call.
    pub added_source_document_ids: Vec<String>,
    /// True when the canonical page was updated in place; false on an idempotent
    /// re-apply where nothing changed.
    pub canonical_updated: bool,
}

/// Apply a `concept_merge` review item: fold subsumed concept pages into the
/// canonical page and delete them.
///
/// The review item is expected to carry:
/// - `proposed_destination` = path (relative to `root`) to the canonical concept page.
/// - `metadata.dependencies` = member candidate names (the canonical candidate
///   plus the ones being subsumed). Each name is slugged to resolve its
///   `wiki/concepts/<slug>.md` path.
///
/// On apply:
/// 1. Each subsumed member file's frontmatter is read; its `aliases`,
///    `source_document_ids`, and `sources` entries are unioned into the
///    canonical page. The member's own name is also added as an alias on the
///    canonical page.
/// 2. Each subsumed member file is deleted.
/// 3. The canonical page's frontmatter is rewritten with the merged fields
///    (sorted + de-duplicated for deterministic output).
///
/// Idempotency: calling apply a second time (after all members have already
/// been removed) is a no-op — it recomputes the same summary, detects nothing
/// to merge, and does not error.
///
/// # Errors
///
/// Returns an error when the review item has no `proposed_destination`, is not
/// a `concept_merge`, or when the canonical page is missing on disk.
pub fn apply_concept_merge(root: &Path, item: &ReviewItem) -> Result<AppliedMerge> {
    if item.kind != ReviewKind::ConceptMerge {
        bail!(
            "apply_concept_merge called on a {:?} review item",
            item.kind
        );
    }

    let canonical_rel = item
        .proposed_destination
        .as_ref()
        .context("review item has no proposed_destination")?
        .clone();
    let canonical_path = root.join(&canonical_rel);

    if !canonical_path.exists() {
        bail!(
            "canonical concept page missing at {}; cannot apply merge '{}'",
            canonical_rel.display(),
            item.metadata.id
        );
    }

    let canonical_slug = canonical_rel
        .file_stem()
        .and_then(|s| s.to_str())
        .map_or_else(String::new, ToString::to_string);

    let (mut fm, body) = read_frontmatter(&canonical_path).with_context(|| {
        format!("read canonical concept page {}", canonical_path.display())
    })?;

    let mut existing = CanonicalFields::read(&fm);
    let mut additions = MergeAdditions::default();
    let mut removed_members: Vec<PathBuf> = Vec::new();

    for member_name in &item.metadata.dependencies {
        let member_slug = slug_from_title(member_name);
        if member_slug == canonical_slug {
            // The canonical candidate shares a file with the canonical page; don't
            // delete it or add its own name as an alias.
            continue;
        }

        let member_rel = PathBuf::from(WIKI_CONCEPTS_DIR).join(format!("{member_slug}.md"));
        let member_path = root.join(&member_rel);

        // Always record the member's name as an alias on the canonical page, even
        // if the member file is already gone (idempotent re-apply still sets it).
        additions.add_alias(&existing.aliases, member_name);

        if !member_path.exists() {
            continue;
        }

        absorb_member_frontmatter(&member_path, &existing, &mut additions)?;

        std::fs::remove_file(&member_path)
            .with_context(|| format!("delete subsumed concept page {}", member_path.display()))?;
        removed_members.push(member_rel);
    }

    let added_aliases: Vec<String> = additions.aliases.into_iter().collect();
    let added_source_document_ids: Vec<String> = additions.source_ids.into_iter().collect();
    let canonical_updated = !added_aliases.is_empty()
        || !added_source_document_ids.is_empty()
        || !additions.sources.is_empty();

    if canonical_updated {
        existing.union_additions(&added_aliases, &added_source_document_ids, additions.sources);
        existing.write_into(&mut fm);
        write_frontmatter(&canonical_path, &fm, body.as_str())
            .with_context(|| format!("rewrite canonical concept page {}", canonical_path.display()))?;
    }

    Ok(AppliedMerge {
        canonical_path: canonical_rel,
        canonical_existed: true,
        removed_members,
        added_aliases,
        added_source_document_ids,
        canonical_updated,
    })
}

#[derive(Default)]
struct CanonicalFields {
    aliases: Vec<String>,
    source_ids: Vec<String>,
    sources: Vec<Value>,
}

impl CanonicalFields {
    fn read(fm: &Mapping) -> Self {
        Self {
            aliases: string_seq(fm, "aliases"),
            source_ids: string_seq(fm, "source_document_ids"),
            sources: fm
                .get(Value::String("sources".into()))
                .and_then(|v| v.as_sequence())
                .cloned()
                .unwrap_or_default(),
        }
    }

    fn union_additions(
        &mut self,
        added_aliases: &[String],
        added_source_ids: &[String],
        added_sources: Vec<Value>,
    ) {
        self.aliases.extend(added_aliases.iter().cloned());
        self.aliases.sort();
        self.aliases.dedup();

        self.source_ids.extend(added_source_ids.iter().cloned());
        self.source_ids.sort();
        self.source_ids.dedup();

        self.sources.extend(added_sources);
    }

    fn write_into(&self, fm: &mut Mapping) {
        if !self.aliases.is_empty() {
            fm.insert(
                Value::String("aliases".into()),
                Value::Sequence(self.aliases.iter().map(|s| Value::String(s.clone())).collect()),
            );
        }
        if !self.source_ids.is_empty() {
            fm.insert(
                Value::String("source_document_ids".into()),
                Value::Sequence(
                    self.source_ids
                        .iter()
                        .map(|s| Value::String(s.clone()))
                        .collect(),
                ),
            );
        }
        if !self.sources.is_empty() {
            fm.insert(
                Value::String("sources".into()),
                Value::Sequence(self.sources.clone()),
            );
        }
    }
}

#[derive(Default)]
struct MergeAdditions {
    aliases: BTreeSet<String>,
    source_ids: BTreeSet<String>,
    sources: Vec<Value>,
}

impl MergeAdditions {
    fn add_alias(&mut self, existing: &[String], alias: &str) {
        if !existing.iter().any(|a| a == alias) {
            self.aliases.insert(alias.to_string());
        }
    }

    fn add_source_id(&mut self, existing: &[String], id: &str) {
        if !existing.iter().any(|a| a == id) {
            self.source_ids.insert(id.to_string());
        }
    }

    fn add_source_entry(&mut self, existing: &[Value], entry: &Value) {
        if !existing.iter().any(|e| e == entry) && !self.sources.iter().any(|e| e == entry) {
            self.sources.push(entry.clone());
        }
    }
}

fn absorb_member_frontmatter(
    member_path: &Path,
    existing: &CanonicalFields,
    additions: &mut MergeAdditions,
) -> Result<()> {
    let (member_fm, _body) = read_frontmatter(member_path)
        .with_context(|| format!("read member concept page {}", member_path.display()))?;

    for alias in string_seq(&member_fm, "aliases") {
        additions.add_alias(&existing.aliases, &alias);
    }
    for id in string_seq(&member_fm, "source_document_ids") {
        additions.add_source_id(&existing.source_ids, &id);
    }
    if let Some(seq) = member_fm
        .get(Value::String("sources".into()))
        .and_then(|v| v.as_sequence())
    {
        for entry in seq {
            additions.add_source_entry(&existing.sources, entry);
        }
    }
    Ok(())
}

fn string_seq(fm: &Mapping, key: &str) -> Vec<String> {
    fm.get(Value::String(key.into()))
        .and_then(|v| v.as_sequence())
        .map(|seq| {
            seq.iter()
                .filter_map(|v| v.as_str().map(ToString::to_string))
                .collect()
        })
        .unwrap_or_default()
}

fn unix_time_ms() -> Result<u64, ConceptMergeError> {
    let duration = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|err| ConceptMergeError::Time(err.to_string()))?;
    let millis = duration
        .as_millis()
        .try_into()
        .context("system time exceeded u64 millisecond range")
        .map_err(|err| ConceptMergeError::Time(err.to_string()))?;
    Ok(millis)
}

#[cfg(test)]
mod tests {
    use super::*;

    use kb_core::Hash;
    use kb_llm::{
        AnswerQuestionRequest, AnswerQuestionResponse, ExtractConceptsRequest,
        ExtractConceptsResponse, GenerateSlidesRequest, GenerateSlidesResponse, LlmAdapter,
        LlmAdapterError, MergeConceptCandidatesResponse, RunHealthCheckRequest,
        RunHealthCheckResponse, SourceAnchor, SummarizeDocumentRequest, SummarizeDocumentResponse,
        TokenUsage,
    };
    use tempfile::tempdir;

    #[derive(Debug)]
    struct FakeAdapter {
        response: MergeConceptCandidatesResponse,
        provenance: ProvenanceRecord,
    }

    impl LlmAdapter for FakeAdapter {
        fn summarize_document(
            &self,
            _request: SummarizeDocumentRequest,
        ) -> Result<(SummarizeDocumentResponse, ProvenanceRecord), LlmAdapterError> {
            unreachable!("unused in concept merge test")
        }

        fn extract_concepts(
            &self,
            _request: ExtractConceptsRequest,
        ) -> Result<(ExtractConceptsResponse, ProvenanceRecord), LlmAdapterError> {
            unreachable!("unused in concept merge test")
        }

        fn merge_concept_candidates(
            &self,
            _request: MergeConceptCandidatesRequest,
        ) -> Result<(MergeConceptCandidatesResponse, ProvenanceRecord), LlmAdapterError> {
            Ok((self.response.clone(), self.provenance.clone()))
        }

        fn answer_question(
            &self,
            _request: AnswerQuestionRequest,
        ) -> Result<(AnswerQuestionResponse, ProvenanceRecord), LlmAdapterError> {
            unreachable!("unused in concept merge test")
        }

        fn generate_slides(
            &self,
            _request: GenerateSlidesRequest,
        ) -> Result<(GenerateSlidesResponse, ProvenanceRecord), LlmAdapterError> {
            unreachable!("unused in concept merge test")
        }

        fn run_health_check(
            &self,
            _request: RunHealthCheckRequest,
        ) -> Result<(RunHealthCheckResponse, ProvenanceRecord), LlmAdapterError> {
            unreachable!("unused in concept merge test")
        }
    }

    fn provenance() -> ProvenanceRecord {
        ProvenanceRecord {
            harness: "opencode".to_string(),
            harness_version: None,
            model: "openai/gpt-5.4".to_string(),
            prompt_template_name: "merge_concept_candidates.md".to_string(),
            prompt_template_hash: Hash::from([9u8; 32]),
            prompt_render_hash: Hash::from([10u8; 32]),
            started_at: 10,
            ended_at: 20,
            latency_ms: 10,
            retries: 0,
            tokens: Some(TokenUsage {
                prompt_tokens: 100,
                completion_tokens: 50,
            }),
            cost_estimate: None,
        }
    }

    fn borrow_checker_candidate() -> ConceptCandidate {
        ConceptCandidate {
            name: "borrow checker".to_string(),
            aliases: vec!["borrowck".to_string()],
            definition_hint: Some("Validates references at compile time.".to_string()),
            source_anchors: vec![SourceAnchor {
                heading_anchor: Some("ownership".to_string()),
                quote: Some("The borrow checker validates references.".to_string()),
            }],
        }
    }

    fn borrowck_candidate() -> ConceptCandidate {
        ConceptCandidate {
            name: "borrowck".to_string(),
            aliases: vec![],
            definition_hint: None,
            source_anchors: vec![SourceAnchor {
                heading_anchor: None,
                quote: Some("borrowck prevents dangling pointers.".to_string()),
            }],
        }
    }

    fn uncertain_candidate() -> ConceptCandidate {
        ConceptCandidate {
            name: "lifetime elision".to_string(),
            aliases: vec![],
            definition_hint: Some("Rules for inferring lifetimes.".to_string()),
            source_anchors: vec![],
        }
    }

    #[test]
    fn confident_groups_become_concept_pages() {
        let dir = tempdir().expect("tempdir");
        let adapter = FakeAdapter {
            response: MergeConceptCandidatesResponse {
                groups: vec![MergeGroup {
                    canonical_name: "Borrow checker".to_string(),
                    aliases: vec!["borrowck".to_string()],
                    members: vec![borrow_checker_candidate(), borrowck_candidate()],
                    confident: true,
                    rationale: None,
                }],
            },
            provenance: provenance(),
        };

        let candidates = vec![borrow_checker_candidate(), borrowck_candidate()];
        let artifact =
            run_concept_merge_pass(&adapter, candidates, dir.path()).expect("run merge pass");

        assert_eq!(artifact.concept_pages.len(), 1);
        assert!(artifact.review_records.is_empty());

        let page = &artifact.concept_pages[0];
        assert_eq!(page.slug, "borrow-checker");
        assert_eq!(page.canonical_name, "Borrow checker");
        assert_eq!(page.aliases, vec!["borrowck"]);
        assert!(page.content.contains("id: concept:borrow-checker"));
        assert!(page.content.contains("name: Borrow checker"));
        assert!(page.content.contains("aliases:"));
        assert!(page.content.contains("- borrowck"));
        assert!(page.content.contains("# Borrow checker"));
        assert!(
            page.content
                .contains("Validates references at compile time.")
        );
        assert_eq!(
            page.path,
            dir.path().join("wiki/concepts/borrow-checker.md")
        );
    }

    #[test]
    fn uncertain_groups_become_review_records() {
        let dir = tempdir().expect("tempdir");
        let adapter = FakeAdapter {
            response: MergeConceptCandidatesResponse {
                groups: vec![MergeGroup {
                    canonical_name: "Lifetime elision".to_string(),
                    aliases: vec![],
                    members: vec![uncertain_candidate()],
                    confident: false,
                    rationale: Some(
                        "Ambiguous whether this is the same as lifetime rules.".to_string(),
                    ),
                }],
            },
            provenance: provenance(),
        };

        let artifact = run_concept_merge_pass(&adapter, vec![uncertain_candidate()], dir.path())
            .expect("run merge pass");

        assert!(artifact.concept_pages.is_empty());
        assert_eq!(artifact.review_records.len(), 1);

        let review = &artifact.review_records[0];
        assert_eq!(review.merge_id(), "merge:lifetime-elision");
        assert_eq!(review.item.kind, kb_core::ReviewKind::ConceptMerge);
        assert_eq!(review.item.status, kb_core::ReviewStatus::Pending);
        assert_eq!(
            review.item.proposed_destination.as_deref(),
            Some(std::path::Path::new("wiki/concepts/lifetime-elision.md"))
        );
        assert!(review.item.comment.contains("Lifetime elision"));
        assert!(review.item.comment.contains("Ambiguous"));

        // Verify the persisted record can round-trip through `load_review_item`,
        // which is how `kb review show/approve` will consume it.
        persist_merge_review(dir.path(), review).expect("persist");
        let loaded = kb_core::load_review_item(dir.path(), review.merge_id())
            .expect("load")
            .expect("item present");
        assert_eq!(loaded.metadata.id, review.merge_id());
        assert_eq!(loaded.kind, kb_core::ReviewKind::ConceptMerge);
    }

    #[test]
    fn candidates_sorted_before_sending_to_adapter() {
        let dir = tempdir().expect("tempdir");
        let adapter = FakeAdapter {
            response: MergeConceptCandidatesResponse { groups: vec![] },
            provenance: provenance(),
        };

        let candidates = vec![borrowck_candidate(), borrow_checker_candidate()];
        let artifact =
            run_concept_merge_pass(&adapter, candidates, dir.path()).expect("run merge pass");

        assert!(artifact.concept_pages.is_empty());
        assert!(artifact.review_records.is_empty());
        assert_eq!(artifact.build_record.pass_name, "merge_concept_candidates");
    }

    #[test]
    fn build_record_lists_concept_page_paths_as_outputs() {
        let dir = tempdir().expect("tempdir");
        let adapter = FakeAdapter {
            response: MergeConceptCandidatesResponse {
                groups: vec![
                    MergeGroup {
                        canonical_name: "Borrow checker".to_string(),
                        aliases: vec!["borrowck".to_string()],
                        members: vec![borrow_checker_candidate()],
                        confident: true,
                        rationale: None,
                    },
                    MergeGroup {
                        canonical_name: "Lifetime elision".to_string(),
                        aliases: vec![],
                        members: vec![uncertain_candidate()],
                        confident: false,
                        rationale: Some("Uncertain".to_string()),
                    },
                ],
            },
            provenance: provenance(),
        };

        let candidates = vec![borrow_checker_candidate(), uncertain_candidate()];
        let artifact =
            run_concept_merge_pass(&adapter, candidates, dir.path()).expect("run merge pass");

        let output_ids = &artifact.build_record.output_ids;
        assert!(
            output_ids.iter().any(|id| id.contains("borrow-checker.md")),
            "output_ids: {output_ids:?}"
        );
        // Uncertain merges now appear in output_ids by their review id, not
        // a filesystem path — `save_review_item` picks the concrete location.
        assert!(
            output_ids
                .iter()
                .any(|id| id == "merge:lifetime-elision"),
            "output_ids: {output_ids:?}"
        );
    }

    #[test]
    fn persist_concept_page_writes_file() {
        let dir = tempdir().expect("tempdir");
        std::fs::create_dir_all(dir.path().join("wiki/concepts")).expect("create dir");

        let page = ConceptPage {
            slug: "borrow-checker".to_string(),
            path: dir.path().join("wiki/concepts/borrow-checker.md"),
            content: "---\nid: concept:borrow-checker\n---\n".to_string(),
            canonical_name: "Borrow checker".to_string(),
            aliases: vec![],
        };

        persist_concept_page(&page).expect("persist concept page");
        let written = std::fs::read_to_string(&page.path).expect("read file");
        assert_eq!(written, page.content);
    }

    #[test]
    fn yaml_sensitive_aliases_produce_valid_frontmatter() {
        let dir = tempdir().expect("tempdir");
        let tricky_aliases = vec![
            "'a".to_string(),
            "'static".to_string(),
            "foo: bar".to_string(),
            "hash#tag".to_string(),
            "line\nwith\nnewlines".to_string(),
        ];
        let adapter = FakeAdapter {
            response: MergeConceptCandidatesResponse {
                groups: vec![MergeGroup {
                    canonical_name: "Named lifetime parameter".to_string(),
                    aliases: tricky_aliases.clone(),
                    members: vec![borrow_checker_candidate()],
                    confident: true,
                    rationale: None,
                }],
            },
            provenance: provenance(),
        };

        let artifact =
            run_concept_merge_pass(&adapter, vec![borrow_checker_candidate()], dir.path())
                .expect("run merge pass");

        assert_eq!(artifact.concept_pages.len(), 1);
        let page = &artifact.concept_pages[0];

        // Extract the frontmatter between the `---` fences and verify it parses.
        let body = page
            .content
            .strip_prefix("---\n")
            .expect("page starts with frontmatter fence");
        let end = body.find("\n---\n").expect("closing frontmatter fence");
        let frontmatter = &body[..=end];

        let parsed: serde_yaml::Mapping =
            serde_yaml::from_str(frontmatter).expect("frontmatter parses as YAML mapping");

        // Round-trip: every alias we put in must come back out exactly.
        let aliases = parsed
            .get(serde_yaml::Value::String("aliases".into()))
            .and_then(|v| v.as_sequence())
            .expect("aliases sequence present");
        let recovered: Vec<String> = aliases
            .iter()
            .map(|v| v.as_str().expect("alias is a string").to_string())
            .collect();
        assert_eq!(recovered, tricky_aliases);

        // id / name round-trip too.
        assert_eq!(
            parsed
                .get(serde_yaml::Value::String("id".into()))
                .and_then(|v| v.as_str()),
            Some("concept:named-lifetime-parameter")
        );
        assert_eq!(
            parsed
                .get(serde_yaml::Value::String("name".into()))
                .and_then(|v| v.as_str()),
            Some("Named lifetime parameter")
        );
    }

    #[test]
    fn confident_pages_carry_source_document_ids_from_origins() {
        use std::collections::HashMap;

        let dir = tempdir().expect("tempdir");
        let adapter = FakeAdapter {
            response: MergeConceptCandidatesResponse {
                groups: vec![MergeGroup {
                    canonical_name: "Borrow checker".to_string(),
                    aliases: vec!["borrowck".to_string()],
                    members: vec![borrow_checker_candidate(), borrowck_candidate()],
                    confident: true,
                    rationale: None,
                }],
            },
            provenance: provenance(),
        };

        let mut origins: HashMap<String, Vec<String>> = HashMap::new();
        origins.insert("borrow checker".to_string(), vec!["src-aaa".to_string()]);
        origins.insert("borrowck".to_string(), vec!["src-bbb".to_string()]);

        let candidates = vec![borrow_checker_candidate(), borrowck_candidate()];
        let artifact = run_concept_merge_pass_with_origins(
            &adapter,
            candidates,
            dir.path(),
            &origins,
        )
        .expect("run merge pass with origins");

        assert_eq!(artifact.concept_pages.len(), 1);
        let page = &artifact.concept_pages[0];

        // Frontmatter must contain a deterministic, sorted, de-duplicated
        // `source_document_ids:` list covering both contributing sources so
        // `kb lint --check missing-citations` accepts the page as grounded.
        assert!(
            page.content.contains("source_document_ids:"),
            "missing source_document_ids field; got:\n{}",
            page.content
        );

        let body = page
            .content
            .strip_prefix("---\n")
            .expect("starts with frontmatter fence");
        let end = body.find("\n---\n").expect("closing fence");
        let frontmatter = &body[..=end];

        let parsed: serde_yaml::Mapping =
            serde_yaml::from_str(frontmatter).expect("valid yaml");
        let ids = parsed
            .get(serde_yaml::Value::String("source_document_ids".into()))
            .and_then(|v| v.as_sequence())
            .expect("source_document_ids sequence present");
        let recovered: Vec<String> = ids
            .iter()
            .map(|v| v.as_str().expect("id is string").to_string())
            .collect();
        assert_eq!(recovered, vec!["src-aaa".to_string(), "src-bbb".to_string()]);
    }

    #[test]
    fn concept_page_omits_source_document_ids_when_origins_empty() {
        use std::collections::HashMap;

        // Default run_concept_merge_pass (no origin provenance) must NOT emit an
        // empty `source_document_ids: []` — we only add the field when we have
        // real source ids. Preserves the existing frontmatter shape for tests
        // and callers that don't thread origin data.
        let dir = tempdir().expect("tempdir");
        let adapter = FakeAdapter {
            response: MergeConceptCandidatesResponse {
                groups: vec![MergeGroup {
                    canonical_name: "Borrow checker".to_string(),
                    aliases: vec![],
                    members: vec![borrow_checker_candidate()],
                    confident: true,
                    rationale: None,
                }],
            },
            provenance: provenance(),
        };

        let origins: HashMap<String, Vec<String>> = HashMap::new();
        let artifact = run_concept_merge_pass_with_origins(
            &adapter,
            vec![borrow_checker_candidate()],
            dir.path(),
            &origins,
        )
        .expect("run merge pass");

        let page = &artifact.concept_pages[0];
        assert!(
            !page.content.contains("source_document_ids"),
            "unexpected source_document_ids with empty origins:\n{}",
            page.content
        );
    }

    fn paxos_candidate(name: &str, hint: &str) -> ConceptCandidate {
        ConceptCandidate {
            name: name.to_string(),
            aliases: vec![],
            definition_hint: Some(hint.to_string()),
            source_anchors: vec![SourceAnchor {
                heading_anchor: None,
                quote: Some(format!("{name}: {hint}")),
            }],
        }
    }

    #[test]
    fn narrow_variant_body_fires_on_paxos_regression() {
        // Reproduces the pass-9 regression: merge output quotes Cheap Paxos's
        // one-liner as the body for the canonical "Paxos" concept. The
        // post-check must flag it via the narrow-variant pattern.
        let hit = narrow_variant_body_match(
            "A Paxos variant that keeps auxiliary nodes out of the steady state \
             and activates them only during failures.",
            "Paxos",
            &[
                "Basic Paxos".to_string(),
                "Multi-Paxos".to_string(),
                "Cheap Paxos".to_string(),
            ],
        );
        // "A Paxos variant …" with canonical == "Paxos" is the noun-is-canonical
        // false-positive suppression case — so it must NOT fire for the first
        // pattern… unless the canonical literally IS a more specific alias.
        // For the real regression, the body copied from Cheap Paxos reads "A
        // Paxos variant …" — the noun is "Paxos", which equals the canonical
        // name, so the narrow_variant_prefix pattern is suppressed. That's the
        // right behaviour; we catch it via the alternate phrasings below.
        assert!(hit.is_none(), "noun-is-canonical suppression should hold");

        // But the observed bug is more typically phrased with the alias name
        // in the slot, e.g. "A Cheap Paxos variant …" — those must fire.
        let hit = narrow_variant_body_match(
            "A Cheap Paxos variant that keeps auxiliary nodes out of the steady state.",
            "Paxos",
            &[
                "Basic Paxos".to_string(),
                "Multi-Paxos".to_string(),
                "Cheap Paxos".to_string(),
            ],
        );
        assert_eq!(hit, Some("narrow_variant_prefix"));

        // "A specialization of X" phrasing also fires.
        let hit = narrow_variant_body_match(
            "A specialization of Paxos that trades liveness for cheaper quorums.",
            "Paxos",
            &[],
        );
        assert_eq!(hit, Some("specialization_prefix"));
    }

    #[test]
    fn narrow_variant_body_allows_general_definition() {
        // The ✓-correct body from the prompt's worked example must not trip
        // the post-check.
        let hit = narrow_variant_body_match(
            "A family of consensus algorithms for replicated logs, with variants \
             including Basic Paxos, Multi-Paxos, and Cheap Paxos.",
            "Paxos",
            &["Basic Paxos".to_string(), "Multi-Paxos".to_string()],
        );
        assert!(hit.is_none(), "general-family body must not be flagged");

        // A neutral single-variant concept page (no aliases, canonical in the
        // slot) is fine.
        let hit = narrow_variant_body_match(
            "A rule-based system for validating references.",
            "Borrow checker",
            &[],
        );
        assert!(hit.is_none());
    }

    #[test]
    fn paxos_family_merge_body_is_general_not_cheap_paxos_specific() {
        // Golden-corpus-style test: three Paxos variants fold into a single
        // "Paxos" concept, and the stubbed merge adapter returns a canonical
        // member whose `definition_hint` is the correct (most-general) body.
        //
        // The assertion captures the bug-contract: the merged body's first
        // phrase must NOT be "A Paxos variant" — that wording describes Cheap
        // Paxos, not Paxos. We test the *contract the merge pipeline emits*,
        // so when the post-check or future prompt tuning regresses, this
        // test catches the wrong body shape.
        let dir = tempdir().expect("tempdir");

        let basic = paxos_candidate(
            "Basic Paxos",
            "A two-phase consensus protocol for agreeing on a single value.",
        );
        let multi = paxos_candidate(
            "Multi-Paxos",
            "An optimization of Paxos that amortizes leader election across a log of values.",
        );
        let cheap = paxos_candidate(
            "Cheap Paxos",
            "A Paxos variant that keeps auxiliary nodes out of the steady state \
             and activates them only during failures.",
        );

        // The canonical "Paxos" member carries the correct, general body.
        // This is what the revised prompt is supposed to produce.
        let canonical_member = ConceptCandidate {
            name: "Paxos".to_string(),
            aliases: vec![
                "Basic Paxos".to_string(),
                "Multi-Paxos".to_string(),
                "Cheap Paxos".to_string(),
            ],
            definition_hint: Some(
                "A family of consensus algorithms for replicated logs, with variants \
                 including Basic Paxos, Multi-Paxos, and Cheap Paxos."
                    .to_string(),
            ),
            source_anchors: vec![],
        };

        let adapter = FakeAdapter {
            response: MergeConceptCandidatesResponse {
                groups: vec![MergeGroup {
                    canonical_name: "Paxos".to_string(),
                    aliases: vec![
                        "Basic Paxos".to_string(),
                        "Multi-Paxos".to_string(),
                        "Cheap Paxos".to_string(),
                    ],
                    members: vec![
                        canonical_member,
                        basic.clone(),
                        multi.clone(),
                        cheap.clone(),
                    ],
                    confident: true,
                    rationale: None,
                }],
            },
            provenance: provenance(),
        };

        let candidates = vec![basic, multi, cheap];
        let artifact =
            run_concept_merge_pass(&adapter, candidates, dir.path()).expect("run merge pass");

        assert_eq!(artifact.concept_pages.len(), 1);
        let page = &artifact.concept_pages[0];
        assert_eq!(page.canonical_name, "Paxos");

        // Contract: the page body's first phrase must NOT be "A Paxos
        // variant" — that's Cheap Paxos, not Paxos.
        let body_after_heading = page
            .content
            .split_once("# Paxos\n")
            .map(|(_, rest)| rest.trim_start())
            .expect("rendered page has heading");
        assert!(
            !body_after_heading.starts_with("A Paxos variant"),
            "body leads with narrow-variant phrasing:\n{body_after_heading}"
        );
        assert!(
            !body_after_heading.starts_with("A Cheap Paxos variant"),
            "body leads with alias-specific phrasing:\n{body_after_heading}"
        );
        // And it must contain the family framing.
        assert!(
            body_after_heading.contains("family of consensus algorithms"),
            "body missing general definition:\n{body_after_heading}"
        );
    }

    #[test]
    fn canonical_prefix_check_flags_raft_leader_completeness_body() {
        // Observed pass-10 failure: wiki/concepts/raft.md body was "A Raft
        // safety property stating that any entry committed in a term must
        // appear in the logs of all leaders elected in later terms." — that
        // describes Leader Completeness, not Raft. The body starts with
        // "A Raft safety property …", which means the subject after the
        // leading article is "Raft safety property", not "Raft" alone, so
        // the canonical-prefix check must flag it.
        assert!(body_missing_canonical_prefix(
            "A Raft safety property stating that any entry committed in a term \
             must appear in the logs of all leaders elected in later terms.",
            "Raft",
        ));
        // A body that actually opens with the canonical name passes.
        assert!(!body_missing_canonical_prefix(
            "Raft is a consensus algorithm for managing a replicated log.",
            "Raft",
        ));
        // Leading article before the canonical is tolerated.
        assert!(!body_missing_canonical_prefix(
            "The borrow checker validates references at compile time.",
            "Borrow checker",
        ));
    }

    #[test]
    fn variant_scoping_check_flags_paxos_basic_only_body() {
        // Observed pass-10 failure: wiki/concepts/paxos.md body described
        // Basic Paxos's single-value agreement even though aliases included
        // Multi-Paxos, Cheap Paxos, EPaxos, Flexible Paxos. No "variant" /
        // "family" / "includes" keyword, and only one alias (Basic Paxos) is
        // named, so the variant-scoping check must flag it.
        let aliases = vec![
            "Basic Paxos".to_string(),
            "Multi-Paxos".to_string(),
            "Cheap Paxos".to_string(),
            "EPaxos".to_string(),
            "Flexible Paxos".to_string(),
        ];
        assert!(body_missing_variant_scoping(
            "Paxos is a two-phase consensus protocol where proposers and \
             acceptors agree on a single value using a Basic Paxos exchange.",
            &aliases,
        ));
        // Mentioning ≥ 2 aliases (without the keyword) is enough to pass.
        assert!(!body_missing_variant_scoping(
            "Paxos covers Basic Paxos and Multi-Paxos under a single umbrella \
             name.",
            &aliases,
        ));
        // The keyword alone is enough to pass.
        assert!(!body_missing_variant_scoping(
            "Paxos is a family of consensus algorithms.",
            &aliases,
        ));
        // Fewer than 3 aliases — the check is suppressed regardless.
        assert!(!body_missing_variant_scoping(
            "Paxos is a consensus protocol.",
            &["Basic Paxos".to_string()],
        ));
    }

    #[test]
    fn alias_in_first_sentence_flags_byzantine_quorum_body() {
        // Observed pass-10 failure: wiki/concepts/quorum.md body led with
        // "Byzantine quorums ..." on a page whose canonical is "Quorum" and
        // aliases are [majority quorum, Byzantine quorum, uniform quorum].
        // The alias-in-first-sentence check must identify "Byzantine quorum".
        let aliases = vec![
            "majority quorum".to_string(),
            "Byzantine quorum".to_string(),
            "uniform quorum".to_string(),
        ];
        let hit = first_sentence_names_alias(
            "Byzantine quorums are sized to tolerate f Byzantine failures; \
             typically ⌈(n+f+1)/2⌉.",
            "Quorum",
            &aliases,
        );
        assert_eq!(hit, Some("Byzantine quorum"));

        // A generic first sentence that doesn't name any alias passes.
        let hit = first_sentence_names_alias(
            "Quorum is a subset of nodes whose agreement is required for a \
             distributed decision.",
            "Quorum",
            &aliases,
        );
        assert!(hit.is_none());

        // Aliases that only appear after the first sentence don't count.
        let hit = first_sentence_names_alias(
            "Quorum is a subset of voting nodes. Byzantine quorums tolerate \
             Byzantine failures.",
            "Quorum",
            &aliases,
        );
        assert!(hit.is_none());
    }

    #[test]
    fn alias_in_first_sentence_flags_zab_leader_election_body() {
        // Observed pass-10 failure: wiki/concepts/leader-election.md body
        // described Zab's specific procedure. Canonical "Leader election" with
        // an alias "Zab leader election" → the first-sentence-names-alias
        // check catches it.
        let aliases = vec![
            "Raft leader election".to_string(),
            "Zab leader election".to_string(),
            "Bully algorithm".to_string(),
        ];
        let hit = first_sentence_names_alias(
            "Zab leader election proceeds in phases using epochs and proposals \
             in ZooKeeper Atomic Broadcast.",
            "Leader election",
            &aliases,
        );
        assert_eq!(hit, Some("Zab leader election"));

        // Also flagged by canonical-prefix (subject is "Zab leader election",
        // not "Leader election"), giving the bug two independent signals.
        assert!(body_missing_canonical_prefix(
            "Zab leader election proceeds in phases using epochs.",
            "Leader election",
        ));
    }

    #[test]
    fn contains_word_respects_word_boundaries() {
        // "paxos" should not match inside "multi-paxos" because the preceding
        // '-' counts as a non-word byte — but the needle "basic paxos" should
        // match "basic paxos" even when it's followed by punctuation.
        assert!(contains_word("paxos is a family", "paxos"));
        assert!(contains_word(
            "variants include basic paxos, multi-paxos, and cheap paxos.",
            "basic paxos",
        ));
        // Word-boundary: "paxos" must not match inside "paxosoid" (pretend).
        assert!(!contains_word("paxosoid is not a word", "paxos"));
        // Empty needle is never a match.
        assert!(!contains_word("anything", ""));
    }

    #[test]
    fn all_four_observed_failures_trigger_at_least_one_post_check() {
        // Golden-corpus-style test: feed the four pass-10 failure bodies
        // through the merge pipeline (via stubbed LLM) and assert the
        // underlying post-check predicates fire for each one. This is the
        // contract the warn! calls in render_concept_page rely on — if these
        // predicates fire, tracing::warn! fires.

        // Case 1: Raft page carrying a Leader-Completeness body.
        let body_raft = "A Raft safety property stating that any entry committed \
             in a term must appear in the logs of all leaders elected in \
             later terms.";
        assert!(
            body_missing_canonical_prefix(body_raft, "Raft"),
            "Raft body should fail canonical-prefix check"
        );

        // Case 2: Paxos page carrying a Basic-Paxos-only body. Aliases span
        // multiple variants; body mentions only one alias and no family
        // keyword.
        let body_paxos = "Paxos is a two-phase consensus protocol where \
             proposers and acceptors agree on a single value using the Basic \
             Paxos exchange.";
        let paxos_aliases = vec![
            "Basic Paxos".to_string(),
            "Multi-Paxos".to_string(),
            "Cheap Paxos".to_string(),
        ];
        assert!(
            body_missing_variant_scoping(body_paxos, &paxos_aliases),
            "Paxos body should fail variant-scoping check"
        );

        // Case 3: Quorum page whose first sentence scopes to Byzantine quorum.
        let body_quorum = "Byzantine quorums are sized to tolerate f Byzantine \
             failures; typically ⌈(n+f+1)/2⌉.";
        let quorum_aliases = vec![
            "majority quorum".to_string(),
            "Byzantine quorum".to_string(),
            "uniform quorum".to_string(),
        ];
        assert!(
            first_sentence_names_alias(body_quorum, "Quorum", &quorum_aliases).is_some(),
            "Quorum body should fail first-sentence-names-alias check"
        );
        assert!(
            body_missing_canonical_prefix(body_quorum, "Quorum"),
            "Quorum body should also fail canonical-prefix check"
        );

        // Case 4: Leader-election page describing Zab's procedure.
        let body_leader = "Zab leader election proceeds in phases using epochs \
             and proposals in ZooKeeper Atomic Broadcast.";
        let leader_aliases = vec![
            "Raft leader election".to_string(),
            "Zab leader election".to_string(),
            "Bully algorithm".to_string(),
        ];
        assert!(
            first_sentence_names_alias(body_leader, "Leader election", &leader_aliases)
                .is_some(),
            "Leader-election body should fail first-sentence-names-alias check"
        );
        assert!(
            body_missing_canonical_prefix(body_leader, "Leader election"),
            "Leader-election body should also fail canonical-prefix check"
        );
    }

    fn bad_group(canonical: &str, aliases: Vec<String>, body: &str) -> MergeGroup {
        MergeGroup {
            canonical_name: canonical.to_string(),
            aliases: aliases.clone(),
            members: vec![ConceptCandidate {
                name: canonical.to_string(),
                aliases,
                definition_hint: Some(body.to_string()),
                source_anchors: vec![],
            }],
            confident: true,
            rationale: None,
        }
    }

    #[test]
    fn golden_corpus_bad_bodies_flow_through_merge_without_blocking() {
        // Stub the LLM adapter to return the four observed bad bodies and
        // assert: (a) the write still happens (warn-only contract), and
        // (b) the underlying predicates that gate the warn! calls fire.
        // tracing-capture infra isn't wired into this crate's test deps, so
        // we assert the gating predicates directly alongside the contract
        // that the concept pages are still emitted.
        let dir = tempdir().expect("tempdir");

        let groups = vec![
            bad_group(
                "Raft",
                vec!["Leader Completeness".to_string()],
                "A Raft safety property stating that any entry committed in a \
                 term must appear in the logs of all leaders elected in later \
                 terms.",
            ),
            bad_group(
                "Paxos",
                vec![
                    "Basic Paxos".to_string(),
                    "Multi-Paxos".to_string(),
                    "Cheap Paxos".to_string(),
                ],
                "Paxos is a two-phase consensus protocol where proposers and \
                 acceptors agree on a single value using the Basic Paxos \
                 exchange.",
            ),
            bad_group(
                "Quorum",
                vec![
                    "majority quorum".to_string(),
                    "Byzantine quorum".to_string(),
                    "uniform quorum".to_string(),
                ],
                "Byzantine quorums are sized to tolerate f Byzantine failures; \
                 typically ⌈(n+f+1)/2⌉.",
            ),
            bad_group(
                "Leader election",
                vec![
                    "Raft leader election".to_string(),
                    "Zab leader election".to_string(),
                    "Bully algorithm".to_string(),
                ],
                "Zab leader election proceeds in phases using epochs and \
                 proposals in ZooKeeper Atomic Broadcast.",
            ),
        ];

        let adapter = FakeAdapter {
            response: MergeConceptCandidatesResponse {
                groups: groups.clone(),
            },
            provenance: provenance(),
        };

        // The merge pass must still produce one concept page per group — the
        // warn checks are warn-only and never block the write.
        let candidates: Vec<ConceptCandidate> = groups
            .iter()
            .flat_map(|g| g.members.clone())
            .collect();
        let artifact = run_concept_merge_pass(&adapter, candidates, dir.path())
            .expect("merge pass runs despite bad bodies");
        assert_eq!(
            artifact.concept_pages.len(),
            groups.len(),
            "every confident group, even with a bad body, must still emit a page"
        );

        // And the gating predicates must fire for each group — this is what
        // the warn! calls key off of, so if these fire, the warn! fires.
        for group in &groups {
            let body = group.members[0]
                .definition_hint
                .as_deref()
                .expect("fixture body present");
            let canonical_prefix = body_missing_canonical_prefix(body, &group.canonical_name);
            let variant_scoping = body_missing_variant_scoping(body, &group.aliases);
            let alias_first = first_sentence_names_alias(
                body,
                &group.canonical_name,
                &group.aliases,
            )
            .is_some();
            let narrow_variant =
                narrow_variant_body_match(body, &group.canonical_name, &group.aliases).is_some();
            assert!(
                canonical_prefix || variant_scoping || alias_first || narrow_variant,
                "group {} had no post-check fire for body: {body}",
                group.canonical_name
            );
        }
    }

    #[test]
    fn parse_merge_concept_candidates_json_roundtrips() {
        use kb_llm::parse_merge_concept_candidates_json;

        let json = r#"{
            "groups": [
                {
                    "canonical_name": "Borrow checker",
                    "aliases": ["borrowck"],
                    "members": [],
                    "confident": true,
                    "rationale": null
                }
            ]
        }"#;

        let response = parse_merge_concept_candidates_json(json).expect("parse");
        assert_eq!(response.groups.len(), 1);
        assert_eq!(response.groups[0].canonical_name, "Borrow checker");
        assert!(response.groups[0].confident);
    }
}
