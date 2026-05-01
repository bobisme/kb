//! Auto-suggest new concept pages discovered during `kb compile`.
//!
//! bn-13zx: rather than waiting for the user to manually author every concept
//! page, this pass extracts candidate keyphrases from each source body via a
//! RAKE-style algorithm, optionally filters them through the configured LLM,
//! and queues survivors as `ReviewKind::ConceptCandidate` review items. The
//! existing `kb review approve <id>` flow then materializes the concept page —
//! we deliberately reuse that pipeline rather than introducing a parallel
//! `.kb/suggestions/` store + new CLI verbs (the bone description sketched
//! both, but the review queue already does the same job and avoids duplicating
//! the approve/reject UX).
//!
//! Detection has two stages, additive:
//!
//! 1. **Regex-based candidates** (already implemented in
//!    `kb_lint::check_missing_concepts_hits`). Matches capitalized multi-word
//!    spans and backtick-quoted identifiers above the corpus-wide
//!    `min_sources` / `min_mentions` thresholds.
//! 2. **RAKE-based candidates** (this module). Picks up lower-case noun
//!    phrases the regex skips — "leader-follower replication", "credential
//!    rotation", "tail-call optimization". Cheaper but noisier; the LLM
//!    filter trims down to the actually useful ones.
//!
//! Both streams converge into a single deduplicated set, the LLM scores
//! the top-K, and survivors become review items keyed by candidate slug
//! (so the existing rejected-item dedup keeps users from re-seeing rejects).

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::Path;

use anyhow::{Context, Result};

use kb_core::{
    EntityMetadata, ReviewItem, ReviewKind, ReviewStatus, Status, normalized_dir,
    save_review_item, slug_from_title,
};
use kb_llm::{ConceptSuggestionInput, FilterConceptSuggestionsRequest, LlmAdapter};

use kb_lint::{ConceptCandidateHit, MissingConceptsConfig, check_missing_concepts_hits};

const WIKI_CONCEPTS_DIR: &str = "wiki/concepts";

/// Configuration for the [`run_concept_suggestions_pass`].
#[derive(Debug, Clone)]
pub struct ConceptSuggestionsOptions {
    /// Whether the pass runs at all. Off-by-default skips it entirely.
    pub enabled: bool,
    /// Cap on how many top RAKE candidates to keep per source. Each
    /// source contributes at most this many phrases to the global pool
    /// before merging with regex-based hits. Default: 8.
    pub max_per_source: usize,
    /// Global cap on candidates fed to the LLM filter. Higher = more
    /// LLM tokens. Default: 24.
    pub top_k_global: usize,
    /// When true, route the merged candidate list through the configured
    /// LLM adapter for usefulness filtering. When false, all candidates
    /// pass through and become review items unchanged.
    pub llm_filter: bool,
    /// Configuration shared with the `missing_concepts` lint (regex-based
    /// signal + min-sources / min-mentions thresholds).
    pub missing_concepts: MissingConceptsConfig,
}

impl Default for ConceptSuggestionsOptions {
    fn default() -> Self {
        Self {
            enabled: true,
            max_per_source: 8,
            top_k_global: 24,
            llm_filter: true,
            missing_concepts: MissingConceptsConfig::default(),
        }
    }
}

/// Summary of work done by the pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ConceptSuggestionStats {
    /// Candidates discovered by the union of regex + RAKE detection.
    pub discovered: usize,
    /// Candidates that survived the LLM filter (or all of them, if the
    /// filter is disabled).
    pub accepted: usize,
    /// Review items written or refreshed in `<root>/reviews/`.
    pub review_items: usize,
}

/// Run the concept-suggestion compile pass.
///
/// Walks normalized sources for keyphrases, optionally filters via LLM,
/// then queues surviving candidates as review items. Idempotent: the
/// existing `save_review_item` machinery dedupes by id (slug), so a second
/// compile against the same corpus refreshes the same items rather than
/// duplicating them, and previously-rejected items stay rejected.
///
/// # Errors
///
/// Returns an error when source bodies cannot be read or review items
/// cannot be persisted. LLM filter failures are downgraded to a warning
/// and the unfiltered candidate set is queued instead — it's better to
/// surface noisy suggestions than to silently drop them.
pub fn run_concept_suggestions_pass(
    root: &Path,
    options: &ConceptSuggestionsOptions,
    adapter: Option<&(dyn LlmAdapter + '_)>,
) -> Result<ConceptSuggestionStats> {
    if !options.enabled {
        return Ok(ConceptSuggestionStats::default());
    }

    let regex_hits = check_missing_concepts_hits(root, &options.missing_concepts)
        .context("regex-based concept-candidate scan")?;

    let rake_hits = collect_rake_candidates(root, options)
        .context("RAKE concept-candidate scan")?;

    let mut merged = merge_candidates(regex_hits, rake_hits, options.top_k_global);

    let discovered = merged.len();

    let accepted_keys: Option<BTreeSet<String>> = if options.llm_filter {
        adapter.and_then(|adapter| match filter_with_llm(adapter, &merged) {
            Ok(keys) => Some(keys),
            Err(err) => {
                tracing::warn!(
                    "concept-suggestion LLM filter failed; queueing unfiltered: {err:#}"
                );
                None
            }
        })
    } else {
        None
    };

    if let Some(keys) = accepted_keys {
        merged.retain(|hit| keys.contains(&slug_from_title(&hit.name)));
    }

    let accepted = merged.len();

    let now = unix_time_ms()?;
    let mut review_items = 0_usize;
    for hit in &merged {
        let item = build_review_item(hit, now);
        save_review_item(root, &item).with_context(|| {
            format!(
                "save concept-suggestion review item for '{}'",
                hit.name
            )
        })?;
        review_items += 1;
    }

    Ok(ConceptSuggestionStats {
        discovered,
        accepted,
        review_items,
    })
}

fn merge_candidates(
    regex_hits: Vec<ConceptCandidateHit>,
    rake_hits: Vec<ConceptCandidateHit>,
    cap: usize,
) -> Vec<ConceptCandidateHit> {
    // Dedup by slug. Regex hits win when both detectors fire on the same
    // term — they carry the canonical capitalization and the cross-source
    // count is already cross-validated.
    let mut by_slug: BTreeMap<String, ConceptCandidateHit> = BTreeMap::new();
    for hit in regex_hits {
        let slug = slug_from_title(&hit.name);
        by_slug.entry(slug).or_insert(hit);
    }
    for hit in rake_hits {
        let slug = slug_from_title(&hit.name);
        by_slug.entry(slug).or_insert(hit);
    }

    let mut merged: Vec<ConceptCandidateHit> = by_slug.into_values().collect();
    // Higher source-count and mention-count first; ties broken by name
    // for deterministic output across runs.
    merged.sort_by(|a, b| {
        b.source_ids
            .len()
            .cmp(&a.source_ids.len())
            .then_with(|| b.mention_count.cmp(&a.mention_count))
            .then_with(|| a.name.cmp(&b.name))
    });
    merged.truncate(cap);
    merged
}

fn build_review_item(hit: &ConceptCandidateHit, now: u64) -> ReviewItem {
    let slug = slug_from_title(&hit.name);
    let id = format!("compile:concept-suggestion:{slug}");
    let destination = std::path::PathBuf::from(WIKI_CONCEPTS_DIR).join(format!("{slug}.md"));

    let mut fingerprint_parts: Vec<&[u8]> = vec![hit.name.as_bytes()];
    for src in &hit.source_ids {
        fingerprint_parts.push(src.as_bytes());
    }
    let fingerprint = kb_core::hash_many(&fingerprint_parts).to_hex();

    let comment = format!(
        "Auto-suggested concept '{}' from {} source(s) ({} mention(s)). \
         Sources: {}. Approve to draft wiki/concepts/{}.md from the source mentions.",
        hit.name,
        hit.source_ids.len(),
        hit.mention_count,
        hit.source_ids.join(", "),
        slug,
    );

    ReviewItem {
        metadata: EntityMetadata {
            id: id.clone(),
            created_at_millis: now,
            updated_at_millis: now,
            source_hashes: vec![fingerprint],
            model_version: None,
            tool_version: Some(format!(
                "{}/{}",
                env!("CARGO_PKG_NAME"),
                env!("CARGO_PKG_VERSION")
            )),
            prompt_template_hash: None,
            dependencies: hit.source_ids.clone(),
            output_paths: vec![destination.clone()],
            status: Status::NeedsReview,
        },
        kind: ReviewKind::ConceptCandidate,
        target_entity_id: id,
        proposed_destination: Some(destination.clone()),
        citations: hit.source_ids.clone(),
        affected_pages: vec![destination],
        created_at_millis: now,
        status: ReviewStatus::Pending,
        comment,
    }
}

fn unix_time_ms() -> Result<u64> {
    Ok(std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .context("system clock returned a value before the Unix epoch")?
        .as_millis()
        .try_into()
        .unwrap_or(u64::MAX))
}

fn filter_with_llm(
    adapter: &(dyn LlmAdapter + '_),
    candidates: &[ConceptCandidateHit],
) -> Result<BTreeSet<String>> {
    if candidates.is_empty() {
        return Ok(BTreeSet::new());
    }

    let inputs: Vec<ConceptSuggestionInput> = candidates
        .iter()
        .map(|hit| ConceptSuggestionInput {
            slug: slug_from_title(&hit.name),
            name: hit.name.clone(),
            source_ids: hit.source_ids.clone(),
            mention_count: hit.mention_count,
        })
        .collect();

    let request = FilterConceptSuggestionsRequest { candidates: inputs };
    let accepted = adapter
        .filter_concept_suggestions(&request)
        .context("LLM filter pass for concept suggestions")?;

    Ok(accepted.into_iter().collect())
}

// ---------------------------------------------------------------------------
// RAKE keyphrase extraction
// ---------------------------------------------------------------------------
//
// Reference: Rose, Engel, Cramer, Cowley, "Automatic Keyword Extraction from
// Individual Documents" (2010). Pure functional implementation, no
// dependency on Python NLP toolchains. The thresholds and stopword list
// are tuned for short prose (kb sources are typically 200-2000 words —
// research-paper or domain notes, not whole books).
//
// Algorithm:
//   1. Split each source body into candidate phrases by cutting on
//      stopwords + punctuation.
//   2. For each *content word* w that appears in any phrase:
//        deg(w)  = sum of |phrase| for every phrase containing w
//        freq(w) = total occurrences of w across all phrases
//        wscore(w) = deg(w) / freq(w)   (favors words sitting in long phrases)
//   3. phrase_score = sum of wscore(w) for every word in the phrase.
//   4. Top-N phrases by score per source.
//
// We DO NOT lemmatize — kb sources mix code identifiers with prose and a
// stemmer would mangle "WAL", "JWT", etc. Casual repetitions like
// "replicate" / "replicated" / "replication" stay as separate phrases;
// the LLM filter dedupes those at the semantic level.

/// English stopwords used to fragment text into RAKE phrases. The list is
/// intentionally a superset of what the regex-based detector uses — RAKE
/// is more aggressive about phrase boundaries because we want it to cut
/// "the leader-follower replication pattern" into "leader-follower
/// replication" rather than treating "the" as content.
const RAKE_STOPWORDS: &[&str] = &[
    // articles / determiners
    "a", "an", "the", "this", "that", "these", "those", "such", "every", "all", "any", "some",
    "no", "another", "either", "neither", "each", "both",
    // conjunctions / prepositions
    "and", "or", "but", "nor", "yet", "so", "for", "to", "of", "in", "on", "at", "by", "with",
    "from", "as", "via", "into", "onto", "over", "under", "between", "among", "across", "around",
    "during", "while", "after", "before", "since", "until", "than", "then", "though", "although",
    "because", "if", "unless", "whether", "however", "moreover", "thus", "hence", "also",
    // be / have / do
    "be", "is", "are", "was", "were", "been", "being", "am", "have", "has", "had", "having", "do",
    "does", "did", "doing", "done",
    // pronouns
    "i", "me", "my", "we", "us", "our", "you", "your", "he", "him", "his", "she", "her", "they",
    "them", "their", "it", "its", "what", "which", "who", "whom", "whose", "where", "when", "why",
    "how",
    // modals / auxiliaries
    "can", "could", "may", "might", "must", "shall", "should", "will", "would", "ought",
    // generic filler that bloats RAKE phrases without carrying meaning
    "use", "used", "using", "make", "made", "makes", "see", "seen", "saw", "say", "said", "get",
    "got", "go", "went", "gone", "come", "came", "take", "took", "taken", "give", "gave", "given",
    "set", "put", "find", "found", "way", "ways", "thing", "things", "section", "chapter",
    "figure", "table", "page", "example", "note", "one", "two", "three", "four", "five", "first",
    "second", "third", "last", "next", "new", "old", "many", "much", "more", "most", "less",
    "least", "few", "several", "still", "even", "just", "only", "very", "really", "quite",
    "rather", "almost", "always", "never", "often", "sometimes", "usually", "perhaps", "maybe",
    "above", "below", "again", "etc",
];

fn rake_stopword_set() -> &'static std::sync::OnceLock<BTreeSet<&'static str>> {
    static STOPWORDS: std::sync::OnceLock<BTreeSet<&'static str>> = std::sync::OnceLock::new();
    STOPWORDS.get_or_init(|| RAKE_STOPWORDS.iter().copied().collect());
    &STOPWORDS
}

/// Minimum word count for a RAKE phrase to be considered. Single-word
/// phrases are too noisy for concept suggestions (every common verb would
/// surface).
const RAKE_MIN_WORDS: usize = 2;

/// Maximum word count for a RAKE phrase. Longer phrases are usually run-on
/// fragments rather than topical concepts. Capping at 4 mirrors the
/// regex-based detector's upper bound.
const RAKE_MAX_WORDS: usize = 4;

/// Minimum total mention count across the corpus for a RAKE phrase to be
/// kept. Below this, the LLM filter wastes tokens on coincidental n-grams
/// that won't repeat.
const RAKE_MIN_CORPUS_MENTIONS: usize = 2;

/// Walk normalized sources, run RAKE over each, and aggregate the top
/// keyphrases into `ConceptCandidateHit`s suitable for the merge pipeline.
fn collect_rake_candidates(
    root: &Path,
    options: &ConceptSuggestionsOptions,
) -> Result<Vec<ConceptCandidateHit>> {
    let normalized_root = normalized_dir(root);
    if !normalized_root.is_dir() {
        return Ok(Vec::new());
    }

    // Concept slugs already on disk — skip RAKE phrases that would just
    // duplicate an existing concept. Cheap to do here so the LLM filter
    // sees a tighter candidate list.
    let existing_slugs = collect_existing_concept_slugs(root).unwrap_or_default();

    let mut by_phrase: BTreeMap<String, AggregatedPhrase> = BTreeMap::new();

    for entry in std::fs::read_dir(&normalized_root)
        .with_context(|| format!("read normalized dir {}", normalized_root.display()))?
    {
        let entry = entry.with_context(|| format!("walk {}", normalized_root.display()))?;
        if !entry.file_type().is_ok_and(|t| t.is_dir()) {
            continue;
        }
        let source_id = entry.file_name().to_string_lossy().into_owned();
        let body_path = entry.path().join("source.md");
        if !body_path.is_file() {
            continue;
        }

        let body = match std::fs::read_to_string(&body_path) {
            Ok(text) => text,
            Err(err) => {
                tracing::warn!(
                    "concept-suggestion: skipping {} (read failed): {err}",
                    body_path.display()
                );
                continue;
            }
        };

        let phrases = rake_keyphrases(&body, options.max_per_source);
        for (phrase, score) in phrases {
            let slug = slug_from_title(&phrase);
            if existing_slugs.contains(&slug) {
                continue;
            }
            let entry = by_phrase
                .entry(phrase.clone())
                .or_insert_with(|| AggregatedPhrase {
                    name: phrase,
                    sources: BTreeSet::new(),
                    mention_count: 0,
                    total_score: 0.0,
                });
            entry.sources.insert(source_id.clone());
            entry.mention_count += 1;
            entry.total_score += score;
        }
    }

    let mut hits: Vec<ConceptCandidateHit> = by_phrase
        .into_values()
        .filter(|phrase| phrase.mention_count >= RAKE_MIN_CORPUS_MENTIONS)
        .map(|phrase| ConceptCandidateHit {
            name: phrase.name,
            source_ids: phrase.sources.into_iter().collect(),
            mention_count: phrase.mention_count,
        })
        .collect();
    hits.sort_by(|a, b| {
        b.source_ids
            .len()
            .cmp(&a.source_ids.len())
            .then_with(|| b.mention_count.cmp(&a.mention_count))
            .then_with(|| a.name.cmp(&b.name))
    });
    Ok(hits)
}

#[derive(Debug, Clone)]
struct AggregatedPhrase {
    name: String,
    sources: BTreeSet<String>,
    mention_count: usize,
    total_score: f32,
}

fn collect_existing_concept_slugs(root: &Path) -> Result<BTreeSet<String>> {
    let mut out = BTreeSet::new();
    let dir = root.join(WIKI_CONCEPTS_DIR);
    if !dir.is_dir() {
        return Ok(out);
    }
    for entry in std::fs::read_dir(&dir)? {
        let entry = entry?;
        if !entry.file_type().is_ok_and(|t| t.is_file()) {
            continue;
        }
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("md") {
            continue;
        }
        if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
            out.insert(stem.to_string());
        }
    }
    Ok(out)
}

/// Run RAKE over `body` and return the top-`limit` (phrase, score) pairs.
///
/// The phrase strings are the raw original casing from the document (with
/// internal whitespace normalized); scores are RAKE phrase scores.
///
/// # Panics
///
/// Panics only if the internal `OnceLock` for the stopword set somehow
/// fails to initialize, which is not reachable in practice.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn rake_keyphrases(body: &str, limit: usize) -> Vec<(String, f32)> {
    let stopwords = rake_stopword_set().get().expect("oncelock initialized");

    let phrases = split_into_phrases(body, stopwords);
    if phrases.is_empty() {
        return Vec::new();
    }

    // Compute degree(w) and freq(w) over the lowercase word forms.
    let mut degree: HashMap<String, usize> = HashMap::new();
    let mut freq: HashMap<String, usize> = HashMap::new();
    for phrase in &phrases {
        let n = phrase.words.len();
        for word in &phrase.words {
            let key = word.to_lowercase();
            *degree.entry(key.clone()).or_insert(0) += n;
            *freq.entry(key).or_insert(0) += 1;
        }
    }

    let word_score = |w: &str| -> f32 {
        let key = w.to_lowercase();
        let d = degree.get(&key).copied().unwrap_or(0);
        let f = freq.get(&key).copied().unwrap_or(1).max(1);
        d as f32 / f as f32
    };

    // Score each phrase. Dedupe by lower-case phrase form so "Replication"
    // and "replication" don't both surface.
    let mut by_phrase: BTreeMap<String, (String, f32, usize)> = BTreeMap::new();
    for phrase in &phrases {
        let words = &phrase.words;
        if !(RAKE_MIN_WORDS..=RAKE_MAX_WORDS).contains(&words.len()) {
            continue;
        }
        let canonical: String = words.join(" ");
        let key = canonical.to_lowercase();
        let score: f32 = words.iter().map(|w| word_score(w)).sum();
        let entry = by_phrase
            .entry(key)
            .or_insert_with(|| (canonical.clone(), 0.0, 0));
        entry.1 += score;
        entry.2 += 1;
    }

    let mut scored: Vec<(String, f32)> = by_phrase
        .into_values()
        .map(|(name, score, _count)| (name, score))
        .collect();
    // Highest score first; ties broken by name asc.
    scored.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });
    scored.truncate(limit);
    scored
}

/// A single RAKE phrase candidate (a contiguous run of content words
/// bounded by stopwords or punctuation in the source text).
#[derive(Debug, Clone)]
struct RakePhrase {
    words: Vec<String>,
}

fn split_into_phrases(body: &str, stopwords: &BTreeSet<&'static str>) -> Vec<RakePhrase> {
    let mut phrases: Vec<RakePhrase> = Vec::new();
    let mut current: Vec<String> = Vec::new();

    for raw_token in tokenize(body) {
        // Punctuation token: cut here.
        if !raw_token.chars().any(char::is_alphanumeric) {
            flush_phrase(&mut current, &mut phrases);
            continue;
        }

        let lower = raw_token.to_lowercase();
        // Strip surrounding non-alnum chars while keeping internal hyphens
        // / underscores / dots — `leader-follower` and `tcp/ip` should stay
        // as single content tokens.
        let trimmed: String = trim_token(&raw_token);
        let trimmed_lower = trimmed.to_lowercase();

        if trimmed.is_empty() {
            flush_phrase(&mut current, &mut phrases);
            continue;
        }

        if stopwords.contains(trimmed_lower.as_str())
            || stopwords.contains(lower.as_str())
            || is_pure_number(&trimmed)
        {
            flush_phrase(&mut current, &mut phrases);
            continue;
        }

        current.push(trimmed);
    }
    flush_phrase(&mut current, &mut phrases);
    phrases
}

fn flush_phrase(current: &mut Vec<String>, phrases: &mut Vec<RakePhrase>) {
    if !current.is_empty() {
        phrases.push(RakePhrase {
            words: std::mem::take(current),
        });
    }
}

/// Whitespace + punctuation tokenizer. Each yielded token is either a
/// run of `is_alphanumeric()`-or-internal-symbol characters (a word) or a
/// single punctuation character (a phrase boundary).
///
/// `-`, `_`, `/`, and `'` stay inside words so identifiers like
/// `leader-follower`, `tcp/ip`, and `it's` survive as single tokens. The
/// period is intentionally a phrase boundary even though that splits
/// version numbers like `v1.0` into pieces — sentence-end detection
/// matters more for RAKE quality than preserving the rare versioned
/// identifier as a single content token.
fn tokenize(body: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    let mut current = String::new();
    let mut in_word = false;
    for ch in body.chars() {
        if ch.is_alphanumeric() || matches!(ch, '-' | '_' | '/' | '\'') {
            current.push(ch);
            in_word = true;
        } else {
            if in_word {
                out.push(std::mem::take(&mut current));
                in_word = false;
            }
            if !ch.is_whitespace() {
                out.push(ch.to_string());
            }
        }
    }
    if in_word {
        out.push(current);
    }
    out
}

fn trim_token(token: &str) -> String {
    token
        .trim_matches(|c: char| !c.is_alphanumeric() && c != '_')
        .to_string()
}

fn is_pure_number(token: &str) -> bool {
    !token.is_empty() && token.chars().all(|c| c.is_ascii_digit() || c == '.')
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn rake_picks_up_multi_word_phrase() {
        let body = "Leader-follower replication is the standard pattern. \
                    Leader-follower replication keeps replicas in sync. \
                    Failover requires confirming replica lag.";
        let hits = rake_keyphrases(body, 5);
        let names: Vec<String> = hits.iter().map(|(n, _)| n.clone()).collect();
        assert!(
            names.iter().any(|n| n.to_lowercase() == "leader-follower replication"),
            "expected 'leader-follower replication' in {names:?}"
        );
    }

    #[test]
    fn rake_drops_short_phrases() {
        let body = "Keep. Short. Phrases. Out.";
        let hits = rake_keyphrases(body, 5);
        assert!(hits.is_empty(), "single-word phrases must be filtered");
    }

    #[test]
    fn rake_dedupes_case_variants() {
        let body = "Authn token rotation. AUTHN TOKEN ROTATION. authn token rotation.";
        let hits = rake_keyphrases(body, 5);
        let count = hits
            .iter()
            .filter(|(n, _)| n.to_lowercase() == "authn token rotation")
            .count();
        assert_eq!(count, 1, "case variants must collapse: {hits:?}");
    }

    #[test]
    fn rake_handles_empty_input() {
        assert!(rake_keyphrases("", 5).is_empty());
        assert!(rake_keyphrases("the and or but", 5).is_empty());
    }

    #[test]
    fn rake_caps_phrase_length_at_four() {
        let body = "alpha beta gamma delta epsilon zeta eta theta. \
                    alpha beta gamma delta epsilon zeta eta theta.";
        let hits = rake_keyphrases(body, 10);
        for (phrase, _) in &hits {
            let words = phrase.split_whitespace().count();
            assert!(
                (RAKE_MIN_WORDS..=RAKE_MAX_WORDS).contains(&words),
                "phrase '{phrase}' has {words} words (allowed {RAKE_MIN_WORDS}..={RAKE_MAX_WORDS})"
            );
        }
    }

    #[test]
    fn rake_ignores_pure_numbers() {
        let body = "Quorum size is 3. Quorum size is 3. Replicas count to 5.";
        let hits = rake_keyphrases(body, 5);
        let has_number_word = hits
            .iter()
            .any(|(n, _)| n.split_whitespace().any(is_pure_number));
        assert!(!has_number_word, "numbers must not appear as content words: {hits:?}");
    }

    #[test]
    fn merge_dedupes_by_slug() {
        let regex_hit = ConceptCandidateHit {
            name: "Authentication Token".to_string(),
            source_ids: vec!["src-a".to_string()],
            mention_count: 4,
        };
        let rake_hit = ConceptCandidateHit {
            name: "authentication token".to_string(),
            source_ids: vec!["src-b".to_string()],
            mention_count: 3,
        };
        let merged = merge_candidates(vec![regex_hit], vec![rake_hit], 10);
        assert_eq!(merged.len(), 1);
        assert_eq!(merged[0].name, "Authentication Token", "regex variant wins");
    }

    #[test]
    fn merge_truncates_to_cap() {
        let many: Vec<ConceptCandidateHit> = (0..20)
            .map(|i| ConceptCandidateHit {
                name: format!("Phrase {i}"),
                source_ids: vec![format!("src-{i}")],
                mention_count: i,
            })
            .collect();
        let merged = merge_candidates(many, Vec::new(), 5);
        assert_eq!(merged.len(), 5);
    }

    #[test]
    fn merge_sorts_by_source_count_then_mention_count() {
        let lo = ConceptCandidateHit {
            name: "Low".to_string(),
            source_ids: vec!["src-1".to_string()],
            mention_count: 100,
        };
        let hi = ConceptCandidateHit {
            name: "High".to_string(),
            source_ids: vec!["src-1".to_string(), "src-2".to_string()],
            mention_count: 2,
        };
        let merged = merge_candidates(vec![lo, hi], Vec::new(), 10);
        assert_eq!(merged[0].name, "High", "more sources beats more mentions");
    }
}
