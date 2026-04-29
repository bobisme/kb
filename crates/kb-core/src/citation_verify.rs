//! Quote-verification primitives for citation trust checks (bn-166d).
//!
//! "Required citations" lints verify that *some* citation exists; this
//! module closes the complementary gap of verifying that quoted spans
//! adjacent to a citation actually appear in the cited source.
//!
//! The pipeline is:
//!
//! 1. [`extract_quote_citations`] scans a body for `"..." [src-id]`,
//!    `[src-id] "..."`, and `"..." (src-id)` patterns and yields
//!    `(quote, src_id)` pairs. Smart quotes are normalized to ASCII
//!    before pattern matching so curly-quoted prose works too.
//! 2. [`normalize_for_match`] produces the comparison form: collapse
//!    whitespace runs, lowercase, drop markdown emphasis (`*`, `_`,
//!    backticks). It is applied to both the extracted quote and the
//!    candidate source body before substring search.
//! 3. [`is_quote_present`] checks `normalize(quote) ⊂ normalize(source)`
//!    with a bounded Levenshtein fuzz allowance proportional to quote
//!    length (≤ `fuzz_per_100_chars` edits per 100 normalized chars).
//!
//! All routines are pure — no IO, no path resolution. Consumers
//! ([`kb-query`](../kb_query/index.html) for `kb ask` answers,
//! [`kb-lint`](../kb_lint/index.html) for compiled concept pages) load
//! the candidate source content themselves and feed bytes in.

use std::sync::LazyLock;

use regex::Regex;

/// Default fuzz allowance (edits per 100 normalized characters). Mirrors
/// the bone's specification — picked to absorb chunk-boundary rewrites
/// from ingestion without admitting wholesale fabrications.
pub const DEFAULT_FUZZ_PER_100_CHARS: u32 = 1;

/// One extracted (quote, src-id) pair, with the byte offset of the quote
/// in the *original* (un-normalized) text so callers can flag the line.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QuoteCitation {
    /// The quoted span exactly as it appeared in the body (smart quotes
    /// folded to ASCII but contents otherwise verbatim).
    pub quote: String,
    /// The bare src-id, e.g. `src-abc`. The leading `src-` prefix is
    /// preserved so callers can route directly to wiki/sources.
    pub src_id: String,
    /// 0-based byte offset of the opening `"` in the (smart-quote-folded)
    /// body. Useful for line attribution in lint reports.
    pub offset: usize,
}

/// Outcome of [`is_quote_present`] — distinguishes exact-substring hits
/// from fuzz-bounded matches so callers can surface "near miss" diagnostics
/// later if needed. Today both shapes count as "verified".
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuoteMatch {
    /// `normalize(quote)` is an exact substring of `normalize(source)`.
    Exact,
    /// Found via bounded-Levenshtein within the allowed edit budget.
    Fuzzy { distance: u32, budget: u32 },
    /// No span in `normalize(source)` was within the edit budget of `normalize(quote)`.
    NotFound,
}

impl QuoteMatch {
    #[must_use]
    pub const fn is_match(self) -> bool {
        !matches!(self, Self::NotFound)
    }
}

/// Fold the typographic quote variants kb-ingest tends to leave behind
/// into ASCII `"` / `'`. Applied to the body before pattern matching so
/// `"foo" [src-x]` and `\u{201c}foo\u{201d} [src-x]` both extract.
#[must_use]
pub fn fold_smart_quotes(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for ch in text.chars() {
        match ch {
            // Double quotes: " " „ ‟ « »
            '\u{201c}' | '\u{201d}' | '\u{201e}' | '\u{201f}' | '\u{00ab}' | '\u{00bb}' => {
                out.push('"');
            }
            // Single quotes / apostrophes: ' ' ‚ ‛
            '\u{2018}' | '\u{2019}' | '\u{201a}' | '\u{201b}' => {
                out.push('\'');
            }
            other => out.push(other),
        }
    }
    out
}

/// Extract every `(quote, src-id)` pair from a body using the three
/// supported syntactic shapes:
///
/// - `"<text>" [src-XXX]`  (citation immediately after closing quote)
/// - `[src-XXX] "<text>"`  (citation immediately before opening quote)
/// - `"<text>" (src-XXX)`  (parenthesized form)
///
/// Smart quotes are normalized first via [`fold_smart_quotes`]. Up to
/// one whitespace run is allowed between the quote and the citation
/// marker — agents tend to soft-wrap and we'd rather match too much
/// than spuriously skip pairs.
///
/// Returned offsets index into the *folded* body, not the original.
#[must_use]
pub fn extract_quote_citations(body: &str) -> Vec<QuoteCitation> {
    let folded = fold_smart_quotes(body);
    let mut out = Vec::new();
    let mut seen: std::collections::HashSet<(String, String)> =
        std::collections::HashSet::new();

    let mut push_unique = |q: &str, id: &str, offset: usize, out: &mut Vec<QuoteCitation>| {
        let key = (q.to_string(), id.to_string());
        if seen.insert(key) {
            out.push(QuoteCitation {
                quote: q.to_string(),
                src_id: id.to_string(),
                offset,
            });
        }
    };

    for cap in QUOTE_THEN_BRACKET_RE.captures_iter(&folded) {
        if let (Some(q), Some(id)) = (cap.get(1), cap.get(2)) {
            push_unique(q.as_str(), id.as_str(), q.start().saturating_sub(1), &mut out);
        }
    }
    for cap in BRACKET_THEN_QUOTE_RE.captures_iter(&folded) {
        if let (Some(id), Some(q)) = (cap.get(1), cap.get(2)) {
            push_unique(q.as_str(), id.as_str(), q.start().saturating_sub(1), &mut out);
        }
    }
    for cap in QUOTE_THEN_PAREN_RE.captures_iter(&folded) {
        if let (Some(q), Some(id)) = (cap.get(1), cap.get(2)) {
            push_unique(q.as_str(), id.as_str(), q.start().saturating_sub(1), &mut out);
        }
    }

    out
}

// `"text" [src-XXX]` — captures: 1=quote-body, 2=src-id (with prefix).
static QUOTE_THEN_BRACKET_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r#""([^"\r\n]+)"\s*\[(src-[A-Za-z0-9][A-Za-z0-9_-]*)\]"#)
        .expect("valid quote-bracket regex")
});

// `[src-XXX] "text"` — captures: 1=src-id, 2=quote-body.
//
// We only allow whitespace (any kind, including line breaks) between the
// closing bracket and the opening quote. Anything else — punctuation,
// "notes:", a sentence-ending word — should not anchor the next quote
// to this citation.
static BRACKET_THEN_QUOTE_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r#"\[(src-[A-Za-z0-9][A-Za-z0-9_-]*)\]\s+"([^"\r\n]+)""#)
        .expect("valid bracket-quote regex")
});

// `"text" (src-XXX)` — captures: 1=quote-body, 2=src-id.
static QUOTE_THEN_PAREN_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r#""([^"\r\n]+)"\s*\((src-[A-Za-z0-9][A-Za-z0-9_-]*)\)"#)
        .expect("valid quote-paren regex")
});

/// Strip markdown emphasis, lowercase, and collapse whitespace.
///
/// Used as a comparison form on both sides — quotes the model produced
/// and source bodies on disk — so cosmetic differences (line wraps,
/// bold/italic, code spans) don't cause false negatives.
///
/// The transformation is conservative: it strips the *delimiters*
/// (`*foo*`, `_bar_`, `` `baz` ``) but keeps the inner text. Rendering
/// always preserves text content, so this matches what a human reader
/// sees.
#[must_use]
pub fn normalize_for_match(text: &str) -> String {
    let folded = fold_smart_quotes(text);
    let stripped: String = folded
        .chars()
        .filter(|ch| !matches!(ch, '*' | '_' | '`'))
        .collect();

    let lowered = stripped.to_lowercase();
    let mut out = String::with_capacity(lowered.len());
    let mut last_was_space = true;
    for ch in lowered.chars() {
        if ch.is_whitespace() {
            if !last_was_space {
                out.push(' ');
                last_was_space = true;
            }
        } else {
            out.push(ch);
            last_was_space = false;
        }
    }
    if out.ends_with(' ') {
        out.pop();
    }
    out
}

/// Bounded Levenshtein distance.
///
/// Returns `Some(d)` when `d <= budget`, else `None`. Allocates two
/// rows of length `n+1`, never the full `n*m` matrix — fine for
/// quote-length spans (typically << 1 KB).
///
/// "Bounded" is enforced by short-circuiting the row when the running
/// minimum exceeds `budget`. That keeps worst-case O(n*budget) when
/// the budget is small (our use case: ~1% of quote length).
#[must_use]
#[allow(clippy::cast_possible_truncation)]
pub fn bounded_levenshtein(a: &str, b: &str, budget: u32) -> Option<u32> {
    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();

    let a_len = u32_from_usize(a_chars.len());
    let b_len = u32_from_usize(b_chars.len());

    if a_len.abs_diff(b_len) > budget {
        return None;
    }
    if a_chars.is_empty() {
        return (b_len <= budget).then_some(b_len);
    }
    if b_chars.is_empty() {
        return (a_len <= budget).then_some(a_len);
    }

    let mut prev: Vec<u32> = (0..=a_len).collect();
    let mut curr: Vec<u32> = vec![0; a_chars.len() + 1];

    for (j, b_ch) in b_chars.iter().enumerate() {
        curr[0] = u32_from_usize(j + 1);
        let mut row_min = curr[0];
        for (i, a_ch) in a_chars.iter().enumerate() {
            let cost = u32::from(a_ch != b_ch);
            let del = prev[i + 1].saturating_add(1);
            let ins = curr[i].saturating_add(1);
            let sub = prev[i].saturating_add(cost);
            curr[i + 1] = del.min(ins).min(sub);
            row_min = row_min.min(curr[i + 1]);
        }
        if row_min > budget {
            return None;
        }
        std::mem::swap(&mut prev, &mut curr);
    }

    let dist = prev[a_chars.len()];
    if dist <= budget { Some(dist) } else { None }
}

/// Saturating-cast a `usize` into a `u32`. Quote spans are small (< 4 GB
/// of characters by construction), so this is exact in practice; the
/// fallback `u32::MAX` exists only to satisfy clippy's truncation lint.
#[inline]
fn u32_from_usize(n: usize) -> u32 {
    u32::try_from(n).unwrap_or(u32::MAX)
}

/// Check whether the (already-normalized) `quote` appears as a substring
/// of (already-normalized) `source` within the configured fuzz budget.
///
/// Strategy:
///
/// 1. Exact substring hit (`source.contains(quote)`) → [`QuoteMatch::Exact`].
/// 2. Else, scan every length-`q` window of `source` (`q = quote.chars().count()`)
///    and compute bounded Levenshtein. First hit within budget is
///    reported as [`QuoteMatch::Fuzzy`]. The window slides one character
///    at a time, so off-by-one whitespace errors get caught.
/// 3. No hit → [`QuoteMatch::NotFound`].
///
/// The budget is `quote.chars().count() / 100 * fuzz_per_100_chars`,
/// rounded down (so quotes < 100 chars admit zero fuzz).
#[must_use]
pub fn is_quote_present(
    normalized_quote: &str,
    normalized_source: &str,
    fuzz_per_100_chars: u32,
) -> QuoteMatch {
    if normalized_quote.is_empty() {
        // An empty quote vacuously matches; callers should filter these
        // out earlier (extract_quote_citations does — empty quotes can't
        // satisfy the `[^"\r\n]+` regex), but be defensive.
        return QuoteMatch::Exact;
    }
    if normalized_source.contains(normalized_quote) {
        return QuoteMatch::Exact;
    }

    let q_len = normalized_quote.chars().count();
    let budget = u32_from_usize(q_len) / 100 * fuzz_per_100_chars;
    if budget == 0 {
        return QuoteMatch::NotFound;
    }

    let s_chars: Vec<char> = normalized_source.chars().collect();
    let budget_usize = budget as usize;
    if s_chars.len() + budget_usize <= q_len {
        return QuoteMatch::NotFound;
    }

    // Slide a window over the source. Each window is `q_len` chars, but
    // we also try `q_len ± budget` so insertions/deletions get covered.
    let min_window = q_len.saturating_sub(budget_usize);
    let max_window = q_len + budget_usize;

    for start in 0..s_chars.len() {
        for window_len in min_window..=max_window {
            let end = start + window_len;
            if end > s_chars.len() {
                break;
            }
            let candidate: String = s_chars[start..end].iter().collect();
            if let Some(d) = bounded_levenshtein(normalized_quote, &candidate, budget) {
                return QuoteMatch::Fuzzy {
                    distance: d,
                    budget,
                };
            }
        }
    }

    QuoteMatch::NotFound
}

/// Convenience: extract pairs from `body`, then for each pair load the
/// source content via `load_source` and verify the quote.
///
/// `load_source` returns `Some(content)` when the src-id resolves to a
/// known source, `None` when the id is unknown. Unknown ids surface as
/// `verified = false` with a distinct `kind` so callers can phrase the
/// failure precisely ("source not found" vs "quote not in source").
#[must_use]
pub fn verify_body_quotes<F>(
    body: &str,
    fuzz_per_100_chars: u32,
    mut load_source: F,
) -> Vec<QuoteVerification>
where
    F: FnMut(&str) -> Option<String>,
{
    extract_quote_citations(body)
        .into_iter()
        .map(|cite| {
            let normalized_quote = normalize_for_match(&cite.quote);
            let outcome = load_source(&cite.src_id).map_or(
                VerificationKind::SourceNotFound,
                |content| {
                    let normalized_source = normalize_for_match(&content);
                    match is_quote_present(
                        &normalized_quote,
                        &normalized_source,
                        fuzz_per_100_chars,
                    ) {
                        QuoteMatch::Exact => VerificationKind::Verified,
                        QuoteMatch::Fuzzy { distance, budget } => {
                            VerificationKind::VerifiedFuzzy { distance, budget }
                        }
                        QuoteMatch::NotFound => VerificationKind::QuoteNotInSource,
                    }
                },
            );
            QuoteVerification {
                quote: cite.quote,
                src_id: cite.src_id,
                offset: cite.offset,
                outcome,
            }
        })
        .collect()
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QuoteVerification {
    pub quote: String,
    pub src_id: String,
    pub offset: usize,
    pub outcome: VerificationKind,
}

impl QuoteVerification {
    #[must_use]
    pub const fn is_verified(&self) -> bool {
        matches!(
            self.outcome,
            VerificationKind::Verified | VerificationKind::VerifiedFuzzy { .. }
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VerificationKind {
    /// Exact substring match found in the normalized source.
    Verified,
    /// Bounded-Levenshtein match within the allowed edit budget.
    VerifiedFuzzy { distance: u32, budget: u32 },
    /// The src-id does not resolve to any known source — this is a
    /// citation problem, not a quote problem, and is surfaced as a
    /// verification failure for the caller to phrase appropriately.
    SourceNotFound,
    /// The source loaded fine, but the quote is not present (even after
    /// normalization and fuzz allowance).
    QuoteNotInSource,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_pairs_quote_then_bracket() {
        let body = r#"He said "the answer is 42" [src-abc] and moved on."#;
        let pairs = extract_quote_citations(body);
        assert_eq!(pairs.len(), 1);
        assert_eq!(pairs[0].quote, "the answer is 42");
        assert_eq!(pairs[0].src_id, "src-abc");
    }

    #[test]
    fn extract_pairs_bracket_then_quote() {
        // The bracket-then-quote shape requires the citation to sit
        // directly before the quote, separated only by whitespace —
        // anything in between (e.g. `notes:`) means the citation isn't
        // anchored to the following quote.
        let body = r#"[src-xyz] "raft uses a leader" — full stop."#;
        let pairs = extract_quote_citations(body);
        assert_eq!(pairs.len(), 1);
        assert_eq!(pairs[0].quote, "raft uses a leader");
        assert_eq!(pairs[0].src_id, "src-xyz");
    }

    #[test]
    fn extract_pairs_quote_then_paren() {
        let body = r#"Per "the spec" (src-spec1), this is required."#;
        let pairs = extract_quote_citations(body);
        assert_eq!(pairs.len(), 1);
        assert_eq!(pairs[0].quote, "the spec");
        assert_eq!(pairs[0].src_id, "src-spec1");
    }

    #[test]
    fn extract_pairs_smart_quotes_are_folded() {
        // U+201C / U+201D
        let body = "He said \u{201c}hello world\u{201d} [src-abc].";
        let pairs = extract_quote_citations(body);
        assert_eq!(pairs.len(), 1);
        assert_eq!(pairs[0].quote, "hello world");
    }

    #[test]
    fn extract_pairs_yields_multiple_in_order() {
        let body = r#"
            "first claim" [src-a].
            And later, [src-b] "second claim".
            Finally "third claim" (src-c)."#;
        let pairs = extract_quote_citations(body);
        let quotes: Vec<&str> = pairs.iter().map(|p| p.quote.as_str()).collect();
        // All three patterns hit; order is per-pattern (bracket-after,
        // bracket-before, paren) which is fine for downstream summary.
        assert!(quotes.contains(&"first claim"));
        assert!(quotes.contains(&"second claim"));
        assert!(quotes.contains(&"third claim"));
        assert_eq!(pairs.len(), 3);
    }

    #[test]
    fn extract_skips_quotes_without_adjacent_citation() {
        let body = r#"He said "no citation here" and that was that. [src-abc] is alone."#;
        let pairs = extract_quote_citations(body);
        assert!(pairs.is_empty());
    }

    #[test]
    fn normalize_collapses_whitespace_and_lowercases() {
        let input = "  Hello   World\nfoo\tbar  ";
        assert_eq!(normalize_for_match(input), "hello world foo bar");
    }

    #[test]
    fn normalize_strips_markdown_emphasis() {
        assert_eq!(
            normalize_for_match("the **bold** and _italic_ and `code`"),
            "the bold and italic and code"
        );
    }

    #[test]
    fn normalize_folds_smart_quotes() {
        let s = "\u{201c}quoted\u{201d}";
        assert_eq!(normalize_for_match(s), "\"quoted\"");
    }

    #[test]
    fn bounded_levenshtein_exact() {
        assert_eq!(bounded_levenshtein("abc", "abc", 0), Some(0));
    }

    #[test]
    fn bounded_levenshtein_under_budget() {
        // one substitution
        assert_eq!(bounded_levenshtein("abc", "abd", 1), Some(1));
        // one insertion
        assert_eq!(bounded_levenshtein("abc", "abcd", 1), Some(1));
        // one deletion
        assert_eq!(bounded_levenshtein("abc", "ac", 1), Some(1));
    }

    #[test]
    fn bounded_levenshtein_over_budget() {
        assert_eq!(bounded_levenshtein("abcdef", "uvwxyz", 1), None);
    }

    #[test]
    fn is_quote_present_exact_match() {
        let src = "the quick brown fox jumps over the lazy dog";
        assert_eq!(
            is_quote_present("brown fox", src, 1),
            QuoteMatch::Exact
        );
    }

    #[test]
    fn is_quote_present_under_fuzz() {
        // 100-char quote with one substitution (1% fuzz allowed).
        let original = "a".repeat(100);
        let mut quote = original.clone();
        quote.replace_range(50..51, "b");
        let result = is_quote_present(&quote, &original, 1);
        assert!(matches!(result, QuoteMatch::Fuzzy { .. }), "got {result:?}");
    }

    #[test]
    fn is_quote_present_exceeds_fuzz() {
        let original = "alpha bravo charlie delta echo";
        // Garbled — many substitutions, < 100 chars so budget is 0.
        let quote = "alphz brovi charlee dxlta xcho";
        assert_eq!(
            is_quote_present(quote, original, 1),
            QuoteMatch::NotFound
        );
    }

    #[test]
    fn is_quote_present_short_quote_no_fuzz() {
        // Short quote (< 100 chars) gets budget 0 — must be exact.
        let result = is_quote_present("hello", "hxllo world", 1);
        assert_eq!(result, QuoteMatch::NotFound);
    }

    #[test]
    fn verify_body_quotes_routes_through_loader() {
        // One quote per line so the bracket-then-quote regex doesn't
        // re-pair an already-cited quote with the *next* line's
        // citation. Real answers tend to put each quote on its own
        // sentence anyway.
        let body = "\"the answer is 42\" [src-good]. Then later, \
                    \"fabricated text\" [src-good]. Lastly, \
                    \"anything\" [src-missing].";
        let verifications = verify_body_quotes(body, 1, |id| {
            (id == "src-good").then(|| "the answer is 42 etc".to_string())
        });
        assert_eq!(verifications.len(), 3, "got: {verifications:?}");
        let by_quote: std::collections::HashMap<&str, &VerificationKind> = verifications
            .iter()
            .map(|v| (v.quote.as_str(), &v.outcome))
            .collect();
        assert!(matches!(by_quote["the answer is 42"], VerificationKind::Verified));
        assert!(matches!(
            by_quote["fabricated text"],
            VerificationKind::QuoteNotInSource
        ));
        assert!(matches!(
            by_quote["anything"],
            VerificationKind::SourceNotFound
        ));
    }

    #[test]
    fn quote_spanning_paragraph_break_verifies_after_normalization() {
        let source = "Lorem ipsum.\n\nThe answer is forty-two\n   and we should be confident.";
        let body = r#"They said "the answer is forty-two and we should be confident" [src-x]."#;
        let pairs = extract_quote_citations(body);
        assert_eq!(pairs.len(), 1);
        let nq = normalize_for_match(&pairs[0].quote);
        let ns = normalize_for_match(source);
        assert!(matches!(is_quote_present(&nq, &ns, 1), QuoteMatch::Exact));
    }
}
