//! Pure ranking-quality metrics used by the golden eval harness.
//!
//! Each function takes a binary-relevance ranking (`true` = relevant /
//! expected, `false` = not relevant) and returns a score in `[0.0, 1.0]`.
//! Empty rankings or empty relevance vectors return `0.0` rather than
//! `NaN` so downstream JSON / markdown rendering never has to special-case
//! it.

/// Precision at K — fraction of the top-`k` ranked items that are relevant.
///
/// `ranking` is `true` for hits that match an expected source/concept and
/// `false` otherwise. `k` is clamped to `ranking.len()` so callers can
/// pass `5` against a 3-result ranking without panicking.
///
/// Returns `0.0` for an empty ranking or `k == 0`.
#[must_use]
#[allow(
    clippy::cast_precision_loss,
    reason = "k and hits are both small (< 100); precision loss is irrelevant for IR metrics"
)]
pub fn precision_at_k(ranking: &[bool], k: usize) -> f32 {
    if ranking.is_empty() || k == 0 {
        return 0.0;
    }
    let take = k.min(ranking.len());
    let hits = ranking[..take].iter().filter(|hit| **hit).count();
    // Use `k` (not `take`) as the denominator so a 2-of-3 result against
    // P@5 reports 0.4, not 0.67. This matches the standard IR convention
    // — short result lists are "missing" the rest, not "perfect".
    hits as f32 / k as f32
}

/// Mean reciprocal rank — `1 / r` where `r` is the 1-indexed position of
/// the first relevant item, or `0.0` if no relevant item is found.
#[must_use]
#[allow(
    clippy::cast_precision_loss,
    reason = "rank index is bounded by ranking length (< 1000 in practice)"
)]
pub fn mean_reciprocal_rank(ranking: &[bool]) -> f32 {
    for (i, hit) in ranking.iter().enumerate() {
        if *hit {
            return 1.0 / (i + 1) as f32;
        }
    }
    0.0
}

/// Normalized DCG at K with binary relevance.
///
/// `relevance[i]` is `1` if the `i`-th ranked item is relevant, else `0`.
/// With binary relevance, the ideal DCG is the same vector sorted
/// descending — i.e. all the `1`s packed at the top. Returns `0.0`
/// when there are no relevant items in the ranking (so the ideal DCG
/// is `0` and the ratio would be undefined).
#[must_use]
pub fn ndcg_at_k(relevance: &[u32], k: usize) -> f32 {
    if relevance.is_empty() || k == 0 {
        return 0.0;
    }
    let take = k.min(relevance.len());
    let dcg_actual = dcg(&relevance[..take]);
    // Build the ideal ranking: same multiset of relevance values, sorted
    // desc, truncated to `k`. With binary relevance this is a contiguous
    // run of 1s followed by 0s.
    let mut ideal = relevance.to_vec();
    ideal.sort_unstable_by(|a, b| b.cmp(a));
    ideal.truncate(take);
    let idcg = dcg(&ideal);
    if idcg <= 0.0 {
        // No relevant items at all (or k too small to capture any). Define
        // nDCG as 0 in this case rather than NaN.
        return 0.0;
    }
    // Final ratio is always in [0, 1] so the f64 -> f32 cast cannot
    // overflow; the precision loss is well below display rounding.
    #[allow(
        clippy::cast_possible_truncation,
        reason = "ratio is bounded [0, 1]; truncation cannot occur"
    )]
    let out = (dcg_actual / idcg) as f32;
    out
}

/// Standard DCG: `sum(rel_i / log2(i + 2))`, 0-indexed `i`.
#[allow(
    clippy::cast_precision_loss,
    reason = "i is bounded by ranking length (< 10_000); f64 mantissa easily covers it"
)]
fn dcg(relevance: &[u32]) -> f64 {
    relevance
        .iter()
        .enumerate()
        .map(|(i, rel)| {
            let denom = ((i + 2) as f64).log2();
            f64::from(*rel) / denom
        })
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32) {
        assert!(
            (a - b).abs() < 1e-4,
            "expected {b}, got {a} (diff {})",
            (a - b).abs()
        );
    }

    #[test]
    fn precision_perfect_ranking() {
        let ranking = vec![true, true, true, true, true];
        approx(precision_at_k(&ranking, 5), 1.0);
        approx(precision_at_k(&ranking, 3), 1.0);
    }

    #[test]
    fn precision_partial_ranking() {
        // 3 of top 5 are relevant -> 0.6.
        let ranking = vec![true, false, true, true, false];
        approx(precision_at_k(&ranking, 5), 0.6);
        approx(precision_at_k(&ranking, 3), 2.0 / 3.0);
        approx(precision_at_k(&ranking, 1), 1.0);
    }

    #[test]
    fn precision_short_ranking_uses_k_as_denominator() {
        // A 3-result list with 2 hits against P@5 should be 0.4, not 0.67.
        let ranking = vec![true, true, false];
        approx(precision_at_k(&ranking, 5), 0.4);
    }

    #[test]
    fn precision_no_results() {
        let ranking: Vec<bool> = Vec::new();
        approx(precision_at_k(&ranking, 5), 0.0);
        approx(precision_at_k(&[true, false], 0), 0.0);
    }

    #[test]
    fn mrr_first_position() {
        approx(mean_reciprocal_rank(&[true, false, false]), 1.0);
    }

    #[test]
    fn mrr_third_position() {
        approx(
            mean_reciprocal_rank(&[false, false, true, false]),
            1.0 / 3.0,
        );
    }

    #[test]
    fn mrr_no_relevant() {
        approx(mean_reciprocal_rank(&[false, false, false]), 0.0);
        approx(mean_reciprocal_rank(&[]), 0.0);
    }

    #[test]
    fn ndcg_perfect_ranking() {
        let relevance = vec![1, 1, 1, 0, 0];
        approx(ndcg_at_k(&relevance, 5), 1.0);
        approx(ndcg_at_k(&relevance, 10), 1.0);
    }

    #[test]
    fn ndcg_reversed_ranking_is_lower() {
        let perfect = vec![1, 1, 1, 0, 0];
        let reversed = vec![0, 0, 1, 1, 1];
        let p = ndcg_at_k(&perfect, 5);
        let r = ndcg_at_k(&reversed, 5);
        assert!(p > r, "perfect ({p}) should beat reversed ({r})");
        assert!(p >= 0.999_9, "perfect should be ~1.0, got {p}");
        assert!(r < 0.7, "reversed should be noticeably lower, got {r}");
    }

    #[test]
    fn ndcg_no_relevant_is_zero() {
        let relevance = vec![0, 0, 0];
        approx(ndcg_at_k(&relevance, 10), 0.0);
        approx(ndcg_at_k(&[], 10), 0.0);
    }

    #[test]
    fn ndcg_known_value() {
        // Single relevant item at position 2 (1-indexed):
        //   DCG  = 1/log2(3) = 0.6309...
        //   IDCG = 1/log2(2) = 1.0  (ideal is to put the 1 first)
        //   nDCG = 0.6309
        let relevance = vec![0, 1, 0];
        let got = ndcg_at_k(&relevance, 3);
        approx(got, 0.6309);
    }

    #[test]
    fn ndcg_partial_match() {
        // Top hit relevant, 3rd relevant -> nDCG is between MRR=1 and 1.0.
        // DCG  = 1/log2(2) + 1/log2(4) = 1 + 0.5 = 1.5
        // IDCG = 1/log2(2) + 1/log2(3) = 1 + 0.6309 = 1.6309
        let relevance = vec![1, 0, 1, 0];
        let got = ndcg_at_k(&relevance, 4);
        approx(got, 1.5 / 1.6309);
    }
}
