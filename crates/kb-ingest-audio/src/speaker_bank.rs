//! Online speaker assignment with centroid updates.
//!
//! Replacement for `pyannote_rs::EmbeddingManager`. The upstream version
//! freezes each speaker's centroid to the *first* embedding it sees, so
//! a noisy or partial-utterance first segment becomes a permanent
//! anchor; subsequent good embeddings of the same speaker fail to
//! match the noisy anchor at any reasonable cosine threshold and get
//! assigned new speaker IDs. On the LiveRamp 75-min recording this
//! over-fragments a 6-speaker meeting into 28 distinct IDs.
//!
//! Our `SpeakerBank` keeps a running mean per speaker, so each new
//! matched embedding pulls the centroid toward a denoised consensus.
//! Calibrated threshold for `wespeaker-voxceleb-resnet34-LM` is around
//! 0.35–0.45 cosine similarity.

#[derive(Debug, Default)]
pub struct SpeakerBank {
    centroids: Vec<Vec<f32>>,
    counts: Vec<usize>,
    threshold: f32,
}

impl SpeakerBank {
    #[must_use]
    pub const fn new(threshold: f32) -> Self {
        Self {
            centroids: Vec::new(),
            counts: Vec::new(),
            threshold,
        }
    }

    /// Find the best matching speaker (cosine similarity above
    /// `threshold`) and update its centroid, or register a new speaker.
    /// Returns the speaker id.
    pub fn assign(&mut self, embedding: &[f32]) -> usize {
        let mut best: Option<(usize, f32)> = None;
        for (i, c) in self.centroids.iter().enumerate() {
            let sim = cosine_similarity(c, embedding);
            if sim > self.threshold && best.is_none_or(|(_, b)| sim > b) {
                best = Some((i, sim));
            }
        }
        if let Some((i, _)) = best {
            self.update_centroid(i, embedding);
            i
        } else {
            self.centroids.push(embedding.to_vec());
            self.counts.push(1);
            self.centroids.len() - 1
        }
    }

    fn update_centroid(&mut self, i: usize, embedding: &[f32]) {
        let n = self.counts[i] as f32;
        let new_n = n + 1.0;
        for (c, &e) in self.centroids[i].iter_mut().zip(embedding.iter()) {
            *c = c.mul_add(n, e) / new_n;
        }
        self.counts[i] += 1;
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.centroids.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.centroids.is_empty()
    }

    /// Iteratively merge centroid pairs whose cosine similarity exceeds
    /// `merge_threshold`. Each merge is weighted by per-speaker segment
    /// count, so a heavily-attested cluster pulls a smaller one toward
    /// itself rather than the reverse.
    ///
    /// Returns a remapping `old_speaker_id → final_speaker_id` where
    /// final ids are renumbered sequentially from 0. Use this to rewrite
    /// labels that were assigned during the (greedy, online) `assign`
    /// phase — the second pass collapses ghost speakers that were
    /// created from noisy initial embeddings before the real centroid
    /// stabilized.
    #[allow(clippy::needless_range_loop)] // iterating two indices into self.centroids
    pub fn agglomerative_merge(&mut self, merge_threshold: f32) -> Vec<usize> {
        let n = self.centroids.len();
        let mut alive = vec![true; n];
        // merge_target[i] = id of the cluster i has been absorbed into
        // (or i itself if it's still its own root).
        let mut merge_target: Vec<usize> = (0..n).collect();

        loop {
            let mut best: Option<(usize, usize, f32)> = None;
            for i in 0..n {
                if !alive[i] {
                    continue;
                }
                for j in (i + 1)..n {
                    if !alive[j] {
                        continue;
                    }
                    let sim = cosine_similarity(&self.centroids[i], &self.centroids[j]);
                    if sim > merge_threshold && best.is_none_or(|(_, _, b)| sim > b) {
                        best = Some((i, j, sim));
                    }
                }
            }
            let Some((i, j, _)) = best else {
                break;
            };
            // Weighted merge of j into i. Take ownership of j's
            // centroid first so the borrow checker is happy with the
            // simultaneous mutable+immutable borrow on self.centroids.
            let ni = self.counts[i] as f32;
            let nj = self.counts[j] as f32;
            let total = ni + nj;
            let j_centroid = std::mem::take(&mut self.centroids[j]);
            for (c_i, c_j) in self.centroids[i].iter_mut().zip(j_centroid.iter()) {
                *c_i = c_i.mul_add(ni, c_j * nj) / total;
            }
            self.counts[i] += self.counts[j];
            alive[j] = false;
            for t in &mut merge_target {
                if *t == j {
                    *t = i;
                }
            }
        }

        // Path-compress: follow chains to terminal roots.
        for orig in 0..n {
            let mut t = merge_target[orig];
            while merge_target[t] != t {
                t = merge_target[t];
            }
            merge_target[orig] = t;
        }

        // Renumber surviving roots sequentially from 0 in their order
        // of first appearance.
        let mut renumber: std::collections::HashMap<usize, usize> =
            std::collections::HashMap::new();
        let mut remapping = vec![0_usize; n];
        for orig in 0..n {
            let root = merge_target[orig];
            let next_id = renumber.len();
            let new_id = *renumber.entry(root).or_insert(next_id);
            remapping[orig] = new_id;
        }
        remapping
    }

    /// Reabsorb clusters with fewer than `min_count` matched segments
    /// into the nearest *larger* surviving cluster, regardless of
    /// cosine similarity. Used as a finishing pass after
    /// `agglomerative_merge` to clear out the long tail of one-off
    /// "ghost" clusters that the principled merge couldn't justify
    /// collapsing on similarity alone but which clearly aren't
    /// independent speakers in a multi-hour meeting.
    ///
    /// `existing` is the remapping returned by an earlier call to
    /// `agglomerative_merge` (so the small/large bookkeeping is done
    /// over the post-merge cluster set, not the raw greedy ids).
    /// Returns a fresh remapping `old_speaker_id → final_speaker_id`,
    /// composed with `existing`.
    #[allow(clippy::needless_range_loop)] // iterating index into self.centroids
    pub fn reabsorb_small_clusters(
        &mut self,
        existing: &[usize],
        min_count: usize,
    ) -> Vec<usize> {
        // Aggregate centroid + count per surviving cluster id.
        let n = self.centroids.len();
        let mut new_count = std::collections::HashMap::<usize, usize>::new();
        let mut new_centroid = std::collections::HashMap::<usize, Vec<f32>>::new();
        for orig in 0..n {
            let new_id = existing[orig];
            *new_count.entry(new_id).or_insert(0) += self.counts[orig];
            // Centroid for the post-merge id is the count-weighted mean
            // of centroids that mapped into it. We average lazily here
            // by accumulating sums and dividing at the end.
            let entry = new_centroid
                .entry(new_id)
                .or_insert_with(|| vec![0.0_f32; self.centroids[orig].len()]);
            let count = self.counts[orig] as f32;
            for (acc, &c) in entry.iter_mut().zip(self.centroids[orig].iter()) {
                *acc = c.mul_add(count, *acc);
            }
        }
        // Finalize means.
        for (id, sum) in &mut new_centroid {
            let count = new_count[id] as f32;
            if count > 0.0 {
                for v in sum.iter_mut() {
                    *v /= count;
                }
            }
        }

        // For each "small" cluster, find the closest "large" cluster.
        let small: Vec<usize> = new_count
            .iter()
            .filter_map(|(&id, &c)| (c < min_count).then_some(id))
            .collect();
        let large: Vec<usize> = new_count
            .iter()
            .filter_map(|(&id, &c)| (c >= min_count).then_some(id))
            .collect();
        let mut absorb: std::collections::HashMap<usize, usize> =
            std::collections::HashMap::new();
        if !large.is_empty() {
            for s in &small {
                let s_centroid = &new_centroid[s];
                let target = large
                    .iter()
                    .max_by(|a, b| {
                        let sa = cosine_similarity(s_centroid, &new_centroid[a]);
                        let sb = cosine_similarity(s_centroid, &new_centroid[b]);
                        sa.partial_cmp(&sb).unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .copied();
                if let Some(t) = target {
                    absorb.insert(*s, t);
                }
            }
        }

        // Compose existing + absorb, then renumber sequentially.
        let mut renumber: std::collections::HashMap<usize, usize> =
            std::collections::HashMap::new();
        let mut remapping = vec![0_usize; n];
        for orig in 0..n {
            let after_merge = existing[orig];
            let final_id = absorb.get(&after_merge).copied().unwrap_or(after_merge);
            let next_id = renumber.len();
            let new_id = *renumber.entry(final_id).or_insert(next_id);
            remapping[orig] = new_id;
        }
        remapping
    }
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0_f32;
    let mut na = 0.0_f32;
    let mut nb = 0.0_f32;
    for (&x, &y) in a.iter().zip(b.iter()) {
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    let denom = na.sqrt() * nb.sqrt();
    if denom == 0.0 {
        0.0
    } else {
        dot / denom
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn cosine_orthogonal_is_zero() {
        let a = [1.0_f32, 0.0, 0.0];
        let b = [0.0_f32, 1.0, 0.0];
        assert!((cosine_similarity(&a, &b)).abs() < 1e-6);
    }

    #[test]
    fn cosine_parallel_is_one() {
        let a = [1.0_f32, 2.0, 3.0];
        let b = [2.0_f32, 4.0, 6.0];
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn assign_creates_new_speaker_when_below_threshold() {
        let mut bank = SpeakerBank::new(0.5);
        let id_a = bank.assign(&[1.0, 0.0, 0.0]);
        let id_b = bank.assign(&[0.0, 1.0, 0.0]);
        assert_ne!(id_a, id_b);
        assert_eq!(bank.len(), 2);
    }

    #[test]
    fn assign_merges_similar_speakers() {
        let mut bank = SpeakerBank::new(0.5);
        let id1 = bank.assign(&[1.0, 0.0, 0.0]);
        let id2 = bank.assign(&[0.99, 0.01, 0.0]); // very similar
        assert_eq!(id1, id2);
        assert_eq!(bank.len(), 1);
    }

    #[test]
    fn centroid_drifts_toward_mean_as_more_match() {
        let mut bank = SpeakerBank::new(0.5);
        // Anchor at a slightly noisy direction.
        let _ = bank.assign(&[1.0, 0.1, 0.0]);
        // Several "clean" matches should pull the centroid toward [1, 0, 0].
        for _ in 0..50 {
            let _ = bank.assign(&[1.0, 0.0, 0.0]);
        }
        // Centroid's y-component should be ~ 0.1/51 ≈ 0.002.
        assert!(bank.centroids[0][1].abs() < 0.01);
        assert_eq!(bank.len(), 1);
    }

    #[test]
    fn agglomerative_merge_collapses_ghost_speakers() {
        // Greedy phase with a strict threshold: a and b are similar
        // (cosine ≈ 0.9) but the threshold of 0.95 splits them into
        // separate clusters. c is orthogonal to both.
        let mut bank = SpeakerBank::new(0.95);
        let id_a = bank.assign(&[1.0_f32, 0.0, 0.0]);
        let id_b = bank.assign(&[0.9_f32, 0.4359, 0.0]); // sim ≈ 0.9 with a
        let id_c = bank.assign(&[0.0_f32, 1.0, 0.0]);
        assert_eq!(bank.len(), 3, "greedy strict threshold creates ghost");

        // Second pass merges centroids that are actually similar.
        let remap = bank.agglomerative_merge(0.85);
        assert_eq!(remap[id_a], remap[id_b], "ghost collapsed into real");
        assert_ne!(remap[id_a], remap[id_c], "distinct speakers stay distinct");
        let surviving: std::collections::BTreeSet<usize> = remap.iter().copied().collect();
        assert_eq!(surviving.into_iter().collect::<Vec<_>>(), vec![0, 1]);
    }

    #[test]
    fn agglomerative_merge_no_op_below_threshold() {
        let mut bank = SpeakerBank::new(0.999);
        let _ = bank.assign(&[1.0, 0.0, 0.0]);
        let _ = bank.assign(&[0.0, 1.0, 0.0]);
        let _ = bank.assign(&[0.0, 0.0, 1.0]);

        let remap = bank.agglomerative_merge(0.999);
        // No merges happen — IDs stay distinct, just renumbered to [0, 1, 2].
        assert_eq!(remap.len(), 3);
        let surviving: std::collections::BTreeSet<usize> = remap.iter().copied().collect();
        assert_eq!(surviving.into_iter().collect::<Vec<_>>(), vec![0, 1, 2]);
    }

    #[test]
    fn reabsorb_pulls_singletons_into_dominant_cluster() {
        // Build a bank where one cluster is heavily attested and a
        // couple of singletons exist around the embedding space.
        let mut bank = SpeakerBank::new(0.999);
        let _ = bank.assign(&[1.0_f32, 0.0, 0.0]);
        for _ in 0..30 {
            let _ = bank.assign(&[1.0_f32, 0.0, 0.0]); // dominant cluster
        }
        // Two singleton ghosts that aren't actually similar enough to
        // merge under the principled threshold (sim ≈ 0.7 with the
        // dominant centroid, far from same-speaker territory).
        let _ = bank.assign(&[0.7_f32, 0.7, 0.0]);
        let _ = bank.assign(&[0.5_f32, 0.5, 0.7]);
        assert_eq!(bank.len(), 3);

        // Identity merge (no real merging — keep all 3 ids).
        let merged = bank.agglomerative_merge(0.999);
        let surviving: std::collections::BTreeSet<usize> = merged.iter().copied().collect();
        assert_eq!(surviving.len(), 3);

        // Reabsorb anything with fewer than 5 segments — both
        // singletons fold into the dominant cluster.
        let final_remap = bank.reabsorb_small_clusters(&merged, 5);
        let survivors: std::collections::BTreeSet<usize> = final_remap.iter().copied().collect();
        assert_eq!(survivors.into_iter().collect::<Vec<_>>(), vec![0]);
    }

    #[test]
    fn agglomerative_merge_weighted_by_count() {
        // 50 samples on speaker A, 1 noisy sample on speaker B, then the
        // merge should pull the result toward A (not midway).
        let mut bank = SpeakerBank::new(0.999);
        // Build A with weight 50 by allowing matches.
        let _ = bank.assign(&[1.0, 0.0, 0.0]);
        for _ in 0..49 {
            let _ = bank.assign(&[1.0, 0.0, 0.0]); // matches at sim=1
        }
        // B is a single noisy sample, just barely above the merge cutoff.
        let _ = bank.assign(&[0.95, 0.31, 0.0]); // sim ≈ 0.95
        assert_eq!(bank.len(), 2);

        let _ = bank.agglomerative_merge(0.9);
        // After merge, the centroid should be very close to A (50 votes
        // dominate 1 vote).
        assert!(bank.centroids[0][0] > 0.99);
        assert!(bank.centroids[0][1] < 0.01);
    }
}
