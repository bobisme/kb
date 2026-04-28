//! Speech-segment detection via the pyannote-segmentation-3.0 ONNX model.
//!
//! This is a from-scratch replacement for `pyannote_rs::get_segments`,
//! which has a long-audio bug where (a) per-window state drift produces
//! empty-slice segments that crash downstream embedding, and (b) some
//! recordings produce zero transitions and therefore zero segments.
//!
//! Our version:
//! - Tracks frame offset *per window* (not as a running accumulator),
//!   so position never drifts.
//! - Guards against zero-length slices when emitting.
//! - Flushes a trailing speech segment at end-of-audio (pyannote-rs
//!   only emits on speech→silence transitions, losing the last segment
//!   when audio ends mid-utterance).
//!
//! Output shape matches `pyannote_rs::Segment` so the rest of the
//! pipeline (embedding extraction + speaker clustering) is unchanged.

use std::path::Path;

use ort::session::{Session, builder::GraphOptimizationLevel};
use ort::value::TensorRef;

/// Per-frame stride of the segmentation model's output, in samples
/// at 16 kHz. ~16.875 ms per output frame.
const FRAME_STRIDE_SAMPLES: usize = 270;

/// Receptive-field offset of the first frame, in samples at 16 kHz.
/// Matches pyannote-rs's value (which matches upstream pyannote).
const FRAME_START_SAMPLES: usize = 721;

/// Window length the segmentation model is trained on, in seconds.
const WINDOW_SECONDS: usize = 10;

#[derive(Debug, Clone)]
pub struct SpeechSegment {
    pub start_seconds: f64,
    pub end_seconds: f64,
    pub samples: Vec<i16>,
}

/// Run the pyannote-segmentation-3.0 ONNX model over `samples` in
/// 10-second windows and return all detected speech regions.
///
/// # Errors
///
/// I/O and ONNX Runtime errors from session creation, inference, or
/// tensor extraction.
pub fn segment_speech(
    samples: &[i16],
    sample_rate: u32,
    model_path: &Path,
) -> Result<Vec<SpeechSegment>, anyhow::Error> {
    let mut session = Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_intra_threads(1)?
        .with_inter_threads(1)?
        .commit_from_file(model_path)?;

    let window_samples = (sample_rate as usize) * WINDOW_SECONDS;
    let n_windows = samples.len().div_ceil(window_samples);
    let padded_len = n_windows * window_samples;
    let mut padded = vec![0i16; padded_len];
    padded[..samples.len()].copy_from_slice(samples);

    let mut segments = Vec::new();
    let mut is_speech = false;
    let mut start_sample: usize = 0;

    for window_idx in 0..n_windows {
        let window_start = window_idx * window_samples;
        let window = &padded[window_start..window_start + window_samples];

        let array = ndarray::Array1::from_iter(window.iter().map(|&x| f32::from(x)));
        let array = array
            .view()
            .insert_axis(ndarray::Axis(0))
            .insert_axis(ndarray::Axis(1));

        let inputs = ort::inputs![TensorRef::from_array_view(array.into_dyn())?];
        let outputs = session.run(inputs)?;
        let tensor = outputs
            .get("output")
            .ok_or_else(|| anyhow::anyhow!("segmentation output tensor not found"))?;
        let (shape, data) = tensor.try_extract_tensor::<f32>()?;

        // Expected shape: [batch=1, n_frames, n_classes=7].
        let n_frames = shape[1] as usize;
        let n_classes = shape[2] as usize;

        for frame_idx in 0..n_frames {
            let row = &data[frame_idx * n_classes..(frame_idx + 1) * n_classes];
            let max_class = argmax(row);
            // Class 0 is non-speech; classes 1..=6 are speech (single
            // speaker or overlap combinations).
            let is_voice = max_class != 0;

            // Center of this frame in absolute samples.
            let frame_sample =
                window_start + FRAME_START_SAMPLES + frame_idx * FRAME_STRIDE_SAMPLES;

            if is_voice {
                if !is_speech {
                    start_sample = frame_sample;
                    is_speech = true;
                }
            } else if is_speech {
                push_segment(
                    &mut segments,
                    &padded,
                    samples.len(),
                    sample_rate,
                    start_sample,
                    frame_sample,
                );
                is_speech = false;
            }
        }
    }

    // Trailing speech segment that runs into end-of-audio.
    if is_speech {
        push_segment(
            &mut segments,
            &padded,
            samples.len(),
            sample_rate,
            start_sample,
            samples.len(),
        );
    }

    Ok(segments)
}

fn argmax(row: &[f32]) -> usize {
    // Hand-rolled rather than `Iterator::max_by` so that ties resolve to
    // the *first* index. That matters at this site because class 0 is
    // "non-speech": if the model is uncertain (all logits ≈ equal),
    // returning the first index biases toward silence rather than
    // arbitrarily inflating speech detections.
    let mut best_idx = 0;
    let mut best_val = f32::NEG_INFINITY;
    for (i, &v) in row.iter().enumerate() {
        if v > best_val {
            best_idx = i;
            best_val = v;
        }
    }
    best_idx
}

fn push_segment(
    segments: &mut Vec<SpeechSegment>,
    padded: &[i16],
    original_len: usize,
    sample_rate: u32,
    start_sample: usize,
    end_sample: usize,
) {
    // Clamp to the original (non-padded) audio bounds and skip empty
    // slices — both root causes of pyannote-rs's downstream crashes.
    let start = start_sample.min(original_len.saturating_sub(1));
    let end = end_sample.min(original_len);
    if end <= start {
        return;
    }
    segments.push(SpeechSegment {
        start_seconds: start_sample as f64 / f64::from(sample_rate),
        end_seconds: end_sample as f64 / f64::from(sample_rate),
        samples: padded[start..end].to_vec(),
    });
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn argmax_picks_largest() {
        assert_eq!(argmax(&[0.1, 0.5, 0.3]), 1);
        assert_eq!(argmax(&[0.0, 0.0, 0.0]), 0);
        assert_eq!(argmax(&[]), 0);
    }

    #[test]
    fn push_segment_skips_empty() {
        let padded = vec![0i16; 16_000];
        let mut out = Vec::new();
        push_segment(&mut out, &padded, 16_000, 16_000, 100, 100);
        assert!(out.is_empty(), "zero-length range must not push a segment");
    }

    #[test]
    fn push_segment_clamps_to_original_len() {
        let padded = vec![1i16; 32_000];
        let mut out = Vec::new();
        // Original audio is shorter than padded; emit using padded data
        // but clamp the slice indices to the original length.
        push_segment(&mut out, &padded, 16_000, 16_000, 0, 32_000);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].samples.len(), 16_000);
        // Timestamps reflect the *uncapped* frame offsets so downstream
        // code can see the model's view of the boundary.
        assert!((out[0].end_seconds - 2.0).abs() < 1e-9);
    }
}
