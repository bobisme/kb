//! End-to-end transcribe pipeline: decode → ASR → diarization → kbtx.

use std::path::{Path, PathBuf};

use kb_core::transcript::{BodyNode, RosterEntry, TranscriptDoc, Turn};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::decode::DecodedAudio;
use crate::models::{ModelHandles, ModelStore};

/// Same threshold the Python prototype uses; see notes/transcript-spec.md.
const INTRA_TURN_GAP_SECONDS: f32 = 10.0;
const MAX_TURN_DURATION_SECONDS: f32 = 300.0;
const DEFAULT_PAUSE_THRESHOLD_SECONDS: f32 = 5.0;

#[derive(Debug, Clone)]
pub struct TranscribeConfig {
    pub audio_path: PathBuf,
    /// Title written into the kbtx frontmatter.
    pub title: String,
    /// Recording date (YYYY-MM-DD).
    pub recording_date: String,
    /// Threshold for synthesized `> [pause: Ns]` action lines. Default 5s.
    pub pause_threshold_seconds: f32,
    /// Override model cache root. Default: ~/.kb/models.
    pub model_cache_root: Option<PathBuf>,
}

impl TranscribeConfig {
    pub fn new(
        audio_path: impl Into<PathBuf>,
        title: impl Into<String>,
        recording_date: impl Into<String>,
    ) -> Self {
        Self {
            audio_path: audio_path.into(),
            title: title.into(),
            recording_date: recording_date.into(),
            pause_threshold_seconds: DEFAULT_PAUSE_THRESHOLD_SECONDS,
            model_cache_root: None,
        }
    }
}

#[derive(Debug, Error)]
pub enum TranscribeError {
    #[error("audio decode failed: {0}")]
    Decode(#[from] crate::decode::DecodeError),
    #[error("model load failed: {0}")]
    Model(#[from] crate::models::ModelError),
    #[error("ASR failed: {0}")]
    Asr(String),
    #[error("diarization failed: {0}")]
    Diarization(String),
    #[error(transparent)]
    Io(#[from] std::io::Error),
}

/// Run the full pipeline. Returns the kbtx-rendered transcript markdown.
///
/// On first call, models are downloaded into the cache. Subsequent calls
/// hit the cache and only do the ASR + diarization work.
///
/// # Errors
///
/// Surfaces decode, model-load, ASR, and diarization errors from the
/// underlying pipeline stages.
pub fn transcribe(cfg: &TranscribeConfig) -> Result<String, TranscribeError> {
    let artifact = produce_artifact(cfg)?;
    Ok(render_artifact(&artifact, cfg))
}

/// Run ASR + diarization and return the raw segments. Cheap to serialize
/// and round-trip back into `render_artifact` for fast re-renders.
///
/// # Errors
///
/// Surfaces decode, model-load, ASR, and diarization errors from the
/// underlying pipeline stages.
pub fn produce_artifact(cfg: &TranscribeConfig) -> Result<TranscriptArtifact, TranscribeError> {
    init_ort_environment();
    let store = ModelStore::new(cfg.model_cache_root.clone())?;
    let handles = store.ensure_all()?;

    let audio = decode_audio(&cfg.audio_path)?;

    let asr_segments = run_asr(&audio, &handles)?;
    // pyannote-rs has a long-audio bug (segments never emitted, or
    // emitted with empty samples) that makes diarization unreliable on
    // anything past ~30s. Tolerate the failure so we still produce a
    // transcript — turns get merged into one per ASR segment with
    // synthesized per-segment speaker IDs (see fallback in
    // assign_speakers_and_merge).
    let diarization_segments = match run_diarization(&audio, &handles) {
        Ok(segs) => segs,
        Err(e) => {
            tracing::warn!(error = %e, "diarization failed; proceeding without speaker labels");
            Vec::new()
        }
    };

    let audio_filename = cfg
        .audio_path
        .file_name()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_default();

    Ok(TranscriptArtifact {
        duration_seconds: audio.duration_seconds,
        sample_rate: audio.sample_rate,
        asr_model: "whisper-large-v3-q5_0".into(),
        diarization_model: "pyannote-segmentation-3.0+wespeaker-resnet34-LM".into(),
        audio_filename,
        asr_segments,
        diarization_segments,
    })
}

/// Render an artifact to kbtx markdown. Pure function — no I/O, no model
/// access — so re-rendering is free once the artifact exists.
#[must_use]
pub fn render_artifact(artifact: &TranscriptArtifact, cfg: &TranscribeConfig) -> String {
    let turns = assign_speakers_and_merge(&artifact.asr_segments, &artifact.diarization_segments);
    let doc = build_kbtx_doc(cfg, artifact, &turns);
    kb_core::transcript::render(&doc)
}

fn decode_audio(path: &Path) -> Result<DecodedAudio, crate::decode::DecodeError> {
    crate::decode::decode_to_mono_16k(path)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AsrSegment {
    pub start_seconds: f32,
    pub end_seconds: f32,
    pub text: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiarizationSegment {
    pub start_seconds: f32,
    pub end_seconds: f32,
    /// `SPEAKER_NN` style label.
    pub speaker: String,
}

/// Raw, deterministic outputs of the ASR + diarization stages, before
/// turn-merging or kbtx rendering. Serializable so callers can cache it
/// to disk and re-render the kbtx without redoing the GPU work.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscriptArtifact {
    pub duration_seconds: f32,
    pub sample_rate: u32,
    pub asr_model: String,
    pub diarization_model: String,
    pub audio_filename: String,
    pub asr_segments: Vec<AsrSegment>,
    pub diarization_segments: Vec<DiarizationSegment>,
}

fn run_asr(audio: &DecodedAudio, handles: &ModelHandles) -> Result<Vec<AsrSegment>, TranscribeError> {
    use whisper_rs::{WhisperContext, WhisperContextParameters};

    // Process audio in independent chunks so a stuck-decoder pathology
    // in one chunk can't poison the rest of the recording. Empirically,
    // once whisper.cpp's decoder locks into a repetition loop in a
    // single `state.full()` call, it stays locked for every subsequent
    // 30s internal chunk — so we lost ~50 minutes of real conversation
    // after the decoder got stuck at 23:37 in the LiveRamp USB
    // recording. Chunking the call boundary breaks the lock.
    //
    // 120 seconds is large enough for whisper to use cross-chunk
    // context within a chunk (its internal stride is 30s) but small
    // enough that one stuck chunk only loses ~2 minutes.
    const CHUNK_SECONDS: f32 = 120.0;

    let model_path = handles.whisper.to_str().ok_or_else(|| {
        TranscribeError::Asr(format!(
            "whisper model path is not valid UTF-8: {}",
            handles.whisper.display()
        ))
    })?;

    tracing::info!(
        samples = audio.samples.len(),
        duration_s = audio.duration_seconds,
        chunk_seconds = CHUNK_SECONDS,
        "running whisper ASR"
    );
    let ctx = WhisperContext::new_with_params(model_path, WhisperContextParameters::default())
        .map_err(|e| TranscribeError::Asr(format!("failed to load whisper model: {e}")))?;
    let mut state = ctx
        .create_state()
        .map_err(|e| TranscribeError::Asr(format!("failed to create whisper state: {e}")))?;

    let chunk_samples = (audio.sample_rate as f32 * CHUNK_SECONDS) as usize;
    let total = audio.samples.len();
    let mut out = Vec::new();
    let mut chunk_start = 0_usize;
    let mut chunk_idx = 0_usize;

    while chunk_start < total {
        let chunk_end = (chunk_start + chunk_samples).min(total);
        let chunk = &audio.samples[chunk_start..chunk_end];
        let chunk_offset_seconds = chunk_start as f32 / audio.sample_rate as f32;

        let params = build_whisper_params();
        state
            .full(params, chunk)
            .map_err(|e| TranscribeError::Asr(format!("whisper inference failed at chunk {chunk_idx}: {e}")))?;

        let n = state
            .full_n_segments()
            .map_err(|e| TranscribeError::Asr(format!("failed to get segment count: {e}")))?;
        for i in 0..n {
            let text = state
                .full_get_segment_text(i)
                .map_err(|e| TranscribeError::Asr(format!("segment text error at {i}: {e}")))?;
            // whisper.cpp returns timestamps in 10 ms units, relative
            // to the start of the slice we passed in. Add the chunk
            // offset to get absolute timestamps.
            let t0 = state
                .full_get_segment_t0(i)
                .map_err(|e| TranscribeError::Asr(format!("segment t0 error at {i}: {e}")))?;
            let t1 = state
                .full_get_segment_t1(i)
                .map_err(|e| TranscribeError::Asr(format!("segment t1 error at {i}: {e}")))?;
            out.push(AsrSegment {
                start_seconds: chunk_offset_seconds + t0 as f32 / 100.0,
                end_seconds: chunk_offset_seconds + t1 as f32 / 100.0,
                text,
            });
        }

        chunk_start = chunk_end;
        chunk_idx += 1;
    }

    // Stuck-decoder post-filter. Beam search + temperature fallback
    // catch most repetition lock-in, but the model occasionally
    // produces a *grammatical* repeated sentence ("Then, once the user
    // click a field, the backend will run a query." × 50) that scores
    // fine on entropy/logprob and survives the in-decoder checks.
    //
    // Two shapes to defend against:
    //
    //   1. Within a single segment — long output containing a chunk
    //      that repeats many times. Catch via duplicated-window count.
    //
    //   2. Across consecutive segments — whisper emits each iteration
    //      as its own short segment, so the within-segment scan sees
    //      just one copy each. Catch via consecutive-text dedup.
    //
    // In both cases, drop the redundant text. The time range becomes a
    // gap, which beats a wall of false transcription.
    let before = out.len();
    out.retain(|s| !looks_like_decoder_repetition(&s.text));
    let dropped_within = before.saturating_sub(out.len());

    let before_dedup = out.len();
    out = collapse_long_runs(out, 4);
    let dropped_dup = before_dedup.saturating_sub(out.len());

    if dropped_within + dropped_dup > 0 {
        tracing::warn!(
            within = dropped_within,
            consecutive = dropped_dup,
            "dropped repetition-hallucinated ASR segments"
        );
    }

    tracing::info!(segments = out.len(), "whisper ASR done");
    Ok(out)
}

/// Build the whisper.cpp `FullParams` we use for each chunk. Factored
/// out because we rebuild fresh params per chunk (the API consumes
/// them) and want a single place to edit the knobs.
fn build_whisper_params<'a, 'b>() -> whisper_rs::FullParams<'a, 'b> {
    use whisper_rs::{FullParams, SamplingStrategy};

    // Beam search rather than greedy: explores multiple decoding
    // paths so the model can back out of repetition lock-in. Beam=5
    // matches faster-whisper / WhisperX defaults.
    let mut params = FullParams::new(SamplingStrategy::BeamSearch {
        beam_size: 5,
        patience: -1.0,
    });
    // Quiet stdout — we expose whisper progress via tracing only.
    params.set_print_special(false);
    params.set_print_progress(false);
    params.set_print_realtime(false);
    params.set_print_timestamps(false);
    // `set_language(None)` tells whisper.cpp to auto-detect. (Don't use
    // `set_detect_language(true)` — that flag tells whisper to *only*
    // detect language and skip transcription, which is why we were
    // getting 0 segments back.)
    params.set_language(None);

    // Anti-hallucination knobs (whisper.cpp main-binary defaults that
    // whisper-rs's `FullParams::new` doesn't apply for us):
    //   temperature_inc = 0.2  (enables fallback retries)
    //   entropy_thold   = 2.4  (triggers fallback on stuck decoder)
    //   logprob_thold   = -1.0 (triggers fallback on low-confidence)
    //   no_speech_thold = 0.6
    //   no_context      = true (each chunk decodes independently)
    params.set_temperature(0.0);
    params.set_temperature_inc(0.2);
    params.set_entropy_thold(2.4);
    params.set_logprob_thold(-1.0);
    params.set_no_speech_thold(0.6);
    params.set_no_context(true);
    params.set_suppress_blank(true);
    params.set_suppress_non_speech_tokens(true);

    // Style anchor for the decoder. whisper-large-v3-q5_0 sometimes
    // outputs unpunctuated lowercase ("uh architecture uh for the
    // segment building product from as a top level we have...") when
    // the prompt context is empty (`no_context=true`) and the audio is
    // continuous fast speech. An initial prompt of well-formatted
    // English biases the BPE tokenizer toward capitalization +
    // sentence punctuation. The text content doesn't appear in the
    // output — it's just a style hint. faster-whisper / WhisperX
    // similarly use a hidden prompt for this purpose.
    params.set_initial_prompt(
        "Hello. This is a recorded meeting. Multiple speakers will discuss \
         technical topics, ask questions, and share their thoughts.",
    );

    params
}

/// Collapse any run of `min_run` or more consecutive ASR segments
/// whose trimmed text is identical, keeping only the first.
///
/// This catches the cross-segment shape of a stuck whisper decoder:
/// the same phrase emitted as its own segment for 5/10/50 audio frames
/// in a row. Earlier passes used a length-based filter ("dedup
/// identical segments ≥ 60 chars"), but that misses short hallucinated
/// runs like "Okay. Okay. Okay. ..." (5 chars per segment) and
/// medium-length runs of grammatical clauses. A run-length condition
/// is more reliable: a real human might say "Yeah. Yeah." (run of 2)
/// or "Yeah. Yeah. Yeah." (run of 3) in genuine emphasis, but four-in-
/// a-row identical segments is essentially always a decoder loop.
fn collapse_long_runs(segs: Vec<AsrSegment>, min_run: usize) -> Vec<AsrSegment> {
    let n = segs.len();
    if n == 0 {
        return segs;
    }
    let mut out: Vec<AsrSegment> = Vec::with_capacity(n);
    let mut i = 0;
    while i < n {
        let cur = segs[i].text.trim().to_string();
        let mut j = i + 1;
        while j < n && segs[j].text.trim() == cur {
            j += 1;
        }
        let run_len = j - i;
        if run_len >= min_run {
            // Keep only the first segment of the run.
            out.push(segs[i].clone());
        } else {
            // Run is short enough to be plausible human emphasis;
            // keep every member.
            out.extend_from_slice(&segs[i..j]);
        }
        i = j;
    }
    out
}

/// Detect the within-segment shape of a stuck whisper decoder: a single
/// substring of meaningful length repeated more than once. Normal
/// English text doesn't reproduce 60-character chunks within a single
/// utterance, so even one duplicate window is enough signal.
fn looks_like_decoder_repetition(text: &str) -> bool {
    const WINDOW: usize = 60;
    let bytes = text.as_bytes();
    if bytes.len() < WINDOW * 2 {
        return false;
    }
    let mut counts: std::collections::HashMap<&[u8], usize> =
        std::collections::HashMap::new();
    for i in 0..=bytes.len() - WINDOW {
        let entry = counts.entry(&bytes[i..i + WINDOW]).or_insert(0);
        *entry += 1;
        if *entry > 1 {
            return true;
        }
    }
    false
}

/// Hook for any process-wide ONNX Runtime setup we want to do before
/// pyannote's models are loaded. Currently a no-op — empirically the
/// ort 2.0-rc.10 CUDA EP produces wrong segmentation outputs on long
/// audio (the per-window argmax saturates on one class, so the iterator
/// never sees a speech→silence transition and emits zero segments).
/// pyannote's models are small enough to run cheaply on CPU; whisper
/// keeps GPU acceleration via the `whisper-rs/cuda` feature regardless.
#[allow(clippy::missing_const_for_fn)]
fn init_ort_environment() {}

fn run_diarization(
    audio: &DecodedAudio,
    handles: &ModelHandles,
) -> Result<Vec<DiarizationSegment>, TranscribeError> {
    use pyannote_rs::EmbeddingExtractor;
    // Cosine threshold for the online (greedy) assign pass.
    // wespeaker-resnet34-LM produces same-speaker similarities in the
    // 0.4-0.6 range and cross-speaker similarities in the 0.1-0.3
    // range, but on meeting audio (cross-talk, mic distance variation,
    // codec compression) same-speaker scores can dip into the
    // borderline. We accept a slightly stricter assign threshold here
    // and rely on the second pass to pull legitimate splits back
    // together.
    const SEARCH_THRESHOLD: f32 = 0.4;
    // After the greedy pass the per-cluster centroids are denoised
    // (running mean of all matched embeddings). Two centroids that
    // ended up in different clusters because their *first* embeddings
    // were noisy may now look very similar. Merge any pair above this
    // threshold.
    const MERGE_THRESHOLD: f32 = 0.4;
    // Final cleanup pass: any cluster with fewer than this many matched
    // segments folds into the nearest *larger* cluster regardless of
    // similarity. Calibrated against the LiveRamp recordings: pinning
    // it at 20 collapsed real-but-quiet meeting participants who only
    // contributed ~10 utterances over 75 minutes; pinning it at 5 still
    // catches the singleton/double-segment ghosts that dominate the
    // long tail without erasing minor real speakers.
    const MIN_SEGMENTS_PER_REAL_SPEAKER: usize = 5;
    // wespeaker embeddings need a meaningful chunk of audio to
    // discriminate speakers reliably (the model card recommends ≥1 s).
    // Below that, embeddings are noisy and the speaker tally explodes.
    // fbank itself only needs ~25 ms but quality is what matters here.
    const MIN_SEGMENT_SAMPLES: usize = 16_000; // 1 s at 16 kHz.

    // pyannote-rs takes i16 samples; convert from our f32 mono PCM.
    let samples_i16: Vec<i16> = audio
        .samples
        .iter()
        .map(|&s| (s.clamp(-1.0, 1.0) * i16::MAX as f32) as i16)
        .collect();

    tracing::info!(samples = samples_i16.len(), "running pyannote diarization");
    let mut embedding_extractor = EmbeddingExtractor::new(&handles.embedding)
        .map_err(|e| TranscribeError::Diarization(format!("failed to load embedding model: {e:?}")))?;
    let mut speaker_bank = crate::speaker_bank::SpeakerBank::new(SEARCH_THRESHOLD);

    let segments = crate::segmentation::segment_speech(
        &samples_i16,
        audio.sample_rate,
        &handles.segmentation,
    )
    .map_err(|e| TranscribeError::Diarization(format!("segmentation failed: {e:?}")))?;

    let total_segments = segments.len();
    // Pass 1: greedy online assignment. Track the raw speaker id (not
    // the formatted string) so we can rewrite labels after the merge
    // pass without re-parsing.
    let mut raw: Vec<(f32, f32, usize)> = Vec::new(); // (start, end, raw_id)
    let mut skipped = 0_usize;
    for segment in segments {
        if segment.samples.len() < MIN_SEGMENT_SAMPLES {
            skipped += 1;
            continue;
        }
        let embedding: Vec<f32> = embedding_extractor
            .compute(&segment.samples)
            .map_err(|e| TranscribeError::Diarization(format!("embedding failed: {e:?}")))?
            .collect();
        let raw_id = speaker_bank.assign(&embedding);
        raw.push((
            segment.start_seconds as f32,
            segment.end_seconds as f32,
            raw_id,
        ));
    }
    if skipped > 0 {
        tracing::debug!(
            skipped,
            total = total_segments,
            "skipped sub-1-s segmentation outputs (too short for stable embedding)"
        );
    }

    // Pass 2a: agglomerative merge on refined centroids. Collapses
    // ghost clusters that were created from a noisy initial embedding
    // before the real centroid stabilized.
    let speakers_before = speaker_bank.len();
    let merged = speaker_bank.agglomerative_merge(MERGE_THRESHOLD);
    let after_merge = merged.iter().copied().collect::<std::collections::BTreeSet<_>>().len();

    // Pass 2b: orphan reabsorption. Any cluster with too few matched
    // segments to plausibly be a real meeting participant gets folded
    // into the nearest larger cluster regardless of similarity.
    let remap = speaker_bank.reabsorb_small_clusters(&merged, MIN_SEGMENTS_PER_REAL_SPEAKER);
    let speakers_after = remap.iter().copied().collect::<std::collections::BTreeSet<_>>().len();
    tracing::info!(
        speakers_before,
        after_agglomerative_merge = after_merge,
        speakers_after,
        "diarization clustering passes"
    );

    let out: Vec<DiarizationSegment> = raw
        .into_iter()
        .map(|(start_seconds, end_seconds, raw_id)| {
            let final_id = remap.get(raw_id).copied().unwrap_or(raw_id);
            DiarizationSegment {
                start_seconds,
                end_seconds,
                speaker: format!("SPEAKER_{final_id:02}"),
            }
        })
        .collect();

    tracing::info!(segments = out.len(), "pyannote diarization done");
    Ok(out)
}

/// Assign each ASR segment to the diarization speaker whose interval
/// covers the segment's midpoint (with greedy fallback to the nearest
/// speaker), then merge consecutive same-speaker segments under the
/// INTRA_TURN_GAP rule from the spec.
///
/// When `diar` is empty (e.g. diarization failed), each ASR segment
/// becomes its own turn under a synthetic per-index speaker id so the
/// merge rule doesn't collapse everything into one giant turn.
fn assign_speakers_and_merge(asr: &[AsrSegment], diar: &[DiarizationSegment]) -> Vec<Turn> {
    let mut turns: Vec<Turn> = Vec::new();
    for (i, seg) in asr.iter().enumerate() {
        let speaker = if diar.is_empty() {
            format!("speaker_unknown_{i:04}")
        } else {
            speaker_for(seg, diar).unwrap_or_else(|| "speaker_unknown".to_string())
        };
        let start = seg.start_seconds;
        let end = seg.end_seconds.max(start);
        let text = seg.text.trim();
        if text.is_empty() {
            continue;
        }
        let can_merge = turns.last().is_some_and(|prev| {
            prev.speaker_id == speaker
                && (start - prev.end_seconds as f32) < INTRA_TURN_GAP_SECONDS
                && (end - prev.start_seconds as f32) < MAX_TURN_DURATION_SECONDS
        });
        if can_merge {
            let last = turns.last_mut().expect("just checked non-empty");
            last.end_seconds = end as u32;
            last.text = format!("{} {}", last.text, text).trim().to_string();
        } else {
            turns.push(Turn {
                speaker_id: speaker,
                start_seconds: start as u32,
                end_seconds: end as u32,
                text: text.to_string(),
            });
        }
    }
    turns
}

fn speaker_for(seg: &AsrSegment, diar: &[DiarizationSegment]) -> Option<String> {
    if diar.is_empty() {
        return None;
    }
    let midpoint = f32::midpoint(seg.start_seconds, seg.end_seconds);
    // Prefer the diar segment that strictly contains the ASR midpoint;
    // fall back to the temporally nearest diar segment so ASR clauses
    // that land in inter-utterance silence still get a real speaker
    // label (rather than the `speaker_unknown` placeholder).
    let containing = diar
        .iter()
        .find(|d| d.start_seconds <= midpoint && midpoint <= d.end_seconds);
    let best = containing.or_else(|| {
        diar.iter().min_by(|a, b| {
            distance_to_segment(midpoint, a)
                .partial_cmp(&distance_to_segment(midpoint, b))
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    });
    best.map(|d| normalize_speaker(&d.speaker))
}

fn distance_to_segment(t: f32, seg: &DiarizationSegment) -> f32 {
    if t < seg.start_seconds {
        seg.start_seconds - t
    } else if t > seg.end_seconds {
        t - seg.end_seconds
    } else {
        0.0
    }
}

fn normalize_speaker(raw: &str) -> String {
    let s = raw.trim().to_lowercase();
    let s: String = s
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() || c == '_' || c == '-' { c } else { '-' })
        .collect();
    let s = s.trim_matches('-').to_string();
    if s.is_empty() {
        "speaker".to_string()
    } else {
        s
    }
}

fn build_kbtx_doc(cfg: &TranscribeConfig, artifact: &TranscriptArtifact, turns: &[Turn]) -> TranscriptDoc {
    use std::fmt::Write as _;
    let unique_speakers = unique_in_order(turns.iter().map(|t| t.speaker_id.as_str()));
    let mut frontmatter = String::new();
    frontmatter.push_str("type: source\n");
    let _ = writeln!(frontmatter, "title: {}", cfg.title);
    let _ = writeln!(frontmatter, "recording_date: {}", cfg.recording_date);
    let _ = writeln!(frontmatter, "audio_file: {}", artifact.audio_filename);
    let _ = writeln!(frontmatter, "duration_seconds: {:.1}", artifact.duration_seconds);
    let _ = writeln!(frontmatter, "speakers_detected: {}", unique_speakers.len());
    let _ = writeln!(frontmatter, "asr_model: {}", artifact.asr_model);
    let _ = writeln!(frontmatter, "diarization_model: {}", artifact.diarization_model);
    frontmatter.push_str("language: en\n");

    let roster: Vec<RosterEntry> = unique_speakers
        .into_iter()
        .map(|id| RosterEntry {
            speaker_id: id.to_string(),
            display_name: None,
            role: None,
        })
        .collect();

    let mut body: Vec<BodyNode> = Vec::new();
    let mut prev_end: Option<u32> = None;
    for turn in turns {
        if let Some(prev) = prev_end {
            let gap = turn.start_seconds.saturating_sub(prev);
            if cfg.pause_threshold_seconds > 0.0 && gap as f32 >= cfg.pause_threshold_seconds {
                body.push(BodyNode::ActionLine(kb_core::transcript::ActionLine {
                    text: format!("pause: {gap}s"),
                }));
            }
        }
        body.push(BodyNode::Turn(turn.clone()));
        prev_end = Some(turn.end_seconds);
    }

    TranscriptDoc {
        frontmatter,
        roster,
        body,
    }
}

fn unique_in_order<'a, I: IntoIterator<Item = &'a str>>(items: I) -> Vec<&'a str> {
    let mut seen = std::collections::BTreeSet::new();
    let mut out = Vec::new();
    for item in items {
        if seen.insert(item) {
            out.push(item);
        }
    }
    out
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn looks_like_decoder_repetition_flags_hallucinated_loop() {
        let phrase = "Then, once the user click a field, the backend will run a query. ";
        let hallucinated = phrase.repeat(20);
        assert!(looks_like_decoder_repetition(&hallucinated));
    }

    #[test]
    fn looks_like_decoder_repetition_passes_normal_speech() {
        let normal = "So what is, I'm trying to remember from back in my LiveRamp days, \
                     like, what exactly is the segment builder as a product? Is that when \
                     somebody logs into a UI and then they choose a field, query the \
                     backend for matching audiences, and surfaces the results to the user?";
        assert!(!looks_like_decoder_repetition(normal));
    }

    #[test]
    fn looks_like_decoder_repetition_passes_short_text() {
        assert!(!looks_like_decoder_repetition("Yeah."));
        assert!(!looks_like_decoder_repetition(""));
    }

    fn make_seg(t: &str) -> AsrSegment {
        AsrSegment {
            start_seconds: 0.0,
            end_seconds: 1.0,
            text: t.into(),
        }
    }

    #[test]
    fn collapse_long_runs_kills_short_repeats_at_run_4() {
        // "Okay." × 15 — the Delta v15 hallucination shape.
        let segs: Vec<AsrSegment> = (0..15).map(|_| make_seg("Okay.")).collect();
        let out = collapse_long_runs(segs, 4);
        assert_eq!(out.len(), 1, "15 identical 5-char segments collapse to 1");
    }

    #[test]
    fn collapse_long_runs_kills_medium_phrase_loops() {
        // 50-char phrase, 7 in a row — also from Delta v15.
        let phrase = "I think that would be great, because then we can...";
        let segs: Vec<AsrSegment> = (0..7).map(|_| make_seg(phrase)).collect();
        let out = collapse_long_runs(segs, 4);
        assert_eq!(out.len(), 1);
    }

    #[test]
    fn collapse_long_runs_keeps_short_runs() {
        // "Yeah." × 3 in a row is plausible human emphasis. Keep all.
        let segs: Vec<AsrSegment> = (0..3).map(|_| make_seg("Yeah.")).collect();
        let out = collapse_long_runs(segs, 4);
        assert_eq!(out.len(), 3, "3-run is below the cutoff");
    }

    #[test]
    fn collapse_long_runs_keeps_distinct_segments() {
        let segs = vec![
            make_seg("Hello there."),
            make_seg("How are you?"),
            make_seg("I'm fine, thanks."),
        ];
        let out = collapse_long_runs(segs, 4);
        assert_eq!(out.len(), 3);
    }

    #[test]
    fn collapse_long_runs_only_collapses_one_run_of_many() {
        // Mixed: 5-run of "X." + 1 of "Y." + 2-run of "Z." → 1 + 1 + 2 = 4
        let mut segs = Vec::new();
        for _ in 0..5 { segs.push(make_seg("X.")); }
        segs.push(make_seg("Y."));
        for _ in 0..2 { segs.push(make_seg("Z.")); }
        let out = collapse_long_runs(segs, 4);
        let texts: Vec<&str> = out.iter().map(|s| s.text.as_str()).collect();
        assert_eq!(texts, vec!["X.", "Y.", "Z.", "Z."]);
    }

    #[test]
    fn merges_same_speaker_within_gap() {
        let asr = vec![
            AsrSegment { start_seconds: 0.0, end_seconds: 5.0, text: "hello".into() },
            AsrSegment { start_seconds: 6.0, end_seconds: 10.0, text: "world".into() },
        ];
        let diar = vec![DiarizationSegment {
            start_seconds: 0.0,
            end_seconds: 30.0,
            speaker: "SPEAKER_00".into(),
        }];
        let turns = assign_speakers_and_merge(&asr, &diar);
        assert_eq!(turns.len(), 1);
        assert_eq!(turns[0].text, "hello world");
        assert_eq!(turns[0].speaker_id, "speaker_00");
    }

    #[test]
    fn splits_on_long_gap() {
        let asr = vec![
            AsrSegment { start_seconds: 0.0, end_seconds: 5.0, text: "hello".into() },
            AsrSegment { start_seconds: 20.0, end_seconds: 25.0, text: "world".into() },
        ];
        let diar = vec![DiarizationSegment {
            start_seconds: 0.0,
            end_seconds: 30.0,
            speaker: "SPEAKER_00".into(),
        }];
        let turns = assign_speakers_and_merge(&asr, &diar);
        assert_eq!(turns.len(), 2);
    }

    #[test]
    fn splits_on_speaker_change() {
        let asr = vec![
            AsrSegment { start_seconds: 0.0, end_seconds: 5.0, text: "hello".into() },
            AsrSegment { start_seconds: 6.0, end_seconds: 10.0, text: "world".into() },
        ];
        let diar = vec![
            DiarizationSegment { start_seconds: 0.0, end_seconds: 5.5, speaker: "SPEAKER_00".into() },
            DiarizationSegment { start_seconds: 5.5, end_seconds: 30.0, speaker: "SPEAKER_01".into() },
        ];
        let turns = assign_speakers_and_merge(&asr, &diar);
        assert_eq!(turns.len(), 2);
        assert_eq!(turns[0].speaker_id, "speaker_00");
        assert_eq!(turns[1].speaker_id, "speaker_01");
    }

    fn sample_artifact() -> TranscriptArtifact {
        TranscriptArtifact {
            duration_seconds: 12.0,
            sample_rate: 16_000,
            asr_model: "whisper-large-v3-q5_0".into(),
            diarization_model: "pyannote-segmentation-3.0+wespeaker-resnet34-LM".into(),
            audio_filename: "demo.m4a".into(),
            asr_segments: vec![
                AsrSegment { start_seconds: 0.0, end_seconds: 5.0, text: "hello".into() },
                AsrSegment { start_seconds: 6.0, end_seconds: 10.0, text: "world".into() },
            ],
            diarization_segments: vec![DiarizationSegment {
                start_seconds: 0.0,
                end_seconds: 30.0,
                speaker: "SPEAKER_00".into(),
            }],
        }
    }

    #[test]
    fn render_artifact_round_trip_through_kbtx_parser() {
        let artifact = sample_artifact();
        let cfg = TranscribeConfig::new("demo.m4a", "demo", "2026-04-27");
        let kbtx = render_artifact(&artifact, &cfg);
        // Round-trip through the kb_core parser to assert the renderer
        // emits well-formed kbtx — the same guarantee the prototype gave.
        let doc = kb_core::transcript::parse(&kbtx).expect("parses");
        assert_eq!(doc.body.len(), 1, "single merged turn expected");
    }

    #[test]
    fn artifact_serde_round_trip() {
        let artifact = sample_artifact();
        let json = serde_json::to_string(&artifact).unwrap();
        let back: TranscriptArtifact = serde_json::from_str(&json).unwrap();
        assert_eq!(back.asr_segments.len(), 2);
        assert_eq!(back.diarization_segments.len(), 1);
        assert_eq!(back.audio_filename, "demo.m4a");
    }
}
