//! Native Rust audio ingestion: ASR + speaker diarization → kbtx transcript.
//!
//! This is the in-tree replacement for the Python `kb-audio-transcribe` tool
//! (bn-3hfw). It produces output byte-equivalent to the prototype's kbtx
//! renderer, so corpus content is stable across the migration.
//!
//! Pipeline:
//!
//! 1. Decode any container (mp3 / m4a / mp4 / wav) → mono 16 kHz f32 PCM
//!    via `symphonia` (pure Rust, no external ffmpeg dependency at runtime).
//! 2. Run whisper-large-v3 via `whisper-rs` (whisper.cpp FFI) for ASR.
//! 3. Run pyannote-segmentation-3.0 + wespeaker via `pyannote-rs`
//!    (ONNX Runtime) for diarization.
//! 4. Word-level speaker assignment + same-speaker turn merging
//!    (matching the prototype's INTRA_TURN_GAP / MAX_TURN_DURATION rules).
//! 5. Render to kbtx via `kb_core::transcript`.
//!
//! Models auto-download to `~/.kb/models/<sha256>` on first use from
//! MIT-licensed mirrors (no HuggingFace authentication required).
//!
//! See `notes/audio-ingest-rust.md` (in this workspace) for design notes
//! and `notes/transcript-spec.md` for the kbtx output format.

#![forbid(unsafe_code)]
// Audio DSP work involves a lot of necessary casts between numeric types
// (f32 PCM ↔ i16 PCM, sample-count usize ↔ duration f32, sample-rate u32
// ↔ float for ratio math). The pedantic/nursery cast lints flag every
// one of those, so we relax them at the crate level.
#![allow(
    clippy::cast_lossless,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::similar_names,
    clippy::doc_markdown
)]

pub mod decode;
pub mod models;
pub mod pipeline;
pub mod segmentation;
pub mod speaker_bank;

pub use pipeline::{
    produce_artifact, render_artifact, transcribe, AsrSegment, DiarizationSegment,
    TranscribeConfig, TranscribeError, TranscriptArtifact,
};
