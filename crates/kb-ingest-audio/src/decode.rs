//! Container-agnostic audio decode → mono f32 PCM @ 16 kHz.
//!
//! Whisper and pyannote both expect 16 kHz mono. We decode whatever
//! format symphonia knows (mp3, m4a, mp4, wav, flac), downmix to mono,
//! and resample to 16 kHz. No ffmpeg runtime dependency.

use std::fs::File;
use std::path::Path;

use thiserror::Error;

const TARGET_SAMPLE_RATE: u32 = 16_000;

#[derive(Debug, Error)]
pub enum DecodeError {
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error("symphonia probe failed: {0}")]
    Probe(String),
    #[error("symphonia format read failed: {0}")]
    Format(String),
    #[error("symphonia decode failed: {0}")]
    Decode(String),
    #[error("no audio track found")]
    NoTrack,
    #[error("source had unsupported sample format")]
    UnsupportedSampleFormat,
}

/// Decoded mono 16 kHz f32 PCM, ready for whisper / pyannote.
#[derive(Debug, Clone)]
pub struct DecodedAudio {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
    pub duration_seconds: f32,
}

/// Decode `path` to mono 16 kHz f32 PCM.
///
/// # Errors
///
/// Returns a `DecodeError` for I/O failures, container probe/format
/// failures, codec decode errors, missing audio tracks, or unsupported
/// sample formats.
pub fn decode_to_mono_16k(path: &Path) -> Result<DecodedAudio, DecodeError> {
    use symphonia::core::codecs::DecoderOptions;
    use symphonia::core::errors::Error as SymphoniaError;
    use symphonia::core::formats::FormatOptions;
    use symphonia::core::io::{MediaSourceStream, MediaSourceStreamOptions};
    use symphonia::core::meta::MetadataOptions;
    use symphonia::core::probe::Hint;

    let file = File::open(path)?;
    let mss = MediaSourceStream::new(Box::new(file), MediaSourceStreamOptions::default());

    let hint = Hint::new();
    let probed = symphonia::default::get_probe()
        .format(&hint, mss, &FormatOptions::default(), &MetadataOptions::default())
        .map_err(|e| DecodeError::Probe(e.to_string()))?;
    let mut format = probed.format;

    let track = format.default_track().ok_or(DecodeError::NoTrack)?;
    let track_id = track.id;
    let codec_params = track.codec_params.clone();
    let mut decoder = symphonia::default::get_codecs()
        .make(&codec_params, &DecoderOptions::default())
        .map_err(|e| DecodeError::Decode(e.to_string()))?;

    let source_rate = codec_params.sample_rate.unwrap_or(TARGET_SAMPLE_RATE);
    let mut mono_at_source_rate: Vec<f32> = Vec::new();

    loop {
        let packet = match format.next_packet() {
            Ok(p) => p,
            Err(SymphoniaError::IoError(e)) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                break;
            }
            Err(SymphoniaError::ResetRequired) => {
                break;
            }
            Err(e) => return Err(DecodeError::Format(e.to_string())),
        };
        if packet.track_id() != track_id {
            continue;
        }
        let decoded = match decoder.decode(&packet) {
            Ok(d) => d,
            Err(SymphoniaError::DecodeError(_)) => continue,
            Err(e) => return Err(DecodeError::Decode(e.to_string())),
        };
        append_mono(&decoded, &mut mono_at_source_rate)?;
    }

    let resampled = if source_rate == TARGET_SAMPLE_RATE {
        mono_at_source_rate
    } else {
        resample_linear(&mono_at_source_rate, source_rate, TARGET_SAMPLE_RATE)
    };

    let duration_seconds = resampled.len() as f32 / TARGET_SAMPLE_RATE as f32;
    Ok(DecodedAudio {
        samples: resampled,
        sample_rate: TARGET_SAMPLE_RATE,
        duration_seconds,
    })
}

fn append_mono(buf: &symphonia::core::audio::AudioBufferRef, out: &mut Vec<f32>) -> Result<(), DecodeError> {
    use symphonia::core::audio::{AudioBufferRef, Signal};
    match buf {
        AudioBufferRef::F32(b) => {
            downmix(b, out);
            Ok(())
        }
        AudioBufferRef::F64(b) => {
            let frames = b.frames();
            let channels = b.spec().channels.count();
            for f in 0..frames {
                let mut acc = 0.0_f64;
                for c in 0..channels {
                    acc += b.chan(c)[f];
                }
                out.push((acc / channels as f64) as f32);
            }
            Ok(())
        }
        AudioBufferRef::S16(b) => {
            let frames = b.frames();
            let channels = b.spec().channels.count();
            for f in 0..frames {
                let mut acc = 0.0_f32;
                for c in 0..channels {
                    acc += b.chan(c)[f] as f32 / i16::MAX as f32;
                }
                out.push(acc / channels as f32);
            }
            Ok(())
        }
        AudioBufferRef::S32(b) => {
            let frames = b.frames();
            let channels = b.spec().channels.count();
            for f in 0..frames {
                let mut acc = 0.0_f32;
                for c in 0..channels {
                    acc += b.chan(c)[f] as f32 / i32::MAX as f32;
                }
                out.push(acc / channels as f32);
            }
            Ok(())
        }
        _ => Err(DecodeError::UnsupportedSampleFormat),
    }
}

fn downmix(buf: &symphonia::core::audio::AudioBuffer<f32>, out: &mut Vec<f32>) {
    use symphonia::core::audio::Signal;
    let frames = buf.frames();
    let channels = buf.spec().channels.count();
    for f in 0..frames {
        let mut acc = 0.0_f32;
        for c in 0..channels {
            acc += buf.chan(c)[f];
        }
        out.push(acc / channels as f32);
    }
}

/// Linear resampler. Good enough for whisper input — whisper itself does
/// its own log-mel front-end so anti-aliasing isn't critical at this
/// stage. If quality issues show up, swap in `rubato` cubic resampling.
fn resample_linear(input: &[f32], from_rate: u32, to_rate: u32) -> Vec<f32> {
    if from_rate == to_rate || input.is_empty() {
        return input.to_vec();
    }
    let ratio = from_rate as f64 / to_rate as f64;
    let out_len = ((input.len() as f64) / ratio).round() as usize;
    let mut out = Vec::with_capacity(out_len);
    for i in 0..out_len {
        let src_pos = (i as f64) * ratio;
        let idx = src_pos.floor() as usize;
        let frac = src_pos - idx as f64;
        let a = input[idx];
        let b = if idx + 1 < input.len() { input[idx + 1] } else { a };
        out.push((b - a).mul_add(frac as f32, a));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resample_identity() {
        let v = vec![1.0, 2.0, 3.0];
        let out = resample_linear(&v, 16_000, 16_000);
        assert_eq!(out, v);
    }

    #[test]
    fn resample_halves_to_double_rate_then_back() {
        let input: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let up = resample_linear(&input, 16_000, 32_000);
        assert!(up.len() >= input.len() * 2 - 1);
        let down = resample_linear(&up, 32_000, 16_000);
        assert!((down.len() as i32 - input.len() as i32).abs() <= 1);
    }
}
