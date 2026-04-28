//! One-shot helper: download every `ModelSpec` into a fresh tmpdir and print
//! the actual blake3 hashes so the const literals can be updated.
//!
//!     cargo run -p kb-ingest-audio --example capture_model_hashes
//!
//! Output is two columns per line: `<spec_name>  blake3$<hex>`.

use std::path::PathBuf;

use kb_ingest_audio::models::{
    ModelStore, SEGMENTATION_3_0, WESPEAKER_RESNET34_LM, WHISPER_LARGE_V3,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::new(
            std::env::var("RUST_LOG").unwrap_or_else(|_| "info".into()),
        ))
        .init();

    let tmpdir = tempfile::tempdir()?;
    let store = ModelStore::new(Some(tmpdir.path().to_path_buf()))?;

    for spec in &[SEGMENTATION_3_0, WESPEAKER_RESNET34_LM, WHISPER_LARGE_V3] {
        let path = store.ensure(spec)?;
        let hash = blake3_of(&path)?;
        println!("{:32}  blake3${hash}", spec.name);
    }
    Ok(())
}

fn blake3_of(path: &PathBuf) -> std::io::Result<String> {
    use std::io::Read;
    let mut file = std::fs::File::open(path)?;
    let mut hasher = blake3::Hasher::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(hasher.finalize().to_hex().to_string())
}
