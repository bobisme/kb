default:
    @just --list

check:
    cargo clippy --locked --workspace --all-targets -- -D warnings
    cargo test --locked --workspace --no-run

build:
    cargo build --locked --workspace

test:
    cargo test --locked --workspace

# auto-detects CUDA, falls back to CPU; override with install-{cuda,cpu}
install:
    #!/usr/bin/env bash
    set -euo pipefail
    if command -v nvcc >/dev/null 2>&1 && command -v nvidia-smi >/dev/null 2>&1; then
        echo "[install] CUDA toolkit + GPU detected — building with GPU acceleration"
        just install-cuda
    else
        echo "[install] No CUDA — building CPU-only kb"
        just install-cpu
    fi

# CPU-only build (portable, no GPU dep)
install-cpu:
    cargo install --locked --path crates/kb-cli

# GPU build; auto-detects CUDA_PATH and the local GPU's compute capability
install-cuda:
    #!/usr/bin/env bash
    set -euo pipefail
    arch=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits | head -1 | tr -d '.')
    cuda_path=$(dirname "$(dirname "$(command -v nvcc)")")
    echo "[install-cuda] CUDA_PATH=$cuda_path  CMAKE_CUDA_ARCHITECTURES=$arch"
    CUDA_PATH="$cuda_path" \
    PATH="$cuda_path/bin:$PATH" \
    CMAKE_CUDA_ARCHITECTURES="$arch" \
    CMAKE_CUDA_STANDARD=17 \
    CMAKE_CXX_STANDARD=17 \
        cargo install --locked --path crates/kb-cli --features cuda
