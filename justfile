default:
    @just --list

check:
    cargo clippy --workspace --all-targets -- -D warnings
    cargo test --workspace --no-run

build:
    cargo build --workspace

test:
    cargo test --workspace

install:
    cargo install --path crates/kb-cli
