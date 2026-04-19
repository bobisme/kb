default:
    @just --list

check:
    cargo clippy --locked --workspace --all-targets -- -D warnings
    cargo test --locked --workspace --no-run

build:
    cargo build --locked --workspace

test:
    cargo test --locked --workspace

install:
    cargo install --locked --path crates/kb-cli
