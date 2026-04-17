#![forbid(unsafe_code)]

use clap::Parser;

#[derive(Parser)]
#[command(name = "kb", version, about = "Personal knowledge base compiler")]
struct Cli {
    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(clap::Subcommand)]
enum Command {}

#[allow(clippy::unnecessary_wraps)]
fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let _cli = Cli::parse();

    Ok(())
}
