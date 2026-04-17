#![forbid(unsafe_code)]

use clap::Parser;
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "kb", version, about = "Personal knowledge base compiler")]
struct Cli {
    /// Root directory of the knowledge base
    #[arg(global = true, long)]
    root: Option<PathBuf>,

    /// Output format
    #[arg(global = true, long, value_parser = ["md", "marp", "json", "png"])]
    format: Option<String>,

    /// LLM model to use
    #[arg(global = true, long)]
    model: Option<String>,

    /// Filter results since a given time
    #[arg(global = true, long)]
    since: Option<String>,

    /// Perform a dry-run without making changes
    #[arg(global = true, long)]
    dry_run: bool,

    /// Output structured JSON
    #[arg(global = true, long)]
    json: bool,

    /// Force operation (overwrite, skip confirmations, etc.)
    #[arg(global = true, long)]
    force: bool,

    /// Enable review mode
    #[arg(global = true, long)]
    review: bool,

    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(clap::Subcommand)]
enum Command {
    /// Initialize a new knowledge base
    Init {
        /// Path to initialize at
        path: Option<PathBuf>,
    },
    /// Ingest documents into the knowledge base
    Ingest {
        /// Files or directories to ingest
        #[arg(required = true)]
        sources: Vec<PathBuf>,
    },
    /// Compile the knowledge base
    Compile,
    /// Query the knowledge base with natural language
    Ask {
        /// Question to ask
        #[arg(required = true)]
        query: String,
    },
    /// Lint knowledge base for issues
    Lint {
        /// Check specific rules
        #[arg(long)]
        rule: Option<String>,
    },
    /// Run health checks on the knowledge base
    Doctor,
    /// Show status of the knowledge base
    Status,
    /// Publish the knowledge base
    Publish {
        /// Destination for publishing
        #[arg(long)]
        dest: Option<String>,
    },
    /// Search the knowledge base
    Search {
        /// Search query
        #[arg(required = true)]
        query: String,
    },
    /// Inspect a document or entity
    Inspect {
        /// Document or entity to inspect
        #[arg(required = true)]
        target: String,
    },
    /// Request or manage reviews
    Review {
        /// Review operation (list, approve, reject, etc.)
        #[arg(required = true)]
        operation: String,
    },
}

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let cli = Cli::parse();

    match cli.command {
        Some(Command::Init { path: _ }) => {
            eprintln!("error: 'init' subcommand not implemented");
            std::process::exit(1);
        }
        Some(Command::Ingest { sources: _ }) => {
            eprintln!("error: 'ingest' subcommand not implemented");
            std::process::exit(1);
        }
        Some(Command::Compile) => {
            eprintln!("error: 'compile' subcommand not implemented");
            std::process::exit(1);
        }
        Some(Command::Ask { query: _ }) => {
            eprintln!("error: 'ask' subcommand not implemented");
            std::process::exit(1);
        }
        Some(Command::Lint { rule: _ }) => {
            eprintln!("error: 'lint' subcommand not implemented");
            std::process::exit(1);
        }
        Some(Command::Doctor) => {
            eprintln!("error: 'doctor' subcommand not implemented");
            std::process::exit(1);
        }
        Some(Command::Status) => {
            eprintln!("error: 'status' subcommand not implemented");
            std::process::exit(1);
        }
        Some(Command::Publish { dest: _ }) => {
            eprintln!("error: 'publish' subcommand not implemented");
            std::process::exit(1);
        }
        Some(Command::Search { query: _ }) => {
            eprintln!("error: 'search' subcommand not implemented");
            std::process::exit(1);
        }
        Some(Command::Inspect { target: _ }) => {
            eprintln!("error: 'inspect' subcommand not implemented");
            std::process::exit(1);
        }
        Some(Command::Review { operation: _ }) => {
            eprintln!("error: 'review' subcommand not implemented");
            std::process::exit(1);
        }
        None => {
            if !cli.dry_run && !cli.force && !cli.review && cli.root.is_none() && cli.format.is_none() && cli.model.is_none() && cli.since.is_none() && !cli.json {
                println!("kb: a personal knowledge base compiler");
                println!("Run 'kb --help' for more information");
                Ok(())
            } else {
                Ok(())
            }
        }
    }
}
