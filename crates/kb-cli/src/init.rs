use std::env;
use std::fs;
use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use kb_core::{Hashes, Manifest};

use crate::config::Config;

// Plain placeholder body — `kb compile` rewrites this file with real index
// content once sources are ingested. No managed region here: an empty region
// with no source frontmatter would trip the missing-citations lint.
const MANAGED_STUB: &str = "\
# Knowledge Base

This is the root index of your knowledge base. Run `kb compile` to populate.
";

pub fn init(root_override: Option<PathBuf>, path_arg: Option<PathBuf>, force: bool) -> Result<()> {
    let target = if let Some(r) = root_override {
        r
    } else if let Some(p) = path_arg {
        p
    } else {
        env::current_dir().context("failed to get current directory")?
    };

    let target = if target.is_absolute() {
        target
    } else {
        env::current_dir()?.join(target)
    };

    let config_path = target.join(Config::FILE_NAME);

    if config_path.exists() && !force {
        bail!(
            "{} already exists. Use --force to overwrite.",
            config_path.display()
        );
    }

    fs::create_dir_all(&target)
        .with_context(|| format!("failed to create directory {}", target.display()))?;

    let default_config = Config::default();
    let toml = default_config.to_toml_string()?;
    fs::write(&config_path, toml)
        .with_context(|| format!("failed to write {}", config_path.display()))?;

    let dirs = [
        "raw/inbox",
        "raw/web",
        "raw/papers",
        "raw/repos",
        "raw/datasets",
        "raw/images",
        "normalized",
        "wiki/concepts",
        "wiki/sources",
        "wiki/projects",
        "wiki/timelines",
        "wiki/people",
        "outputs/questions",
        "outputs/reports",
        "outputs/slides",
        "outputs/figures",
        "reviews/promotions",
        "reviews/merges",
        "reviews/aliases",
        "reviews/canonicalization",
        "prompts",
        "state/indexes",
        "state/jobs",
        "state/locks",
        "state/build_records",
        "state/concept_candidates",
        "cache",
        "logs",
        "trash",
    ];

    for dir in dirs {
        let dir_path = target.join(dir);
        fs::create_dir_all(&dir_path)
            .with_context(|| format!("failed to create directory {}", dir_path.display()))?;
    }

    Manifest::default().save(&target)?;
    Hashes::default().save(&target)?;

    let index_path = target.join("wiki/index.md");
    if !index_path.exists() || force {
        fs::write(&index_path, MANAGED_STUB)
            .with_context(|| format!("failed to write {}", index_path.display()))?;
    }

    println!("Initialized empty knowledge base at {}", target.display());

    Ok(())
}
