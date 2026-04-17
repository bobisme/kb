use assert_cmd::Command;
use std::path::{Path, PathBuf};
use tempfile::TempDir;

/// Create a temporary knowledge base directory for testing.
///
/// Returns the `TempDir` and its path. The directory is automatically cleaned up
/// when the `TempDir` is dropped.
pub fn make_temp_kb() -> (TempDir, PathBuf) {
    let temp_dir = TempDir::new().expect("failed to create temp directory");
    let path = temp_dir.path().to_path_buf();
    (temp_dir, path)
}

/// Create an `assert_cmd` Command for the kb CLI with --root set.
///
/// The command is configured to:
/// - Use the kb binary
/// - Point to the specified root directory via --root
/// - Capture stdout/stderr for assertions
pub fn kb_cmd(root: &Path) -> Command {
    let mut cmd = Command::cargo_bin("kb").expect("failed to find kb binary");
    cmd.arg("--root").arg(root);
    cmd
}
