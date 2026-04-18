use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize, de::DeserializeOwned};

use crate::fs::atomic_write;
use crate::{BuildRecord, EntityId, Hash, hash_file};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct Manifest {
    pub artifacts: BTreeMap<PathBuf, BuildRecord>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct Hashes {
    pub inputs: BTreeMap<PathBuf, Hash>,
}

#[must_use]
pub fn manifest_path(root: &Path) -> PathBuf {
    root.join("state").join("manifest.json")
}

#[must_use]
pub fn hashes_path(root: &Path) -> PathBuf {
    root.join("state").join("hashes.json")
}

impl Manifest {
    /// Load the build manifest from disk, returning an empty manifest when absent.
    ///
    /// # Errors
    /// Returns an error when the manifest exists but cannot be read or decoded.
    pub fn load(root: &Path) -> Result<Self> {
        load_json_file(&manifest_path(root), "manifest")
    }

    /// Persist the build manifest atomically.
    ///
    /// # Errors
    /// Returns an error when the manifest cannot be serialized or written.
    pub fn save(&self, root: &Path) -> Result<()> {
        write_json_file(&manifest_path(root), "manifest", self)
    }

    #[must_use]
    pub fn stale_artifacts(&self, changed_entity_ids: &BTreeSet<EntityId>) -> BTreeSet<PathBuf> {
        let mut frontier = changed_entity_ids.clone();
        let mut stale = BTreeSet::new();
        let mut made_progress = true;

        while made_progress {
            made_progress = false;

            for (artifact_path, record) in &self.artifacts {
                if stale.contains(artifact_path) {
                    continue;
                }

                if record.input_ids.iter().any(|input| frontier.contains(input)) {
                    stale.insert(artifact_path.clone());
                    frontier.extend(record.output_ids.iter().cloned());
                    made_progress = true;
                }
            }
        }

        stale
    }
}

impl Hashes {
    /// Load the tracked input hashes from disk, returning an empty set when absent.
    ///
    /// # Errors
    /// Returns an error when the hashes file exists but cannot be read or decoded.
    pub fn load(root: &Path) -> Result<Self> {
        load_json_file(&hashes_path(root), "hashes")
    }

    /// Persist the tracked input hashes atomically.
    ///
    /// # Errors
    /// Returns an error when the hashes cannot be serialized or written.
    pub fn save(&self, root: &Path) -> Result<()> {
        write_json_file(&hashes_path(root), "hashes", self)
    }

    /// Recompute hashes for the provided inputs, returning the paths whose hash changed,
    /// were newly added, or were removed from the tracked set.
    ///
    /// # Errors
    /// Returns an error when any tracked file cannot be hashed.
    pub fn reconcile_inputs<I>(&mut self, paths: I) -> Result<BTreeSet<PathBuf>>
    where
        I: IntoIterator<Item = PathBuf>,
    {
        let tracked: BTreeSet<_> = paths.into_iter().collect();
        let mut changed = BTreeSet::new();
        let mut next = BTreeMap::new();

        for path in &tracked {
            let hash = hash_file(path)
                .with_context(|| format!("hash tracked input {}", path.display()))?;
            if self.inputs.get(path) != Some(&hash) {
                changed.insert(path.clone());
            }
            next.insert(path.clone(), hash);
        }

        for previous_path in self.inputs.keys() {
            if !tracked.contains(previous_path) {
                changed.insert(previous_path.clone());
            }
        }

        self.inputs = next;
        Ok(changed)
    }
}

fn load_json_file<T>(path: &Path, label: &str) -> Result<T>
where
    T: DeserializeOwned + Default,
{
    if !path.exists() {
        return Ok(T::default());
    }

    let raw = fs::read_to_string(path).with_context(|| format!("read {label} {}", path.display()))?;
    serde_json::from_str(&raw)
        .with_context(|| format!("deserialize {label} {}", path.display()))
}

fn write_json_file<T>(path: &Path, label: &str, value: &T) -> Result<()>
where
    T: Serialize,
{
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("create {} directory {}", label, parent.display()))?;
    }

    let mut json = serde_json::to_vec_pretty(value)
        .with_context(|| format!("serialize {label} {}", path.display()))?;
    json.push(b'\n');
    atomic_write(path, &json).with_context(|| format!("write {label} {}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{BuildRecord, EntityMetadata, Status};
    use tempfile::tempdir;

    fn metadata(id: &str) -> EntityMetadata {
        EntityMetadata {
            id: id.to_string(),
            created_at_millis: 1,
            updated_at_millis: 1,
            source_hashes: vec!["hash-1".to_string()],
            model_version: None,
            tool_version: Some("kb-test".to_string()),
            prompt_template_hash: None,
            dependencies: Vec::new(),
            output_paths: Vec::new(),
            status: Status::Fresh,
        }
    }

    fn record(input_ids: &[&str], output_ids: &[&str], manifest_hash: &str) -> BuildRecord {
        BuildRecord {
            metadata: metadata(manifest_hash),
            input_ids: input_ids.iter().map(|id| (*id).to_string()).collect(),
            output_ids: output_ids.iter().map(|id| (*id).to_string()).collect(),
            manifest_hash: manifest_hash.to_string(),
        }
    }

    #[test]
    fn manifest_save_is_stable_across_identical_writes() {
        let dir = tempdir().expect("tempdir");
        let root = dir.path();
        let mut manifest = Manifest::default();
        manifest.artifacts.insert(
            PathBuf::from("wiki/index.md"),
            record(&["src-a"], &["page-a"], "manifest-a"),
        );
        manifest.artifacts.insert(
            PathBuf::from("wiki/concepts/rust.md"),
            record(&["page-a"], &["concept-rust"], "manifest-b"),
        );

        manifest.save(root).expect("save manifest first time");
        let first = fs::read_to_string(manifest_path(root)).expect("read first manifest");

        let loaded = Manifest::load(root).expect("load manifest");
        assert_eq!(loaded, manifest);

        loaded.save(root).expect("save manifest second time");
        let second = fs::read_to_string(manifest_path(root)).expect("read second manifest");
        assert_eq!(first, second);
    }

    #[test]
    fn corrupt_manifest_reports_clear_error() {
        let dir = tempdir().expect("tempdir");
        let root = dir.path();
        let path = manifest_path(root);
        fs::create_dir_all(path.parent().expect("manifest parent")).expect("create state dir");
        fs::write(&path, "{not-json}\n").expect("write corrupt manifest");

        let err = Manifest::load(root).expect_err("corrupt manifest should fail");
        let message = err.to_string();
        assert!(message.contains("deserialize manifest"));
        assert!(message.contains("state/manifest.json"));
    }

    #[test]
    fn reconcile_inputs_detects_added_changed_and_removed_files() {
        let dir = tempdir().expect("tempdir");
        let source = dir.path().join("source.md");
        fs::write(&source, "hello\n").expect("write source");

        let mut hashes = Hashes::default();
        let changed = hashes
            .reconcile_inputs(vec![source.clone()])
            .expect("hash initial input");
        assert_eq!(changed, BTreeSet::from([source.clone()]));

        let changed = hashes
            .reconcile_inputs(vec![source.clone()])
            .expect("rehash unchanged input");
        assert!(changed.is_empty());

        fs::write(&source, "hello world\n").expect("update source");
        let changed = hashes
            .reconcile_inputs(vec![source.clone()])
            .expect("rehash changed input");
        assert_eq!(changed, BTreeSet::from([source.clone()]));

        let changed = hashes
            .reconcile_inputs(Vec::new())
            .expect("remove tracked input");
        assert_eq!(changed, BTreeSet::from([source]));
    }

    #[test]
    fn stale_artifacts_propagates_through_dependents() {
        let manifest = Manifest {
            artifacts: BTreeMap::from([
                (
                    PathBuf::from("wiki/sources/source-a.md"),
                    record(&["src-a"], &["page-a"], "manifest-a"),
                ),
                (
                    PathBuf::from("wiki/concepts/concept-a.md"),
                    record(&["page-a"], &["concept-a"], "manifest-b"),
                ),
                (
                    PathBuf::from("outputs/reports/report.md"),
                    record(&["concept-a"], &["report-a"], "manifest-c"),
                ),
                (
                    PathBuf::from("wiki/sources/source-b.md"),
                    record(&["src-b"], &["page-b"], "manifest-d"),
                ),
            ]),
        };

        let stale = manifest.stale_artifacts(&BTreeSet::from(["src-a".to_string()]));
        assert_eq!(
            stale,
            BTreeSet::from([
                PathBuf::from("outputs/reports/report.md"),
                PathBuf::from("wiki/concepts/concept-a.md"),
                PathBuf::from("wiki/sources/source-a.md"),
            ])
        );
    }
}
