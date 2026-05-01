//! `kb ask --session` state — multi-turn conversation history (bn-o6wv).
//!
//! Sessions live at `<root>/.kb/sessions/<id>.json`. Each turn records who
//! spoke, the text, and (for user turns) the retrieval ids that grounded
//! the next answer or (for assistant turns) the citation labels that
//! actually grounded the answer. The schema is intentionally narrow:
//! callers are expected to add new fields with `#[serde(default)]` so old
//! files stay readable.
//!
//! Storage uses a temp-file + rename atomic write so a crash between
//! flushes never leaves a half-serialized JSON file behind, and so a
//! reader that happens to open the path mid-write still sees the previous
//! complete file.

use std::fs;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};

/// Filesystem-friendly session id pattern.
///
/// `[A-Za-z0-9._-]+`, capped at 64 chars. Long enough to hold a terseid
/// with a slug suffix, narrow enough that a malicious id can't escape
/// the sessions dir or shadow a system file.
const SESSION_ID_MAX: usize = 64;

/// `<root>/.kb/sessions` — the directory holding per-session JSON files.
#[must_use]
pub fn sessions_dir(root: &Path) -> PathBuf {
    root.join(".kb").join("sessions")
}

/// `<root>/.kb/sessions/<id>.json` — the path for a given session id.
///
/// Caller is responsible for validating the id with [`validate_session_id`]
/// first. The path is constructed unconditionally so the on-disk shape is
/// deterministic for both readers and writers.
#[must_use]
pub fn session_path(root: &Path, id: &str) -> PathBuf {
    sessions_dir(root).join(format!("{id}.json"))
}

/// Reject ids that would escape the sessions dir or land in the wrong
/// place. Keep it conservative — alphanum, dot, dash, underscore.
///
/// # Errors
///
/// Returns an error if the id is empty, longer than 64 chars, or
/// contains any character outside `[A-Za-z0-9._-]`. Also rejects
/// dot-only ids (`.`, `..`) which would otherwise build a path
/// pointing at the parent directory.
pub fn validate_session_id(id: &str) -> Result<()> {
    if id.is_empty() {
        bail!("session id must not be empty");
    }
    if id.len() > SESSION_ID_MAX {
        bail!("session id must be {SESSION_ID_MAX} chars or fewer (got {})", id.len());
    }
    if id == "." || id == ".." {
        bail!("session id must not be '.' or '..'");
    }
    if !id
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-'))
    {
        bail!(
            "session id may only contain ASCII letters, digits, '.', '_', or '-' (got {id:?})"
        );
    }
    Ok(())
}

/// Whether a turn was authored by the user or the assistant.
///
/// Stored as a snake-case discriminant so JSON is readable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TurnRole {
    User,
    Assistant,
}

/// One turn in a session — a single user question or assistant reply.
///
/// `retrieved_ids` is populated on user turns (after the rewrite pass
/// runs hybrid retrieval); `citations` is populated on assistant turns
/// (parsed from the LLM's `[N]` markers via the existing manifest). The
/// other field is left empty for that role to keep the schema flat.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Turn {
    pub role: TurnRole,
    pub text: String,
    /// Retrieval candidate ids fetched for this turn (user turns only).
    #[serde(default)]
    pub retrieved_ids: Vec<String>,
    /// Citation labels actually used in the answer (assistant turns only).
    #[serde(default)]
    pub citations: Vec<String>,
}

/// A full session — a sequence of turns plus metadata.
///
/// New fields should default-construct via `#[serde(default)]` so older
/// session files keep deserializing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Session {
    pub id: String,
    pub created_at_millis: i64,
    pub updated_at_millis: i64,
    #[serde(default)]
    pub turns: Vec<Turn>,
}

impl Session {
    /// Build an empty session with `created_at = updated_at = now`.
    ///
    /// # Errors
    ///
    /// Returns an error when the system clock is before the Unix epoch
    /// (in which case nothing else in kb would work either).
    pub fn new(id: impl Into<String>) -> Result<Self> {
        let now = now_millis()?;
        Ok(Self {
            id: id.into(),
            created_at_millis: now,
            updated_at_millis: now,
            turns: Vec::new(),
        })
    }

    /// Append a turn and bump `updated_at_millis`.
    ///
    /// # Errors
    ///
    /// Same as [`Session::new`].
    pub fn push_turn(&mut self, turn: Turn) -> Result<()> {
        self.turns.push(turn);
        self.updated_at_millis = now_millis()?;
        Ok(())
    }
}

/// Load a session from `<root>/.kb/sessions/<id>.json`.
///
/// # Errors
///
/// Returns `Ok(None)` when the file doesn't exist (callers treat that
/// as "fresh session"). Returns an error when the file exists but
/// cannot be read or parsed — that's a corrupt session and the user
/// should be told rather than silently losing prior turns.
pub fn load(root: &Path, id: &str) -> Result<Option<Session>> {
    validate_session_id(id)?;
    let path = session_path(root, id);
    match fs::read_to_string(&path) {
        Ok(text) => {
            let session: Session = serde_json::from_str(&text)
                .with_context(|| format!("parse session file {}", path.display()))?;
            Ok(Some(session))
        }
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(err) => {
            Err(err).with_context(|| format!("read session file {}", path.display()))
        }
    }
}

/// Atomic write: serialize to a temp file under the same directory,
/// fsync it, then rename over the destination. The rename is atomic on
/// POSIX, so a reader either sees the previous complete file or the new
/// one — never a half-written blob.
///
/// On failure (serialization, write, or rename), best-effort removes
/// the temp file so we don't leak `*.tmp-*` siblings.
///
/// # Errors
///
/// Returns an error when the target directory cannot be created, the
/// session cannot be serialized, the temp file cannot be written, or
/// the rename fails.
pub fn save(root: &Path, session: &Session) -> Result<()> {
    validate_session_id(&session.id)?;
    let dir = sessions_dir(root);
    fs::create_dir_all(&dir)
        .with_context(|| format!("create sessions dir {}", dir.display()))?;
    let final_path = session_path(root, &session.id);
    let tmp_name = format!(
        ".{id}.json.tmp-{pid}-{nanos}",
        id = session.id,
        pid = std::process::id(),
        nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos()),
    );
    let tmp_path = dir.join(tmp_name);

    // Pretty-print so a curious user can `cat` the file and read it.
    let body = serde_json::to_vec_pretty(session).context("serialize session")?;

    let write_then_rename = || -> Result<()> {
        let mut file = fs::File::create(&tmp_path)
            .with_context(|| format!("create temp file {}", tmp_path.display()))?;
        file.write_all(&body)
            .with_context(|| format!("write temp file {}", tmp_path.display()))?;
        file.sync_all()
            .with_context(|| format!("fsync temp file {}", tmp_path.display()))?;
        drop(file);
        fs::rename(&tmp_path, &final_path).with_context(|| {
            format!(
                "rename {} -> {}",
                tmp_path.display(),
                final_path.display()
            )
        })?;
        Ok(())
    };

    if let Err(err) = write_then_rename() {
        // Best-effort cleanup; if the rename succeeded we already moved
        // the file away and the unlink will be a no-op NotFound.
        let _ = fs::remove_file(&tmp_path);
        return Err(err);
    }
    Ok(())
}

/// One row in the `kb session list` output.
#[derive(Debug, Clone)]
pub struct SessionListEntry {
    pub id: String,
    pub updated_at_millis: i64,
    pub turn_count: usize,
}

/// Walk `<root>/.kb/sessions/` and return one entry per `<id>.json`,
/// sorted newest-first by `updated_at_millis`.
///
/// Files that fail to parse are skipped silently — a future "fsck"
/// command can surface those, but `kb session list` should keep working
/// even if one file gets corrupted.
///
/// # Errors
///
/// Returns an error only when the sessions directory itself cannot be
/// read (e.g. permission denied). A missing directory yields an empty
/// list because the user just hasn't started any sessions yet.
pub fn list(root: &Path) -> Result<Vec<SessionListEntry>> {
    let dir = sessions_dir(root);
    if !dir.exists() {
        return Ok(Vec::new());
    }
    let mut entries = Vec::new();
    for raw in fs::read_dir(&dir)
        .with_context(|| format!("read sessions dir {}", dir.display()))?
    {
        let Ok(raw) = raw else {
            continue;
        };
        let path = raw.path();
        if path.extension().and_then(|s| s.to_str()) != Some("json") {
            continue;
        }
        // Skip atomic-write temp files (start with `.`).
        if path
            .file_name()
            .and_then(|s| s.to_str())
            .is_some_and(|n| n.starts_with('.'))
        {
            continue;
        }
        let Some(id) = path
            .file_stem()
            .and_then(|s| s.to_str())
            .map(str::to_owned)
        else {
            continue;
        };
        if validate_session_id(&id).is_err() {
            continue;
        }
        let Ok(text) = fs::read_to_string(&path) else {
            continue;
        };
        let Ok(session) = serde_json::from_str::<Session>(&text) else {
            continue;
        };
        entries.push(SessionListEntry {
            id: session.id,
            updated_at_millis: session.updated_at_millis,
            turn_count: session.turns.len(),
        });
    }
    entries.sort_by_key(|e| std::cmp::Reverse(e.updated_at_millis));
    Ok(entries)
}

/// Generate a fresh session id using `terseid`. The id is short, sortable,
/// and collision-free against the on-disk session list.
#[must_use]
pub fn generate_session_id(root: &Path) -> String {
    let dir = sessions_dir(root);
    let item_count = if dir.exists() {
        fs::read_dir(&dir).map_or(0, |rd| rd.filter_map(Result::ok).count())
    } else {
        0
    };
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos());
    let pid = std::process::id();
    let generator = terseid::IdGenerator::new(terseid::IdConfig::new("s"));
    generator.generate(
        |nonce| format!("{now}|{pid}|session|{nonce}").into_bytes(),
        item_count,
        |candidate| dir.join(format!("{candidate}.json")).exists(),
    )
}

/// Wrap `SystemTime::now()` so callers don't have to deal with the
/// epoch/error plumbing every time they push a turn. Returns
/// milliseconds-since-epoch as `i64` to match the `EntityMetadata`
/// convention used elsewhere in the codebase.
fn now_millis() -> Result<i64> {
    let millis = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .context("system time is before UNIX_EPOCH")?
        .as_millis();
    i64::try_from(millis).context("timestamp overflows i64 milliseconds")
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn round_trip_serialize_deserialize() {
        let mut s = Session::new("s-test").expect("new session");
        s.push_turn(Turn {
            role: TurnRole::User,
            text: "what is consensus?".to_string(),
            retrieved_ids: vec!["src-paxos".into(), "src-raft".into()],
            citations: vec![],
        })
        .expect("push user turn");
        s.push_turn(Turn {
            role: TurnRole::Assistant,
            text: "Consensus is the problem of agreeing... [1]".into(),
            retrieved_ids: vec![],
            citations: vec!["wiki/sources/src-paxos.md".into()],
        })
        .expect("push assistant turn");

        let body = serde_json::to_string(&s).expect("serialize");
        let parsed: Session = serde_json::from_str(&body).expect("deserialize");
        assert_eq!(parsed.id, "s-test");
        assert_eq!(parsed.turns.len(), 2);
        assert_eq!(parsed.turns[0].role, TurnRole::User);
        assert_eq!(parsed.turns[0].retrieved_ids.len(), 2);
        assert_eq!(parsed.turns[1].role, TurnRole::Assistant);
        assert_eq!(parsed.turns[1].citations.len(), 1);
    }

    #[test]
    fn save_and_load_round_trip() {
        let tmp = TempDir::new().expect("tempdir");
        let mut s = Session::new("s-rt").expect("new");
        s.push_turn(Turn {
            role: TurnRole::User,
            text: "hi".into(),
            retrieved_ids: vec![],
            citations: vec![],
        })
        .expect("push");

        save(tmp.path(), &s).expect("save");
        let loaded = load(tmp.path(), "s-rt").expect("load").expect("present");
        assert_eq!(loaded.id, "s-rt");
        assert_eq!(loaded.turns.len(), 1);
        assert_eq!(loaded.turns[0].text, "hi");
    }

    #[test]
    fn load_missing_returns_none() {
        let tmp = TempDir::new().expect("tempdir");
        assert!(load(tmp.path(), "s-missing").expect("load").is_none());
    }

    #[test]
    fn save_atomic_no_temp_file_left_on_success() {
        let tmp = TempDir::new().expect("tempdir");
        let s = Session::new("s-atomic").expect("new");
        save(tmp.path(), &s).expect("save");

        let dir = sessions_dir(tmp.path());
        let entries: Vec<_> = fs::read_dir(&dir)
            .expect("read dir")
            .filter_map(Result::ok)
            .collect();
        // Only the final file should remain — no `.tmp-` siblings.
        assert_eq!(entries.len(), 1);
        let only = &entries[0];
        let name = only.file_name().to_string_lossy().into_owned();
        assert_eq!(name, "s-atomic.json");
    }

    #[test]
    fn save_overwrites_existing_atomically() {
        let tmp = TempDir::new().expect("tempdir");
        let mut s = Session::new("s-over").expect("new");
        save(tmp.path(), &s).expect("save initial");
        s.push_turn(Turn {
            role: TurnRole::User,
            text: "new turn".into(),
            retrieved_ids: vec![],
            citations: vec![],
        })
        .expect("push");
        save(tmp.path(), &s).expect("save updated");

        let loaded = load(tmp.path(), "s-over").expect("load").expect("present");
        assert_eq!(loaded.turns.len(), 1);

        // Still only one final file in the dir.
        let dir = sessions_dir(tmp.path());
        let count = fs::read_dir(&dir)
            .expect("read dir")
            .filter_map(Result::ok)
            .count();
        assert_eq!(count, 1);
    }

    #[test]
    fn list_returns_sessions_newest_first() {
        let tmp = TempDir::new().expect("tempdir");
        let mut a = Session::new("a-one").expect("new");
        a.updated_at_millis = 1_000;
        save(tmp.path(), &a).expect("save a");
        let mut b = Session::new("b-two").expect("new");
        b.updated_at_millis = 2_000;
        save(tmp.path(), &b).expect("save b");

        let entries = list(tmp.path()).expect("list");
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].id, "b-two");
        assert_eq!(entries[1].id, "a-one");
    }

    #[test]
    fn list_on_missing_dir_is_empty() {
        let tmp = TempDir::new().expect("tempdir");
        assert!(list(tmp.path()).expect("list").is_empty());
    }

    #[test]
    fn validate_session_id_rejects_path_escape() {
        assert!(validate_session_id("../etc").is_err());
        assert!(validate_session_id("../").is_err());
        assert!(validate_session_id("foo/bar").is_err());
        assert!(validate_session_id("..").is_err());
        assert!(validate_session_id(".").is_err());
        assert!(validate_session_id("").is_err());
    }

    #[test]
    fn validate_session_id_accepts_typical_ids() {
        assert!(validate_session_id("s-1abc").is_ok());
        assert!(validate_session_id("work_on_x").is_ok());
        assert!(validate_session_id("research.2026").is_ok());
    }

    #[test]
    fn defaults_let_old_files_parse() {
        // A session with no `turns` field at all (older schema) must still
        // parse — `#[serde(default)]` should fill in an empty Vec.
        let raw = r#"{"id":"s-old","created_at_millis":1,"updated_at_millis":2}"#;
        let parsed: Session = serde_json::from_str(raw).expect("parse minimal session");
        assert_eq!(parsed.id, "s-old");
        assert!(parsed.turns.is_empty());
    }

    #[test]
    fn generate_session_id_starts_with_s() {
        let tmp = TempDir::new().expect("tempdir");
        let id = generate_session_id(tmp.path());
        assert!(id.starts_with('s'), "id should start with 's': {id}");
        assert!(validate_session_id(&id).is_ok(), "id should validate: {id}");
    }
}
