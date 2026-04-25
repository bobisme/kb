//! `SQLite` `vec0` extension loader for kb's semantic search.
//!
//! Provides [`register_auto_extension`] which installs sqlite-vec as a
//! process-wide auto extension via `sqlite3_auto_extension`. After a
//! successful registration every new `rusqlite::Connection` (regardless of
//! how it is opened) gains access to functions like `vec_version()`,
//! `vec_f32()`, and `vec_distance_cosine()`.
//!
//! Mirrors the shape of the `bones-sqlite-vec` crate so call sites in kb
//! can be reasoned about by anyone familiar with bones.
//!
//! # Opt-out
//!
//! Set `KB_SQLITE_VEC_AUTO=0` (or `false`/`off`) before any connection is
//! opened to skip registration entirely. The semantic search path falls
//! back to a Rust-side cosine implementation when sqlite-vec is not
//! available, so this is a clean override for environments where the
//! extension fails to load (uncommon, but useful for diagnosing breakage).

#![forbid(unsafe_op_in_unsafe_fn)]

use std::sync::OnceLock;

const AUTO_ENABLE_ENV: &str = "KB_SQLITE_VEC_AUTO";

static REGISTRATION: OnceLock<Result<(), String>> = OnceLock::new();

/// Register sqlite-vec as a process-wide `SQLite` auto extension.
///
/// Idempotent: subsequent calls return the cached result of the first
/// invocation. Safe to call from any thread; the underlying
/// `OnceLock` serializes the registration.
///
/// # Errors
///
/// Returns an error when auto-registration is disabled via
/// `KB_SQLITE_VEC_AUTO` or when `sqlite3_auto_extension` fails.
pub fn register_auto_extension() -> Result<(), String> {
    if matches!(
        std::env::var(AUTO_ENABLE_ENV).ok().as_deref(),
        Some("0" | "false" | "off")
    ) {
        return Err(format!(
            "sqlite-vec auto-extension disabled by {AUTO_ENABLE_ENV}"
        ));
    }

    REGISTRATION.get_or_init(register_once).clone()
}

fn register_once() -> Result<(), String> {
    // sqlite-vec's entrypoint signature differs from rusqlite's
    // sqlite3_auto_extension by an extra `*const c_char` parameter; the
    // transmute aligns the two. This is the same pattern used by
    // `bones-sqlite-vec` and follows the recommendation from the
    // sqlite-vec README for static linking via rusqlite.
    let entrypoint: unsafe extern "C" fn(
        *mut rusqlite::ffi::sqlite3,
        *mut *const std::os::raw::c_char,
        *const rusqlite::ffi::sqlite3_api_routines,
    ) -> std::os::raw::c_int =
        unsafe { std::mem::transmute(sqlite_vec::sqlite3_vec_init as *const ()) };

    // SAFETY: `sqlite3_auto_extension` is documented as thread-safe and the
    // transmuted function pointer is the entrypoint sqlite-vec exports for
    // exactly this purpose.
    let rc = unsafe { rusqlite::ffi::sqlite3_auto_extension(Some(entrypoint)) };
    if rc == rusqlite::ffi::SQLITE_OK {
        Ok(())
    } else {
        Err(format!("sqlite3_auto_extension failed with rc={rc}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rusqlite::Connection;

    #[test]
    fn registration_makes_vec_version_available() {
        let result = register_auto_extension();
        assert!(result.is_ok(), "registration failed: {result:?}");

        let conn = Connection::open_in_memory().expect("open in-memory sqlite");
        let version: String = conn
            .query_row("SELECT vec_version()", [], |row| row.get(0))
            .expect("vec_version() should be available after registration");
        assert!(
            !version.is_empty(),
            "vec_version() returned empty string: {version}"
        );
    }
}
