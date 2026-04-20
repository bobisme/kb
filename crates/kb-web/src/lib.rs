#![forbid(unsafe_code)]

//! Minimal read-only web UI for browsing a kb root and running
//! `kb search` / `kb ask` queries from a browser.
//!
//! The crate exposes:
//!
//! - [`router`]: builds an [`axum::Router`] given a [`WebState`].
//! - [`serve`]: binds an HTTP listener on `host:port` and serves the router.
//!
//! The router is separate from the listener so integration tests can drive
//! it via `tower::ServiceExt::oneshot` without a real network socket.

mod markdown;
mod server;

pub use server::{WebState, router, serve};
