//! Per-pass progress rendering for `run_compile`.
//!
//! Two renderers are provided:
//!
//! * [`LineLogReporter`] — plain `[run]`/`[ok]` lines on stderr (with an
//!   optional `quiet` mode that reduces output to a single "compiling..." +
//!   "done" pair). Used for piped stdout, `--json`, `--quiet`, and as the
//!   default inside library tests.
//! * [`IndicatifReporter`] — interactive multi-progress bars on a TTY.
//!
//! The CLI dispatcher picks between them (see `run_compile_action` in
//! `kb-cli`). The pipeline layer only ever talks to the [`ProgressReporter`]
//! trait and in parallel forwards every event as plain text to the
//! [`LogSink`](crate::pipeline::LogSink) so `state/jobs/*.log` stays readable
//! regardless of which renderer rendered it for the user.
use std::collections::HashMap;
use std::sync::Mutex;
use std::time::Duration;

use indicatif::{MultiProgress, ProgressBar, ProgressStyle};

/// Receives per-pass lifecycle events from `run_compile`.
///
/// Implementations own any rendering state (progress bars, spinner handles,
/// etc.) and are `Send + Sync` because the pipeline is called from a mutating
/// job handle that may live behind an `Arc`.
pub trait ProgressReporter: Send + Sync {
    /// Announce the start of a pass. `total` is the number of items we expect
    /// to process (0 means "unknown / indivisible", which renderers may treat
    /// as a spinner).
    fn pass_start(&self, pass: &str, total: usize);

    /// Announce the start of a single unit of work inside a pass (a document,
    /// a candidate batch, etc.). `item` is a short label (typically a doc id).
    fn pass_item_start(&self, pass: &str, item: &str);

    /// Announce the completion of a single unit of work.
    fn pass_item_done(&self, pass: &str, item: &str, elapsed: Duration);

    /// Announce the completion of a pass with the number of items actually
    /// affected and the total elapsed time for the pass.
    fn pass_done(&self, pass: &str, affected: usize, elapsed: Duration);

    /// One-shot info message (e.g. the initial "compile: N source(s), K
    /// stale" banner). Renderers that show bars print above them; plain-text
    /// renderers write a line to stderr.
    fn info(&self, message: &str);

    /// Formatted error line for a failed pass. Separate from `info` so
    /// indicatif renderers can style the output (e.g. red) without inspecting
    /// message text.
    fn error(&self, message: &str);
}

/// Plain line-by-line reporter.
///
/// Writes `[run]`/`[ok]`/`[err]` lines to stderr and preserves the exact
/// string format that `state/jobs/*.log` has always contained — integration
/// tests grep for these strings.
pub struct LineLogReporter {
    quiet: bool,
}

impl LineLogReporter {
    /// Create a verbose line reporter (one `[run]`/`[ok]` line per item).
    #[must_use]
    pub const fn new() -> Self {
        Self { quiet: false }
    }

    /// Create a reduced-output reporter: a single `compiling...` banner and a
    /// `done` summary, with no per-pass chatter. Used under `--quiet`.
    #[must_use]
    pub const fn quiet() -> Self {
        Self { quiet: true }
    }
}

impl Default for LineLogReporter {
    fn default() -> Self {
        Self::new()
    }
}

impl ProgressReporter for LineLogReporter {
    fn pass_start(&self, _pass: &str, _total: usize) {
        // Per-item `[run]` lines are emitted in `pass_item_start`, so there is
        // nothing to announce at the pass level for the verbose format. Quiet
        // mode swallows this entirely.
    }

    fn pass_item_start(&self, pass: &str, item: &str) {
        if self.quiet {
            return;
        }
        eprintln!("  [run] {pass}: {item}...");
    }

    fn pass_item_done(&self, pass: &str, item: &str, elapsed: Duration) {
        if self.quiet {
            return;
        }
        // Preserve the double-space `[ok]  ` alignment — existing tests and
        // users are already keyed on this layout.
        if item.is_empty() {
            eprintln!("  [ok]  {pass} ({:.1}s)", elapsed.as_secs_f64());
        } else {
            eprintln!(
                "  [ok]  {pass}: {item} ({:.1}s)",
                elapsed.as_secs_f64()
            );
        }
    }

    fn pass_done(&self, _pass: &str, _affected: usize, _elapsed: Duration) {
        // Pass-level completion is implicit in the per-item `[ok]` lines for
        // the verbose formatter.
    }

    fn info(&self, message: &str) {
        if self.quiet {
            // In quiet mode we still want the user to see that compile
            // started and finished; the initial banner (from pipeline.rs)
            // plus the final render() output cover both.
            return;
        }
        eprintln!("{message}");
    }

    fn error(&self, message: &str) {
        // Errors should always surface, even in quiet mode.
        eprintln!("{message}");
    }
}

/// Interactive indicatif renderer. Owns a `MultiProgress` and creates a fresh
/// bar (or spinner for opaque passes) per `pass_start`. Bars are finished on
/// `pass_done` and removed from the `MultiProgress`.
pub struct IndicatifReporter {
    multi: MultiProgress,
    bars: Mutex<HashMap<String, ProgressBar>>,
}

impl IndicatifReporter {
    /// Create an indicatif reporter attached to a fresh `MultiProgress`
    /// targeting stderr.
    #[must_use]
    pub fn new() -> Self {
        Self {
            // Draw to stderr so stdout stays reserved for the final
            // `report.render()` output (or JSON).
            multi: MultiProgress::with_draw_target(indicatif::ProgressDrawTarget::stderr()),
            bars: Mutex::new(HashMap::new()),
        }
    }

    fn bar_style() -> ProgressStyle {
        ProgressStyle::with_template(
            "  {prefix:20.cyan} [{bar:30.cyan/blue}] {pos}/{len} {msg} ({elapsed})",
        )
        .unwrap_or_else(|_| ProgressStyle::default_bar())
        .progress_chars("=>-")
    }

    fn spinner_style() -> ProgressStyle {
        ProgressStyle::with_template("  {prefix:20.cyan} {spinner} {msg} ({elapsed})")
            .unwrap_or_else(|_| ProgressStyle::default_spinner())
    }
}

impl Default for IndicatifReporter {
    fn default() -> Self {
        Self::new()
    }
}

impl ProgressReporter for IndicatifReporter {
    fn pass_start(&self, pass: &str, total: usize) {
        // total==0 or 1 means "no per-item granularity worth drawing a bar
        // over" (e.g. `concept_merge` is a single opaque LLM call). Render a
        // spinner with steady tick instead so the user sees progress.
        let bar = if total <= 1 {
            let pb = self.multi.add(ProgressBar::new_spinner());
            pb.set_style(Self::spinner_style());
            pb.enable_steady_tick(Duration::from_millis(120));
            pb
        } else {
            // Casting to u64 is safe: `total` comes from Vec::len() which is
            // bounded by `isize::MAX`, well under u64::MAX.
            let pb = self.multi.add(ProgressBar::new(total as u64));
            pb.set_style(Self::bar_style());
            pb
        };
        bar.set_prefix(pass.to_string());
        bar.set_message("");
        if let Ok(mut bars) = self.bars.lock() {
            bars.insert(pass.to_string(), bar);
        }
    }

    fn pass_item_start(&self, pass: &str, item: &str) {
        if let Ok(bars) = self.bars.lock() {
            if let Some(bar) = bars.get(pass) {
                bar.set_message(item.to_string());
            }
        }
    }

    fn pass_item_done(&self, pass: &str, _item: &str, _elapsed: Duration) {
        if let Ok(bars) = self.bars.lock() {
            if let Some(bar) = bars.get(pass) {
                // inc(1) is a no-op for spinner bars (unbounded length), so
                // passes using total==0/1 are still handled correctly.
                bar.inc(1);
            }
        }
    }

    fn pass_done(&self, pass: &str, affected: usize, elapsed: Duration) {
        if let Ok(mut bars) = self.bars.lock() {
            if let Some(bar) = bars.remove(pass) {
                bar.finish_with_message(format!(
                    "done ({affected} affected, {:.1}s)",
                    elapsed.as_secs_f64()
                ));
            }
        }
    }

    fn info(&self, message: &str) {
        // println! on the MultiProgress prints above the bars without
        // corrupting their redraw.
        let _ = self.multi.println(message);
    }

    fn error(&self, message: &str) {
        let _ = self.multi.println(message);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn line_log_reporter_quiet_suppresses_per_item_lines() {
        // Smoke test — we can't capture stderr inside the library, but this
        // exercises the code paths and makes sure nothing panics.
        let reporter = LineLogReporter::quiet();
        reporter.pass_start("source_summary", 3);
        reporter.pass_item_start("source_summary", "doc-a");
        reporter.pass_item_done("source_summary", "doc-a", Duration::from_millis(10));
        reporter.pass_done("source_summary", 3, Duration::from_millis(30));
        reporter.info("compile: 1 source(s), 1 stale");
    }

    #[test]
    fn indicatif_reporter_handles_spinner_pass_without_panic() {
        // total==1 should render as a spinner; ensure nothing panics when we
        // then call pass_item_done (which would inc a bar but is a no-op for
        // an unbounded spinner).
        let reporter = IndicatifReporter::new();
        reporter.pass_start("concept_merge", 1);
        reporter.pass_item_start("concept_merge", "merging 54 candidates");
        reporter.pass_item_done("concept_merge", "merging 54 candidates", Duration::from_millis(10));
        reporter.pass_done("concept_merge", 1, Duration::from_millis(30));
    }
}
