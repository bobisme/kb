# KB Obsidian Plugin

Thin Obsidian plugin wrapping the `kb` CLI. Every command shells out — there
is no duplicated logic on the JS side.

## Commands

All commands are available from the Obsidian command palette (`Ctrl/Cmd + P`):

| Command                       | What it does                                                                                      |
|-------------------------------|---------------------------------------------------------------------------------------------------|
| `KB Compile`                  | Run `kb compile`. Status bar shows progress.                                                      |
| `KB Ask`                      | Prompt for a question, run `kb ask --dry-run`, insert the preview at the cursor.                  |
| `KB Inspect`                  | Run `kb inspect <current-file>` and insert the output at the cursor.                              |
| `KB Chat (external terminal)` | Launch `kb chat` in an external terminal (x-terminal-emulator, etc.); fall back to a notice.      |
| `KB Ingest: current note`     | Run `kb ingest <abs-path-of-current-note>`; show output in a modal.                               |
| `KB Ingest: clipboard URL`    | If the clipboard holds a URL, run `kb ingest <url>`; otherwise show "clipboard has no URL".       |
| `KB Promote this question`    | If the current note is an ask artifact, run `kb review approve review-<question-id>`.             |
| `KB Search`                   | Prompt for a query, run `kb search --json`, show ranked results in a modal; click to open.        |

## Promote detection

`KB Promote this question` identifies ask artifacts by inspecting frontmatter in this order:

1. Explicit `promotion_id`, `review_id`, or `promotion-id` — used directly.
2. `type: question_answer` plus a `question_id` — derives `review-<question_id>`.
3. `kind: ask` or `kind: question_answer` plus a `question_id` — same derivation.

If none of the above matches, the command surfaces a notice: "not a promotable artifact".

## Installation

1. Clone this repository into your Obsidian vault's `.obsidian/plugins/kb-obsidian` directory, **or** clone elsewhere and symlink.
2. `npm install`
3. `npm run build` — produces `main.js` next to `manifest.json`.
4. Enable the plugin in Obsidian: Settings → Community plugins → toggle "KB Obsidian".

The plugin is desktop-only (`isDesktopOnly: true` in `manifest.json`) because it shells out via Node's `child_process`.

## Configuration

Open Settings → Community plugins → KB Obsidian → options:

- **KB path** — path to the `kb` binary. Default: `kb` (PATH lookup).
- **KB root** — path to the KB root (the directory containing `kb.toml`). Blank triggers auto-detection: walk up from the vault root looking for `kb.toml`.
- **Model override** — passed to `kb` as `--model <value>` on every invocation. Blank uses the `kb.toml` default.
- **Terminal command** — executable used to launch `kb chat`. Blank tries `x-terminal-emulator` first and then `gnome-terminal`, `konsole`, `kitty`, `alacritty`, `wezterm`, `foot`, `xterm` in order.

## Development

```bash
npm install
npm run dev      # esbuild watch mode
npm run build    # tsc --noEmit + production esbuild
```

The plugin compiles to `main.js` which is loaded by Obsidian.

## Technical notes

- Built with TypeScript, bundled with esbuild (`cjs`, target `ES6`).
- Runs kb commands via `child_process.execSync` with `cwd` set to the
  resolved KB root. `--model` is threaded through as a global flag.
- `KB Chat` uses `spawn` with `detached: true` + `stdio: "ignore"` so
  closing Obsidian does not kill the chat session.
- `KB Search` parses the kb `--json` envelope (`{ schema_version, command, data, ... }`) and tolerates a bare array as a fallback for future schema changes.
