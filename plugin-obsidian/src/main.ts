import {
  App,
  Plugin,
  PluginSettingTab,
  Setting,
  Editor,
  MarkdownView,
  Modal,
  Notice,
  TFile,
} from "obsidian";
import { execSync, spawn } from "child_process";
import * as path from "path";
import * as fs from "fs";

interface KbPluginSettings {
  /** Path to the `kb` binary. Defaults to `kb` (PATH lookup). */
  kbPath: string;
  /**
   * KB root directory. Empty string means auto-detect by walking up from
   * the vault root looking for `kb.toml`.
   */
  kbRoot: string;
  /**
   * LLM model override. Empty string means use the kb.toml default.
   * When set, it is passed to every invocation via the global `--model` flag.
   */
  model: string;
  /**
   * External terminal emulator for `kb chat`. Empty string means
   * auto-detect (`x-terminal-emulator` first, then fallbacks).
   */
  terminalCommand: string;
}

const DEFAULT_SETTINGS: KbPluginSettings = {
  kbPath: "kb",
  kbRoot: "",
  model: "",
  terminalCommand: "",
};

/**
 * URL detection — lax. We accept http(s), file://, and bare domains that
 * look URL-ish. The CLI is the final judge, so we only need to cheaply
 * distinguish "plausible URL" from "arbitrary clipboard text".
 */
const URL_PATTERN = /^(https?|file|ftp):\/\/\S+$/i;

export default class KbPlugin extends Plugin {
  settings: KbPluginSettings = DEFAULT_SETTINGS;
  statusBar: HTMLElement | null = null;

  async onload() {
    await this.loadSettings();

    // ---- Existing v0.0.1 commands (preserved verbatim in behavior) ----

    this.addCommand({
      id: "kb-compile",
      name: "KB Compile",
      callback: () => this.executeCompile(),
    });

    this.addCommand({
      id: "kb-ask",
      name: "KB Ask",
      editorCallback: (editor: Editor, _view: MarkdownView) =>
        this.executeAsk(editor),
    });

    this.addCommand({
      id: "kb-inspect",
      name: "KB Inspect",
      editorCallback: (editor: Editor, view: MarkdownView) =>
        this.executeInspect(editor, view),
    });

    // ---- New commands added for bn-6y85 ----

    this.addCommand({
      id: "kb-chat",
      name: "KB Chat (external terminal)",
      callback: () => this.executeChat(),
    });

    this.addCommand({
      id: "kb-ingest-current-note",
      name: "KB Ingest: current note",
      callback: () => this.executeIngestCurrentNote(),
    });

    this.addCommand({
      id: "kb-ingest-clipboard-url",
      name: "KB Ingest: clipboard URL",
      callback: () => this.executeIngestClipboardUrl(),
    });

    this.addCommand({
      id: "kb-promote-this-question",
      name: "KB Promote this question",
      callback: () => this.executePromoteThisQuestion(),
    });

    this.addCommand({
      id: "kb-search",
      name: "KB Search",
      callback: () => this.executeSearch(),
    });

    this.statusBar = this.addStatusBarItem();
    this.statusBar.setText("KB ready");

    this.addSettingTab(new KbSettingTab(this.app, this));
  }

  onunload() {}

  async loadSettings() {
    this.settings = Object.assign(
      {},
      DEFAULT_SETTINGS,
      await this.loadData(),
    );
  }

  async saveSettings() {
    await this.saveData(this.settings);
  }

  // -------------------- path / environment helpers --------------------

  private getVaultRoot(): string {
    // FileSystemAdapter exposes basePath. On mobile there's no such adapter,
    // but the plugin is desktop-only (see manifest).
    const adapter = this.app.vault.adapter as { basePath?: string };
    if (adapter?.basePath) return adapter.basePath;
    return process.cwd();
  }

  /**
   * Resolve the KB root. Priority:
   *   1. Explicit `settings.kbRoot` if set.
   *   2. First ancestor of the vault root containing `kb.toml`.
   *   3. Vault root itself (CLI will complain if misconfigured).
   */
  private resolveKbRoot(): string {
    if (this.settings.kbRoot.trim()) return this.settings.kbRoot.trim();
    const vault = this.getVaultRoot();
    let dir = vault;
    // Cap the walk to avoid O(filesystem-root) in pathological cases.
    for (let i = 0; i < 40; i++) {
      const candidate = path.join(dir, "kb.toml");
      try {
        if (fs.existsSync(candidate)) return dir;
      } catch {
        // ignore — fall through to parent
      }
      const parent = path.dirname(dir);
      if (parent === dir) break;
      dir = parent;
    }
    return vault;
  }

  /**
   * Prepend `--model <name>` to any subcommand args when the user has
   * configured a model override. `--model` is a global flag on the kb CLI,
   * so it must come before the subcommand name.
   */
  private withGlobalArgs(subcommand: string, args: string[]): string[] {
    const globals: string[] = [];
    if (this.settings.model.trim()) {
      globals.push("--model", this.settings.model.trim());
    }
    return [...globals, subcommand, ...args];
  }

  private quoteArg(s: string): string {
    // POSIX single-quote escaping. Safe because execSync parses via /bin/sh.
    return `'${s.replace(/'/g, `'\\''`)}'`;
  }

  private buildCommandLine(subcommand: string, args: string[]): string {
    const parts = [this.settings.kbPath, ...this.withGlobalArgs(subcommand, args)];
    return parts.map((p, i) => (i === 0 ? p : this.quoteArg(p))).join(" ");
  }

  private executeCommand(subcommand: string, args: string[]): string {
    const cmd = this.buildCommandLine(subcommand, args);
    try {
      return execSync(cmd, {
        cwd: this.resolveKbRoot(),
        encoding: "utf-8",
        stdio: ["pipe", "pipe", "pipe"],
        maxBuffer: 16 * 1024 * 1024,
      });
    } catch (error) {
      const err = error as { stderr?: Buffer | string; message?: string };
      const message = err.stderr
        ? err.stderr.toString()
        : err.message || "Unknown error";
      throw new Error(`KB command failed: ${message}`);
    }
  }

  // -------------------- legacy commands --------------------

  private async executeCompile() {
    this.statusBar?.setText("KB: Compiling...");

    try {
      this.executeCommand("compile", []);
      this.statusBar?.setText("KB: Compile complete");
      new Notice("KB compile complete");
      setTimeout(() => this.statusBar?.setText("KB ready"), 3000);
    } catch (error) {
      this.statusBar?.setText("KB: Compile failed");
      new Notice(`KB compile failed: ${(error as Error).message}`);
    }
  }

  private async executeAsk(editor: Editor) {
    const question = await this.promptForInput("Enter your question for KB:");
    if (!question) return;

    this.statusBar?.setText("KB: Asking...");

    try {
      // --dry-run preserved from v0.0.1 behavior: the existing contract was
      // "insert a preview". Changing it here would break users of the prior
      // scaffold; bn-6y85 extends the plugin, it doesn't redefine Ask.
      const result = this.executeCommand("ask", [question, "--dry-run"]);
      editor.replaceSelection(result);
      this.statusBar?.setText("KB ready");
      new Notice("KB answer inserted");
    } catch (error) {
      this.statusBar?.setText("KB: Ask failed");
      new Notice(`KB ask failed: ${(error as Error).message}`);
    }
  }

  private async executeInspect(editor: Editor, view: MarkdownView) {
    const filePath = view.file?.path;
    if (!filePath) {
      new Notice("No file open");
      return;
    }

    this.statusBar?.setText("KB: Inspecting...");

    try {
      const result = this.executeCommand("inspect", [filePath]);
      editor.replaceSelection(result);
      this.statusBar?.setText("KB ready");
      new Notice("KB inspection inserted");
    } catch (error) {
      this.statusBar?.setText("KB: Inspect failed");
      new Notice(`KB inspect failed: ${(error as Error).message}`);
    }
  }

  // -------------------- new commands (bn-6y85) --------------------

  /**
   * Launch `kb chat` in an external terminal. We try the configured terminal
   * (if any), then `x-terminal-emulator`, then a small list of common
   * emulators. If none spawn, we fall back to a Notice with the exact command
   * the user should run themselves — the chat command is interactive, so
   * running it inside Obsidian is not meaningful.
   */
  private async executeChat() {
    const kbCmd = this.buildCommandLine("chat", []);
    const cwd = this.resolveKbRoot();

    const configured = this.settings.terminalCommand.trim();
    const candidates = configured
      ? [configured]
      : [
          "x-terminal-emulator",
          "gnome-terminal",
          "konsole",
          "kitty",
          "alacritty",
          "wezterm",
          "foot",
          "xterm",
        ];

    for (const term of candidates) {
      if (this.tryLaunchTerminal(term, kbCmd, cwd)) {
        new Notice(`KB chat launched in ${term}`);
        return;
      }
    }

    // No terminal worked. Show the exact command so the user can run it.
    new Notice(
      `No terminal emulator found. Run manually:\n${kbCmd}\n(cwd: ${cwd})`,
      10000,
    );
  }

  /**
   * Each emulator takes a different "run this command" flag. Rather than
   * curate a flag table we hand the emulator a `bash -lc` subshell that
   * runs the kb command and waits for a keypress — this works for every
   * emulator in the fallback list via their `-e` convention.
   *
   * Returns true if spawn() did not synchronously throw (ENOENT etc).
   * Actual success can only be observed by the user — errors after spawn
   * are surfaced inside the terminal.
   */
  private tryLaunchTerminal(term: string, kbCmd: string, cwd: string): boolean {
    const inner = `${kbCmd}; echo; echo '[kb chat exited — press Enter]'; read _`;
    try {
      const child = spawn(term, ["-e", "bash", "-lc", inner], {
        cwd,
        detached: true,
        stdio: "ignore",
      });
      child.on("error", () => {
        // Silently swallow — we'll try the next candidate. Asynchronous so
        // it cannot affect the synchronous "did spawn throw?" check.
      });
      child.unref();
      return true;
    } catch {
      return false;
    }
  }

  private async executeIngestCurrentNote() {
    const file = this.app.workspace.getActiveFile();
    if (!file) {
      new Notice("No active note");
      return;
    }
    const vault = this.getVaultRoot();
    const absPath = path.join(vault, file.path);

    this.statusBar?.setText("KB: Ingesting...");
    try {
      const out = this.executeCommand("ingest", [absPath]);
      this.statusBar?.setText("KB ready");
      new Notice(`KB ingest complete: ${file.path}`);
      if (out.trim()) new OutputModal(this.app, "KB ingest", out).open();
    } catch (error) {
      this.statusBar?.setText("KB: Ingest failed");
      new Notice(`KB ingest failed: ${(error as Error).message}`);
    }
  }

  private async executeIngestClipboardUrl() {
    let clip = "";
    try {
      clip = (await navigator.clipboard.readText()).trim();
    } catch {
      new Notice("Clipboard read denied");
      return;
    }
    if (!clip) {
      new Notice("Clipboard is empty");
      return;
    }
    if (!URL_PATTERN.test(clip)) {
      new Notice("clipboard has no URL");
      return;
    }

    this.statusBar?.setText("KB: Ingesting URL...");
    try {
      const out = this.executeCommand("ingest", [clip]);
      this.statusBar?.setText("KB ready");
      new Notice(`KB ingest complete: ${clip}`);
      if (out.trim()) new OutputModal(this.app, "KB ingest", out).open();
    } catch (error) {
      this.statusBar?.setText("KB: Ingest failed");
      new Notice(`KB ingest failed: ${(error as Error).message}`);
    }
  }

  /**
   * For an ask artifact, approve its promotion review item.
   *
   * Detection order (first match wins):
   *   1. frontmatter `promotion_id: <id>` or `review_id: <id>` (explicit)
   *   2. frontmatter `type: question_answer` → derive `review-{question_id}`
   *   3. frontmatter `kind: ask` (spec wording) → derive `review-{question_id}`
   *
   * Falls back with a Notice if the note doesn't look like an ask artifact.
   */
  private async executePromoteThisQuestion() {
    const file = this.app.workspace.getActiveFile();
    if (!file) {
      new Notice("No active note");
      return;
    }
    const fm = this.getFrontmatter(file);
    if (!fm) {
      new Notice("not a promotable artifact");
      return;
    }

    const explicit =
      this.fmString(fm, "promotion_id") ??
      this.fmString(fm, "review_id") ??
      this.fmString(fm, "promotion-id");

    let reviewId: string | undefined = explicit;
    if (!reviewId) {
      const type = this.fmString(fm, "type");
      const kind = this.fmString(fm, "kind");
      const questionId = this.fmString(fm, "question_id");
      const isAskArtifact =
        type === "question_answer" ||
        kind === "ask" ||
        kind === "question_answer";
      if (isAskArtifact && questionId) {
        reviewId = `review-${questionId}`;
      }
    }

    if (!reviewId) {
      new Notice("not a promotable artifact");
      return;
    }

    this.statusBar?.setText("KB: Approving...");
    try {
      const out = this.executeCommand("review", ["approve", reviewId]);
      this.statusBar?.setText("KB ready");
      new Notice(`KB promote approved: ${reviewId}`);
      if (out.trim()) new OutputModal(this.app, "KB review approve", out).open();
    } catch (error) {
      this.statusBar?.setText("KB: Approve failed");
      new Notice(`KB review approve failed: ${(error as Error).message}`);
    }
  }

  private getFrontmatter(file: TFile): Record<string, unknown> | null {
    const cache = this.app.metadataCache.getFileCache(file);
    const fm = cache?.frontmatter;
    return fm ? (fm as Record<string, unknown>) : null;
  }

  private fmString(
    fm: Record<string, unknown>,
    key: string,
  ): string | undefined {
    const v = fm[key];
    return typeof v === "string" && v.trim().length > 0 ? v : undefined;
  }

  private async executeSearch() {
    const query = await this.promptForInput("KB search query:");
    if (!query) return;

    this.statusBar?.setText("KB: Searching...");
    try {
      // `--json` is a global flag on kb; add it after the subcommand name so
      // its position relative to the query argument doesn't matter. clap
      // accepts global flags anywhere — but keeping them up-front is clearer.
      const raw = this.executeCommand("search", ["--json", query]);
      this.statusBar?.setText("KB ready");
      const results = parseSearchJson(raw);
      if (results.length === 0) {
        new Notice(`No KB results for: ${query}`);
        return;
      }
      new SearchResultsModal(this.app, query, results, (id) =>
        this.openSearchResult(id),
      ).open();
    } catch (error) {
      this.statusBar?.setText("KB: Search failed");
      new Notice(`KB search failed: ${(error as Error).message}`);
    }
  }

  /**
   * Open a search result. The `id` is a relative path from the KB root.
   * If that path happens to live inside the vault, we open it as a TFile;
   * otherwise we fall back to `openLinkText` (Obsidian will create or locate
   * the best match, or no-op if it can't).
   */
  private openSearchResult(id: string) {
    const kbRoot = this.resolveKbRoot();
    const vaultRoot = this.getVaultRoot();
    const abs = path.join(kbRoot, id);
    const rel = path.relative(vaultRoot, abs);

    // If the path escapes the vault, `relative` yields something starting
    // with `..`. In that case we don't have a TFile — use openLinkText as
    // a best-effort fallback.
    const inVault = !rel.startsWith("..") && !path.isAbsolute(rel);
    if (inVault) {
      const file = this.app.vault.getAbstractFileByPath(rel);
      if (file instanceof TFile) {
        this.app.workspace.getLeaf(false).openFile(file);
        return;
      }
    }
    this.app.workspace.openLinkText(id, "", false);
  }

  // -------------------- prompt UI --------------------

  private promptForInput(message: string): Promise<string | null> {
    return new Promise((resolve) => {
      const dialog = document.createElement("div");
      dialog.className = "kb-input-dialog";
      dialog.style.cssText = `
        position: fixed;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        background: var(--background-primary);
        border: 1px solid var(--divider-color);
        border-radius: 8px;
        padding: 20px;
        z-index: 10000;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        min-width: 300px;
      `;

      const title = document.createElement("div");
      title.textContent = message;
      title.style.marginBottom = "10px";

      const input = document.createElement("input");
      input.type = "text";
      input.style.cssText = `
        width: 100%;
        padding: 8px;
        border: 1px solid var(--divider-color);
        border-radius: 4px;
        font-family: var(--font-monospace);
      `;

      const buttonContainer = document.createElement("div");
      buttonContainer.style.cssText =
        "margin-top: 10px; display: flex; gap: 10px; justify-content: flex-end;";

      const okBtn = document.createElement("button");
      okBtn.textContent = "OK";
      okBtn.style.cssText = "padding: 6px 12px; cursor: pointer;";
      okBtn.onclick = () => {
        document.body.removeChild(dialog);
        resolve(input.value || null);
      };

      const cancelBtn = document.createElement("button");
      cancelBtn.textContent = "Cancel";
      cancelBtn.style.cssText = "padding: 6px 12px; cursor: pointer;";
      cancelBtn.onclick = () => {
        document.body.removeChild(dialog);
        resolve(null);
      };

      buttonContainer.appendChild(okBtn);
      buttonContainer.appendChild(cancelBtn);

      dialog.appendChild(title);
      dialog.appendChild(input);
      dialog.appendChild(buttonContainer);
      document.body.appendChild(dialog);

      input.focus();
      input.onkeydown = (e) => {
        if (e.key === "Enter") okBtn.click();
        if (e.key === "Escape") cancelBtn.click();
      };
    });
  }
}

// -------------------- helper types and parsers --------------------

interface KbSearchResult {
  id: string;
  title: string;
  score: number;
  reasons?: string[];
}

/**
 * The kb CLI's --json output wraps payloads in an envelope
 * `{ schema_version, command, data, warnings, errors }`. For `kb search` the
 * `data` field is a `Vec<SearchResult>`. We tolerate a bare array too, in
 * case the envelope shape changes in future versions.
 */
function parseSearchJson(raw: string): KbSearchResult[] {
  let parsed: unknown;
  try {
    parsed = JSON.parse(raw);
  } catch {
    return [];
  }

  const isResult = (v: unknown): v is KbSearchResult =>
    typeof v === "object" &&
    v !== null &&
    typeof (v as KbSearchResult).id === "string" &&
    typeof (v as KbSearchResult).title === "string";

  if (Array.isArray(parsed)) return parsed.filter(isResult);

  if (typeof parsed === "object" && parsed !== null) {
    const data = (parsed as { data?: unknown }).data;
    if (Array.isArray(data)) return data.filter(isResult);
  }
  return [];
}

// -------------------- modals --------------------

class OutputModal extends Modal {
  constructor(
    app: App,
    private title_: string,
    private body: string,
  ) {
    super(app);
  }

  onOpen() {
    const { contentEl } = this;
    contentEl.createEl("h3", { text: this.title_ });
    const pre = contentEl.createEl("pre");
    pre.style.cssText =
      "max-height: 60vh; overflow: auto; white-space: pre-wrap; padding: 8px; background: var(--background-secondary); border-radius: 4px;";
    pre.textContent = this.body;
  }

  onClose() {
    this.contentEl.empty();
  }
}

class SearchResultsModal extends Modal {
  constructor(
    app: App,
    private query: string,
    private results: KbSearchResult[],
    private onPick: (id: string) => void,
  ) {
    super(app);
  }

  onOpen() {
    const { contentEl } = this;
    contentEl.createEl("h3", { text: `KB search: ${this.query}` });

    const list = contentEl.createEl("div");
    list.style.cssText = "display: flex; flex-direction: column; gap: 6px; max-height: 60vh; overflow: auto;";

    for (const r of this.results) {
      const row = list.createEl("div");
      row.style.cssText =
        "padding: 8px; border: 1px solid var(--divider-color); border-radius: 4px; cursor: pointer;";
      row.addEventListener("mouseenter", () => {
        row.style.background = "var(--background-secondary)";
      });
      row.addEventListener("mouseleave", () => {
        row.style.background = "";
      });
      row.addEventListener("click", () => {
        this.close();
        this.onPick(r.id);
      });

      const titleEl = row.createEl("div", { text: r.title });
      titleEl.style.cssText = "font-weight: 600;";

      const meta = row.createEl("div", {
        text: `${r.id} [score: ${r.score}]`,
      });
      meta.style.cssText =
        "font-size: 0.85em; color: var(--text-muted); font-family: var(--font-monospace);";

      if (r.reasons && r.reasons.length > 0) {
        const reason = row.createEl("div", { text: r.reasons.join(" · ") });
        reason.style.cssText =
          "font-size: 0.8em; color: var(--text-faint); margin-top: 4px;";
      }
    }
  }

  onClose() {
    this.contentEl.empty();
  }
}

// -------------------- settings tab --------------------

class KbSettingTab extends PluginSettingTab {
  plugin: KbPlugin;

  constructor(app: App, plugin: KbPlugin) {
    super(app, plugin);
    this.plugin = plugin;
  }

  display(): void {
    const { containerEl } = this;

    containerEl.empty();

    new Setting(containerEl)
      .setName("KB path")
      .setDesc("Path to the kb binary (defaults to 'kb' in PATH)")
      .addText((text) =>
        text
          .setPlaceholder("kb")
          .setValue(this.plugin.settings.kbPath)
          .onChange(async (value) => {
            this.plugin.settings.kbPath = value;
            await this.plugin.saveSettings();
          }),
      );

    new Setting(containerEl)
      .setName("KB root")
      .setDesc(
        "Path to the KB root (containing kb.toml). Leave blank to auto-detect by walking up from the vault root.",
      )
      .addText((text) =>
        text
          .setPlaceholder("(auto-detect)")
          .setValue(this.plugin.settings.kbRoot)
          .onChange(async (value) => {
            this.plugin.settings.kbRoot = value;
            await this.plugin.saveSettings();
          }),
      );

    new Setting(containerEl)
      .setName("Model override")
      .setDesc(
        "LLM model passed to kb via --model (blank to use kb.toml default).",
      )
      .addText((text) =>
        text
          .setPlaceholder("(kb.toml default)")
          .setValue(this.plugin.settings.model)
          .onChange(async (value) => {
            this.plugin.settings.model = value;
            await this.plugin.saveSettings();
          }),
      );

    new Setting(containerEl)
      .setName("Terminal command")
      .setDesc(
        "External terminal emulator for 'KB Chat' (blank = try x-terminal-emulator, then common fallbacks).",
      )
      .addText((text) =>
        text
          .setPlaceholder("x-terminal-emulator")
          .setValue(this.plugin.settings.terminalCommand)
          .onChange(async (value) => {
            this.plugin.settings.terminalCommand = value;
            await this.plugin.saveSettings();
          }),
      );
  }
}
