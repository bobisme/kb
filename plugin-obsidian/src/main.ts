import { App, Plugin, PluginSettingTab, Setting, Editor, MarkdownView, Notice } from "obsidian";
import { execSync } from "child_process";

interface KbPluginSettings {
  kbPath: string;
}

const DEFAULT_SETTINGS: KbPluginSettings = {
  kbPath: "kb",
};

export default class KbPlugin extends Plugin {
  settings: KbPluginSettings;
  statusBar: any;

  async onload() {
    await this.loadSettings();

    // Register Compile command
    this.addCommand({
      id: "kb-compile",
      name: "KB Compile",
      callback: () => this.executeCompile(),
    });

    // Register Ask command
    this.addCommand({
      id: "kb-ask",
      name: "KB Ask",
      editorCallback: (editor: Editor, _view: MarkdownView) =>
        this.executeAsk(editor),
    });

    // Register Inspect command
    this.addCommand({
      id: "kb-inspect",
      name: "KB Inspect",
      editorCallback: (editor: Editor, view: MarkdownView) =>
        this.executeInspect(editor, view),
    });

    // Status bar item
    this.statusBar = this.addStatusBarItem();
    this.statusBar.setText("KB ready");

    // Add settings tab
    this.addSettingTab(new KbSettingTab(this.app, this));
  }

  onunload() {}

  async loadSettings() {
    this.settings = Object.assign(
      {},
      DEFAULT_SETTINGS,
      await this.loadData()
    );
  }

  async saveSettings() {
    await this.saveData(this.settings);
  }

  private getVaultRoot(): string {
    // Try to get basePath from adapter (works for FileSystemAdapter)
    const adapter = this.app.vault.adapter as any;
    if (adapter?.basePath) {
      return adapter.basePath;
    }
    // Fallback to current working directory
    return process.cwd();
  }

  private executeCommand(args: string[]): string {
    try {
      const cmd = `${this.settings.kbPath} ${args.join(" ")}`;
      const result = execSync(cmd, {
        cwd: this.getVaultRoot(),
        encoding: "utf-8",
        stdio: ["pipe", "pipe", "pipe"],
      });
      return result;
    } catch (error: any) {
      const message = error.stderr
        ? error.stderr.toString()
        : error.message || "Unknown error";
      throw new Error(`KB command failed: ${message}`);
    }
  }

  private async executeCompile() {
    this.statusBar.setText("KB: Compiling...");

    try {
      const result = this.executeCommand(["compile"]);
      this.statusBar.setText("KB: Compile complete");
      new Notice("KB compile complete");

      // Keep status for 3 seconds, then reset
      setTimeout(() => {
        this.statusBar.setText("KB ready");
      }, 3000);
    } catch (error: any) {
      this.statusBar.setText("KB: Compile failed");
      new Notice(`KB compile failed: ${error.message}`);
    }
  }

  private async executeAsk(editor: Editor) {
    // Prompt for question
    const question = await this.promptForInput("Enter your question for KB:");
    if (!question) return;

    this.statusBar.setText("KB: Asking...");

    try {
      const result = this.executeCommand(["ask", question, "--dry-run"]);
      editor.replaceSelection(result);
      this.statusBar.setText("KB ready");
      new Notice("KB answer inserted");
    } catch (error: any) {
      this.statusBar.setText("KB: Ask failed");
      new Notice(`KB ask failed: ${error.message}`);
    }
  }

  private async executeInspect(editor: Editor, view: MarkdownView) {
    const filePath = view.file?.path;
    if (!filePath) {
      new Notice("No file open");
      return;
    }

    this.statusBar.setText("KB: Inspecting...");

    try {
      const result = this.executeCommand(["inspect", filePath]);
      editor.replaceSelection(result);
      this.statusBar.setText("KB ready");
      new Notice("KB inspection inserted");
    } catch (error: any) {
      this.statusBar.setText("KB: Inspect failed");
      new Notice(`KB inspect failed: ${error.message}`);
    }
  }

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
      buttonContainer.style.cssText = "margin-top: 10px; display: flex; gap: 10px; justify-content: flex-end;";

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
      input.onkeypress = (e) => {
        if (e.key === "Enter") okBtn.click();
        if (e.key === "Escape") cancelBtn.click();
      };
    });
  }
}

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
          })
      );
  }
}
