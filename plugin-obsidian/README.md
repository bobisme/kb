# KB Obsidian Plugin

Thin Obsidian plugin wrapper that exposes `kb` commands as Obsidian editor commands.

## Features

- **KB Compile** - Run `kb compile` and show status in the status bar
- **KB Ask** - Ask the knowledge base a question and insert the answer at the cursor
- **KB Inspect** - Inspect the current file in the knowledge base and insert the result

## Installation

1. Clone this repository into your Obsidian vault's `.obsidian/plugins/kb-obsidian` directory
2. Run `npm install` to install dependencies
3. Run `npm run build` to build the plugin
4. Reload Obsidian (or toggle the plugin off/on in settings)

## Usage

All three commands are available in the Obsidian command palette:

- Open command palette (`Ctrl/Cmd + P`) and search for:
  - `KB Compile` - compiles the knowledge base
  - `KB Ask` - prompts for a question and inserts the answer
  - `KB Inspect` - shows the inspection details for the current file

## Configuration

You can configure the path to the `kb` binary in the plugin settings. By default, it uses `kb` from your PATH.

## Development

```bash
npm install
npm run dev      # Watch mode
npm run build    # Production build
```

The plugin compiles to `main.js` which is loaded by Obsidian.

## Technical Details

- Built with TypeScript
- Uses esbuild for bundling
- Shells out to the `kb` CLI using Node.js `child_process`
- Integrates with Obsidian's editor, status bar, and settings UI
