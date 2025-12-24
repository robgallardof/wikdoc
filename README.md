# Wikdoc

Wikdoc turns **any folder** (C#, Java, TS/JS, Python, etc.) into a **local, searchable knowledge base** you can chat with.

It does **not** “train” an LLM in the fine-tuning sense — instead it builds a **local index (embeddings)** and uses **RAG** (retrieve + answer) so you can ask questions about your code, configs, docs, and architecture.

---

## Requirements

### 1) Python
- **Python 3.11** recommended (Windows/macOS/Linux).

### 2) Ollama (models + local API)
Wikdoc uses Ollama for:
- **Embeddings** (indexing)
- **LLM** (answers)

Install Ollama and make sure this works:
```bash
ollama --version
```

Pull recommended models:
```bash
ollama pull nomic-embed-text
ollama pull qwen2.5-coder:7b
```

### 3) Git (for updates)
Git is required to pull the repository and refresh your install with the latest changes.

---

## Install

From the project folder (where `pyproject.toml` lives):
```bash
pip install -e .
```

Verify:
```bash
wikdoc --help
```

### Update
Pull the latest code and reinstall to pick up dependency changes:
```bash
git pull
pip install -e . --upgrade
```

> Tip: run the update before launching `wikdoc start` so the menu uses the newest code.

### Uninstall
Remove Wikdoc from your environment:
```bash
pip uninstall wikdoc
```
Optional: delete any local/global stores you no longer need at `<workspace>/.wikdoc` and `~/.wikdoc`.

---

## Quick start (recommended)

### 1) Launch the interactive menu
```bash
wikdoc start
```

### 2) In the menu, do this order:
1. **Workspace** → set the folder you want to index
2. **Index workspace** → builds the embeddings + SQLite store
3. **Chat** → ask questions (multi-turn, best experience)
4. **Open Web UI** → optional, launches the browser UI in a new tab

> Tip: **“Ask a question”** is a *single-shot* prompt.  
> If you want a normal conversation, use **Chat**.

---

## What gets indexed?

### File extensions (languages)
By default, Wikdoc includes many extensions, including:
- **JavaScript:** `.js`
- **TypeScript:** `.ts`, `.tsx`
- **C#:** `.cs`, `.csproj`, `.sln`
- **Python:** `.py`
- and more…

The defaults are in: `wikdoc/config.py` → `DEFAULT_INCLUDE_EXT`.

You can override per run:
```bash
wikdoc index "C:\path\to\workspace" --include-ext js,ts,tsx
```

### Ignored folders / files
Yes — you should ignore heavy/generated folders like `node_modules`.

Wikdoc already ignores common junk by default (including `**/node_modules/**`, `bin/`, `obj/`, `.git/`, lockfiles, minified assets, etc.).

The defaults are in: `wikdoc/config.py` → `DEFAULT_EXCLUDE_GLOBS`.

You can add your own excludes:
```bash
wikdoc index "C:\path\to\workspace" --exclude-globs "**/vendor/**,**/.next/**"
```

### Workspace settings (recommended)
You can keep indexing rules in your workspace so you don’t have to repeat them.
Create a file at:
```
<workspace>/.wikdoc/settings.json
```

Example:
```json
{
  "include_ext": ["cs", "csproj", "sln", "json", "md"],
  "exclude_globs": ["**/*.appsettings.json", "**/node_modules/**"],
  "max_file_mb": 2.0,
  "chunk_chars": 6000,
  "chunk_overlap_chars": 800
}
```

This keeps defaults in code (best practice) and lets you override per workspace
without changing global behavior.

---

## CLI commands (optional)

> You can use the menu for everything, but the CLI is useful for scripting.

### Index
```bash
wikdoc index "C:\path\to\workspace" --local-store
```

### Single question
```bash
wikdoc ask "C:\path\to\workspace" "Where is authentication configured?" --local-store
```

### Chat (interactive)
```bash
wikdoc chat "C:\path\to\workspace" --local-store
```

### Generate Markdown docs
```bash
wikdoc docs "C:\path\to\workspace" --out ./docs --template wiki --local-store
```

---

## Packs (export / import an index)

Packs let you **share** an index with a friend (or move it between machines) without re-indexing.

- Export creates a `.wikdocpack.zip`
- Import restores it into your store

### Export
```bash
wikdoc export-pack "C:\path\to\workspace" "C:\Users\you\workspace.wikdocpack.zip"
```

### Import
```bash
wikdoc import-pack "C:\Users\you\workspace.wikdocpack.zip" "C:\path\to\workspace"
```

**Local vs Global store notes**
- Local store: `<workspace>/.wikdoc/...`
- Global store: `~/.wikdoc/...`

If you export while using the “wrong” scope, Wikdoc will auto-detect the index location and still export.

---

## Opcional: UI en el navegador

Wikdoc ahora incluye una UI ligera basada en Gradio para que puedas preguntar desde el navegador sin depender de Open WebUI.

- Desde el menú interactivo (`wikdoc start`), usa la opción **Open Web UI (browser)** para abrir una pestaña automáticamente (127.0.0.1:7860).
- Desde la CLI puedes hacer que se abra el navegador con `--browser`:
  ```bash
  wikdoc webui "C:\\path\\to\\workspace" --local-store --browser
  ```

- Host/puerto: `--host 0.0.0.0 --port 7860` si quieres abrirlo a tu red local.
- Asegúrate de haber indexado primero el workspace con `wikdoc index`.

### 2) Usa cualquier cliente OpenAI-compatible (opcional)

Si prefieres otro dashboard (LM Studio, etc.), puedes seguir usando el endpoint OpenAI-compatible de Wikdoc:
```bash
wikdoc serve
```

- **Base URL:** `http://127.0.0.1:17863/v1`
- **API key:** cualquiera (se ignora)
- **Model:** elige uno de los `wikdoc:<workspace_id>` devueltos por `/v1/models`

Para servir un workspace en su store local:
```bash
wikdoc serve --local-store --path "C:\\path\\to\\workspace"
```

---

## Troubleshooting

### Ollama 500 errors during indexing
Usually means the embed model is missing or the chunk is too big.
Try:
- `ollama pull nomic-embed-text`
- Re-run indexing with smaller chunks by updating `.wikdoc/settings.json`:
```json
{
  "chunk_chars": 3000,
  "chunk_overlap_chars": 400
}
```

### Export says the index DB is missing
That means the workspace was never indexed **in that store scope** (Wikdoc looks for `chunks.sqlite3` or legacy `store.sqlite`).
Run **Index workspace** first, or switch Local/Global store.
(Export also tries to auto-detect local vs global if you picked the wrong one.)

---

## Project structure

- `wikdoc/scan.py` — finds files in your folder
- `wikdoc/chunking/` — splits files into chunks
- `wikdoc/embeddings/` — Ollama embeddings client
- `wikdoc/vectordb/` — SQLite + NumPy vector store
- `wikdoc/rag/` — retrieve + answer
- `wikdoc/ui/` — Rich-based interactive menu

---

## License

MIT
