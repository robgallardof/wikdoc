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
wikdoc index "C:\path\to\workspace" --exclude "**/vendor/**,**/.next/**"
```

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
wikdoc pack-export "C:\path\to\workspace" --out "C:\Users\you\workspace.wikdocpack.zip"
```

### Import
```bash
wikdoc pack-import "C:\path\to\workspace" --in "C:\Users\you\workspace.wikdocpack.zip"
```

**Local vs Global store notes**
- Local store: `<workspace>/.wikdoc/...`
- Global store: `~/.wikdoc/...`

If you export while using the “wrong” scope, Wikdoc will auto-detect the index location and still export.

---

## Open WebUI (optional UI)

Wikdoc can launch **Open WebUI** (a web UI) and connect it to **Wikdoc’s OpenAI-compatible RAG API**, so your chats can use the indexed workspace context.

You can do it from the menu:
- **Ollama / Open WebUI** → Start Open WebUI

Or manually:
```bash
pip install open-webui
open-webui serve
```

Then open:
- `http://127.0.0.1:8080`

### Connect Open WebUI to Wikdoc (recommended)

1) **Index your workspace** (Menu → `Index workspace`).

2) Start the **Wikdoc API server**:
```bash
wikdoc serve
```

3) In Open WebUI, go to:
`Settings → Connections → Direct Connections` (or “OpenAI API”).

Set:
- **OpenAI API Base URL:** `http://127.0.0.1:17863/v1`
- **API Key:** anything (not used)

4) Refresh models in Open WebUI and select a model like:
- `wikdoc:<workspace_id>`

Notes:
- `wikdoc serve` defaults to the **global store** (`~/.wikdoc`).
- If you indexed using **Local store**, start the API like this:
```bash
wikdoc serve --local-store --path "C:\path\to\workspace"
```

---

## Troubleshooting

### Ollama 500 errors during indexing
Usually means the embed model is missing or the chunk is too big.
Try:
- `ollama pull nomic-embed-text`
- Re-run indexing with smaller chunks:
```bash
wikdoc index "C:\path\to\workspace" --chunk-chars 3000 --chunk-overlap 400
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
