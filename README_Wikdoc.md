# Wikdoc


## Optional: Open WebUI (browser UI)

Wikdoc is a CLI/TUI tool. If you want a **browser UI** for local chat with your Ollama models, install **Open WebUI** and point it at your Ollama server.

### Install
```bash
pip install -U open-webui
```

### Run
```bash
open-webui serve
```

Open it at:
- http://127.0.0.1:8080

### From Wikdoc
Use:
- Main Menu → `10) Open WebUI (install/run)`

Wikdoc launches it best-effort and sets `OLLAMA_BASE_URL` to your configured Ollama host.


## Use Open WebUI with your Wikdoc index (RAG)

Open WebUI is a browser UI. By default it can chat with **models**, but it does not know about your Wikdoc index.
Wikdoc can expose a **local RAG API** (OpenAI-compatible schema) so Open WebUI can query your indexed codebase.

### 1) Index your workspace (Wikdoc)
From `wikdoc start`:
- Workspace → set path
- Index workspace

### 2) Start the Wikdoc RAG API
From `wikdoc start`:
- `10) Web UI Integration`
- `1) Start Wikdoc RAG API`

This starts a local API like:
- `http://127.0.0.1:17863/v1`

### 3) Connect Open WebUI to the Wikdoc API
In Open WebUI settings:
- Provider: **OpenAI-compatible**
- Base URL: `http://127.0.0.1:17863/v1`
- API key: any value (Wikdoc ignores it)

Then select a model named like:
- `wikdoc:<workspace_id>`

Each Wikdoc workspace acts like a separate "knowledge base" you can select in Open WebUI.
