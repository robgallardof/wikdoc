# Wikdoc


## Install / Update / Uninstall

- Install: `pip install -e .` (run from the repository root)
- Update: `git pull && pip install -e . --upgrade` (Git is required)
- Uninstall: `pip uninstall wikdoc`


## Optional: browser UI

If you want a **built-in browser UI**, start the lightweight Gradio front-end:
- From the interactive menu (`wikdoc start`), pick **Open Web UI (browser)** to open a tab automatically.
- From the CLI you can auto-open the browser with `--browser`:
  ```bash
  wikdoc webui /path/to/workspace --local-store --browser
  ```

- Change host/port with `--host` and `--port` (default 127.0.0.1:7860).
- Index the workspace first via `wikdoc index`.

Prefer an external dashboard? You can still expose the OpenAI-compatible API:
```bash
wikdoc serve
```

- Base URL: `http://127.0.0.1:17863/v1`
- API key: any value (ignored)
- Model: pick one of the `wikdoc:<workspace_id>` entries exposed by `/v1/models`

For a local-store index:
```bash
wikdoc serve --local-store --path /path/to/workspace
```

## Timeouts

If Ollama responses time out, increase the chat client timeout before launching Wikdoc:
```bash
export WIKDOC_CHAT_TIMEOUT=600  # seconds
```
