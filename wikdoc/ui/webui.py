"""Lightweight browser UI for chatting with a Wikdoc workspace.

This uses Gradio to render a minimal chat page that runs the same
RAG pipeline as the CLI, so you can ask questions from a browser
without depending on external dashboards.
"""

from __future__ import annotations

from typing import Optional

import gradio as gr

from ..config import BackendOptions, RuntimeOptions, Workspace
from ..embeddings.ollama import OllamaEmbedder
from ..packs import write_workspace_json
from ..rag.answer import OllamaLLM
from ..rag.prompt import build_prompt
from ..rag.retrieve import retrieve
from ..vectordb.sqlite_numpy import SQLiteNumpyVectorStore
from ..store.layout import StoreLayout, default_store_dir


class WorkspaceChatSession:
    """Encapsulates the components needed to answer questions for a workspace."""

    def __init__(
        self,
        path: str,
        local_store: bool,
        top_k: int,
        llm_model: str,
        embed_model: str,
        ollama_host: str,
        name: Optional[str] = None,
    ) -> None:
        self.workspace = Workspace.from_path(path, name=name)
        store_root = default_store_dir(local_store=local_store, workspace_root=self.workspace.root)
        self.layout = StoreLayout(base_dir=store_root)
        self.wdir = self.layout.ensure(self.workspace)

        backend = BackendOptions(
            embedder="ollama",
            llm="ollama",
            model="",
            embed_model=embed_model,
            ollama_host=ollama_host,
        )
        try:
            write_workspace_json(self.wdir, self.workspace, backend)
        except Exception:
            # Metadata persistence is best-effort and should not block the UI.
            pass

        self.store = SQLiteNumpyVectorStore(store_dir=self.wdir)
        self.embedder = OllamaEmbedder(host=ollama_host, model=embed_model)
        self.llm = OllamaLLM(host=ollama_host, model=llm_model)
        self.runtime = RuntimeOptions(top_k=top_k)

    def answer(self, question: str) -> tuple[str, list[tuple[str, str]]]:
        """Return an answer and the top sources."""

        qvec = self.embedder.embed([question])[0]
        hits = retrieve(self.store, qvec, top_k=self.runtime.top_k)

        if not hits:
            return "No relevant context found in the index.", []

        messages = build_prompt(question=question, hits=hits, max_context_chars=self.runtime.max_context_chars)
        answer = self.llm.chat(messages)
        sources = [(f"{h.path}:{h.start_line}-{h.end_line}", f"score={h.score:.3f}") for h in hits[: min(10, len(hits))]]
        return answer, sources

    def close(self) -> None:
        self.store.close()


def _format_sources(sources: list[tuple[str, str]]) -> str:
    if not sources:
        return "No relevant sources for this question."
    lines = ["### Sources", "", *(f"- `{src}` ({score})" for src, score in sources)]
    return "\n".join(lines)


def launch_webui(
    path: str,
    local_store: bool,
    top_k: int,
    llm_model: str,
    embed_model: str,
    ollama_host: str,
    host: str,
    port: int,
    name: Optional[str] = None,
    open_browser: bool = False,
) -> None:
    """Start a minimal browser UI for chatting with a workspace."""

    session = WorkspaceChatSession(
        path=path,
        local_store=local_store,
        top_k=top_k,
        llm_model=llm_model,
        embed_model=embed_model,
        ollama_host=ollama_host,
        name=name,
    )

    description = (
        "## Wikdoc Web UI\n"
        "Chat with your indexed workspace using Ollama. Ask a question in the textbox and get an answer with sources."
    )

    with gr.Blocks(title="Wikdoc Web UI") as demo:
        gr.Markdown(description)
        chat = gr.Chatbot(label="Chat")
        sources_md = gr.Markdown("", label="Sources")
        question = gr.Textbox(
            label="Question",
            placeholder="What does the authentication service do?",
            lines=2,
        )

        def respond(message: str, history: list[tuple[str, str]]):
            answer, sources = session.answer(message)
            history = history + [(message, answer)]
            return history, _format_sources(sources)

        question.submit(respond, [question, chat], [chat, sources_md])

    try:
        demo.queue().launch(
            server_name=host,
            server_port=port,
            show_error=True,
            inbrowser=open_browser,
        )
    finally:
        session.close()
