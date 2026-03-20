#!/usr/bin/env python3
"""
Flask-based Local RAG Web App
==============================
Uses your existing RAG code paths:
  - document loading via `src/cli.py` (cli.load_documents)
  - retrieval + answer generation via `src/rag_pipeline.py`

UI:
  - GET  /                serves `templates/index.html`

API:
  - POST /api/init       { model_path, data_dir, show_sources }
  - GET  /api/init_status?id=...
  - POST /api/chat      { query }
  - GET  /api/chat_status?id=...
"""

from __future__ import annotations

import json
import os
import threading
import time
import traceback
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Optional

from flask import Flask, jsonify, render_template, request

import cli as cli_module
from document_store import DocumentStore
from local_llm import LocalLLM
from rag_pipeline import RAGPipeline


def build_help_text() -> str:
    # Match `src/cli.py` `print_help()` content.
    return (
        "Commands:\n"
        "  /help     Show this help message\n"
        "  /sources  Toggle source display on/off\n"
        "  /quit     Exit the program\n"
        "\n"
        "Just type your question and press Enter!\n"
    )


HTML_TITLE = "Local RAG Web App"

DEFAULT_MODEL_PATH: str = ""
DEFAULT_DATA_DIR: str = ""


def _auto_detect_paths() -> tuple[str, str]:
    """
    Best-effort auto-detection using project conventions:
      - model:   models/*.gguf
      - data dir: data/
    """
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    model_dir = os.path.join(repo_root, "models")
    data_dir = os.path.join(repo_root, "data")

    model_path = ""
    if os.path.isdir(model_dir):
        candidates = [f for f in os.listdir(model_dir) if f.lower().endswith(".gguf")]
        if candidates:
            # Prefer common model naming patterns first.
            prefer_keywords = ["tinyllama", "llama", "mistral"]
            candidates.sort()

            def score(name: str) -> int:
                low = name.lower()
                for i, kw in enumerate(prefer_keywords):
                    if kw in low:
                        return 100 - i
                return 0

            best = max(candidates, key=score)
            model_path = os.path.join(model_dir, best)

    detected_data = data_dir if os.path.isdir(data_dir) else ""
    return model_path, detected_data


def get_default_paths() -> tuple[str, str]:
    # Command args override auto-detection. Env vars can also override.
    global DEFAULT_MODEL_PATH, DEFAULT_DATA_DIR

    model = os.environ.get("RAG_MODEL_PATH") or DEFAULT_MODEL_PATH
    data_dir = os.environ.get("RAG_DATA_DIR") or DEFAULT_DATA_DIR

    if not model or not data_dir:
        auto_model, auto_data = _auto_detect_paths()
        model = model or auto_model
        data_dir = data_dir or auto_data

    return model, data_dir


@dataclass
class Job:
    status: str  # queued|running|done|failed
    message: str = ""
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class AppState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.pipeline: Optional[RAGPipeline] = None
        self.show_sources: bool = True
        self.init_job: Optional[str] = None
        self.chat_job: Optional[str] = None
        self.jobs: Dict[str, Job] = {}

    def new_job_id(self) -> str:
        return uuid.uuid4().hex


STATE = AppState()

app = Flask(__name__, template_folder="templates", static_folder="static")


@app.get("/")
def index():
    default_model_path, default_data_dir = get_default_paths()
    return render_template(
        "index.html",
        title=HTML_TITLE,
        default_model_path=default_model_path,
        default_data_dir=default_data_dir,
    )


@app.get("/api/health")
def health():
    return jsonify({"ok": True})


def _get_job(job_id: str) -> Job:
    job = STATE.jobs.get(job_id)
    if not job:
        raise KeyError(job_id)
    return job


@app.post("/api/init")
def api_init():
    body = request.get_json(silent=True) or {}
    model_path = str(body.get("model_path") or "").strip()
    data_dir = str(body.get("data_dir") or "").strip()
    show_sources = bool(body.get("show_sources", True))

    if not model_path:
        return jsonify({"error": "Missing model_path"}), 400
    if not os.path.exists(model_path):
        return jsonify({"error": f"Model not found: {model_path}"}), 400
    if not data_dir or not os.path.isdir(data_dir):
        return jsonify({"error": f"Data folder not found: {data_dir}"}), 400

    with STATE.lock:
        # Prevent concurrent init jobs.
        if STATE.init_job:
            existing = STATE.jobs.get(STATE.init_job)
            if existing and existing.status in ("queued", "running"):
                return jsonify({"error": "Initialization already running"}), 409

        STATE.show_sources = show_sources
        job_id = STATE.new_job_id()
        STATE.init_job = job_id
        STATE.jobs[job_id] = Job(status="queued", message="Queued")

    t = threading.Thread(
        target=_init_worker,
        args=(job_id, model_path, data_dir),
        daemon=True,
    )
    t.start()
    return jsonify({"id": job_id})


@app.get("/api/init_status")
def api_init_status():
    job_id = str(request.args.get("id") or "").strip()
    if not job_id:
        return jsonify({"error": "Missing id"}), 400

    try:
        job = _get_job(job_id)
    except KeyError:
        return jsonify({"error": "Job not found"}), 404

    with STATE.lock:
        payload: Dict[str, Any] = {"status": job.status, "message": job.message}
        if job.status == "done" and job.result is not None:
            payload.update(job.result)
        if job.status == "failed":
            payload["error"] = job.error or "Unknown error"
        return jsonify(payload)


@app.post("/api/chat")
def api_chat():
    body = request.get_json(silent=True) or {}
    query = str(body.get("query") or "").strip()
    if not query:
        return jsonify({"error": "Missing query"}), 400

    with STATE.lock:
        if STATE.pipeline is None:
            return jsonify({"error": "System not initialized yet"}), 409

        # Only allow one chat job at a time (single-user local app).
        if STATE.chat_job:
            existing = STATE.jobs.get(STATE.chat_job)
            if existing and existing.status in ("queued", "running"):
                return jsonify({"error": "Another chat is running"}), 409

        job_id = STATE.new_job_id()
        STATE.chat_job = job_id
        STATE.jobs[job_id] = Job(status="queued", message="Queued")

    t = threading.Thread(
        target=_chat_worker,
        args=(job_id, query),
        daemon=True,
    )
    t.start()
    return jsonify({"id": job_id})


@app.get("/api/chat_status")
def api_chat_status():
    job_id = str(request.args.get("id") or "").strip()
    if not job_id:
        return jsonify({"error": "Missing id"}), 400

    try:
        job = _get_job(job_id)
    except KeyError:
        return jsonify({"error": "Job not found"}), 404

    with STATE.lock:
        payload: Dict[str, Any] = {"status": job.status, "message": job.message}
        if job.status == "done" and job.result is not None:
            payload.update(job.result)
        if job.status == "failed":
            payload["error"] = job.error or "Unknown error"
        return jsonify(payload)


def _init_worker(job_id: str, model_path: str, data_dir: str) -> None:
    try:
        with STATE.lock:
            STATE.jobs[job_id].status = "running"
            STATE.jobs[job_id].message = "Loading embedding model..."

        doc_store = DocumentStore()

        with STATE.lock:
            STATE.jobs[job_id].message = "Loading documents..."

        texts, sources = cli_module.load_documents(data_dir)
        if not texts:
            raise RuntimeError("No `.txt` or `.md` files found in the data folder.")

        with STATE.lock:
            STATE.jobs[job_id].message = f"Indexing {len(sources)} sources..."
        doc_store.add_documents(texts, sources)

        with STATE.lock:
            STATE.jobs[job_id].message = "Loading local LLM..."
        llm = LocalLLM(model_path=model_path)

        with STATE.lock:
            STATE.jobs[job_id].message = "Building pipeline..."
        pipeline = RAGPipeline(doc_store, llm)

        with STATE.lock:
            STATE.pipeline = pipeline
            STATE.jobs[job_id].status = "done"
            STATE.jobs[job_id].message = "Ready"
            STATE.jobs[job_id].result = {
                "show_sources": STATE.show_sources,
                "help_text": build_help_text(),
            }
    except Exception as e:
        with STATE.lock:
            STATE.jobs[job_id].status = "failed"
            STATE.jobs[job_id].error = str(e)
            STATE.jobs[job_id].message = "Failed"
            STATE.jobs[job_id].result = None
            STATE.pipeline = None


def _chat_worker(job_id: str, query: str) -> None:
    start_time = time.time()
    try:
        with STATE.lock:
            STATE.jobs[job_id].status = "running"
            STATE.jobs[job_id].message = "Processing..."
            pipeline = STATE.pipeline

        if pipeline is None:
            raise RuntimeError("Pipeline not ready.")

        # CLI-style commands start with "/"
        if query.startswith("/"):
            cmd = query.strip().lower()
            if cmd in ("/quit", "/exit", "/q"):
                result = {
                    "answer": "Close the browser tab (or refresh) to exit.",
                    "sources": [],
                    "confidence": 0.0,
                    "elapsed": time.time() - start_time,
                    "show_sources": STATE.show_sources,
                }
                with STATE.lock:
                    STATE.jobs[job_id].status = "done"
                    STATE.jobs[job_id].message = "Done"
                    STATE.jobs[job_id].result = result
                    STATE.chat_job = None
                return

            if cmd == "/help":
                result = {
                    "answer": build_help_text(),
                    "sources": [],
                    "confidence": 0.0,
                    "elapsed": time.time() - start_time,
                    "show_sources": STATE.show_sources,
                }
                with STATE.lock:
                    STATE.jobs[job_id].status = "done"
                    STATE.jobs[job_id].message = "Done"
                    STATE.jobs[job_id].result = result
                    STATE.chat_job = None
                return

            if cmd == "/sources":
                with STATE.lock:
                    STATE.show_sources = not STATE.show_sources
                    status = "ON" if STATE.show_sources else "OFF"
                result = {
                    "answer": f"Source display: {status}",
                    "sources": [],
                    "confidence": 0.0,
                    "elapsed": time.time() - start_time,
                    "show_sources": STATE.show_sources,
                }
                with STATE.lock:
                    STATE.jobs[job_id].status = "done"
                    STATE.jobs[job_id].message = "Done"
                    STATE.jobs[job_id].result = result
                    STATE.chat_job = None
                return

            result = {
                "answer": f"Unknown command: {query.strip()}\nType /help for available commands.",
                "sources": [],
                "confidence": 0.0,
                "elapsed": time.time() - start_time,
                "show_sources": STATE.show_sources,
            }
            with STATE.lock:
                STATE.jobs[job_id].status = "done"
                STATE.jobs[job_id].message = "Done"
                STATE.jobs[job_id].result = result
                STATE.chat_job = None
            return

        with STATE.lock:
            STATE.jobs[job_id].message = "Thinking..."

        response = pipeline.query(query)
        elapsed = time.time() - start_time

        sources_out = [
            {
                "source": s.source,
                "chunk_id": s.chunk_id,
                "similarity_score": s.similarity_score,
            }
            for s in response.sources
        ]

        result = {
            "answer": response.answer,
            "sources": sources_out,
            "confidence": float(response.confidence),
            "elapsed": float(elapsed),
            "show_sources": STATE.show_sources,
        }

        with STATE.lock:
            STATE.jobs[job_id].status = "done"
            STATE.jobs[job_id].message = "Done"
            STATE.jobs[job_id].result = result
            STATE.chat_job = None
    except Exception as e:
        with STATE.lock:
            STATE.jobs[job_id].status = "failed"
            STATE.jobs[job_id].error = str(e)
            STATE.jobs[job_id].message = "Failed"
            STATE.jobs[job_id].result = None
            STATE.chat_job = None


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Local RAG Web App (Flask)")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=7860, help="Port to listen on")
    parser.add_argument("--model", default="", help="Path to .gguf model file (optional)")
    parser.add_argument("--data", default="", help="Path to documents folder (optional)")
    args = parser.parse_args()

    global DEFAULT_MODEL_PATH, DEFAULT_DATA_DIR
    DEFAULT_MODEL_PATH = args.model
    DEFAULT_DATA_DIR = args.data

    # `use_reloader=False` prevents duplicate init/chat workers during dev reload.
    app.run(host=args.host, port=args.port, threaded=True, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()

