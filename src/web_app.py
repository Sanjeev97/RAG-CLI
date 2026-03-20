#!/usr/bin/env python3
"""
Local RAG Web App (stdlib HTTP server)
=======================================
Web-based replacement for the CLI chat loop.

Endpoints
---------
GET  /                 -> HTML UI
GET  /api/health       -> {"ok": true}
POST /api/init        -> start pipeline init in background
GET  /api/init_status -> poll init job status/result
POST /api/chat        -> start chat/query in background
GET  /api/chat_status -> poll chat job status/result
"""

from __future__ import annotations

import argparse
import json
import os
import threading
import time
import traceback
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, urlparse

from document_store import DocumentStore
from local_llm import LocalLLM
import cli as cli_module
from rag_pipeline import RAGPipeline


HTML_PAGE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>RAG Web App</title>
  <style>
    :root { --bg: #0b1220; --panel: #111a2e; --text: #e6edf3; --muted: #9fb3c8; --accent: #3b82f6; --danger: #ef4444; --ok: #22c55e; }
    body { margin: 0; font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; background: var(--bg); color: var(--text); }
    .wrap { display: grid; grid-template-rows: auto 1fr auto; gap: 10px; height: 100vh; padding: 14px; box-sizing: border-box; }
    header { display: flex; align-items: center; justify-content: space-between; gap: 10px; padding: 10px 12px; background: var(--panel); border-radius: 10px; }
    header .title { font-weight: 700; }
    .grid { display: grid; grid-template-columns: 360px 1fr; gap: 12px; min-height: 0; }
    .panel { background: var(--panel); border-radius: 10px; padding: 12px; min-height: 0; }
    .row { display: flex; gap: 8px; align-items: center; margin: 8px 0; }
    label { color: var(--muted); font-size: 13px; width: 120px; flex: 0 0 auto; }
    input[type="text"], input[type="file"]{ width: 100%; padding: 9px 10px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.08); background: rgba(255,255,255,0.04); color: var(--text); }
    button { padding: 9px 12px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.12); background: rgba(255,255,255,0.06); color: var(--text); cursor: pointer; }
    button.primary { border-color: rgba(59,130,246,0.6); background: rgba(59,130,246,0.2); }
    button.danger { border-color: rgba(239,68,68,0.6); background: rgba(239,68,68,0.15); }
    button:disabled { opacity: 0.6; cursor: not-allowed; }
    .chat { display: flex; flex-direction: column; gap: 10px; }
    .status { color: var(--muted); font-size: 13px; }
    #log { flex: 1; overflow: auto; background: rgba(255,255,255,0.03); border-radius: 10px; padding: 12px; }
    .msg { margin: 10px 0; white-space: pre-wrap; line-height: 1.35; }
    .msg.user { color: var(--accent); }
    .msg.assistant { color: var(--ok); }
    .msg.system { color: var(--muted); }
    .sources { margin-top: 8px; padding-left: 14px; border-left: 3px solid rgba(106,27,154,0.5); color: var(--muted); font-size: 13px; }
    .confidence { margin-top: 6px; color: var(--muted); font-size: 13px; }
    .bar { display: inline-block; vertical-align: middle; letter-spacing: 0.5px; }
    .small { font-size: 13px; color: var(--muted); }
    textarea { width: 100%; min-height: 38px; max-height: 120px; resize: vertical; padding: 9px 10px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.08); background: rgba(255,255,255,0.04); color: var(--text); }
    .footer { display: flex; gap: 10px; align-items: center; }
    .footer .spacer { flex: 1; }
  </style>
</head>
<body>
  <div class="wrap">
    <header>
      <div class="title">Local RAG Web App</div>
      <div class="status" id="status">Idle</div>
    </header>
    <div class="grid">
      <div class="panel">
        <div class="row">
          <label>Model (.gguf)</label>
          <input id="modelPath" type="text" placeholder="C:\\path\\to\\model.gguf" />
        </div>
        <div class="row">
          <label>Data folder</label>
          <input id="dataDir" type="text" placeholder="C:\\path\\to\\data" />
        </div>
        <div class="row">
          <label>Options</label>
          <div class="small">
            <div><input id="showSources" type="checkbox" checked /> Show sources (server-side /sources)</div>
          </div>
        </div>
        <div class="row">
          <button class="primary" id="btnInit">Load & Start</button>
          <button class="danger" id="btnStop" disabled>Stop</button>
        </div>
        <div class="small" style="margin-top: 10px;">
          <div><b>Commands</b></div>
          <div>/help, /sources, /quit</div>
        </div>
      </div>
      <div class="panel chat">
        <div id="log"></div>
        <div class="footer">
          <textarea id="query" placeholder="Type your question and press Enter..."></textarea>
          <div class="spacer"></div>
          <button id="btnSend" class="primary" disabled>Send</button>
        </div>
      </div>
    </div>
  </div>

  <script>
    const logEl = document.getElementById('log');
    const statusEl = document.getElementById('status');
    const modelPathEl = document.getElementById('modelPath');
    const dataDirEl = document.getElementById('dataDir');
    const showSourcesEl = document.getElementById('showSources');
    const queryEl = document.getElementById('query');
    const btnSendEl = document.getElementById('btnSend');
    const btnInitEl = document.getElementById('btnInit');
    const btnStopEl = document.getElementById('btnStop');

    let initJobId = null;
    let chatJobId = null;
    let pipelineReady = false;

    function setStatus(text){ statusEl.textContent = text; }

    function appendMsg(text, cls="system"){
      const div = document.createElement('div');
      div.className = `msg ${cls}`;
      div.textContent = text;
      logEl.appendChild(div);
      logEl.scrollTop = logEl.scrollHeight;
      return div;
    }

    function appendSources(sources){
      const box = document.createElement('div');
      box.className = 'sources';
      if(!sources || sources.length === 0){
        box.textContent = '(no sources)';
        return box;
      }
      for(const s of sources){
        const el = document.createElement('div');
        const rel = (typeof s.similarity_score === 'number') ? s.similarity_score.toFixed(2) : String(s.similarity_score);
        el.textContent = `- ${s.source} (relevance: ${rel})`;
        box.appendChild(el);
      }
      return box;
    }

    function confidenceBar(conf){
      if(typeof conf !== 'number') return '';
      const filled = Math.max(0, Math.min(10, Math.floor(conf * 10)));
      const empty = 10 - filled;
      const bar = '█'.repeat(filled) + '░'.repeat(empty);
      return `${bar} ${(conf*100).toFixed(0)}%`;
    }

    async function waitInit(){
      if(!initJobId) return;
      while(true){
        const r = await fetch(`/api/init_status?id=${encodeURIComponent(initJobId)}`);
        const data = await r.json();
        if(data.status === 'running'){
          setStatus(data.message || 'Initializing...');
          await new Promise(res => setTimeout(res, 1200));
          continue;
        }
        if(data.status === 'done'){
          pipelineReady = true;
          btnSendEl.disabled = false;
          btnInitEl.disabled = true;
          btnStopEl.disabled = false;
          setStatus('Ready');
          appendMsg('System ready.', 'system');
          showSourcesEl.checked = data.show_sources;
          appendMsg(data.help_text, 'system');
          return;
        }
        if(data.status === 'failed'){
          setStatus('Initialization failed');
          btnSendEl.disabled = true;
          btnInitEl.disabled = false;
          btnStopEl.disabled = true;
          appendMsg('Initialization error: ' + (data.error || 'Unknown error'), 'system');
          return;
        }
      }
    }

    async function pollChat(){
      if(!chatJobId) return;
      while(true){
        const r = await fetch(`/api/chat_status?id=${encodeURIComponent(chatJobId)}`);
        const data = await r.json();
        if(data.status === 'running'){
          setStatus(data.message || 'Thinking...');
          await new Promise(res => setTimeout(res, 600));
          continue;
        }
        if(data.status === 'done'){
          setStatus('Ready');
          const msg = appendMsg('Assistant: ' + (data.answer || ''), 'assistant');
          if(data.sources && showSourcesEl.checked){
            msg.appendChild(appendSources(data.sources));
          }
          if(typeof data.confidence === 'number' && data.confidence > 0){
            const confDiv = document.createElement('div');
            confDiv.className = 'confidence';
            confDiv.textContent = 'Confidence: ' + confidenceBar(data.confidence);
            msg.appendChild(confDiv);
          }
          if(typeof data.elapsed === 'number'){
            appendMsg('Response time: ' + data.elapsed.toFixed(1) + 's', 'system');
          }
          if(typeof data.show_sources === 'boolean'){
            showSourcesEl.checked = data.show_sources;
          }
          btnSendEl.disabled = false;
          chatJobId = null;
          return;
        }
        if(data.status === 'failed'){
          setStatus('Error');
          appendMsg('Chat error: ' + (data.error || 'Unknown error'), 'system');
          btnSendEl.disabled = false;
          chatJobId = null;
          return;
        }
      }
    }

    async function startInit(){
      const modelPath = modelPathEl.value.trim();
      const dataDir = dataDirEl.value.trim();
      if(!modelPath){ alert('Please enter model path'); return; }
      if(!dataDir){ alert('Please enter data folder'); return; }

      btnInitEl.disabled = true;
      btnSendEl.disabled = true;
      btnStopEl.disabled = false;
      setStatus('Initializing...');
      appendMsg('Initializing RAG system...', 'system');

      const payload = {
        model_path: modelPath,
        data_dir: dataDir,
        show_sources: !!showSourcesEl.checked
      };
      const r = await fetch('/api/init', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify(payload) });
      const data = await r.json();
      if(!r.ok){
        setStatus('Initialization failed');
        btnInitEl.disabled = false;
        btnStopEl.disabled = true;
        appendMsg('Initialization error: ' + (data.error || 'Unknown error'), 'system');
        return;
      }
      initJobId = data.id;
      await waitInit();
    }

    async function startChat(){
      const query = queryEl.value.trim();
      if(!query) return;
      if(!pipelineReady){ alert('Initialize the system first'); return; }
      if(chatJobId) return;

      btnSendEl.disabled = true;
      setStatus('Thinking...');
      appendMsg('You: ' + query, 'user');
      queryEl.value = '';

      const r = await fetch('/api/chat', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({ query }) });
      const data = await r.json();
      if(!r.ok){
        btnSendEl.disabled = false;
        setStatus('Error');
        appendMsg('Chat error: ' + (data.error || 'Unknown error'), 'system');
        return;
      }
      chatJobId = data.id;
      await pollChat();
    }

    btnInitEl.addEventListener('click', startInit);
    btnSendEl.addEventListener('click', startChat);
    queryEl.addEventListener('keydown', (e) => {
      if(e.key === 'Enter' && !e.shiftKey){
        e.preventDefault();
        startChat();
      }
    });

    btnStopEl.addEventListener('click', () => {
      // This is a lightweight stop: it clears UI, and server will finish any current job.
      logEl.textContent = '';
      appendMsg('Stopped (current job may still be running).', 'system');
      btnSendEl.disabled = !pipelineReady;
      setStatus('Idle');
    });

    appendMsg('Local RAG Web UI loaded.', 'system');
    appendMsg('Use "Load & Start" to initialize the system.', 'system');
    appendMsg('Tip: commands like /help are supported.', 'system');
  </script>
</body>
</html>
"""


def load_documents(data_dir: str) -> Tuple[List[str], List[str]]:
    # Reuse the exact same document loading behavior as `src/cli.py`.
    return cli_module.load_documents(data_dir)


def build_help_text(pipeline_ready: bool) -> str:
    base = (
        "Commands:\n"
        "  /help     Show this help message\n"
        "  /sources  Toggle source display on/off\n"
        "  /quit     Exit the program (close/refresh browser tab)\n"
    )
    return base


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

    def new_job(self) -> str:
        return str(time.time_ns())


STATE = AppState()


class Handler(BaseHTTPRequestHandler):
    server_version = "RAGWebApp/0.1"

    def _send_json(self, status_code: int, payload: Dict[str, Any]) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _read_json(self) -> Dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length > 0 else b"{}"
        if not raw:
            return {}
        try:
            return json.loads(raw.decode("utf-8"))
        except Exception:
            return {}

    def log_message(self, fmt: str, *args: Any) -> None:
        # Keep server logs minimal; errors will be returned in JSON.
        return

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        qs = parse_qs(parsed.query)

        if path == "/":
            page = HTML_PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(page)))
            self.end_headers()
            self.wfile.write(page)
            return

        if path == "/api/health":
            self._send_json(200, {"ok": True})
            return

        if path == "/api/init_status":
            job_id = (qs.get("id") or [None])[0]
            if not job_id:
                self._send_json(400, {"error": "Missing id"})
                return
            job = STATE.jobs.get(job_id)
            if not job:
                self._send_json(404, {"error": "Job not found"})
                return
            with STATE.lock:
                payload = {
                    "status": job.status,
                    "message": job.message,
                }
                if job.status == "done" and job.result is not None:
                    payload.update(job.result)
                if job.status == "failed":
                    payload["error"] = job.error or "Unknown error"
                return self._send_json(200, payload)

        if path == "/api/chat_status":
            job_id = (qs.get("id") or [None])[0]
            if not job_id:
                self._send_json(400, {"error": "Missing id"})
                return
            job = STATE.jobs.get(job_id)
            if not job:
                self._send_json(404, {"error": "Job not found"})
                return
            with STATE.lock:
                payload = {"status": job.status, "message": job.message}
                if job.status == "done" and job.result is not None:
                    payload.update(job.result)
                if job.status == "failed":
                    payload["error"] = job.error or "Unknown error"
                return self._send_json(200, payload)

        self._send_json(404, {"error": f"Not found: {path}"})

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/api/init":
            body = self._read_json()
            model_path = (body.get("model_path") or "").strip()
            data_dir = (body.get("data_dir") or "").strip()
            show_sources = bool(body.get("show_sources", True))

            if not model_path:
                self._send_json(400, {"error": "Missing model_path"})
                return
            if not os.path.exists(model_path):
                self._send_json(400, {"error": f"Model not found: {model_path}"})
                return
            if not data_dir or not os.path.isdir(data_dir):
                self._send_json(400, {"error": f"Data folder not found: {data_dir}"})
                return

            with STATE.lock:
                # If init is running, refuse a new one.
                if STATE.init_job:
                    j = STATE.jobs.get(STATE.init_job)
                    if j and j.status in ("queued", "running"):
                        self._send_json(409, {"error": "Initialization already running"})
                        return

                STATE.show_sources = show_sources
                job_id = STATE.new_job()
                STATE.init_job = job_id
                STATE.jobs[job_id] = Job(status="queued", message="Queued")

            threading.Thread(
                target=self._init_worker,
                    args=(job_id, model_path, data_dir),
                daemon=True,
            ).start()

            self._send_json(200, {"id": job_id})
            return

        if path == "/api/chat":
            body = self._read_json()
            query = (body.get("query") or "").strip()
            if not query:
                self._send_json(400, {"error": "Missing query"})
                return

            with STATE.lock:
                if STATE.pipeline is None:
                    self._send_json(409, {"error": "System not initialized yet"})
                    return

                # Only one chat job at a time (single-user local app).
                if STATE.chat_job:
                    j = STATE.jobs.get(STATE.chat_job)
                    if j and j.status in ("queued", "running"):
                        self._send_json(409, {"error": "Another chat is running"})
                        return

                job_id = STATE.new_job()
                STATE.chat_job = job_id
                STATE.jobs[job_id] = Job(status="queued", message="Queued")

            threading.Thread(
                target=self._chat_worker,
                args=(job_id, query),
                daemon=True,
            ).start()

            self._send_json(200, {"id": job_id})
            return

        self._send_json(404, {"error": f"Not found: {path}"})

    def _init_worker(
        self,
        job_id: str,
        model_path: str,
        data_dir: str,
    ) -> None:
        try:
            with STATE.lock:
                STATE.jobs[job_id].status = "running"
                STATE.jobs[job_id].message = "Loading embedding model..."

            doc_store = DocumentStore()

            with STATE.lock:
                STATE.jobs[job_id].message = "Loading documents..."

            texts, sources = load_documents(data_dir)
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
                    "help_text": build_help_text(pipeline_ready=True),
                }
        except Exception as e:
            err = f"{e}\n\n{traceback.format_exc()}"
            with STATE.lock:
                STATE.jobs[job_id].status = "failed"
                STATE.jobs[job_id].error = str(e)
                STATE.jobs[job_id].message = "Failed"
                STATE.pipeline = None

    def _chat_worker(self, job_id: str, query: str) -> None:
        start_time = time.time()
        try:
            with STATE.lock:
                STATE.jobs[job_id].status = "running"
                STATE.jobs[job_id].message = "Processing..."
                pipeline = STATE.pipeline

            if pipeline is None:
                raise RuntimeError("Pipeline not ready.")

            # Commands start with "/"
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
                elif cmd == "/help":
                    result = {
                        "answer": build_help_text(pipeline_ready=True),
                        "sources": [],
                        "confidence": 0.0,
                        "elapsed": time.time() - start_time,
                        "show_sources": STATE.show_sources,
                    }
                elif cmd == "/sources":
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
                else:
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
                return

            # Normal query
            with STATE.lock:
                STATE.jobs[job_id].message = "Thinking..."

            response = pipeline.query(query)
            elapsed = time.time() - start_time

            sources_out = []
            for s in response.sources:
                sources_out.append(
                    {
                        "source": s.source,
                        "chunk_id": s.chunk_id,
                        "similarity_score": s.similarity_score,
                    }
                )

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
                STATE.chat_job = None


def main() -> None:
    parser = argparse.ArgumentParser(description="Local RAG Web App")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=7860, help="Port to listen on")
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"RAG web app running at http://{args.host}:{args.port}/")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    # Deprecated: this file previously started a stdlib HTTP server.
    # Use the Flask app instead.
    from app import main as flask_main

    flask_main()

