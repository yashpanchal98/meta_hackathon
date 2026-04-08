"""
OpenEnv Benchmark
=================
Launches Gradio non-blocking (prevent_thread_lock=True), then adds
REST API routes to demo.app (FastAPI) which is only available post-launch.
"""

from __future__ import annotations

import threading
import time

import gradio as gr
from fastapi import Request
from fastapi.responses import JSONResponse

from environment import OpenEnvEnvironment, Action, TaskName

# ---------------------------------------------------------------------------
# Global env state
# ---------------------------------------------------------------------------
_lock = threading.Lock()
_env: OpenEnvEnvironment | None = None

# ---------------------------------------------------------------------------
# Minimal Gradio UI
# ---------------------------------------------------------------------------
with gr.Blocks(title="OpenEnv Benchmark") as demo:
    gr.Markdown("""
## OpenEnv Benchmark API

| Endpoint | Method | Body |
|---|---|---|
| `/reset` | POST | `{"task": "email_triage"}` |
| `/step`  | POST | `{"action": {...}}` |
| `/state` | GET  | — |
| `/health`| GET  | — |

Tasks: `email_triage` · `data_cleaning` · `code_review`
    """)

# ---------------------------------------------------------------------------
# Entry point — launch non-blocking so demo.app is set, then add routes
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        prevent_thread_lock=True,
    )

    # demo.app is now the live FastAPI app — add REST routes
    app = demo.app

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post("/reset")
    async def reset_env(request: Request):
        global _env
        try:
            body = await request.json()
        except Exception:
            body = {}
        task_str = body.get("task", "email_triage")
        if task_str not in {t.value for t in TaskName}:
            task_str = "email_triage"
        with _lock:
            _env = OpenEnvEnvironment(task_str)
            obs = _env.reset()
        return {
            "observation": obs.content,
            "metadata": obs.metadata,
            "task": task_str,
            "step": obs.step,
            "done": obs.done,
        }

    @app.post("/step")
    async def step_env(request: Request):
        global _env
        try:
            body = await request.json()
        except Exception:
            return JSONResponse(status_code=400, content={"error": "Invalid JSON"})
        with _lock:
            if _env is None:
                return JSONResponse(status_code=400, content={"error": "Call /reset first"})
            try:
                result = _env.step(
                    Action(task=_env.task_name, payload=body.get("action", {}))
                )
            except Exception as exc:
                return JSONResponse(status_code=400, content={"error": str(exc)})
        return {
            "observation": result.observation.content,
            "reward": result.reward.value,
            "done": result.done,
            "info": result.info,
            "step": result.observation.step,
        }

    @app.get("/state")
    def get_state():
        with _lock:
            if _env is None:
                return {"error": "No active environment"}
            return _env.state()

    # Block the main thread so the process stays alive
    threading.Event().wait()
