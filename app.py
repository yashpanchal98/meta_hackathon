"""
OpenEnv Benchmark
=================
Gradio app with OpenEnv REST API routes mounted on the same server.
HF Gradio SDK starts this with demo.launch() on port 7860.
Custom routes /reset /step /health are added to demo.app (FastAPI).
"""

from __future__ import annotations

import threading
import gradio as gr
from fastapi import Request
from fastapi.responses import JSONResponse, HTMLResponse

from environment import OpenEnvEnvironment, Action, TaskName

# ---------------------------------------------------------------------------
# Global env state
# ---------------------------------------------------------------------------
_lock = threading.Lock()
_env: OpenEnvEnvironment | None = None

# ---------------------------------------------------------------------------
# Minimal Gradio UI (required so HF's SDK detects the app as valid)
# ---------------------------------------------------------------------------
with gr.Blocks(title="OpenEnv Benchmark") as demo:
    gr.Markdown("""
## OpenEnv Benchmark API

Real-world agent evaluation environment.

| Endpoint | Method | Description |
|---|---|---|
| `/reset` | POST | Reset env — body: `{"task": "email_triage"}` |
| `/step`  | POST | Step env — body: `{"action": {...}}` |
| `/state` | GET  | Current environment state |
| `/health`| GET  | Liveness check |

**Tasks:** `email_triage` · `data_cleaning` · `code_review`
    """)

# ---------------------------------------------------------------------------
# REST API routes added to Gradio's internal FastAPI app
# ---------------------------------------------------------------------------

@demo.app.get("/health")
def health():
    return {"status": "ok"}


@demo.app.post("/reset")
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


@demo.app.post("/step")
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
            result = _env.step(Action(task=_env.task_name, payload=body.get("action", {})))
        except Exception as exc:
            return JSONResponse(status_code=400, content={"error": str(exc)})
    return {
        "observation": result.observation.content,
        "reward": result.reward.value,
        "done": result.done,
        "info": result.info,
        "step": result.observation.step,
    }


@demo.app.get("/state")
def get_state():
    with _lock:
        if _env is None:
            return {"error": "No active environment"}
        return _env.state()


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
