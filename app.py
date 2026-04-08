"""
OpenEnv Benchmark — FastAPI Server
====================================
REST API for the OpenEnv hackathon evaluation.

Endpoints:
  GET  /health  — liveness check
  POST /reset   — reset environment, returns initial observation
  POST /step    — take a step, returns (obs, reward, done, info)
  GET  /state   — current environment state
  GET  /        — simple HTML info page
"""

from __future__ import annotations

import threading
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse

from environment import OpenEnvEnvironment, Action, TaskName

app = FastAPI(title="OpenEnv Benchmark")

_lock = threading.Lock()
_env: OpenEnvEnvironment | None = None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
def root():
    return """
    <html><head><title>OpenEnv Benchmark</title></head>
    <body style="font-family:sans-serif;max-width:600px;margin:40px auto">
    <h1>OpenEnv Benchmark API</h1>
    <p>Real-world agent evaluation: email triage, data cleaning, code review.</p>
    <h3>Endpoints</h3>
    <ul>
      <li><code>POST /reset</code> — body: <code>{"task": "email_triage"}</code></li>
      <li><code>POST /step</code>  — body: <code>{"action": {...}}</code></li>
      <li><code>GET  /state</code> — current environment state</li>
      <li><code>GET  /health</code> — liveness check</li>
    </ul>
    <p>Tasks: <code>email_triage</code> · <code>data_cleaning</code> · <code>code_review</code></p>
    </body></html>
    """


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
        return JSONResponse(status_code=400, content={"error": "Invalid JSON body"})

    action_payload = body.get("action", {})

    with _lock:
        if _env is None:
            return JSONResponse(status_code=400, content={"error": "Call /reset first"})
        try:
            action = Action(task=_env.task_name, payload=action_payload)
            result = _env.step(action)
        except Exception as exc:
            return JSONResponse(status_code=400, content={"error": str(exc)})

    return {
        "observation": result.observation.content,
        "metadata": result.observation.metadata,
        "reward": result.reward.value,
        "reward_breakdown": result.reward.breakdown,
        "done": result.done,
        "info": result.info,
        "step": result.observation.step,
    }


@app.get("/state")
def get_state():
    with _lock:
        if _env is None:
            return {"error": "No active environment. Call /reset first."}
        return _env.state()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
