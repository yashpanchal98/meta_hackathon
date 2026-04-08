"""
OpenEnv Benchmark — FastAPI served via Gradio SDK infrastructure
"""

from __future__ import annotations

import threading
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse

from environment import OpenEnvEnvironment, Action, TaskName

app = FastAPI(title="OpenEnv Benchmark")

_lock = threading.Lock()
_env: OpenEnvEnvironment | None = None


@app.get("/", response_class=HTMLResponse)
def root():
    return """<html><body style="font-family:sans-serif;max-width:600px;margin:40px auto">
    <h1>OpenEnv Benchmark API</h1>
    <ul>
      <li><code>POST /reset</code> — body: {"task": "email_triage"}</li>
      <li><code>POST /step</code>  — body: {"action": {...}}</li>
      <li><code>GET  /state</code></li>
      <li><code>GET  /health</code></li>
    </ul></body></html>"""


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
    return {"observation": obs.content, "metadata": obs.metadata,
            "task": task_str, "step": obs.step, "done": obs.done}


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
            result = _env.step(Action(task=_env.task_name, payload=body.get("action", {})))
        except Exception as exc:
            return JSONResponse(status_code=400, content={"error": str(exc)})
    return {"observation": result.observation.content, "reward": result.reward.value,
            "done": result.done, "info": result.info, "step": result.observation.step}


@app.get("/state")
def get_state():
    with _lock:
        if _env is None:
            return {"error": "No active environment"}
        return _env.state()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
