from __future__ import annotations
import json, threading
import gradio as gr
from starlette.routing import Route
from starlette.requests import Request
from starlette.responses import JSONResponse
from environment import OpenEnvEnvironment, Action, TaskName

_lock = threading.Lock()
_env: OpenEnvEnvironment | None = None

# ---------------------------------------------------------------------------
# Handler functions (plain Starlette, no FastAPI decorator needed)
# ---------------------------------------------------------------------------

async def handle_reset(request: Request):
    global _env
    try:
        body = json.loads(await request.body())
    except Exception:
        body = {}
    task_str = body.get("task", "email_triage")
    if task_str not in {t.value for t in TaskName}:
        task_str = "email_triage"
    with _lock:
        _env = OpenEnvEnvironment(task_str)
        obs = _env.reset()
    return JSONResponse({"observation": obs.content, "metadata": obs.metadata,
                         "task": task_str, "step": obs.step, "done": obs.done})

async def handle_step(request: Request):
    global _env
    try:
        body = json.loads(await request.body())
    except Exception:
        return JSONResponse({"error": "Invalid JSON"}, status_code=400)
    with _lock:
        if _env is None:
            return JSONResponse({"error": "Call /reset first"}, status_code=400)
        try:
            result = _env.step(Action(task=_env.task_name, payload=body.get("action", {})))
        except Exception as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
    return JSONResponse({"observation": result.observation.content,
                         "reward": result.reward.value, "done": result.done,
                         "info": result.info, "step": result.observation.step})

async def handle_health(request: Request):
    return JSONResponse({"status": "ok"})

async def handle_state(request: Request):
    with _lock:
        return JSONResponse(_env.state() if _env else {"error": "call /reset first"})

# ---------------------------------------------------------------------------
# Minimal Gradio UI
# ---------------------------------------------------------------------------
with gr.Blocks(title="OpenEnv Benchmark") as demo:
    gr.Markdown("## OpenEnv Benchmark\nAPI: `POST /reset` · `POST /step` · `GET /health`")

# ---------------------------------------------------------------------------
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, prevent_thread_lock=True)

    # Insert our routes at position 0 — checked BEFORE Gradio's own /reset handler
    custom_routes = [
        Route("/reset",  handle_reset,  methods=["POST"]),
        Route("/step",   handle_step,   methods=["POST"]),
        Route("/health", handle_health, methods=["GET"]),
        Route("/state",  handle_state,  methods=["GET"]),
    ]
    for i, route in enumerate(custom_routes):
        demo.app.router.routes.insert(i, route)

    threading.Event().wait()
