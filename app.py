from __future__ import annotations
import json, threading
import gradio as gr
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
from environment import OpenEnvEnvironment, Action, TaskName

_lock = threading.Lock()
_env: OpenEnvEnvironment | None = None

class OpenEnvMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        path = request.url.path
        method = request.method

        if method == "GET" and path == "/health":
            return JSONResponse({"status": "ok"})

        if method == "GET" and path == "/state":
            with _lock:
                return JSONResponse(_env.state() if _env else {"error": "call /reset first"})

        if method == "POST" and path == "/reset":
            try:
                body = json.loads(await request.body())
            except Exception:
                body = {}
            task_str = body.get("task", "email_triage")
            if task_str not in {t.value for t in TaskName}:
                task_str = "email_triage"
            global _env
            with _lock:
                _env = OpenEnvEnvironment(task_str)
                obs = _env.reset()
            return JSONResponse({"observation": obs.content, "metadata": obs.metadata,
                                 "task": task_str, "step": obs.step, "done": obs.done})

        if method == "POST" and path == "/step":
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

        return await call_next(request)


with gr.Blocks(title="OpenEnv Benchmark") as demo:
    gr.Markdown("## OpenEnv Benchmark\nAPI: `POST /reset` · `POST /step` · `GET /health` · `GET /state`")

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, prevent_thread_lock=True)
    demo.app.add_middleware(OpenEnvMiddleware)
    threading.Event().wait()
