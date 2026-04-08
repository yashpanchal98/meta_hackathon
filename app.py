"""
OpenEnv Benchmark — Gradio Demo
================================
Interactive UI for the real-world AI agent evaluation environment.
Supports three tasks: Email Triage, Data Cleaning, Code Review.
"""

import json
import os
import textwrap

import gradio as gr

from environment import OpenEnvEnvironment, Action, TaskName
from environment.tasks.email_triage import EMAILS
from environment.tasks.code_review import PR_DIFF
from environment.tasks.data_cleaning import _RAW_CSV

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt_reward(reward) -> str:
    lines = [f"**Score:** `{reward.value:.4f}`"]
    if reward.breakdown:
        lines.append("**Breakdown:**")
        for k, v in reward.breakdown.items():
            lines.append(f"  - `{k}`: `{v:+.4f}`")
    if reward.message:
        lines.append(f"**Details:** {reward.message}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Task 1 — Email Triage
# ---------------------------------------------------------------------------

EMAIL_EXAMPLE = json.dumps(
    {
        "classifications": [
            {"email_id": "e01", "priority": "urgent",  "category": "action_required"},
            {"email_id": "e02", "priority": "low",     "category": "newsletter"},
            {"email_id": "e03", "priority": "urgent",  "category": "action_required"},
            {"email_id": "e04", "priority": "low",     "category": "spam"},
            {"email_id": "e05", "priority": "normal",  "category": "fyi"},
            {"email_id": "e06", "priority": "urgent",  "category": "action_required"},
            {"email_id": "e07", "priority": "normal",  "category": "action_required"},
            {"email_id": "e08", "priority": "normal",  "category": "fyi"},
            {"email_id": "e09", "priority": "low",     "category": "spam"},
            {"email_id": "e10", "priority": "normal",  "category": "action_required"},
        ]
    },
    indent=2,
)

def run_email_triage(action_json: str) -> str:
    try:
        payload = json.loads(action_json)
    except json.JSONDecodeError as e:
        return f"JSON parse error: {e}"
    env = OpenEnvEnvironment(TaskName.EMAIL_TRIAGE)
    env.reset()
    try:
        result = env.step(Action(task=TaskName.EMAIL_TRIAGE, payload=payload))
    except Exception as e:
        return f"Error: {e}"
    return _fmt_reward(result.reward)


def email_task_description() -> str:
    lines = ["### Emails to Classify\n"]
    for e in EMAILS:
        lines.append(
            f"**[{e['id']}]** `{e['from']}`  \n"
            f"**Subject:** {e['subject']}  \n"
            f"**Body:** {e['body']}\n"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Task 2 — Data Cleaning (multi-step, stateful per session)
# ---------------------------------------------------------------------------

_dc_env: OpenEnvEnvironment | None = None

def reset_data_cleaning():
    global _dc_env
    _dc_env = OpenEnvEnvironment(TaskName.DATA_CLEANING)
    obs = _dc_env.reset()
    return obs.content, ""

def step_data_cleaning(action_json: str) -> tuple[str, str]:
    global _dc_env
    if _dc_env is None:
        return "Click **Reset** first.", ""
    try:
        payload = json.loads(action_json)
    except json.JSONDecodeError as e:
        return f"JSON parse error: {e}", ""
    try:
        result = _dc_env.step(Action(task=TaskName.DATA_CLEANING, payload=payload))
    except RuntimeError as e:
        return f"Episode ended: {e}", ""
    return result.observation.content, _fmt_reward(result.reward)


# ---------------------------------------------------------------------------
# Task 3 — Code Review (multi-step, stateful per session)
# ---------------------------------------------------------------------------

_cr_env: OpenEnvEnvironment | None = None

def reset_code_review():
    global _cr_env
    _cr_env = OpenEnvEnvironment(TaskName.CODE_REVIEW)
    obs = _cr_env.reset()
    return obs.content, ""

def step_code_review(action_json: str) -> tuple[str, str]:
    global _cr_env
    if _cr_env is None:
        return "Click **Reset** first.", ""
    try:
        payload = json.loads(action_json)
    except json.JSONDecodeError as e:
        return f"JSON parse error: {e}", ""
    try:
        result = _cr_env.step(Action(task=TaskName.CODE_REVIEW, payload=payload))
    except RuntimeError as e:
        return f"Episode ended: {e}", ""
    return result.observation.content, _fmt_reward(result.reward)


# ---------------------------------------------------------------------------
# LLM Inference (optional — requires HF_TOKEN)
# ---------------------------------------------------------------------------

def run_llm_inference(api_token: str, base_url: str, model_id: str, task: str) -> str:
    if not api_token.strip():
        return "Please enter your API token."
    if not base_url.strip():
        return "Please enter a base URL."
    try:
        from openai import OpenAI
    except ImportError:
        return "openai package not available."

    from inference import (
        run_email_triage as _et,
        run_data_cleaning as _dc,
        run_code_review as _cr,
    )

    import os, subprocess, sys
    env = os.environ.copy()
    env["HF_TOKEN"]     = api_token.strip()
    env["API_BASE_URL"] = base_url.strip()
    env["MODEL_NAME"]   = model_id.strip()
    # Only run the selected task by temporarily patching TASKS via env
    env["OPENENV_TASK"] = task
    try:
        result = subprocess.run(
            [sys.executable, "inference_single.py"],
            capture_output=True, text=True, timeout=120, env=env
        )
        output = result.stdout or result.stderr or "No output."
        return f"```\n{output}\n```"
    except subprocess.TimeoutExpired:
        return "Timed out after 120 seconds."
    except Exception as e:
        return f"Error: {e}"


# ---------------------------------------------------------------------------
# Build Gradio UI
# ---------------------------------------------------------------------------

with gr.Blocks(title="OpenEnv Benchmark") as demo:
    gr.Markdown(
        """
# OpenEnv Benchmark — Real-World Agent Evaluation

Three tasks of increasing difficulty to evaluate AI agent capabilities.
| Task | Difficulty | Steps | Score Range |
|---|---|---|---|
| Email Triage | Easy | 1 | −1.0 → 1.0 |
| Data Cleaning | Medium | ≤10 | 0.0 → 1.0 |
| Code Review | Hard | ≤15 | −1.0 → 1.0 |
        """
    )

    # ------------------------------------------------------------------
    with gr.Tab("Task 1 — Email Triage"):
        gr.Markdown(
            "Classify 10 emails by **priority** (`urgent`/`normal`/`low`) and "
            "**category** (`action_required`/`fyi`/`spam`/`newsletter`). "
            "Paste your JSON answer below and click **Grade**."
        )
        with gr.Row():
            with gr.Column():
                gr.Markdown(email_task_description())
            with gr.Column():
                et_input = gr.Code(
                    label="Your classifications (JSON)",
                    language="json",
                    value=EMAIL_EXAMPLE,
                    lines=30,
                )
                et_btn = gr.Button("Grade", variant="primary")
                et_output = gr.Markdown(label="Score")
        et_btn.click(run_email_triage, inputs=et_input, outputs=et_output)

    # ------------------------------------------------------------------
    with gr.Tab("Task 2 — Data Cleaning"):
        gr.Markdown(
            "Fix the messy CSV by submitting one operation at a time. "
            "Click **Reset** to start, then submit operations as JSON."
        )
        gr.Markdown(
            "**Available operations:** `drop_duplicates` · `fill_nulls` · "
            "`fix_type` · `normalize_case` · `drop_outliers` · `submit`"
        )
        gr.Markdown(f"**Raw CSV:**\n```csv\n{_RAW_CSV}```")
        with gr.Row():
            with gr.Column():
                dc_input = gr.Code(
                    label="Operation (JSON)",
                    language="json",
                    value='{"operation": "drop_duplicates"}',
                    lines=8,
                )
                with gr.Row():
                    dc_reset_btn = gr.Button("Reset Episode")
                    dc_step_btn = gr.Button("Submit Operation", variant="primary")
            with gr.Column():
                dc_obs = gr.Markdown(label="Observation")
                dc_reward = gr.Markdown(label="Reward")
        dc_reset_btn.click(reset_data_cleaning, outputs=[dc_obs, dc_reward])
        dc_step_btn.click(step_data_cleaning, inputs=dc_input, outputs=[dc_obs, dc_reward])

    # ------------------------------------------------------------------
    with gr.Tab("Task 3 — Code Review"):
        gr.Markdown(
            "Review the PR diff below. There are **5 seeded bugs** (B1–B5). "
            "Use `inspect`, `report_bug`, and `submit` actions."
        )
        with gr.Row():
            with gr.Column():
                gr.Textbox(value=PR_DIFF, label="PR Diff", lines=40, interactive=False)
            with gr.Column():
                cr_input = gr.Code(
                    label="Action (JSON)",
                    language="json",
                    value='{"action": "report_bug", "bug_id": "B1", "line": 8, "description": "SQL injection via f-string", "fix": "Use parameterised query"}',
                    lines=8,
                )
                with gr.Row():
                    cr_reset_btn = gr.Button("Reset Episode")
                    cr_step_btn = gr.Button("Submit Action", variant="primary")
                cr_obs = gr.Markdown(label="Observation")
                cr_reward = gr.Markdown(label="Reward")
        cr_reset_btn.click(reset_code_review, outputs=[cr_obs, cr_reward])
        cr_step_btn.click(step_code_review, inputs=cr_input, outputs=[cr_obs, cr_reward])

    # ------------------------------------------------------------------
    with gr.Tab("Run LLM Inference"):
        gr.Markdown(
            "Plug in any OpenAI-compatible API to run the benchmark automatically.\n\n"
            "| Provider | Base URL | Free? |\n"
            "|---|---|---|\n"
            "| **Groq** | `https://api.groq.com/openai/v1` | Yes |\n"
            "| **Together AI** | `https://api.together.xyz/v1` | Trial credits |\n"
            "| **HF Router** | `https://router.huggingface.co/v1` | Limited free tier |\n"
            "| **OpenAI** | `https://api.openai.com/v1` | Paid |\n"
        )
        with gr.Row():
            llm_token = gr.Textbox(label="API Token", type="password", placeholder="your token...")
            llm_base_url = gr.Textbox(
                label="Base URL",
                value="https://api.groq.com/openai/v1",
                placeholder="https://api.groq.com/openai/v1",
            )
        with gr.Row():
            llm_model = gr.Textbox(
                label="Model ID",
                value="llama-3.1-8b-instant",
                placeholder="model name for chosen provider",
            )
            llm_task = gr.Dropdown(
                choices=["email_triage", "data_cleaning", "code_review"],
                value="email_triage",
                label="Task",
            )
        gr.Markdown(
            "**Groq model names:** `llama-3.1-8b-instant` · `llama-3.3-70b-versatile` · `gemma2-9b-it` · `mixtral-8x7b-32768`"
        )
        llm_btn = gr.Button("Run Benchmark", variant="primary")
        llm_output = gr.Markdown(label="Result")
        llm_btn.click(run_llm_inference, inputs=[llm_token, llm_base_url, llm_model, llm_task], outputs=llm_output)

    # ------------------------------------------------------------------
    gr.Markdown(
        "---\n"
        "**Tags:** `openenv` · `agent-eval` · `nlp` · `code-review` · `data-cleaning`  \n"
        "Built for the META Hackathon."
    )

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())
