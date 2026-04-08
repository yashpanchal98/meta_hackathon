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

def run_llm_inference(hf_token: str, model_id: str, task: str) -> str:
    if not hf_token.strip():
        return "Please enter your HF token."
    try:
        from openai import OpenAI
    except ImportError:
        return "openai package not available."

    from inference import (
        run_email_triage as _et,
        run_data_cleaning as _dc,
        run_code_review as _cr,
        SYSTEM_PROMPTS,
    )

    client = OpenAI(api_key=hf_token.strip(), base_url="https://router.huggingface.co/v1")
    task_map = {
        "email_triage": _et,
        "data_cleaning": _dc,
        "code_review": _cr,
    }
    runner = task_map.get(task)
    if runner is None:
        return f"Unknown task: {task}"
    try:
        score = runner(client, model_id.strip())
        return f"**Task:** {task}\n**Model:** {model_id}\n**Score:** `{score:.4f}`"
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
            "Plug in any HF-compatible model to run the full benchmark automatically. "
            "Requires a [Hugging Face token](https://huggingface.co/settings/tokens)."
        )
        with gr.Row():
            llm_token = gr.Textbox(label="HF Token", type="password", placeholder="hf_...")
            llm_model = gr.Textbox(
                label="Model ID",
                value="Qwen/Qwen2.5-7B-Instruct",
                placeholder="org/model-name",
            )
            llm_task = gr.Dropdown(
                choices=["email_triage", "data_cleaning", "code_review"],
                value="email_triage",
                label="Task",
            )
        llm_btn = gr.Button("Run Benchmark", variant="primary")
        llm_output = gr.Markdown(label="Result")
        llm_btn.click(run_llm_inference, inputs=[llm_token, llm_model, llm_task], outputs=llm_output)

    # ------------------------------------------------------------------
    gr.Markdown(
        "---\n"
        "**Tags:** `openenv` · `agent-eval` · `nlp` · `code-review` · `data-cleaning`  \n"
        "Built for the META Hackathon."
    )

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())
