"""
OpenEnv Hackathon — Inference Script
=====================================
Runs three tasks (email_triage, data_cleaning, code_review) through an LLM
and emits the required [START] / [STEP] / [END] log lines to stdout.

Environment variables:
    API_BASE_URL   LLM endpoint  (default: https://router.huggingface.co/v1)
    MODEL_NAME     Model ID      (default: Qwen/Qwen2.5-7B-Instruct)
    HF_TOKEN       API key       (required, no default)
"""

from __future__ import annotations

import json
import os
import sys
import textwrap
from typing import Any

from openai import OpenAI

from environment import OpenEnvEnvironment, Action, TaskName

# ---------------------------------------------------------------------------
# Environment variables (hackathon-required names)
# ---------------------------------------------------------------------------

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-7B-Instruct")
HF_TOKEN     = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

ENV_NAME = "openenv"

# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def log_start(task: str) -> None:
    print(f"[START] task={task} env={ENV_NAME} model={MODEL_NAME}", flush=True)

def log_step(step: int, action: Any, reward: float, done: bool, error: str | None) -> None:
    action_str = json.dumps(action, separators=(",", ":")) if not isinstance(action, str) else action
    # Flatten to single line — no embedded newlines allowed
    action_str = action_str.replace("\n", " ")
    err_str = error if error else "null"
    done_str = "true" if done else "false"
    print(
        f"[STEP] step={step} action={action_str} reward={reward:.2f} done={done_str} error={err_str}",
        flush=True,
    )

def log_end(success: bool, steps: int, rewards: list[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    success_str = "true" if success else "false"
    print(f"[END] success={success_str} steps={steps} rewards={rewards_str}", flush=True)

# ---------------------------------------------------------------------------
# JSON parser — grabs the first complete JSON object, ignores trailing text
# ---------------------------------------------------------------------------

def parse_json_response(text: str) -> dict[str, Any]:
    text = text.strip()
    if "```" in text:
        text = "\n".join(l for l in text.split("\n") if not l.strip().startswith("```")).strip()
    start = text.find("{")
    if start == -1:
        raise ValueError(f"No JSON object found: {text[:200]!r}")
    obj, _ = json.JSONDecoder().raw_decode(text, start)
    return obj

# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPTS: dict[str, str] = {
    "email_triage": textwrap.dedent("""\
        You are an email triage assistant.
        Classify each email with:
          - priority: "urgent" | "normal" | "low"
          - category: "action_required" | "fyi" | "spam" | "newsletter"

        Respond ONLY with a valid JSON object in this exact format:
        {"classifications": [{"email_id": "e01", "priority": "urgent", "category": "action_required"}, ...]}
        No explanation. No markdown. Just the JSON.
    """),

    "data_cleaning": textwrap.dedent("""\
        You are a data engineer fixing a dirty CSV. Submit one operation per turn.

        Operations (ONE JSON object per response, no other text):
          {"operation": "drop_duplicates"}
          {"operation": "fill_nulls",     "column": "<col>", "value": "<val>"}
          {"operation": "fix_type",       "column": "<col>", "transform": "strip_dollar"|"to_int"}
          {"operation": "normalize_case", "column": "<col>", "case": "lower"}
          {"operation": "drop_outliers",  "column": "<col>", "min": <n>, "max": <n>}
          {"operation": "submit"}

        Use "submit" as your final operation. ONE JSON object only.
    """),

    "code_review": textwrap.dedent("""\
        You are a security-focused code reviewer. There are exactly 5 bugs (B1-B5).

        Each turn: output EXACTLY ONE JSON object, nothing else — no prose, no markdown.

        Actions:
          {"action": "inspect", "file": "app/db.py"}
          {"action": "report_bug", "bug_id": "B1", "line": 8, "description": "<desc>", "fix": "<fix>"}
          {"action": "submit"}

        Strategy:
        1. Inspect files one at a time.
        2. Report each bug with report_bug (one per turn).
        3. After all 5 bugs reported, output {"action": "submit"}.

        ONE JSON object per response. No other text.
    """),
}

# ---------------------------------------------------------------------------
# Per-task episode runners
# ---------------------------------------------------------------------------

def run_episode(task_name: TaskName, max_turns: int) -> None:
    task_str = task_name.value
    log_start(task_str)

    env = OpenEnvEnvironment(task_name)
    obs = env.reset()

    messages = [
        {"role": "system", "content": SYSTEM_PROMPTS[task_str]},
        {"role": "user",   "content": obs.content},
    ]

    step       = 0
    rewards:   list[float] = []
    last_error: str | None = None
    success    = False

    try:
        for _ in range(max_turns):
            response = client.chat.completions.create(
                model=MODEL_NAME, messages=messages, temperature=0
            )
            raw = response.choices[0].message.content or ""
            messages.append({"role": "assistant", "content": raw})

            last_error = None
            try:
                payload = parse_json_response(raw)
            except ValueError as e:
                last_error = str(e)[:120]
                # Force close the episode gracefully on parse failure
                payload = {"operation": "submit"} if task_str == "data_cleaning" else {"action": "submit"}

            action = Action(task=task_name, payload=payload)
            result = env.step(action)
            step += 1
            reward = result.reward.value
            rewards.append(reward)

            log_step(step, payload, reward, result.done, last_error)
            messages.append({"role": "user", "content": result.observation.content})

            if result.done:
                success = reward > 0
                break

    except Exception as exc:
        last_error = str(exc)[:120]
        log_step(step + 1, "exception", 0.0, True, last_error)
    finally:
        log_end(success, step, rewards)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

TASKS = [
    (TaskName.EMAIL_TRIAGE,  1),
    (TaskName.DATA_CLEANING, 10),
    (TaskName.CODE_REVIEW,   15),
]

if __name__ == "__main__":
    for task_name, max_turns in TASKS:
        run_episode(task_name, max_turns)
        print(flush=True)  # blank line between tasks
