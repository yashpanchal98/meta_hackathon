"""
Baseline inference script
=========================
Runs an LLM (via OpenAI-compatible API) through all three environment tasks
and prints final scores.

Usage::

    HF_TOKEN=hf_xxx python inference.py [--model <model-id>] [--base-url <url>]

Environment variables:
    HF_TOKEN       Hugging Face API token (required)
    HF_BASE_URL    Override inference endpoint (optional)
    MODEL_ID       Override model ID (optional)

Default endpoint: https://api-inference.huggingface.co/v1
Default model:    meta-llama/Meta-Llama-3.1-8B-Instruct
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap
from typing import Any

try:
    from openai import OpenAI
except ImportError:
    print("ERROR: openai package not installed. Run: pip install openai")
    sys.exit(1)

from environment import OpenEnvEnvironment, Action, TaskName


DEFAULT_BASE_URL = "https://router.huggingface.co/v1"
DEFAULT_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"

BENCHMARK_MODELS = [
    "meta-llama/Llama-3.1-8B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-Coder-7B-Instruct",
]

# ---------------------------------------------------------------------------
# Task-specific system prompts & action parsers
# ---------------------------------------------------------------------------

SYSTEM_PROMPTS: dict[str, str] = {
    "email_triage": textwrap.dedent("""\
        You are an email triage assistant.
        You will receive a list of emails. Classify each one with:
          - priority: "urgent" | "normal" | "low"
          - category: "action_required" | "fyi" | "spam" | "newsletter"

        Respond ONLY with a valid JSON object in this exact format:
        {
          "classifications": [
            {"email_id": "e01", "priority": "urgent", "category": "action_required"},
            ...
          ]
        }
        No explanation. No markdown. Just the JSON.
    """),

    "data_cleaning": textwrap.dedent("""\
        You are a data engineer. You will receive a dirty CSV and must clean it
        by submitting operations one at a time.

        Available operations (respond with ONE JSON object per turn):
          {"operation": "drop_duplicates"}
          {"operation": "fill_nulls",    "column": "<col>", "value": "<val>"}
          {"operation": "fix_type",      "column": "<col>", "transform": "strip_dollar"|"to_int"}
          {"operation": "normalize_case","column": "<col>", "case": "lower"|"upper"|"title"}
          {"operation": "drop_outliers", "column": "<col>", "min": <num>, "max": <num>}
          {"operation": "submit"}   <- use this as your LAST operation

        Respond ONLY with a single JSON object. No explanation. No markdown.
    """),

    "code_review": textwrap.dedent("""\
        You are a security-focused code reviewer.
        You will review a pull request diff. There are exactly 5 bugs seeded.

        Available actions (respond with ONE JSON object per turn):
          {"action": "inspect", "file": "<path>"}
          {"action": "report_bug", "bug_id": "B1", "line": <n>,
           "description": "<desc>", "fix": "<fix>"}
          {"action": "submit"}   <- use when you have reported all bugs

        Bug IDs to use: B1, B2, B3, B4, B5 (in any order).
        Respond ONLY with a single JSON object. No explanation. No markdown.
    """),
}


def parse_json_response(text: str) -> dict[str, Any]:
    """Extract the first JSON object from model output."""
    text = text.strip()
    # Strip markdown code fences if present
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(
            l for l in lines if not l.strip().startswith("```")
        ).strip()
    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end == 0:
        raise ValueError(f"No JSON object found in response: {text[:200]!r}")
    return json.loads(text[start:end])


# ---------------------------------------------------------------------------
# Per-task runners
# ---------------------------------------------------------------------------

def run_email_triage(client: OpenAI, model: str) -> float:
    env = OpenEnvEnvironment(TaskName.EMAIL_TRIAGE)
    obs = env.reset()

    messages = [
        {"role": "system", "content": SYSTEM_PROMPTS["email_triage"]},
        {"role": "user", "content": obs.content},
    ]
    response = client.chat.completions.create(model=model, messages=messages, temperature=0)
    raw = response.choices[0].message.content or ""

    try:
        payload = parse_json_response(raw)
    except ValueError as e:
        print(f"  [email_triage] Parse error: {e}")
        payload = {"classifications": []}

    action = Action(task=TaskName.EMAIL_TRIAGE, payload=payload)
    result = env.step(action)
    score = result.info.get("cumulative_reward", result.reward.value)
    print(f"  [email_triage] Score: {score:.4f} | {result.reward.message[:120]}")
    return score


def run_data_cleaning(client: OpenAI, model: str, max_turns: int = 10) -> float:
    env = OpenEnvEnvironment(TaskName.DATA_CLEANING)
    obs = env.reset()

    messages = [
        {"role": "system", "content": SYSTEM_PROMPTS["data_cleaning"]},
        {"role": "user", "content": obs.content},
    ]

    final_score = 0.0
    for turn in range(max_turns):
        response = client.chat.completions.create(model=model, messages=messages, temperature=0)
        raw = response.choices[0].message.content or ""
        messages.append({"role": "assistant", "content": raw})

        try:
            payload = parse_json_response(raw)
        except ValueError as e:
            print(f"  [data_cleaning] Turn {turn+1} parse error: {e}")
            payload = {"operation": "submit"}

        action = Action(task=TaskName.DATA_CLEANING, payload=payload)
        result = env.step(action)
        messages.append({"role": "user", "content": result.observation.content})

        if result.done:
            final_score = result.info.get("cumulative_reward", result.reward.value)
            print(f"  [data_cleaning] Score: {final_score:.4f} | {result.reward.message[:120]}")
            break

    return final_score


def run_code_review(client: OpenAI, model: str, max_turns: int = 15) -> float:
    env = OpenEnvEnvironment(TaskName.CODE_REVIEW)
    obs = env.reset()

    messages = [
        {"role": "system", "content": SYSTEM_PROMPTS["code_review"]},
        {"role": "user", "content": obs.content},
    ]

    final_score = 0.0
    for turn in range(max_turns):
        response = client.chat.completions.create(model=model, messages=messages, temperature=0)
        raw = response.choices[0].message.content or ""
        messages.append({"role": "assistant", "content": raw})

        try:
            payload = parse_json_response(raw)
        except ValueError as e:
            print(f"  [code_review] Turn {turn+1} parse error: {e}")
            payload = {"action": "submit"}

        action = Action(task=TaskName.CODE_REVIEW, payload=payload)
        result = env.step(action)
        messages.append({"role": "user", "content": result.observation.content})

        if result.done:
            final_score = result.info.get("cumulative_reward", result.reward.value)
            print(f"  [code_review] Score: {final_score:.4f} | {result.reward.message[:120]}")
            break

    return final_score


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

RUNNERS = {
    "email_triage": run_email_triage,
    "data_cleaning": run_data_cleaning,
    "code_review": run_code_review,
}


def run_model(client: OpenAI, model: str) -> dict[str, float]:
    """Run all tasks for a single model and return scores."""
    scores: dict[str, float] = {}
    for task_name, runner in RUNNERS.items():
        print(f"  [{task_name}]")
        try:
            scores[task_name] = runner(client, model)
        except Exception as exc:
            print(f"  ERROR: {exc}")
            scores[task_name] = 0.0
    return scores


def print_leaderboard(results: dict[str, dict[str, float]]) -> None:
    tasks = list(RUNNERS.keys())
    col_w = 22

    header = f"  {'Model':<40}" + "".join(f"{t:<{col_w}}" for t in tasks) + f"{'AVERAGE':<12}"
    print("\n" + "=" * len(header))
    print("BENCHMARK LEADERBOARD")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    ranked = sorted(
        results.items(),
        key=lambda kv: sum(kv[1].values()) / len(kv[1]),
        reverse=True,
    )
    for model, scores in ranked:
        avg = sum(scores.values()) / len(scores)
        short = model.split("/")[-1][:38]
        row = f"  {short:<40}" + "".join(f"{scores.get(t, 0):<{col_w}.4f}" for t in tasks) + f"{avg:<12.4f}"
        print(row)

    print("=" * len(header))


def main() -> None:
    parser = argparse.ArgumentParser(description="OpenEnv baseline inference script")
    parser.add_argument(
        "--models", nargs="+",
        default=BENCHMARK_MODELS,
        help="List of model IDs to benchmark (default: 3 preset models)",
    )
    parser.add_argument("--base-url", default=os.getenv("HF_BASE_URL", DEFAULT_BASE_URL))
    args = parser.parse_args()

    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        print("ERROR: HF_TOKEN environment variable not set.")
        sys.exit(1)

    client = OpenAI(api_key=hf_token, base_url=args.base_url)
    print(f"API URL : {args.base_url}")
    print(f"Models  : {args.models}")
    print("=" * 60)

    all_results: dict[str, dict[str, float]] = {}
    for model in args.models:
        print(f"\nBenchmarking: {model}")
        print("-" * 60)
        all_results[model] = run_model(client, model)

    print_leaderboard(all_results)


if __name__ == "__main__":
    main()
