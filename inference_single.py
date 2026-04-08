"""Runs a single task from inference.py — used by the Gradio UI."""
import os
from inference import run_episode, TASKS
from environment import TaskName

task_name_str = os.getenv("OPENENV_TASK", "email_triage")
task_map = {t.value: (t, turns) for t, turns in TASKS}

if task_name_str in task_map:
    task, max_turns = task_map[task_name_str]
    run_episode(task, max_turns)
else:
    print(f"Unknown task: {task_name_str}")
