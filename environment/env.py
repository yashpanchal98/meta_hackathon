"""
OpenEnv-compliant environment for AI agent evaluation.
Supports three tasks: email_triage, data_cleaning, code_review.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from .tasks.email_triage import EmailTriageTask
from .tasks.data_cleaning import DataCleaningTask
from .tasks.code_review import CodeReviewTask


class TaskName(str, Enum):
    EMAIL_TRIAGE = "email_triage"
    DATA_CLEANING = "data_cleaning"
    CODE_REVIEW = "code_review"


class Observation(BaseModel):
    task: TaskName
    content: str = Field(description="Primary observable content for the agent")
    metadata: dict[str, Any] = Field(default_factory=dict)
    step: int = Field(default=0)
    done: bool = Field(default=False)


class Action(BaseModel):
    task: TaskName
    payload: dict[str, Any] = Field(
        description="Task-specific action payload. See task docstrings for schema."
    )


class Reward(BaseModel):
    value: float = Field(ge=-1.0, le=1.0, description="Normalised reward in [-1, 1]")
    breakdown: dict[str, float] = Field(
        default_factory=dict, description="Per-criterion reward breakdown"
    )
    message: str = Field(default="", description="Human-readable explanation")


class StepResult(BaseModel):
    observation: Observation
    reward: Reward
    done: bool
    info: dict[str, Any] = Field(default_factory=dict)


_TASK_REGISTRY = {
    TaskName.EMAIL_TRIAGE: EmailTriageTask,
    TaskName.DATA_CLEANING: DataCleaningTask,
    TaskName.CODE_REVIEW: CodeReviewTask,
}


class OpenEnvEnvironment:
    """
    OpenEnv-standard environment class.

    Usage::

        env = OpenEnvEnvironment(task="email_triage")
        obs = env.reset()
        result = env.step(Action(task="email_triage", payload={...}))
    """

    version: str = "1.0.0"

    def __init__(self, task: TaskName | str) -> None:
        self.task_name = TaskName(task)
        self._task = _TASK_REGISTRY[self.task_name]()
        self._step_count = 0
        self._done = False
        self._cumulative_reward = 0.0

    # ------------------------------------------------------------------
    # OpenEnv interface
    # ------------------------------------------------------------------

    def reset(self) -> Observation:
        """Reset the environment and return the initial observation."""
        self._step_count = 0
        self._done = False
        self._cumulative_reward = 0.0
        raw_obs = self._task.reset()
        return Observation(
            task=self.task_name,
            content=raw_obs["content"],
            metadata=raw_obs.get("metadata", {}),
            step=0,
            done=False,
        )

    def step(self, action: Action) -> StepResult:
        """
        Apply an action and return (observation, reward, done, info).

        Returns a StepResult containing all four values.
        """
        if self._done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")
        if action.task != self.task_name:
            raise ValueError(
                f"Action task '{action.task}' does not match environment task '{self.task_name}'"
            )

        raw_obs, reward_val, done, info = self._task.step(action.payload)
        self._step_count += 1
        self._done = done
        self._cumulative_reward += reward_val.value

        obs = Observation(
            task=self.task_name,
            content=raw_obs["content"],
            metadata=raw_obs.get("metadata", {}),
            step=self._step_count,
            done=done,
        )
        return StepResult(
            observation=obs,
            reward=reward_val,
            done=done,
            info={**info, "cumulative_reward": self._cumulative_reward},
        )

    def state(self) -> dict[str, Any]:
        """Return the full internal state (for debugging / logging)."""
        return {
            "task": self.task_name,
            "step": self._step_count,
            "done": self._done,
            "cumulative_reward": self._cumulative_reward,
            "task_state": self._task.state(),
        }
