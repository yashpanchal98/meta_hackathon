"""
Data Cleaning Task (Medium)
===========================
The agent receives a messy CSV (as a string) and must return a cleaned version.

Problems seeded in the data:
  1. Missing values (represented as "", "N/A", "null")
  2. Type errors (age stored as float "25.0", salary with "$" prefix)
  3. Duplicate rows (exact duplicates)
  4. Inconsistent casing in categorical columns (e.g. "male" vs "Male")
  5. Out-of-range values (negative age, salary=0)

Action payload schema (multi-step — submit operations one at a time)::

    # Step 1–N: apply cleaning operation
    {"operation": "drop_duplicates"}
    {"operation": "fill_nulls",   "column": "age",    "value": 30}
    {"operation": "fix_type",     "column": "salary",  "transform": "strip_dollar"}
    {"operation": "normalize_case","column": "gender", "case": "lower"}
    {"operation": "drop_outliers","column": "age",    "min": 0, "max": 120}
    # Final step: submit the cleaned CSV
    {"operation": "submit", "csv": "<csv string>"}

The episode ends when the agent submits or exceeds MAX_STEPS.
"""

from __future__ import annotations

import copy
import csv
import io
from typing import Any

from ..graders.data_cleaning_grader import DataCleaningGrader


_RAW_CSV = """\
id,name,age,gender,salary,department
1,Alice,28,Female,$72000,Engineering
2,Bob,35.0,male,$85000,Marketing
3,Carol,29,Female,$68000,Engineering
1,Alice,28,Female,$72000,Engineering
4,Dave,-5,Male,$91000,Engineering
5,Eve,31,female,,$Sales
6,Frank,42,Male,$105000,null
7,Grace,N/A,Female,$58000,HR
8,Hank,27,male,0,Marketing
9,Ivy,33,Female,$77000,Engineering
10,Jack,55,Male,$N/A,HR
"""

MAX_STEPS = 10


class DataCleaningTask:
    """Multi-step data cleaning task."""

    def __init__(self) -> None:
        self._grader = DataCleaningGrader(_RAW_CSV)
        self._step = 0
        self._done = False
        self._operations: list[dict] = []
        self._current_csv = _RAW_CSV
        self._reward_so_far = 0.0

    def reset(self) -> dict[str, Any]:
        self._step = 0
        self._done = False
        self._operations = []
        self._current_csv = _RAW_CSV
        self._reward_so_far = 0.0
        return {
            "content": (
                "You are a data engineer. The following CSV contains quality issues.\n"
                "Clean it by submitting operations one at a time.\n\n"
                f"```csv\n{_RAW_CSV}```\n\n"
                "Known issues to fix:\n"
                "- Duplicate rows\n"
                "- Missing/null values (fill with sensible defaults or drop)\n"
                "- Type inconsistencies (salary has '$' prefix, age stored as float)\n"
                "- Inconsistent casing in 'gender' column\n"
                "- Out-of-range values (negative age, zero salary)\n\n"
                "Available operations: drop_duplicates, fill_nulls, fix_type, "
                "normalize_case, drop_outliers, submit.\n"
                f"You have {MAX_STEPS} steps. Use 'submit' as your final step."
            ),
            "metadata": {"raw_csv": _RAW_CSV, "max_steps": MAX_STEPS},
        }

    def step(self, payload: dict[str, Any]) -> tuple[dict, Any, bool, dict]:
        from ..env import Reward

        op = payload.get("operation", "")
        self._step += 1
        self._operations.append(payload)

        # Apply operation to current CSV state
        try:
            result_csv, op_reward, op_msg = _apply_operation(
                self._current_csv, payload
            )
            self._current_csv = result_csv
        except Exception as exc:
            op_reward = -0.05
            op_msg = f"Operation failed: {exc}"

        if op == "submit" or self._step >= MAX_STEPS:
            # Final grading
            final_reward, breakdown, message = self._grader.grade(self._current_csv)
            self._done = True
            reward = Reward(
                value=final_reward,
                breakdown=breakdown,
                message=message,
            )
            obs = {
                "content": f"Grading complete. Score: {final_reward:.2f}\n{message}",
                "metadata": {"breakdown": breakdown},
            }
            return obs, reward, True, {}

        # Intermediate reward for a valid operation
        intermediate = max(-0.05, min(0.05, op_reward))
        reward = Reward(
            value=intermediate,
            breakdown={"operation": intermediate},
            message=op_msg,
        )
        obs = {
            "content": (
                f"Operation '{op}' applied. {op_msg}\n\n"
                f"Current CSV:\n```csv\n{self._current_csv}```\n"
                f"Steps remaining: {MAX_STEPS - self._step}"
            ),
            "metadata": {"current_csv": self._current_csv, "steps_remaining": MAX_STEPS - self._step},
        }
        return obs, reward, False, {}

    def state(self) -> dict[str, Any]:
        return {
            "step": self._step,
            "done": self._done,
            "operations": self._operations,
            "current_csv": self._current_csv,
        }


# ---------------------------------------------------------------------------
# Simple in-memory operation engine
# ---------------------------------------------------------------------------

def _parse_csv(text: str) -> tuple[list[str], list[dict]]:
    reader = csv.DictReader(io.StringIO(text.strip()))
    rows = list(reader)
    fieldnames = reader.fieldnames or []
    return list(fieldnames), rows


def _to_csv(fieldnames: list[str], rows: list[dict]) -> str:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue()


def _apply_operation(csv_text: str, payload: dict) -> tuple[str, float, str]:
    op = payload.get("operation", "")
    fieldnames, rows = _parse_csv(csv_text)

    if op == "drop_duplicates":
        seen: set[tuple] = set()
        unique = []
        for row in rows:
            key = tuple(row.values())
            if key not in seen:
                seen.add(key)
                unique.append(row)
        removed = len(rows) - len(unique)
        return _to_csv(fieldnames, unique), 0.05 if removed > 0 else 0.0, f"Removed {removed} duplicate(s)."

    elif op == "fill_nulls":
        col = payload.get("column", "")
        val = str(payload.get("value", ""))
        null_tokens = {"", "n/a", "null", "none", "na"}
        count = 0
        for row in rows:
            if col in row and row[col].strip().lower() in null_tokens:
                row[col] = val
                count += 1
        return _to_csv(fieldnames, rows), 0.03 if count > 0 else 0.0, f"Filled {count} null(s) in '{col}'."

    elif op == "fix_type":
        col = payload.get("column", "")
        transform = payload.get("transform", "")
        count = 0
        for row in rows:
            if col not in row:
                continue
            v = row[col].strip()
            if transform == "strip_dollar" and v.startswith("$"):
                row[col] = v[1:]
                count += 1
            elif transform == "to_int":
                try:
                    row[col] = str(int(float(v)))
                    count += 1
                except ValueError:
                    pass
        return _to_csv(fieldnames, rows), 0.03 if count > 0 else 0.0, f"Fixed {count} value(s) in '{col}'."

    elif op == "normalize_case":
        col = payload.get("column", "")
        case = payload.get("case", "lower")
        count = 0
        for row in rows:
            if col in row:
                old = row[col]
                row[col] = old.lower() if case == "lower" else old.upper() if case == "upper" else old.title()
                if row[col] != old:
                    count += 1
        return _to_csv(fieldnames, rows), 0.02 if count > 0 else 0.0, f"Normalised {count} value(s) in '{col}'."

    elif op == "drop_outliers":
        col = payload.get("column", "")
        min_val = payload.get("min")
        max_val = payload.get("max")
        kept = []
        removed = 0
        for row in rows:
            try:
                v = float(row.get(col, ""))
                if (min_val is not None and v < min_val) or (max_val is not None and v > max_val):
                    removed += 1
                    continue
            except ValueError:
                pass
            kept.append(row)
        return _to_csv(fieldnames, kept), 0.03 if removed > 0 else 0.0, f"Dropped {removed} outlier row(s) in '{col}'."

    elif op == "submit":
        # Caller handles grading; return as-is
        submitted = payload.get("csv", csv_text)
        return submitted, 0.0, "Submission received."

    else:
        raise ValueError(f"Unknown operation: '{op}'")
