"""
Data Cleaning Grader
====================
Evaluates cleaned CSV against a set of deterministic checks.

Checks (each worth equal weight, total normalised to [0, 1]):
  C1  No duplicate rows
  C2  No null/empty/N/A values in any cell
  C3  'salary' column: all values are numeric integers (no $ prefix, no N/A)
  C4  'age' column: all values are integers (no .0 suffix)
  C5  'gender' column: consistent casing (all lowercase or all title-case)
  C6  No rows with age < 0 or age > 120
  C7  No rows with salary <= 0
  C8  Row count is reasonable (between 7 and 10 — original has 11 with 1 dupe)
"""

from __future__ import annotations

import csv
import io
from typing import Any


_NULL_TOKENS = {"", "n/a", "null", "none", "na", "nan"}


class DataCleaningGrader:
    def __init__(self, raw_csv: str) -> None:
        self._raw_csv = raw_csv  # kept for reference

    def grade(self, cleaned_csv: str) -> tuple[float, dict[str, float], str]:
        try:
            fieldnames, rows = _parse(cleaned_csv)
        except Exception as exc:
            return -1.0, {}, f"Could not parse CSV: {exc}"

        checks: dict[str, tuple[bool, str]] = {}

        # C1 – no duplicates
        seen: set[tuple] = set()
        has_dup = False
        for row in rows:
            key = tuple(row.values())
            if key in seen:
                has_dup = True
                break
            seen.add(key)
        checks["C1_no_duplicates"] = (not has_dup, "no duplicates" if not has_dup else "duplicates present")

        # C2 – no nulls
        null_found = any(
            v.strip().lower() in _NULL_TOKENS
            for row in rows
            for v in row.values()
        )
        checks["C2_no_nulls"] = (not null_found, "no nulls" if not null_found else "null values present")

        # C3 – salary numeric integers
        salary_ok = True
        for row in rows:
            val = row.get("salary", "").strip()
            if val.startswith("$"):
                salary_ok = False
                break
            try:
                int(val)
            except ValueError:
                salary_ok = False
                break
        checks["C3_salary_numeric"] = (salary_ok, "salary numeric" if salary_ok else "salary has non-numeric values")

        # C4 – age integers
        age_ok = True
        for row in rows:
            val = row.get("age", "").strip()
            try:
                if "." in val:
                    age_ok = False
                    break
                int(val)
            except ValueError:
                age_ok = False
                break
        checks["C4_age_integer"] = (age_ok, "age integer" if age_ok else "age has float/non-int values")

        # C5 – consistent gender casing
        genders = [row.get("gender", "").strip() for row in rows if row.get("gender", "").strip()]
        casing_ok = len(set(g.lower() for g in genders)) > 0 and (
            all(g == g.lower() for g in genders) or all(g == g.title() for g in genders)
        )
        checks["C5_gender_casing"] = (casing_ok, "gender consistent" if casing_ok else "gender casing inconsistent")

        # C6 – age in range
        age_range_ok = True
        for row in rows:
            try:
                age = int(float(row.get("age", "0")))
                if age < 0 or age > 120:
                    age_range_ok = False
                    break
            except ValueError:
                pass
        checks["C6_age_range"] = (age_range_ok, "age in range" if age_range_ok else "age out of range")

        # C7 – salary > 0
        salary_pos = True
        for row in rows:
            try:
                sal = int(float(row.get("salary", "1").replace("$", "")))
                if sal <= 0:
                    salary_pos = False
                    break
            except ValueError:
                pass
        checks["C7_salary_positive"] = (salary_pos, "salary positive" if salary_pos else "zero/negative salary")

        # C8 – row count reasonable
        row_count_ok = 7 <= len(rows) <= 10
        checks["C8_row_count"] = (row_count_ok, f"{len(rows)} rows (expected 7–10)" )

        passed = sum(1 for ok, _ in checks.values() if ok)
        total = len(checks)
        score = passed / total
        breakdown = {k: 1.0 / total if ok else 0.0 for k, (ok, _) in checks.items()}
        messages = [msg for _, (_, msg) in checks.items()]
        return round(score, 4), breakdown, " | ".join(messages)


def _parse(text: str) -> tuple[list[str], list[dict]]:
    reader = csv.DictReader(io.StringIO(text.strip()))
    rows = list(reader)
    return list(reader.fieldnames or []), rows
