"""
Code Review Grader
==================
Scoring per bug:
  +0.15  bug_id correctly identified (present in seeded set, not duplicated)
  +0.05  fix description contains at least one correct keyword (case-insensitive)
  -0.05  false positive (bug_id not in seeded set, or duplicate report)

Final score = sum / max_possible, clamped to [0, 1].
max_possible = 5 bugs * (0.15 + 0.05) = 1.0
"""

from __future__ import annotations

from typing import Any


class CodeReviewGrader:
    def __init__(self, seeded_bugs: dict[str, dict]) -> None:
        self._bugs = seeded_bugs  # {bug_id: {fix_keywords, ...}}

    def score_report(
        self, bug_id: str, description: str, fix: str
    ) -> tuple[float, str]:
        """Incremental score for a single report (used during the episode)."""
        if bug_id not in self._bugs:
            return -0.05, f"'{bug_id}' is not a known bug — false positive."
        bug = self._bugs[bug_id]
        text = (description + " " + fix).lower()
        kw_match = any(kw.lower() in text for kw in bug["fix_keywords"])
        score = 0.15 + (0.05 if kw_match else 0.0)
        msg = f"'{bug_id}' found (+0.15)"
        if kw_match:
            msg += " + fix keyword matched (+0.05)"
        return score, msg

    def grade(
        self, reports: list[dict[str, Any]]
    ) -> tuple[float, dict[str, float], str]:
        """Final grading after submit."""
        seen: set[str] = set()
        breakdown: dict[str, float] = {}
        messages: list[str] = []
        total = 0.0

        for report in reports:
            bug_id = report.get("bug_id", "").upper()
            description = report.get("description", "")
            fix = report.get("fix", "")

            if bug_id in seen:
                breakdown[f"{bug_id}_dup"] = -0.05
                total -= 0.05
                messages.append(f"{bug_id}: duplicate report (-0.05)")
                continue
            seen.add(bug_id)

            if bug_id not in self._bugs:
                breakdown[bug_id] = -0.05
                total -= 0.05
                messages.append(f"{bug_id}: false positive (-0.05)")
                continue

            bug = self._bugs[bug_id]
            text = (description + " " + fix).lower()
            kw_match = any(kw.lower() in text for kw in bug["fix_keywords"])
            score = 0.15 + (0.05 if kw_match else 0.0)
            breakdown[bug_id] = score
            total += score
            messages.append(
                f"{bug_id}: +{score:.2f}" + (" (fix keyword matched)" if kw_match else "")
            )

        # Missing bugs
        for bug_id in self._bugs:
            if bug_id not in seen:
                breakdown[f"{bug_id}_missed"] = 0.0
                messages.append(f"{bug_id}: missed (0)")

        max_possible = len(self._bugs) * 0.20  # 5 * 0.20 = 1.0
        normalised = max(0.0, min(1.0, total / max_possible))
        return round(normalised, 4), breakdown, " | ".join(messages)
