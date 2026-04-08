"""
Email Triage Grader
===================
Scoring (normalised to [-1, 1]):
  +0.08  correct priority AND correct category
  +0.04  correct priority only
  +0.04  correct category only
  -0.10  marked urgent email as spam (severe penalty)
  -0.05  any other wrong classification

Max achievable score = 10 * 0.08 = 0.80 (capped at 1.0 after normalisation).
"""

from __future__ import annotations

from typing import Any


class EmailTriageGrader:
    def __init__(self, emails: list[dict[str, Any]]) -> None:
        self._ground_truth: dict[str, dict] = {
            e["id"]: {"priority": e["_priority"], "category": e["_category"]}
            for e in emails
        }

    def grade(
        self, classifications: list[dict[str, Any]]
    ) -> tuple[float, dict[str, float], str]:
        """
        Returns (normalised_reward, breakdown, message).
        normalised_reward is clamped to [-1, 1].
        """
        submitted: dict[str, dict] = {
            c["email_id"]: c for c in classifications if "email_id" in c
        }

        raw = 0.0
        breakdown: dict[str, float] = {}
        messages: list[str] = []

        for eid, truth in self._ground_truth.items():
            pred = submitted.get(eid)
            if pred is None:
                raw -= 0.05
                breakdown[eid] = -0.05
                messages.append(f"{eid}: missing")
                continue

            correct_priority = pred.get("priority", "").lower() == truth["priority"]
            correct_category = pred.get("category", "").lower() == truth["category"]

            # Severe penalty: urgent email classified as spam
            if truth["priority"] == "urgent" and pred.get("category", "").lower() == "spam":
                score = -0.10
                messages.append(f"{eid}: URGENT→spam penalty")
            elif correct_priority and correct_category:
                score = 0.08
                messages.append(f"{eid}: both correct")
            elif correct_priority:
                score = 0.04
                messages.append(f"{eid}: priority ok, category wrong")
            elif correct_category:
                score = 0.04
                messages.append(f"{eid}: category ok, priority wrong")
            else:
                score = -0.05
                messages.append(f"{eid}: both wrong")

            raw += score
            breakdown[eid] = score

        # Penalise extra phantom emails
        for eid in submitted:
            if eid not in self._ground_truth:
                raw -= 0.02
                breakdown[eid] = -0.02
                messages.append(f"{eid}: phantom email")

        # Normalise: max possible raw = 0.8, scale to 1.0
        normalised = round(max(-1.0, min(1.0, raw / 0.8)), 4)
        return normalised, breakdown, " | ".join(messages)
