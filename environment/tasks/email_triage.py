"""
Email Triage Task (Easy)
========================
The agent receives a batch of 10 emails and must classify each one with:
  - priority: "urgent" | "normal" | "low"
  - category: "action_required" | "fyi" | "spam" | "newsletter"

Action payload schema::

    {
        "classifications": [
            {"email_id": "e1", "priority": "urgent", "category": "action_required"},
            ...
        ]
    }

The episode ends after a single step (batch classification).
"""

from __future__ import annotations

from typing import Any

from ..graders.email_triage_grader import EmailTriageGrader


EMAILS: list[dict[str, Any]] = [
    {
        "id": "e01",
        "subject": "URGENT: Production server down",
        "from": "ops-alerts@company.com",
        "body": "The primary production server has been unresponsive for 10 minutes. Immediate action required.",
        "_priority": "urgent",
        "_category": "action_required",
    },
    {
        "id": "e02",
        "subject": "Weekly newsletter: Top 10 AI papers",
        "from": "digest@aiweekly.io",
        "body": "This week's top AI research highlights...",
        "_priority": "low",
        "_category": "newsletter",
    },
    {
        "id": "e03",
        "subject": "Invoice #4521 due in 2 days",
        "from": "billing@vendor.com",
        "body": "Please process payment for invoice #4521 totalling $3,200 before Friday.",
        "_priority": "urgent",
        "_category": "action_required",
    },
    {
        "id": "e04",
        "subject": "You've won a $500 gift card!",
        "from": "prizes@totally-legit.biz",
        "body": "Congratulations! Click here to claim your prize...",
        "_priority": "low",
        "_category": "spam",
    },
    {
        "id": "e05",
        "subject": "Q3 financial report available",
        "from": "finance@company.com",
        "body": "The Q3 financial report has been published to the shared drive. No action needed.",
        "_priority": "normal",
        "_category": "fyi",
    },
    {
        "id": "e06",
        "subject": "Security alert: Unusual login detected",
        "from": "security@company.com",
        "body": "We detected a login from an unrecognised device. Please verify or reset your password immediately.",
        "_priority": "urgent",
        "_category": "action_required",
    },
    {
        "id": "e07",
        "subject": "Team lunch this Friday",
        "from": "hr@company.com",
        "body": "Just a reminder about the team lunch on Friday at noon. RSVP by Wednesday.",
        "_priority": "normal",
        "_category": "action_required",
    },
    {
        "id": "e08",
        "subject": "Congratulations on your promotion!",
        "from": "ceo@company.com",
        "body": "It is my pleasure to announce your promotion to Senior Engineer, effective next month.",
        "_priority": "normal",
        "_category": "fyi",
    },
    {
        "id": "e09",
        "subject": "Special offer: 50% off SaaS tools",
        "from": "deals@saas-promo.com",
        "body": "Limited time offer for premium SaaS subscriptions...",
        "_priority": "low",
        "_category": "spam",
    },
    {
        "id": "e10",
        "subject": "Code review requested: PR #847",
        "from": "github-noreply@github.com",
        "body": "Alice has requested your review on PR #847: 'Add OAuth2 support'. Please review at your earliest convenience.",
        "_priority": "normal",
        "_category": "action_required",
    },
]


class EmailTriageTask:
    """Single-step batch email classification task."""

    MAX_STEPS = 1

    def __init__(self) -> None:
        self._grader = EmailTriageGrader(EMAILS)
        self._done = False
        self._last_result: dict[str, Any] = {}

    def reset(self) -> dict[str, Any]:
        self._done = False
        self._last_result = {}
        visible = [
            {k: v for k, v in e.items() if not k.startswith("_")}
            for e in EMAILS
        ]
        return {
            "content": (
                "You are an email assistant. Classify each email below.\n\n"
                + _format_emails(visible)
                + "\n\nFor each email provide: priority (urgent/normal/low) "
                "and category (action_required/fyi/spam/newsletter)."
            ),
            "metadata": {"email_count": len(EMAILS), "emails": visible},
        }

    def step(self, payload: dict[str, Any]) -> tuple[dict, Any, bool, dict]:
        from ..graders.email_triage_grader import EmailTriageGrader
        from ..env import Reward

        classifications = payload.get("classifications", [])
        reward, breakdown, message = self._grader.grade(classifications)
        self._done = True
        self._last_result = {"classifications": classifications, "breakdown": breakdown}

        obs = {
            "content": f"Grading complete. Score: {reward:.2f}\n{message}",
            "metadata": {"breakdown": breakdown},
        }
        return obs, Reward(value=reward, breakdown=breakdown, message=message), True, {}

    def state(self) -> dict[str, Any]:
        return {"done": self._done, "last_result": self._last_result}


def _format_emails(emails: list[dict]) -> str:
    lines = []
    for e in emails:
        lines.append(
            f"[{e['id']}] FROM: {e['from']}\n"
            f"    SUBJECT: {e['subject']}\n"
            f"    BODY: {e['body']}"
        )
    return "\n\n".join(lines)
