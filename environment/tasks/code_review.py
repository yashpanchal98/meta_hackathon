"""
Code Review Task (Hard)
=======================
The agent receives a Python pull request diff with 5 seeded bugs and must
identify them and suggest fixes.

Bugs seeded:
  B1 - SQL injection via string formatting in a query
  B2 - Off-by-one error in a pagination slice
  B3 - Mutable default argument in function signature
  B4 - Missing authentication check before a sensitive endpoint
  B5 - Division by zero when denominator can be 0

Action payload schema (multi-step)::

    # Inspect a specific file (costs 0 steps)
    {"action": "inspect", "file": "app/db.py"}

    # Submit a finding
    {"action": "report_bug", "bug_id": "B1", "line": 12,
     "description": "SQL injection via f-string", "fix": "Use parameterised query"}

    # Final submit (ends episode)
    {"action": "submit"}

Partial credit:
  - +0.15 per correctly identified bug (matched by bug_id in the seeded set)
  - +0.05 bonus if the fix description mentions the right keyword
  - -0.05 per false-positive report (bug_id not in seeded set)
  - Episode ends after MAX_STEPS or explicit submit.
"""

from __future__ import annotations

from typing import Any

from ..graders.code_review_grader import CodeReviewGrader


PR_DIFF = """\
diff --git a/app/db.py b/app/db.py
index 1a2b3c4..5d6e7f8 100644
--- a/app/db.py
+++ b/app/db.py
@@ -1,20 +1,30 @@
 import sqlite3
+from typing import Optional

 def get_user(conn: sqlite3.Connection, username: str) -> Optional[dict]:
-    cursor = conn.execute("SELECT * FROM users WHERE username = ?", (username,))
+    # B1: SQL injection — user input interpolated directly into query string
+    cursor = conn.execute(f"SELECT * FROM users WHERE username = '{username}'")
     row = cursor.fetchone()
     return dict(row) if row else None

 def paginate(items: list, page: int, page_size: int = 10) -> list:
-    start = (page - 1) * page_size
-    end = start + page_size
+    start = page * page_size          # B2: off-by-one — page 1 skips first page_size items
+    end = start + page_size
     return items[start:end]

diff --git a/app/utils.py b/app/utils.py
index aabbcc..ddeeff 100644
--- a/app/utils.py
+++ b/app/utils.py
@@ -1,15 +1,20 @@
 from typing import List

-def merge_tags(new_tags: List[str], existing: List[str] | None = None) -> List[str]:
+# B3: mutable default argument — default list is shared across all calls
+def merge_tags(new_tags: List[str], existing: List[str] = []) -> List[str]:
     if existing is None:
         existing = []
     return existing + new_tags

 def compute_rate(success: int, total: int) -> float:
-    return success / total
+    # B5: ZeroDivisionError when total == 0
+    return success / total

diff --git a/app/views.py b/app/views.py
index 112233..445566 100644
--- a/app/views.py
+++ b/app/views.py
@@ -1,20 +1,25 @@
 from flask import request, jsonify, g
 from .auth import require_auth

 @app.route("/admin/delete_user", methods=["POST"])
-@require_auth(role="admin")
+# B4: @require_auth decorator removed — endpoint is now unauthenticated
 def delete_user():
     user_id = request.json.get("user_id")
     db.delete_user(g.conn, user_id)
     return jsonify({"status": "deleted"})
"""

SEEDED_BUGS: dict[str, dict] = {
    "B1": {
        "file": "app/db.py",
        "line": 8,
        "description": "SQL injection via f-string interpolation",
        "fix_keywords": ["parameteris", "parameteriz", "placeholder", "prepared", "?"],
    },
    "B2": {
        "file": "app/db.py",
        "line": 14,
        "description": "Off-by-one in pagination — page 1 skips first page",
        "fix_keywords": ["page - 1", "(page-1)", "page_size * (page - 1)"],
    },
    "B3": {
        "file": "app/utils.py",
        "line": 4,
        "description": "Mutable default argument shared across calls",
        "fix_keywords": ["none", "None", "if existing is None", "default=None"],
    },
    "B4": {
        "file": "app/views.py",
        "line": 5,
        "description": "Missing authentication decorator on admin endpoint",
        "fix_keywords": ["require_auth", "auth", "decorator", "authentication", "@"],
    },
    "B5": {
        "file": "app/utils.py",
        "line": 11,
        "description": "Division by zero when total == 0",
        "fix_keywords": ["total == 0", "total != 0", "if total", "guard", "zero"],
    },
}

MAX_STEPS = 15

FILES = {
    "app/db.py": """\
import sqlite3
from typing import Optional

def get_user(conn: sqlite3.Connection, username: str) -> Optional[dict]:
    # BUG B1 — interpolated query
    cursor = conn.execute(f"SELECT * FROM users WHERE username = '{username}'")
    row = cursor.fetchone()
    return dict(row) if row else None

def paginate(items: list, page: int, page_size: int = 10) -> list:
    start = page * page_size  # BUG B2
    end = start + page_size
    return items[start:end]
""",
    "app/utils.py": """\
from typing import List

# BUG B3 — mutable default
def merge_tags(new_tags: List[str], existing: List[str] = []) -> List[str]:
    if existing is None:
        existing = []
    return existing + new_tags

def compute_rate(success: int, total: int) -> float:
    # BUG B5 — no zero guard
    return success / total
""",
    "app/views.py": """\
from flask import request, jsonify, g
from .auth import require_auth

# BUG B4 — missing @require_auth
@app.route("/admin/delete_user", methods=["POST"])
def delete_user():
    user_id = request.json.get("user_id")
    db.delete_user(g.conn, user_id)
    return jsonify({"status": "deleted"})
""",
}


class CodeReviewTask:
    """Multi-step code review task with 5 seeded bugs."""

    def __init__(self) -> None:
        self._grader = CodeReviewGrader(SEEDED_BUGS)
        self._step = 0
        self._done = False
        self._reports: list[dict] = []

    def reset(self) -> dict[str, Any]:
        self._step = 0
        self._done = False
        self._reports = []
        return {
            "content": (
                "You are a security-focused code reviewer.\n"
                "A pull request has been opened. Review it and report all bugs.\n\n"
                f"PR DIFF:\n```diff\n{PR_DIFF}```\n\n"
                "Use 'inspect' to view a full file, 'report_bug' to record a finding, "
                "and 'submit' when done.\n"
                f"You have {MAX_STEPS} steps."
            ),
            "metadata": {
                "files": list(FILES.keys()),
                "max_steps": MAX_STEPS,
                "bug_count_hint": len(SEEDED_BUGS),
            },
        }

    def step(self, payload: dict[str, Any]) -> tuple[dict, Any, bool, dict]:
        from ..env import Reward

        action = payload.get("action", "")
        self._step += 1

        if action == "inspect":
            fname = payload.get("file", "")
            content = FILES.get(fname, f"File '{fname}' not found.")
            obs = {
                "content": f"File: {fname}\n```python\n{content}```",
                "metadata": {"file": fname},
            }
            return obs, Reward(value=0.0, breakdown={}, message="File inspected."), False, {}

        elif action == "report_bug":
            bug_id = payload.get("bug_id", "").upper()
            description = payload.get("description", "")
            fix = payload.get("fix", "")
            self._reports.append({"bug_id": bug_id, "description": description, "fix": fix})

            incremental, msg = self._grader.score_report(bug_id, description, fix)
            obs = {
                "content": f"Bug report '{bug_id}' recorded. {msg}",
                "metadata": {"bug_id": bug_id, "incremental_score": incremental},
            }
            done = self._step >= MAX_STEPS
            self._done = done
            return obs, Reward(value=incremental, breakdown={"report": incremental}, message=msg), done, {}

        elif action == "submit" or self._step >= MAX_STEPS:
            final_reward, breakdown, message = self._grader.grade(self._reports)
            self._done = True
            obs = {
                "content": f"Review complete. Score: {final_reward:.2f}\n{message}",
                "metadata": {"breakdown": breakdown},
            }
            return obs, Reward(value=final_reward, breakdown=breakdown, message=message), True, {}

        else:
            obs = {"content": f"Unknown action '{action}'.", "metadata": {}}
            return obs, Reward(value=-0.02, breakdown={}, message="Invalid action."), False, {}

    def state(self) -> dict[str, Any]:
        return {"step": self._step, "done": self._done, "reports": self._reports}
