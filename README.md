---
title: Meta Hackathon OpenEnv Benchmark
emoji: 🤖
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
license: mit
tags:
  - openenv
  - agent-eval
  - nlp
  - code-review
  - data-cleaning
---

# Real-World Agent Eval — OpenEnv Benchmark

An [OpenEnv](https://openenv.dev)-compliant benchmark environment for evaluating AI agents on practical real-world tasks.

## Tasks

| Task | Difficulty | Steps | Score Range |
|---|---|---|---|
| Email Triage | Easy | 1 | −1.0 → 1.0 |
| Data Cleaning | Medium | ≤10 | 0.0 → 1.0 |
| Code Review | Hard | ≤15 | −1.0 → 1.0 |

### Email Triage
Classify 10 emails by **priority** (`urgent`/`normal`/`low`) and **category** (`action_required`/`fyi`/`spam`/`newsletter`). Special −0.10 penalty for marking an urgent email as spam.

### Data Cleaning
Fix a seeded CSV with duplicate rows, null values, `$`-prefixed salary, float ages, inconsistent gender casing, and out-of-range values. Submit operations one at a time; graded on 8 deterministic checks.

### Code Review
Review a Python PR diff with 5 seeded bugs:
- **B1** SQL injection via f-string interpolation
- **B2** Off-by-one in pagination
- **B3** Mutable default argument
- **B4** Missing `@require_auth` decorator on admin endpoint
- **B5** Division by zero

+0.15 per identified bug, +0.05 bonus if fix description contains the right keyword. −0.05 for false positives.

## Reward Design

All rewards are normalised to `[−1, 1]` (or `[0, 1]` for data cleaning). Graders are fully **deterministic** — no LLM-as-judge. Intermediate rewards are issued after each step in multi-step tasks.
