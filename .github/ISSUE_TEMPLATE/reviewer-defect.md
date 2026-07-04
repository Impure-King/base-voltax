---
name: "Code review defect (Sylvia)"
about: A confirmed defect identified during code review that outlives a PR
title: "[SEVERITY] <short title>"
labels: ["bug", "from:sylvia", "source:review"]
---

## Severity
<BLOCKER | MAJOR | MINOR>

## Location
`path/to/file.py:L120–L138` (function/class name)

## Observation
<What the code does, factually.>

## Why it matters
<Concrete consequence — correctness, perf, reproducibility, security, etc.>

## Minimal repro
<Commands or artifact path under `.sandbox/sylvia/repro/`.>

## Suggested direction
<A direction, not a rewrite. Reference a pattern or specific technique.>

## Confidence
<High / Medium / Low, with one-line reason.>
