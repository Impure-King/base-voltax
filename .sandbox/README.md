# `.sandbox/` — Agent Working Directories

This directory is scratch space for the pod's subagents (Sylvia and Ethan).
**Everything here except this README is gitignored.** Nothing under `.sandbox/`
gets committed.

The layout is standardized so that:

- Subagents always know where their artifacts live.
- Akame and the mentee always know where to look for a subagent's work.
- Nothing subagents produce accidentally pollutes the source tree.

## Layout

```
.sandbox/
├── README.md                    # this file (committed)
│
├── sylvia/                      # Code Reviewer
│   ├── review-runs/             # git worktrees of PR branches for execution
│   │   └── pr-<num>/
│   ├── repro/                   # minimal repros of defects found
│   │   └── issue-<num>/
│   ├── reports/                 # structured review reports
│   │   ├── pr-<num>.md
│   │   └── audit-<YYYY-MM-DD>.md
│   └── issues-drafted/          # local copies of issues filed via gh
│       └── <slug>.md
│
└── ethan/                       # Design Partner
    ├── projects/                # sandbox projects for realistic usage
    │   └── <project-name>/
    ├── benchmarks/              # benchmark runs with raw output
    │   └── <YYYY-MM-DD>-<name>/
    │       ├── env.txt          # environment fingerprint
    │       ├── repro.sh         # command to re-run
    │       ├── raw.log          # verbatim stdout/stderr
    │       └── summary.md       # honest read of the numbers
    ├── notes/                   # freeform session logs
    │   └── session-log.md
    └── issues-drafted/          # local copies of issues filed via gh
        └── <slug>.md
```

## Working with `git worktree` (Sylvia)

Sylvia uses `git worktree` to check out PR branches into
`.sandbox/sylvia/review-runs/pr-<num>/` without disturbing the mentee's
working tree:

```bash
git worktree add .sandbox/sylvia/review-runs/pr-42 origin/pull/42/head
# review work here
git worktree remove .sandbox/sylvia/review-runs/pr-42
```

List active worktrees: `git worktree list`.
Prune stale ones: `git worktree prune`.

## `.gitignore` rule

```
.sandbox/*
!.sandbox/README.md
```

This ignores everything under `.sandbox/` **except** this README.

## Verifying the ignore rule works

```bash
git check-ignore -v .sandbox/sylvia/reports/dummy.md
# Should print a line pointing at the .gitignore rule.
```

If it prints nothing, the rule isn't matching — check `.gitignore`.

## Access conventions

* **Sylvia** works only inside `.sandbox/sylvia/`.
* **Ethan** works only inside `.sandbox/ethan/`.
* **Akame** may read anywhere under `.sandbox/` to understand what
  subagents have found, but does not write to it.
* **The mentee** may read or write anywhere — but the whole point of
  sandboxing is that you don't need to touch it in normal use. If you find
  yourself editing files under `.sandbox/`, something has gone off-pattern.
