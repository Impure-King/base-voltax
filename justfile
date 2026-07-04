# Pod-of-Three ritual commands

# Start a daily standup with Akame
standup:
    opencode run --agent Akame "Standup: what did we finish since last session, what's on deck today, blockers?"

# Kick off a review session on a PR
review PR:
    git worktree add .sandbox/sylvia/review-runs/pr-{{PR}} origin/pull/{{PR}}/head || true
    opencode run --agent Sylvia "Review PR #{{PR}}. Worktree at .sandbox/sylvia/review-runs/pr-{{PR}}."

# Clean up a review worktree
review-done PR:
    git worktree remove .sandbox/sylvia/review-runs/pr-{{PR}}

# Kick off a design-partner session on a sandbox project
ethan-session PROJECT:
    opencode run --agent Ethan "Session in .sandbox/ethan/projects/{{PROJECT}}. Log to notes/session-log.md."

# Ask an agent to respond to an issue
check-issue AGENT NUM:
    opencode run --agent {{AGENT}} "Please check GitHub issue #{{NUM}} and respond as appropriate."
