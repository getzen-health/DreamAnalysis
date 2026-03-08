#!/usr/bin/env bash
# issue-demon.sh — Autonomous GitHub issue execution pipeline
#
# Usage: ./scripts/issue-demon.sh
#
# RULES (DO NOT VIOLATE):
#   - Model selection: Haiku for eval/commenting, Sonnet for implementation, NEVER Opus
#   - MAX 2 agents in parallel
#   - MAX 10 agents per session
#   - Classify scope in orchestrator (no agent needed) before dispatching
#   - Run tests ONCE after a batch, not inside each agent
#   - Stop after each cycle — wait for user confirmation before next
#
# Scope classification:
#   quick-win: <50 lines, 1 file → dispatch Sonnet agent
#   medium:    2-3 files, self-contained → dispatch Sonnet agent
#   large:     multi-component, unclear scope → comment + skip (no agent)

set -euo pipefail

REPO="LakshmiSravyaVedantham/DreamAnalysis"
SESSION_AGENTS=0
MAX_AGENTS=10
MAX_PARALLEL=2

echo "================================="
echo "ISSUE DEMON — $(date '+%Y-%m-%d %H:%M:%S')"
echo "Repo: $REPO | Max agents: $MAX_AGENTS | Max parallel: $MAX_PARALLEL"
echo "Models: Haiku=eval/comment, Sonnet=impl, Opus=FORBIDDEN"
echo "================================="

# Fetch open issues (orchestrator does this directly — no agent)
issues=$(gh issue list \
  --repo "$REPO" \
  --state open \
  --limit 20 \
  --json number,title,labels \
  2>&1)

count=$(echo "$issues" | python3 -c "import json,sys; print(len(json.load(sys.stdin)))")
echo "Open issues: $count"
echo "$issues" | python3 -c "
import json, sys
data = json.load(sys.stdin)
for i in data:
    labels = [l['name'] for l in i.get('labels', [])]
    print(f'  #{i[\"number\"]:3d}  [{\" \".join(labels) or \"no-label\"}]  {i[\"title\"]}')
"

echo ""
echo "Handing issue list to orchestrator (Claude) for scope classification."
echo "Orchestrator reads relevant files directly, then dispatches MAX $MAX_PARALLEL agents."
echo ""
echo "$issues"
