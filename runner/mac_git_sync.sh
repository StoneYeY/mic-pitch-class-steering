#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Runs on the Mac, inside the repo clone.  Every INTERVAL seconds:
#   git add -A  ->  commit (if anything changed)  ->  pull --rebase  ->  push
# It is the "hands" of the cloud session: Claude writes files into this clone
# through the Cowork folder bridge, this loop publishes them to GitHub, and it
# also brings back what the GPU runner pushes (jobs/done, results).
#
#   bash runner/mac_git_sync.sh          (leave it running in its own tab; Ctrl-C stops)
# ---------------------------------------------------------------------------
set -u
BRANCH="${W2S_BRANCH:-icassp}"
INTERVAL="${SYNC_INTERVAL:-20}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO" || exit 1
say() { printf '[%s] %s\n' "$(date '+%H:%M:%S')" "$*"; }

git rev-parse --is-inside-work-tree >/dev/null 2>&1 || { echo "not a git repo: $REPO"; exit 1; }
git config user.name  >/dev/null || git config user.name  "w2s-mac"
git config user.email >/dev/null || git config user.email "w2s-mac@local"
git config pull.rebase true

# make sure we are on the sprint branch (create from current HEAD if needed)
if ! git show-ref --verify --quiet "refs/heads/$BRANCH"; then
  git fetch -q origin "$BRANCH" 2>/dev/null && git checkout -q -b "$BRANCH" "origin/$BRANCH" || git checkout -q -b "$BRANCH"
else
  git checkout -q "$BRANCH"
fi
git push -q -u origin "$BRANCH" 2>/dev/null || true
say "syncing $REPO on branch $BRANCH every ${INTERVAL}s"

while true; do
  git add -A >/dev/null 2>&1
  if ! git diff --cached --quiet; then
    git commit -q -m "mac: $(date '+%m-%d %H:%M:%S') $(git diff --cached --name-only | head -3 | tr '\n' ' ')" && say "committed"
  fi
  if ! git pull -q --rebase --autostash origin "$BRANCH" 2>/tmp/w2s_pull.err; then
    say "pull failed: $(tail -1 /tmp/w2s_pull.err)"; git rebase --abort 2>/dev/null
  fi
  if [ "$(git rev-list --count "origin/$BRANCH..HEAD" 2>/dev/null || echo 0)" != "0" ]; then
    git push -q origin "$BRANCH" 2>/tmp/w2s_push.err && say "pushed" || say "push failed: $(tail -1 /tmp/w2s_push.err)"
  fi
  st="$(tr -d '\n' < runner/status.json 2>/dev/null | cut -c1-150)"
  say "ok  ${st:-(runner not reporting yet)}"
  sleep "$INTERVAL"
done
