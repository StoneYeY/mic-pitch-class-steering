#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# w2s job runner — runs on the GPU machine, transport = GitHub.
#
#   one-time:  git clone <repo> w2s && cd w2s && git checkout icassp
#              tmux new -d -s w2s 'bash runner/runner.sh'
#   watch:     tmux attach -t w2s        stop: touch runner/STOP
#   cancel:    (from anywhere) commit an empty file jobs/queue/<name>.cancel
#
# Loop: git pull -> run every jobs/queue/*.sh without jobs/done/<name>.exit
#       -> commit + push jobs/done/*, results/*, runner/status.json.
# Contract for jobs:
#   * cwd = repo root; runner/env.sh is sourced first (conda etc.); PYTHONPATH includes the root
#   * small outputs (csv/json/png/txt) -> results/<job>/   (pushed)
#   * big outputs (wav/npz/pt)         -> runs/<job>/      (git-ignored, stays here)
#   * be idempotent (skip work whose outputs exist) — jobs may be re-run after a crash
# ---------------------------------------------------------------------------
set -u
W2S_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$W2S_ROOT" || exit 1
export W2S_ROOT
BRANCH="${W2S_BRANCH:-icassp}"
POLL="${RUNNER_POLL:-20}"
JOB_TIMEOUT="${RUNNER_JOB_TIMEOUT:-12h}"
LOG_TAIL="${RUNNER_LOG_TAIL:-4000}"
PROGRESS_PUSH="${RUNNER_PROGRESS_PUSH:-300}"     # seconds between progress pushes while a job runs
QUEUE="jobs/queue"; DONE="jobs/done"; LOGS="runner/logs"
HOST="$(hostname)"
GIT="git -c user.name=w2s-gpu -c user.email=w2s-gpu@local"
mkdir -p "$QUEUE" "$DONE" "$LOGS" results runs

say() { printf '[%s] %s\n' "$(date '+%m-%d %H:%M:%S')" "$*"; }

sync_pull() {
  $GIT pull -q --rebase --autostash origin "$BRANCH" 2>/tmp/w2s_pull.err && return 0
  say "pull failed: $(tail -1 /tmp/w2s_pull.err)"; git rebase --abort 2>/dev/null; return 1
}

push_results() {   # $1 = commit message
  $GIT add -A "$DONE" results runner/status.json 2>/dev/null || true
  git diff --cached --quiet && return 0
  $GIT commit -q -m "$1" || return 1
  local n
  for n in 1 2 3 4 5; do
    sync_pull && $GIT push -q origin "HEAD:$BRANCH" 2>/tmp/w2s_push.err && { say "pushed: $1"; return 0; }
    say "push failed ($n): $(tail -1 /tmp/w2s_push.err 2>/dev/null)"; sleep 15
  done
  return 1
}

write_status() {  # $1=state $2=job
  local gpu
  gpu="$(nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null | head -1 || echo n/a)"
  printf '{"host":"%s","state":"%s","job":"%s","gpu":"%s","time":"%s","commit":"%s","queue":%s,"done":%s}\n' \
    "$HOST" "$1" "$2" "$gpu" "$(date -Is)" "$(git rev-parse --short HEAD 2>/dev/null)" \
    "$(ls "$QUEUE"/*.sh 2>/dev/null | wc -l | tr -d ' ')" "$(ls "$DONE"/*.exit 2>/dev/null | wc -l | tr -d ' ')" \
    > runner/status.json
}

git fetch -q origin "$BRANCH" 2>/dev/null || true
git checkout -q "$BRANCH" 2>/dev/null || git checkout -q -b "$BRANCH" "origin/$BRANCH" || { say "branch $BRANCH not found"; exit 1; }
git config pull.rebase true
say "runner up on $HOST at $W2S_ROOT (branch $BRANCH, poll ${POLL}s, timeout $JOB_TIMEOUT)"
[ -f runner/env.sh ] && say "sourcing runner/env.sh for every job"
write_status idle ""; push_results "gpu: runner up on $HOST" || true

while true; do
  if [ -f runner/STOP ]; then say "STOP found, exiting"; rm -f runner/STOP; write_status stopped ""; push_results "gpu: runner stopped" || true; exit 0; fi
  sync_pull || { sleep "$POLL"; continue; }
  ran_any=0
  for job in $(ls "$QUEUE"/*.sh 2>/dev/null | sort); do
    name="$(basename "$job" .sh)"
    [ -f "$DONE/$name.exit" ] && continue
    [ -f "$QUEUE/$name.cancel" ] && { echo "cancelled before start" > "$DONE/$name.exit"; push_results "gpu: $name cancelled" || true; continue; }
    ran_any=1
    say "=== START $name ==="
    write_status running "$name"; push_results "gpu: start $name" || true
    start_ts=$(date +%s); last_push=$start_ts
    (
      set -o pipefail
      [ -f runner/env.sh ] && source runner/env.sh
      export W2S_JOB="$name" PYTHONUNBUFFERED=1 PYTHONPATH="$W2S_ROOT${PYTHONPATH:+:$PYTHONPATH}"
      export W2S_RESULTS="results/$name" W2S_RUNS="runs/$name"
      mkdir -p "results/$name" "runs/$name"
      exec timeout --foreground "$JOB_TIMEOUT" bash "$job"
    ) > "$LOGS/$name.full.log" 2>&1 &
    jpid=$!
    while kill -0 "$jpid" 2>/dev/null; do
      sleep 30
      write_status running "$name"
      now=$(date +%s)
      if (( now - last_push >= PROGRESS_PUSH )); then
        tail -n 80 "$LOGS/$name.full.log" > "$DONE/$name.progress" 2>/dev/null
        sync_pull; push_results "gpu: progress $name ($(( (now-start_ts)/60 ))min)" || true
        last_push=$now
        if [ -f "$QUEUE/$name.cancel" ]; then
          say "cancel requested for $name"; pkill -TERM -P "$jpid" 2>/dev/null; kill -TERM "$jpid" 2>/dev/null
          sleep 5; pkill -KILL -P "$jpid" 2>/dev/null; kill -KILL "$jpid" 2>/dev/null
        fi
      fi
    done
    wait "$jpid"; code=$?
    dur=$(( $(date +%s) - start_ts ))
    { echo "# job=$name exit=$code duration_s=$dur host=$HOST finished=$(date -Is)"
      echo "# (last $LOG_TAIL lines; full log: runner/logs/$name.full.log on $HOST)"
      tail -n "$LOG_TAIL" "$LOGS/$name.full.log"; } > "$DONE/$name.log"
    echo "$code" > "$DONE/$name.exit"
    rm -f "$DONE/$name.progress"
    say "=== END $name exit=$code (${dur}s) ==="
    write_status idle ""; push_results "gpu: done $name exit=$code (${dur}s)" || true
    [ -f runner/STOP ] && break
  done
  [ "$ran_any" = 0 ] && sleep "$POLL"
done
