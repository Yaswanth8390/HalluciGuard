#!/usr/bin/env bash
# Serve the open agent models on BOTH Kaggle T4s: one Ollama server per GPU, each holding the same models.
#
#   GPU0 -> http://localhost:11434/v1      GPU1 -> http://localhost:11435/v1
#
# blackboard_core.py gets both URLs (LLM_BASE_URL="http://localhost:11434/v1,http://localhost:11435/v1")
# and sends every call to the less busy server, so both GPUs work in parallel for every role, and a crashed
# server is skipped. Per GPU (T4, ~15 GB usable), agents phase:
#   llama3.1:8b  Q4 ~4.9 GB + KV (2 parallel x 4096 ctx, q8_0) ~0.5 GB
#   mistral:7b   Q4 ~4.4 GB + KV                                ~0.5 GB   -> ~11 GB per GPU,
# plus OLLAMA_GPU_OVERHEAD (1.5 GB) kept free, so Ollama never fills a GPU to the brim.
#
# Staying on the GPU matters: if a model does not fit, Ollama silently runs part of it on the CPU. That is
# ~10x slower and fills host RAM until Kaggle restarts the notebook. After loading, this script checks that
# every model runs "100% GPU"; if not, it restarts the servers with 1 parallel request per model.
# A watchdog (background) restarts a server that crashed and reloads its models; meanwhile the client
# fails over to the other GPU.
#
#   bash scripts/kaggle_serve_agents.sh            # = agents: install, start both servers, load agents
#   bash scripts/kaggle_serve_agents.sh judge      # unload agents, load the judge on both GPUs
#   bash scripts/kaggle_serve_agents.sh agents     # back to the agents
#   bash scripts/kaggle_serve_agents.sh status     # models, GPU/CPU split, GPU memory, host RAM
#   bash scripts/kaggle_serve_agents.sh stop       # stop servers + watchdog
#
# Run it AFTER the Qwen generation step (stage 1): Qwen is sharded over both GPUs and must have exited.
set -euo pipefail

MODE="${1:-agents}"
AGENT_MODELS="${AGENT_MODELS:-llama3.1:8b mistral:7b}"
JUDGE_MODELS="${JUDGE_MODELS:-gemma2:9b}"
PORTS=(11434 11435)
NGPU=$(nvidia-smi -L 2>/dev/null | wc -l || echo 1)
[ "$NGPU" -ge 2 ] || PORTS=(11434)
LOGDIR=/kaggle/working
[ -d "$LOGDIR" ] || LOGDIR=.
SCRIPT="$(readlink -f "$0")"
MODELS_FILE="$LOGDIR/ollama_models.txt"      # what the watchdog reloads after a restart
PARALLEL_FILE="$LOGDIR/ollama_parallel.txt"
WATCHDOG_PID="$LOGDIR/ollama_watchdog.pid"
CONFIG_FILE="$LOGDIR/ollama_config.txt"       # settings the running servers were started with
PARALLEL="${OLLAMA_NUM_PARALLEL:-$(cat "$PARALLEL_FILE" 2>/dev/null || echo 2)}"

install() {
  if ! command -v ollama >/dev/null 2>&1; then
    echo "[serve] installing ollama..."
    (apt-get install -y -qq zstd >/dev/null 2>&1 || sudo apt-get install -y -qq zstd >/dev/null 2>&1 || true)
    curl -fsSL https://ollama.com/install.sh | sh >/dev/null
  fi
}

up() { curl -s -m 5 "http://localhost:$1/api/version" >/dev/null 2>&1; }

start_one() {   # start the server for GPU $1 (no-op if it is already up)
  local i=$1 port=${PORTS[$1]}
  up "$port" && return 0
  echo "[serve] starting ollama on GPU$i -> port $port, $PARALLEL parallel/model (log: $LOGDIR/ollama_gpu$i.log)"
  CUDA_VISIBLE_DEVICES=$i \
  OLLAMA_HOST=127.0.0.1:$port \
  OLLAMA_MAX_LOADED_MODELS="${OLLAMA_MAX_LOADED_MODELS:-2}" \
  OLLAMA_NUM_PARALLEL="$PARALLEL" \
  OLLAMA_KEEP_ALIVE="${OLLAMA_KEEP_ALIVE:--1}" \
  OLLAMA_CONTEXT_LENGTH="${OLLAMA_CONTEXT_LENGTH:-4096}" \
  OLLAMA_FLASH_ATTENTION="${OLLAMA_FLASH_ATTENTION:-1}" \
  OLLAMA_KV_CACHE_TYPE="${OLLAMA_KV_CACHE_TYPE:-q8_0}" \
  OLLAMA_GPU_OVERHEAD="${OLLAMA_GPU_OVERHEAD:-1610612736}" \
    setsid nohup ollama serve >> "$LOGDIR/ollama_gpu$i.log" 2>&1 < /dev/null &
  echo $! > "$LOGDIR/ollama_gpu$i.pid"
  for _ in $(seq 1 60); do up "$port" && return 0; sleep 2; done
  echo "[serve] server on port $port did not start"; tail -20 "$LOGDIR/ollama_gpu$i.log"; return 1
}

start_servers() { for i in "${!PORTS[@]}"; do start_one "$i"; done; }

config() { echo "$PARALLEL ${OLLAMA_CONTEXT_LENGTH:-4096} ${OLLAMA_GPU_OVERHEAD:-1610612736} ${OLLAMA_KV_CACHE_TYPE:-q8_0}"; }

restart_if_stale() {   # servers started by an older version / other settings: restart them with these
  local any=1
  for port in "${PORTS[@]}"; do up "$port" && any=0; done
  if [ "$any" -eq 0 ] && [ "$(cat "$CONFIG_FILE" 2>/dev/null)" != "$(config)" ]; then
    echo "[serve] servers are running with other settings; restarting them"
    stop_servers
  fi
}

stop_servers() {
  for i in "${!PORTS[@]}"; do
    [ -f "$LOGDIR/ollama_gpu$i.pid" ] && kill "$(cat "$LOGDIR/ollama_gpu$i.pid")" 2>/dev/null || true
  done
  pkill -f "ollama serve" 2>/dev/null || true
  pkill -f "ollama runner" 2>/dev/null || true
  for port in "${PORTS[@]}"; do for _ in $(seq 1 15); do up "$port" || break; sleep 1; done; done
  sleep 3
}

pull() {   # the model store is shared, pulling through one server is enough
  for m in "$@"; do
    echo "[serve] pulling $m ..."
    OLLAMA_HOST=127.0.0.1:${PORTS[0]} ollama pull "$m" >/dev/null
  done
}

unload() {
  for port in "${PORTS[@]}"; do
    for m in "$@"; do
      curl -s "http://localhost:$port/api/generate" -d "{\"model\": \"$m\", \"keep_alive\": 0}" >/dev/null || true
    done
  done
}

load_on() {   # load models $2.. on the server at port $1 (one after the other, so memory is estimated cleanly)
  local port=$1; shift
  for m in "$@"; do
    curl -s -m 600 "http://localhost:$port/v1/chat/completions" -H "Content-Type: application/json" \
      -d "{\"model\": \"$m\", \"messages\": [{\"role\": \"user\", \"content\": \"Say OK.\"}], \"max_tokens\": 5}" \
      | python3 -c "import sys, json; r = json.load(sys.stdin); print('[serve] port $port  $m ->', r['choices'][0]['message']['content'].strip())" \
      || echo "[serve] port $port  $m -> FAILED to load"
  done
}

load() {   # every model on every server; the servers load in parallel
  local pids=()
  for port in "${PORTS[@]}"; do load_on "$port" "$@" & pids+=($!); done
  for p in "${pids[@]}"; do wait "$p" || true; done
}

gpu_only() {   # true if every expected model is loaded and runs 100% on its GPU
  local ok=0 n
  for port in "${PORTS[@]}"; do
    n=$(OLLAMA_HOST=127.0.0.1:$port ollama ps 2>/dev/null | tail -n +2 | grep -c "100% GPU" || true)
    if [ "$n" -lt "$#" ]; then
      echo "[serve] port $port: only $n of $# models fully on the GPU:"
      OLLAMA_HOST=127.0.0.1:$port ollama ps 2>/dev/null || true
      ok=1
    fi
  done
  return $ok
}

bring_up() {   # start servers, load models, make sure nothing spills to the CPU
  start_servers
  load "$@"
  if ! gpu_only "$@"; then
    if [ "$PARALLEL" -gt 1 ]; then
      echo "[serve] part of a model runs on the CPU (slow, fills RAM). Restarting with 1 parallel request per model."
      stop_servers
      PARALLEL=1
      start_servers
      load "$@"
      gpu_only "$@" || echo "[serve] WARNING: still not fully on the GPU. Try OLLAMA_CONTEXT_LENGTH=3072 bash $0 $MODE"
    else
      echo "[serve] WARNING: not fully on the GPU. Try OLLAMA_CONTEXT_LENGTH=3072 bash $0 $MODE"
    fi
  fi
  echo "$PARALLEL" > "$PARALLEL_FILE"
  config > "$CONFIG_FILE"
  echo "$*" > "$MODELS_FILE"
}

start_watchdog() {
  if [ -f "$WATCHDOG_PID" ] && kill -0 "$(cat "$WATCHDOG_PID")" 2>/dev/null; then return 0; fi
  setsid nohup bash "$SCRIPT" watchdog >> "$LOGDIR/ollama_watchdog.log" 2>&1 < /dev/null &
  echo $! > "$WATCHDOG_PID"
  echo "[serve] watchdog started (log: $LOGDIR/ollama_watchdog.log)"
}

watchdog() {   # restart a crashed server and reload its models; the client uses the other GPU meanwhile
  while true; do
    sleep 30
    [ -f "$MODELS_FILE" ] || continue
    read -r -a models < "$MODELS_FILE" || continue
    for i in "${!PORTS[@]}"; do
      port=${PORTS[$i]}
      if ! up "$port"; then
        echo "$(date '+%F %T') GPU$i server (port $port) is down; restarting"
        tail -5 "$LOGDIR/ollama_gpu$i.log" || true
        if start_one "$i"; then load_on "$port" "${models[@]}" || true; fi
      fi
    done
  done
}

status() {
  for i in "${!PORTS[@]}"; do
    echo "--- GPU$i (port ${PORTS[$i]}) ---"
    OLLAMA_HOST=127.0.0.1:${PORTS[$i]} ollama ps || echo "(server down)"
  done
  nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv || true
  free -g | head -2 || true
}

case "$MODE" in
  agents)
    install; restart_if_stale; start_servers; pull $AGENT_MODELS
    unload $JUDGE_MODELS; bring_up $AGENT_MODELS; start_watchdog; status ;;
  judge)
    install; restart_if_stale; start_servers; pull $JUDGE_MODELS
    unload $AGENT_MODELS; bring_up $JUDGE_MODELS; start_watchdog; status ;;
  status)
    status ;;
  watchdog)
    watchdog ;;
  stop)
    [ -f "$WATCHDOG_PID" ] && kill "$(cat "$WATCHDOG_PID")" 2>/dev/null || true
    stop_servers; echo "[serve] stopped"; exit 0 ;;
  *) echo "usage: $0 [agents|judge|status|stop]"; exit 2 ;;
esac
URLS=$(printf 'http://localhost:%s/v1,' "${PORTS[@]}"); URLS=${URLS%,}
echo "[serve] ready ($MODE, $PARALLEL parallel/model). LLM_BASE_URL=$URLS"
