#!/usr/bin/env bash
# Serve the open agent models on BOTH Kaggle T4s: one Ollama server per GPU, each holding the same models.
#
#   GPU0 -> http://localhost:11434/v1      GPU1 -> http://localhost:11435/v1
#
# blackboard_core.py gets both URLs (LLM_BASE_URL="http://localhost:11434/v1,http://localhost:11435/v1")
# and sends every call to the less busy server, so both GPUs work in parallel for every role, and a crashed
# server is skipped. Per GPU (T4, ~15 GB usable), agents phase:
#   llama3.1:8b  Q4 ~4.9 GB + KV (4 parallel x 4096 ctx, q8_0) ~1 GB
#   mistral:7b   Q4 ~4.4 GB + KV                                ~1 GB      -> ~12-13 GB per GPU
# The FActScore judge (gemma2:9b) is only needed for scoring, so it gets its own phase (agents unloaded).
#
#   bash scripts/kaggle_serve_agents.sh            # = agents: install, start both servers, load agents
#   bash scripts/kaggle_serve_agents.sh judge      # unload agents, load the judge on both GPUs
#   bash scripts/kaggle_serve_agents.sh agents     # back to the agents
#   bash scripts/kaggle_serve_agents.sh status
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

install() {
  if ! command -v ollama >/dev/null 2>&1; then
    echo "[serve] installing ollama..."
    (apt-get install -y -qq zstd >/dev/null 2>&1 || sudo apt-get install -y -qq zstd >/dev/null 2>&1 || true)
    curl -fsSL https://ollama.com/install.sh | sh >/dev/null
  fi
}

up() { curl -s "http://localhost:$1/api/version" >/dev/null 2>&1; }

start_servers() {
  for i in "${!PORTS[@]}"; do
    port=${PORTS[$i]}
    if up "$port"; then continue; fi
    echo "[serve] starting ollama on GPU$i -> port $port (log: $LOGDIR/ollama_gpu$i.log)"
    CUDA_VISIBLE_DEVICES=$i \
    OLLAMA_HOST=127.0.0.1:$port \
    OLLAMA_MAX_LOADED_MODELS="${OLLAMA_MAX_LOADED_MODELS:-2}" \
    OLLAMA_NUM_PARALLEL="${OLLAMA_NUM_PARALLEL:-4}" \
    OLLAMA_KEEP_ALIVE="${OLLAMA_KEEP_ALIVE:--1}" \
    OLLAMA_CONTEXT_LENGTH="${OLLAMA_CONTEXT_LENGTH:-4096}" \
    OLLAMA_FLASH_ATTENTION="${OLLAMA_FLASH_ATTENTION:-1}" \
    OLLAMA_KV_CACHE_TYPE="${OLLAMA_KV_CACHE_TYPE:-q8_0}" \
      setsid nohup ollama serve > "$LOGDIR/ollama_gpu$i.log" 2>&1 < /dev/null &
  done
  for port in "${PORTS[@]}"; do
    for _ in $(seq 1 60); do up "$port" && break; sleep 2; done
    up "$port" || { echo "[serve] server on port $port did not start"; tail -20 "$LOGDIR"/ollama_gpu*.log; exit 1; }
  done
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

load() {   # load every model on every server (in parallel) and check it answers
  pids=()
  for port in "${PORTS[@]}"; do
    for m in "$@"; do
      ( curl -s "http://localhost:$port/v1/chat/completions" -H "Content-Type: application/json" \
          -d "{\"model\": \"$m\", \"messages\": [{\"role\": \"user\", \"content\": \"Say OK.\"}], \"max_tokens\": 5}" \
          | python3 -c "import sys, json; r = json.load(sys.stdin); print('[serve] port $port  $m ->', r['choices'][0]['message']['content'].strip())" ) &
      pids+=($!)
    done
  done
  for p in "${pids[@]}"; do wait "$p"; done
}

status() {
  for i in "${!PORTS[@]}"; do
    echo "--- GPU$i (port ${PORTS[$i]}) ---"
    OLLAMA_HOST=127.0.0.1:${PORTS[$i]} ollama ps || true
  done
  nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv
}

case "$MODE" in
  agents)
    install; start_servers; pull $AGENT_MODELS
    unload $JUDGE_MODELS; load $AGENT_MODELS; status ;;
  judge)
    install; start_servers; pull $JUDGE_MODELS
    unload $AGENT_MODELS; load $JUDGE_MODELS; status ;;
  status)
    status ;;
  *) echo "usage: $0 [agents|judge|status]"; exit 2 ;;
esac
URLS=$(printf 'http://localhost:%s/v1,' "${PORTS[@]}"); URLS=${URLS%,}
echo "[serve] ready ($MODE). LLM_BASE_URL=$URLS"
