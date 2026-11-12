#!/usr/bin/env bash
# Start an Ollama server on a Kaggle (or Colab) GPU box and pull the open agent models.
# Ollama serves an OpenAI-compatible API at http://localhost:11434/v1, which blackboard_core.py talks to
# through LLM_BASE_URL / LLM_<ROLE>_MODEL (see notebooks/kaggle_eval.ipynb for the role mapping).
#
#   bash scripts/kaggle_serve_agents.sh                    # default models
#   MODELS="llama3.1:8b mistral:7b" bash scripts/kaggle_serve_agents.sh
#
# Run it AFTER the Qwen generation step (stage 1): Qwen needs both GPUs, the agents need them afterwards.
set -euo pipefail

MODELS="${MODELS:-llama3.1:8b mistral:7b gemma2:9b}"

if ! command -v ollama >/dev/null 2>&1; then
  echo "[serve] installing ollama..."
  (apt-get install -y -qq zstd >/dev/null 2>&1 || sudo apt-get install -y -qq zstd >/dev/null 2>&1 || true)
  curl -fsSL https://ollama.com/install.sh | sh
fi

if ! curl -s http://localhost:11434/api/version >/dev/null 2>&1; then
  echo "[serve] starting ollama serve (log: /kaggle/working/ollama.log)..."
  export OLLAMA_HOST=127.0.0.1:11434
  export OLLAMA_MAX_LOADED_MODELS="${OLLAMA_MAX_LOADED_MODELS:-3}"   # keep all agent models resident
  export OLLAMA_NUM_PARALLEL="${OLLAMA_NUM_PARALLEL:-4}"             # concurrent requests per model
  export OLLAMA_KEEP_ALIVE="${OLLAMA_KEEP_ALIVE:--1}"                # never unload between calls
  export OLLAMA_CONTEXT_LENGTH="${OLLAMA_CONTEXT_LENGTH:-8192}"      # evidence prompts need > 4k tokens
  LOG=/kaggle/working/ollama.log
  [ -d /kaggle/working ] || LOG=./ollama.log
  setsid nohup ollama serve > "$LOG" 2>&1 < /dev/null &
  for i in $(seq 1 60); do
    curl -s http://localhost:11434/api/version >/dev/null 2>&1 && break
    sleep 2
  done
fi
curl -s http://localhost:11434/api/version && echo

for m in $MODELS; do
  echo "[serve] pulling $m ..."
  ollama pull "$m" >/dev/null
done

# warm up: load every model once so the first eval call does not time out
for m in $MODELS; do
  curl -s http://localhost:11434/v1/chat/completions -H "Content-Type: application/json" \
    -d "{\"model\": \"$m\", \"messages\": [{\"role\": \"user\", \"content\": \"Say OK.\"}], \"max_tokens\": 5}" \
    | python3 -c "import sys, json; r = json.load(sys.stdin); print('[serve] $m ->', r['choices'][0]['message']['content'].strip())"
done
ollama ps || true
echo "[serve] ready: http://localhost:11434/v1"
