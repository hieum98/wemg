#!/usr/bin/env bash
# Launch the three local SGLang servers that langgraph_coe/config.eval.yaml expects.
#
#   ./setup/sglang_up.sh          # start all three, wait until each answers /v1/models
#   ./setup/sglang_up.sh --down   # stop them
#   ./setup/sglang_up.sh --check  # just report which endpoints are reachable
#
# ── the tier -> model -> port mapping ────────────────────────────────────────────
#   heavy, plan, consolidate  Qwen3-8B    :30000   the roles that reason over evidence
#   medium                    Qwen3-4B    :30001   extraction, verification, scoring
#   light                     Qwen3-1.7B  :30002   bounded classification / IE
#
# Ports must match `llm.tiers.*.api_base` in config.eval.yaml. Override with
# QWEN8B_PORT / QWEN4B_PORT / QWEN17B_PORT if these collide with something.
#
# ── why --reasoning-parser qwen3 is not optional ────────────────────────────────
# Without it SGLang leaves the chain-of-thought INLINE in `content`, wrapped in
# <think>...</think>, and every structured-output parse then has to survive a reasoning
# trace prepended to its JSON. With it, SGLang splits the trace into `reasoning_content`
# and leaves `content` clean, which is what langgraph_coe's parsers assume.
# See https://docs.sglang.io/advanced_features/separate_reasoning.html
#
# Reasoning is then toggled PER TIER by `enable_thinking`, which llm.py forwards as
# `chat_template_kwargs={"enable_thinking": ...}` — the Qwen3 chat template reads it.
# (`reasoning_effort` is the Bedrock-only equivalent and is inert here; see
# TierConfig.reasoning_effort.)
#
# `--enable-custom-logit-processor` is required only if a tier sets `thinking_budget`.
# config.eval.yaml leaves it null, so it is off by default here; uncomment per server if
# you want to cap reasoning length.
set -uo pipefail

MODEL_ROOT="${MODEL_ROOT:-Qwen}"     # HF org, or a local directory of checkpoints
PY="${PY:-python3}"
LOG_DIR="${LOG_DIR:-/tmp}"

# tier-group : model : port : extra args
SERVERS=(
  "qwen8b:${MODEL_ROOT}/Qwen3-8B:${QWEN8B_PORT:-30000}"
  "qwen4b:${MODEL_ROOT}/Qwen3-4B:${QWEN4B_PORT:-30001}"
  "qwen17b:${MODEL_ROOT}/Qwen3-1.7B:${QWEN17B_PORT:-30002}"
)

reachable () {  # port
    timeout 3 curl -sf "http://localhost:$1/v1/models" >/dev/null 2>&1
}

case "${1:-}" in
  --down)
    for s in "${SERVERS[@]}"; do
        IFS=: read -r name _model port <<< "$s"
        # Match the port to avoid killing an unrelated SGLang server. `pkill -f` on the
        # module name alone would also match the shell running this script.
        pids=$(ps -eo pid,args | awk -v p="--port $port" '$0 ~ /sglang\.launch_server/ && index($0, p) {print $1}')
        if [ -n "$pids" ]; then kill $pids && echo "  stopped $name (port $port)"; else echo "  $name not running"; fi
    done
    exit 0
    ;;
  --check)
    for s in "${SERVERS[@]}"; do
        IFS=: read -r name model port <<< "$s"
        if reachable "$port"; then echo "  OK          $name  $model  :$port"
        else echo "  UNREACHABLE $name  $model  :$port"; fi
    done
    exit 0
    ;;
esac

for s in "${SERVERS[@]}"; do
    IFS=: read -r name model port <<< "$s"
    if reachable "$port"; then
        echo "  $name already up on :$port"
        continue
    fi
    echo "  starting $name ($model) on :$port -> $LOG_DIR/sglang_$name.log"
    setsid nohup "$PY" -m sglang.launch_server \
        --model-path "$model" \
        --port "$port" \
        --host 127.0.0.1 \
        --reasoning-parser qwen3 \
        > "$LOG_DIR/sglang_$name.log" 2>&1 < /dev/null &
        # --enable-custom-logit-processor   # only needed for thinking_budget
done

echo "  waiting for endpoints (up to 10 min; first load includes weight download)"
for s in "${SERVERS[@]}"; do
    IFS=: read -r name _model port <<< "$s"
    for _ in $(seq 1 200); do
        reachable "$port" && { echo "  ready: $name :$port"; break; }
        sleep 3
    done
    reachable "$port" || echo "  TIMEOUT: $name :$port — see $LOG_DIR/sglang_$name.log"
done
