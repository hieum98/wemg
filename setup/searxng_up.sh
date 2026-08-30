#!/usr/bin/env bash
# Start (or restart) the local SearXNG metasearch instance used as the free web
# search tier by langgraph_coe/tools/web.py.
#
#   ./setup/searxng_up.sh          # start, wait for health, print a probe result
#   ./setup/searxng_up.sh --down   # stop and remove the container
#
# The instance binds to 127.0.0.1 only: it is an internal fallback, not a service.
set -euo pipefail

NAME="${SEARXNG_CONTAINER:-searxng}"
PORT="${SEARXNG_PORT:-8080}"
IMAGE="${SEARXNG_IMAGE:-searxng/searxng:latest}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONF="$HERE/searxng"

if [[ "${1:-}" == "--down" ]]; then
  docker rm -f "$NAME" >/dev/null 2>&1 && echo "[searxng] removed container '$NAME'" || echo "[searxng] no container '$NAME'"
  exit 0
fi

# The secret only has to be stable for the life of the container (it signs
# session state we never use), so a per-start random value is fine.
SECRET="${SEARXNG_SECRET:-$(openssl rand -hex 32)}"

docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run -d --name "$NAME" --restart unless-stopped \
  -p "127.0.0.1:${PORT}:8080" \
  -v "$CONF/settings.yml:/etc/searxng/settings.yml:ro" \
  -e "SEARXNG_SECRET=$SECRET" \
  -e "SEARXNG_BASE_URL=http://localhost:${PORT}/" \
  "$IMAGE" >/dev/null

printf '[searxng] starting'
for _ in $(seq 1 40); do
  if curl -fsS -m 3 "http://localhost:${PORT}/healthz" >/dev/null 2>&1; then
    echo " — healthy on http://localhost:${PORT}"
    # A health check only proves the process is up; the thing that actually
    # breaks is the JSON format being disabled, so probe a real search.
    n=$(curl -fsS -m 25 "http://localhost:${PORT}/search?q=capital+of+Bolivia&format=json" \
        | python3 -c 'import json,sys; print(len(json.load(sys.stdin).get("results",[])))' 2>/dev/null || echo 0)
    echo "[searxng] json probe: ${n} results"
    [[ "$n" -gt 0 ]] || { echo "[searxng] WARNING: 0 results — check engine reachability with 'docker logs $NAME'"; exit 1; }
    exit 0
  fi
  printf '.'
  sleep 1
done

echo " — FAILED to become healthy; last logs:" >&2
docker logs --tail 30 "$NAME" >&2
exit 1
