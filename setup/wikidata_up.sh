#!/usr/bin/env bash
# Local Wikidata SPARQL endpoint (QLever) for langgraph_coe's KG retrieval.
#
#   ./setup/wikidata_up.sh              # check: probe the endpoint (default)
#   ./setup/wikidata_up.sh status       # container, restart policy, index integrity
#   ./setup/wikidata_up.sh start
#   ./setup/wikidata_up.sh stop
#   ./setup/wikidata_up.sh build        # rebuild the index from the dump (~1h40m)
#
# ── which engine, and why it matters ────────────────────────────────────────────
# This is **QLever**, serving plain `/` on port 7001. docs/deploy_local_wikidata-v2.md
# describes a *different* engine (QEndpoint) at `/api/endpoint/sparql` — its prebuilt-HDT
# sources are unreachable from this box, which is why QLever is used instead. The path
# difference is load-bearing: point `wikidata.sparql_endpoint` at the bare origin
# (`http://localhost:7001`), NOT at a `/api/endpoint/sparql` suffix.
#
# ── what is indexed (a filtered subset, deliberately) ──────────────────────────
# 2,064,307,080 triples, ~72 GB index, all 6 permutations. Kept: `wdt:` direct claims,
# English `rdfs:label` / `schema:description` / `skos:altLabel`, and `schema:about`. That is
# exactly what langgraph_coe/tools/wikidata_backend.py queries — `fetch_outgoing` and
# `fetch_incoming` filter on `prop/direct/`. Full truthy (~6.5 B triples) needs ~200 GB of
# index and does not fit the 248 GB disk.
#
# Entity *search* and label lookup (`wbsearchentities` / `wbgetentities`) are NOT SPARQL and
# still hit the public MediaWiki API, which rate-limits; wikidata_backend.py falls back to
# resolving labels against this endpoint when the public API answers 429.
#
# ── two operational traps, both learned the hard way ────────────────────────────
# 1. `sg docker -c` is required. The `ubuntu` user's docker group membership is not active
#    in already-running shells, so a bare `docker` call fails with a permission error.
# 2. The QLever CLI DELETES AND RECREATES the container on every start, which drops the
#    restart policy. `start` below re-applies `--restart unless-stopped` afterwards; without
#    that the endpoint silently fails to come back after a reboot.
set -uo pipefail

PORT="${QLEVER_PORT:-7001}"
NAME="${QLEVER_DATASET:-wikidata-truthy}"
CONTAINER="${QLEVER_CONTAINER:-qlever.server.$NAME}"
IDX_DIR="${QLEVER_DIR:-/home/ubuntu/wikidata/qlever-truthy}"
QLEVER="${QLEVER_BIN:-/home/ubuntu/.local/bin/qlever}"
# Fast mirror: measured ~250 MB/s here, against a crawl from the canonical host.
DUMP_URL="${WIKIDATA_DUMP_URL:-https://dumps.wikimedia.your.org/wikidatawiki/entities/latest-truthy.nt.gz}"
ENDPOINT="http://localhost:$PORT"

# Every docker/qlever call goes through this. See trap 1 above.
dock () { sg docker -c "$*"; }

# Nearest existing ancestor of a path. `df` on a nonexistent path prints nothing, and the
# index dir (and often its parent) may be missing — which is exactly when the free-space
# figure matters most, because it decides whether a rebuild can even fit.
existing_ancestor () {
    local d="$1"
    while [ -n "$d" ] && [ "$d" != "/" ] && [ ! -d "$d" ]; do d=$(dirname "$d"); done
    printf '%s' "${d:-/}"
}

disk_free () { df -h "$(existing_ancestor "$1")" 2>/dev/null | tail -1 | awk '{print $4}'; }

probe () {  # -> prints a binding, or fails
    timeout "${1:-10}" curl -sf -G "$ENDPOINT" \
        --data-urlencode 'query=SELECT * WHERE { ?s ?p ?o } LIMIT 1' \
        -H 'Accept: application/sparql-results+json' 2>/dev/null
}

cmd_check () {
    printf 'endpoint %s ... ' "$ENDPOINT"
    if probe 10 | grep -q '"bindings"'; then
        echo "OK"
        local n
        n=$(timeout 60 curl -sf -G "$ENDPOINT" \
            --data-urlencode 'query=SELECT (COUNT(*) AS ?n) WHERE { ?s ?p ?o }' \
            -H 'Accept: application/sparql-results+json' 2>/dev/null \
            | grep -oE '"value":"[0-9]+"' | head -1 | grep -oE '[0-9]+')
        [ -n "$n" ] && echo "  triples: $n"
        return 0
    fi
    echo "UNREACHABLE"
    echo "  langgraph_coe will fall back to public Wikidata (slow, rate-limited) or, if"
    echo "  wikidata.sparql_endpoint is set and dead, KG retrieval returns nothing."
    echo "  Try: $0 start   (or $0 build if no index exists yet)"
    return 1
}

cmd_status () {
    cmd_check || true
    echo
    echo "container:"
    dock "docker ps -a --filter name=^${CONTAINER}\$ --format '  {{.Names}}  {{.Status}}  {{.Ports}}'" 2>/dev/null \
        | grep . || echo "  (no container named $CONTAINER)"
    local policy
    policy=$(dock "docker inspect $CONTAINER --format '{{.HostConfig.RestartPolicy.Name}}'" 2>/dev/null | tr -d '\r')
    if [ -n "$policy" ]; then
        printf '  restart policy: %s' "$policy"
        [ "$policy" = "unless-stopped" ] && echo "  OK" || echo "  <- will NOT survive a reboot; run '$0 fix-policy'"
    fi

    # Integrity: the container bind-mounts the index dir. If that directory has been
    # deleted from the host while the container ran, QLever keeps serving from the still-open
    # (unlinked) inodes and looks perfectly healthy — but the endpoint is UNRECOVERABLE,
    # because any stop or reboot loses it and there is nothing on disk to restart from.
    # This is not hypothetical; it is the exact state this box was found in.
    echo
    echo "index directory:"
    local src
    src=$(dock "docker inspect $CONTAINER --format '{{range .Mounts}}{{if eq .Destination \"/index\"}}{{.Source}}{{end}}{{end}}'" 2>/dev/null | tr -d '\r')
    src="${src:-$IDX_DIR}"
    if [ -d "$src" ]; then
        echo "  $src  ($(du -sh "$src" 2>/dev/null | cut -f1) on disk)  OK"
    else
        echo "  $src  ** MISSING FROM THE HOST **"
        echo "  The container may still be answering queries from unlinked inodes it holds"
        echo "  open. Do NOT stop or reboot expecting it to come back: there is no index on"
        echo "  disk to restart from, and 'docker exec' into it already fails."
        echo "  Recover with: $0 build   (~1h40m, needs ~72 GB free)"
    fi
    local anc; anc=$(existing_ancestor "$src")
    echo
    echo "disk: $(disk_free "$src") free at $anc"
}

cmd_fix_policy () {
    dock "docker update --restart unless-stopped $CONTAINER" >/dev/null 2>&1 \
        && echo "restart policy set to unless-stopped" \
        || echo "could not update policy (is $CONTAINER present?)"
}

cmd_start () {
    if probe 5 | grep -q '"bindings"'; then echo "already serving on $ENDPOINT"; exit 0; fi
    [ -d "$IDX_DIR" ] || { echo "no index dir at $IDX_DIR — run '$0 build' first"; exit 1; }
    ( cd "$IDX_DIR" && dock "$QLEVER start" ) || { echo "start failed"; exit 1; }
    # Trap 2: the CLI recreated the container, so the restart policy is gone.
    cmd_fix_policy
    for _ in $(seq 1 30); do
        probe 5 | grep -q '"bindings"' && { echo "serving on $ENDPOINT"; exit 0; }
        sleep 2
    done
    echo "started but not answering yet; check: cd $IDX_DIR && sg docker -c '$QLEVER log'"
}

cmd_stop () {
    if [ ! -d "$IDX_DIR" ]; then
        echo "REFUSING to stop: $IDX_DIR is missing from the host, so this endpoint cannot"
        echo "be restarted (see '$0 status'). Stopping it now would require a ~1h40m rebuild."
        echo "Override with FORCE=1 if that is genuinely what you want."
        [ "${FORCE:-0}" = "1" ] || exit 1
    fi
    ( cd "${IDX_DIR:-/tmp}" 2>/dev/null || cd /tmp; dock "$QLEVER stop" ) || dock "docker stop $CONTAINER"
    echo "stopped"
}

# The one piece of custom logic here, kept in a function so it can be unit-tested on a
# sample without a 50 GB download: see setup/tests/test_wikidata_filter.sh
filter_program () {
cat <<'AWK'
# Keep exactly what wikidata_backend.py queries, drop the rest. Cuts ~6.5 B triples to
# ~2.06 B. $2 is the predicate in N-Triples (`<s> <p> <o> .`).
$2 == "<http://www.w3.org/2000/01/rdf-schema#label>"      { if ($0 ~ /@en \.[[:space:]]*$/) print; next }
$2 == "<http://schema.org/description>"                    { if ($0 ~ /@en \.[[:space:]]*$/) print; next }
$2 == "<http://www.w3.org/2004/02/skos/core#altLabel>"     { if ($0 ~ /@en \.[[:space:]]*$/) print; next }
$2 == "<http://schema.org/about>"                          { print; next }
index($2, "<http://www.wikidata.org/prop/direct/") == 1    { print; next }
AWK
}

cmd_build () {
    cat <<EOF
Rebuild the local Wikidata index. Measured on this box: ~1h40m total, needs ~72 GB free
for the index plus room for the download.

  1. download  latest-truthy.nt.gz from the fast mirror        ~8 min
  2. filter    to the subset wikidata_backend.py queries      ~34 min
  3. index     qlever index (6 permutations)                  ~56 min
  4. start     mmap'd, so startup is seconds

Target dir : $IDX_DIR
Dump       : $DUMP_URL
Disk free  : $(disk_free "$IDX_DIR") (at $(existing_ancestor "$IDX_DIR"))
EOF
    if [ "${1:-}" != "--yes" ]; then
        echo
        echo "This downloads tens of GB and runs for ~1h40m. Re-run with --yes to proceed."
        exit 0
    fi

    mkdir -p "$IDX_DIR" || exit 1
    cd "$IDX_DIR" || exit 1

    if [ ! -f Qleverfile ]; then
        echo "[1/4] writing Qleverfile"
        # Bootstrap from QLever's own preset, then override for the filtered input and the
        # budgets this box uses. MEMORY_FOR_QUERIES / CACHE_MAX_SIZE are query and
        # result-cache budgets, NOT a data preload — QLever mmaps the index.
        "$QLEVER" setup-config wikidata >/dev/null 2>&1 || true
        cat > Qleverfile <<EOF
[data]
NAME             = $NAME
GET_DATA_CMD     = curl -L -o latest-truthy.nt.gz '$DUMP_URL'
DESCRIPTION      = Wikidata truthy, filtered to wdt: claims + English labels/descriptions/altLabels + schema:about

[index]
INPUT_FILES      = filtered.nt.gz
CAT_INPUT_FILES  = zcat filtered.nt.gz
SETTINGS_JSON    = { "languages-internal": [], "prefixes-external": [""], "ascii-prefixes-only": false, "num-triples-per-batch": 5000000 }

[server]
PORT             = $PORT
ACCESS_TOKEN     = $NAME
MEMORY_FOR_QUERIES = 20G
CACHE_MAX_SIZE     = 10G
TIMEOUT          = 120s

[runtime]
SYSTEM           = docker
IMAGE            = adfreiburg/qlever
EOF
    fi

    if [ ! -f latest-truthy.nt.gz ] && [ ! -f filtered.nt.gz ]; then
        echo "[2/4] downloading dump (~8 min on the fast mirror)"
        curl -L --fail -o latest-truthy.nt.gz "$DUMP_URL" || { echo "download failed"; exit 1; }
    fi

    if [ ! -f filtered.nt.gz ]; then
        echo "[3/4] filtering (~34 min; bound by gzip decompression)"
        filter_program > filter.awk
        # mawk where available: measurably faster than gawk on this shape of line.
        AWKBIN=$(command -v mawk || command -v awk)
        zcat latest-truthy.nt.gz | "$AWKBIN" -f filter.awk | gzip -1 > filtered.nt.gz.part \
            && mv filtered.nt.gz.part filtered.nt.gz \
            || { echo "filter failed"; rm -f filtered.nt.gz.part; exit 1; }
        # The dump is the largest file here and is not needed again; the index is built from
        # filtered.nt.gz. Removing it is what made room for the FAISS index previously.
        echo "  (keeping latest-truthy.nt.gz; delete it to reclaim disk once indexing succeeds)"
    fi

    echo "[4/4] building index (~56 min)"
    dock "$QLEVER index" || { echo "index failed — see $IDX_DIR/*.log"; exit 1; }
    cmd_start
}

case "${1:-check}" in
    check)      cmd_check ;;
    status)     cmd_status ;;
    start)      cmd_start ;;
    stop)       cmd_stop ;;
    fix-policy) cmd_fix_policy ;;
    build)      shift; cmd_build "${1:-}" ;;
    *) sed -n '2,10p' "$0"; exit 2 ;;
esac
