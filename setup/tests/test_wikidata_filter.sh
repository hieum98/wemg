#!/usr/bin/env bash
# Unit test for the dump filter in setup/wikidata_up.sh.
#
# The filter is the only custom logic in the Wikidata setup path, and it is the one part
# that cannot be checked by running it: a real pass takes ~34 minutes over a ~50 GB dump.
# So it is tested here against a hand-built sample covering each keep and drop rule.
#
# Getting it wrong is expensive and quiet. Dropping `prop/direct/` would empty the KG
# surface (wikidata_backend.py's fetch_outgoing/fetch_incoming filter on exactly that
# prefix) and every hop would retrieve nothing; keeping non-English labels or the full
# statement model would inflate 2.06 B triples back toward 6.5 B and overflow the disk.
#
#   ./setup/tests/test_wikidata_filter.sh
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Reuse the real filter rather than a copy, so this cannot drift from what ships.
# shellcheck disable=SC1090
filter_program () { :; }
eval "$(sed -n '/^filter_program () {$/,/^}$/p' "$HERE/../wikidata_up.sh")"

PROG=$(mktemp); filter_program > "$PROG"
AWKBIN=$(command -v mawk || command -v awk)
pass=0; fail=0

check () {  # expect(keep|drop)  description  line
    local expect="$1" desc="$2" line="$3" out
    out=$(printf '%s\n' "$line" | "$AWKBIN" -f "$PROG")
    local got="drop"; [ -n "$out" ] && got="keep"
    if [ "$got" = "$expect" ]; then
        pass=$((pass+1))
    else
        fail=$((fail+1)); printf '  FAIL (%s, got %s): %s\n    %s\n' "$expect" "$got" "$desc" "$line"
    fi
}

echo "filter rules:"

# ── keep: the direct claims the KG tools actually query ───────────────────────
check keep "wdt: direct claim" \
  '<http://www.wikidata.org/entity/Q42> <http://www.wikidata.org/prop/direct/P31> <http://www.wikidata.org/entity/Q5> .'
check keep "wdt: claim with a literal object" \
  '<http://www.wikidata.org/entity/Q42> <http://www.wikidata.org/prop/direct/P569> "1952-03-11T00:00:00Z"^^<http://www.w3.org/2001/XMLSchema#dateTime> .'

# ── keep: English labels / descriptions / aliases, and sitelink subjects ──────
check keep "English rdfs:label" \
  '<http://www.wikidata.org/entity/Q42> <http://www.w3.org/2000/01/rdf-schema#label> "Douglas Adams"@en .'
check keep "English schema:description" \
  '<http://www.wikidata.org/entity/Q42> <http://schema.org/description> "English writer"@en .'
check keep "English skos:altLabel" \
  '<http://www.wikidata.org/entity/Q42> <http://www.w3.org/2004/02/skos/core#altLabel> "Douglas Noel Adams"@en .'
check keep "schema:about (sitelink), untagged" \
  '<https://en.wikipedia.org/wiki/Douglas_Adams> <http://schema.org/about> <http://www.wikidata.org/entity/Q42> .'

# ── drop: other languages. This is where most of the volume is. ───────────────
check drop "German label" \
  '<http://www.wikidata.org/entity/Q42> <http://www.w3.org/2000/01/rdf-schema#label> "Douglas Adams"@de .'
check drop "Japanese description" \
  '<http://www.wikidata.org/entity/Q42> <http://schema.org/description> "イギリスの作家"@ja .'
check drop "French altLabel" \
  '<http://www.wikidata.org/entity/Q42> <http://www.w3.org/2004/02/skos/core#altLabel> "Douglas Noel Adams"@fr .'
# A label whose *text* contains "@en" but whose tag is not @en must still be dropped.
check drop "label containing the literal text @en but tagged @de" \
  '<http://www.wikidata.org/entity/Q1> <http://www.w3.org/2000/01/rdf-schema#label> "mail@example.com"@de .'

# ── drop: the reified statement model and the non-truthy predicates ───────────
check drop "p: statement node" \
  '<http://www.wikidata.org/entity/Q42> <http://www.wikidata.org/prop/P31> <http://www.wikidata.org/entity/statement/Q42-abc> .'
check drop "ps: statement value" \
  '<http://www.wikidata.org/entity/statement/Q42-abc> <http://www.wikidata.org/prop/statement/P31> <http://www.wikidata.org/entity/Q5> .'
check drop "prov:wasDerivedFrom" \
  '<http://www.wikidata.org/entity/statement/Q42-abc> <http://www.w3.org/ns/prov#wasDerivedFrom> <http://www.wikidata.org/reference/xyz> .'
check drop "schema:dateModified" \
  '<http://www.wikidata.org/entity/Q42> <http://schema.org/dateModified> "2024-01-01T00:00:00Z"^^<http://www.w3.org/2001/XMLSchema#dateTime> .'
check drop "rdf:type" \
  '<http://www.wikidata.org/entity/Q42> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://wikiba.se/ontology#Item> .'
# `prop/direct-normalized/` is a DIFFERENT predicate namespace and must not be swept in by
# a loose prefix match — the rule is anchored at position 1 and compares the full prefix.
check drop "prop/direct-normalized/ is not prop/direct/" \
  '<http://www.wikidata.org/entity/Q42> <http://www.wikidata.org/prop/direct-normalized/P213> "0000000121441970" .'
check drop "blank line" ''

rm -f "$PROG"
echo
printf '%d passed, %d failed\n' "$pass" "$fail"
[ "$fail" -eq 0 ]
