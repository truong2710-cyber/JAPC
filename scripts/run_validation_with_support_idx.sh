#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./scripts/run_validation_with_support_idx.sh <result_tag> <support_idx>
#   ./scripts/run_validation_with_support_idx.sh <result_tag> start end
#   ./scripts/run_validation_with_support_idx.sh <result_tag> all
#
# Examples:
#   ./scripts/run_validation_with_support_idx.sh results_ssl_alp_setting2_set1 64
#   ./scripts/run_validation_with_support_idx.sh results_ssl_alp_setting2_set1 0 64
#   ./scripts/run_validation_with_support_idx.sh results_ssl_alp_setting2_set1 all
#
# With extra Sacred overrides:
#   ./scripts/run_validation_with_support_idx.sh results_ssl_alp_setting2_set1 0 64 lr=0.001 batch_size=4

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
CFG_FILE="$ROOT_DIR/config_ssl_upload.py"

usage() {
    echo "Usage: $0 <result_tag> <support_idx> [sacred_overrides...]" >&2
    echo "       $0 <result_tag> start end [sacred_overrides...]" >&2
    echo "       $0 <result_tag> all [sacred_overrides...]    # iterate 0..64" >&2
    echo "" >&2
    echo "Example:" >&2
    echo "       $0 results_ssl_alp_setting2_set1 64" >&2
    echo "       $0 results_ssl_alp_setting2_set1 0 64" >&2
    echo "       $0 results_ssl_alp_setting2_set1 all" >&2
    exit 2
}

ARGS=("$@")

if [ "${#ARGS[@]}" -lt 2 ]; then
    usage
fi

RESULT_TAG="${ARGS[0]}"

# Convert:
#   results_ssl_alp_setting2_set1
# to:
#   ssl_alp_setting2_set1
#
# Then use it for:
#   validation_results_ssl_alp_setting2_set1.json
#   ../runs_ssl_alp_setting2_set1/...
RUN_TAG="${RESULT_TAG#results_}"

# Extract set number from tags like:
#   results_ssl_alp_setting2_set1
#   results_ssl_alp_setting2_set2
#   results_ssl_alp_setting2_set10
if [[ "$RESULT_TAG" =~ _set([0-9]+)($|_) ]]; then
    SET_NUM="${BASH_REMATCH[1]}"
else
    echo "Could not extract set number from RESULT_TAG: $RESULT_TAG" >&2
    echo "Expected tag containing: _set0, _set1, _set2, _set10, etc." >&2
    exit 2
fi

RESULTS_FILE="$ROOT_DIR/validation_${RESULT_TAG}.json"
RUNS_DIR="$ROOT_DIR/runs_${RUN_TAG}"
METRICS_GLOB="${RUNS_DIR}/mySSL__CURVAS_Superpix_sets_${SET_NUM}_1shot/**/metrics.json"

if [ "${ARGS[1]}" = "all" ]; then
    START=0
    END=64
    OFFSET=2
elif [ "${#ARGS[@]}" -ge 3 ] && [[ "${ARGS[2]}" =~ ^[0-9]+$ ]]; then
    START="${ARGS[1]}"
    END="${ARGS[2]}"
    OFFSET=3
else
    START="${ARGS[1]}"
    END="${ARGS[1]}"
    OFFSET=2
fi

# Additional Sacred overrides can follow the indices and will be forwarded.
OVERRIDES=()
if [ "${#ARGS[@]}" -gt "$OFFSET" ]; then
    OVERRIDES=("${ARGS[@]:$OFFSET}")
fi

echo "Result tag:      $RESULT_TAG"
echo "Run tag:         $RUN_TAG"
echo "Results file:    $RESULTS_FILE"
echo "Metrics pattern: $METRICS_GLOB"
echo ""

for SUPPORT_IDX in $(seq "$START" "$END"); do
    echo ""
    echo "=== Running support_idx = ${SUPPORT_IDX} ==="
    echo "Running validation.py with Sacred config override: support_idx=[${SUPPORT_IDX}] ${OVERRIDES[*]}"

    CMD=(python3 validation.py with "support_idx=[${SUPPORT_IDX}]")

    if [ "${#OVERRIDES[@]}" -gt 0 ]; then
        for o in "${OVERRIDES[@]}"; do
            CMD+=("$o")
        done
    fi

    set +e
    (cd "$ROOT_DIR" && "${CMD[@]}")
    RC=$?
    set -e

    if [ "$RC" -ne 0 ]; then
        echo "validation.py exited with code $RC for support_idx=${SUPPORT_IDX}" >&2
        continue
    fi

    echo "Searching for latest metrics.json using:"
    echo "$METRICS_GLOB"

    LATEST_METRICS=$(cd "$ROOT_DIR" && python3 - <<PY
import glob
import os

pattern = r'''$METRICS_GLOB'''
files = glob.glob(pattern, recursive=True)

if not files:
    print('')
else:
    latest = max(files, key=os.path.getmtime)
    print(latest)
PY
)

    if [ -z "$LATEST_METRICS" ]; then
        echo "No metrics.json found for support_idx=${SUPPORT_IDX}. Skipping append." >&2
        continue
    fi

    echo "Found metrics: $LATEST_METRICS"
    echo "Extracting mar_val_batches_classDice and appending to $RESULTS_FILE"

    python3 - <<PY
import json
import os

metrics_path = os.path.join(r'''$ROOT_DIR''', r'''$LATEST_METRICS''')

with open(metrics_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

key = 'mar_val_batches_classDice'
val = data.get(key)

entry = {
    'support_idx': int(${SUPPORT_IDX}),
    key: val,
    'metrics_path': metrics_path,
    'result_tag': r'''$RESULT_TAG'''
}

outf = r'''$RESULTS_FILE'''

if os.path.exists(outf):
    with open(outf, 'r', encoding='utf-8') as f:
        arr = json.load(f)

    if not isinstance(arr, list):
        arr = [arr]
else:
    arr = []

arr.append(entry)

with open(outf, 'w', encoding='utf-8') as f:
    json.dump(arr, f, indent=2)

print('Appended entry to', outf)
print(json.dumps(entry, indent=2))
PY

done

echo ""
echo "Done."
exit 0