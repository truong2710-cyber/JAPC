#!/usr/bin/env bash
set -euo pipefail

# Usage: ./scripts/run_validation_with_support_idx.sh <support_idx>
# Example: ./scripts/run_validation_with_support_idx.sh 64

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
CFG_FILE="$ROOT_DIR/config_ssl_upload.py"
RESULTS_FILE="$ROOT_DIR/validation_results_ssl_alp_setting2_set1_ours.json"

usage() {
    echo "Usage: $0 <support_idx>" >&2
    echo "       $0 start end" >&2
    echo "       $0 all    # iterate 0..64" >&2
    exit 2
}

ARGS=("$@")
if [ "${#ARGS[@]}" -eq 0 ]; then
    usage
fi
if [ "${ARGS[0]}" = "all" ]; then
    START=0
    END=64
    OFFSET=1
elif [ "${#ARGS[@]}" -ge 2 ] && [[ "${ARGS[1]}" =~ ^[0-9]+$ ]]; then
    START="${ARGS[0]}"
    END="${ARGS[1]}"
    OFFSET=2
else
    START="${ARGS[0]}"
    END="${ARGS[0]}"
    OFFSET=1
fi

# Additional Sacred overrides can follow the indices and will be forwarded.
OVERRIDES=( )
if [ "${#ARGS[@]}" -gt "$OFFSET" ]; then
    OVERRIDES=( "${ARGS[@]:$OFFSET}" )
fi

for SUPPORT_IDX in $(seq "$START" "$END"); do
        echo "\n=== Running support_idx = ${SUPPORT_IDX} ==="
        echo "Running validation.py with Sacred config override (support_idx=[${SUPPORT_IDX}] ${OVERRIDES[*]})"
        CMD=(python3 validation.py with "support_idx=[${SUPPORT_IDX}]")
        if [ "${#OVERRIDES[@]}" -gt 0 ]; then
            for o in "${OVERRIDES[@]}"; do
                CMD+=("$o")
            done
        fi
        (cd "$ROOT_DIR" && "${CMD[@]}")
        RC=$?
        if [ $RC -ne 0 ]; then
            echo "validation.py exited with code $RC for support_idx=${SUPPORT_IDX}" >&2
        fi

    echo "Searching for latest metrics.json under runs/"
        LATEST_METRICS=$(python3 - <<PY
import glob,os
files = glob.glob('../runs_ssl_alp_setting2_set1_ours/mySSL__CURVAS_Superpix_sets_1_1shot/**/metrics.json', recursive=True)
if not files:
        print('')
else:
        latest = max(files, key=os.path.getmtime)
        print(latest)
PY
)

    if [ -z "$LATEST_METRICS" ]; then
        echo "No metrics.json found under runs/ for support_idx=${SUPPORT_IDX}. Skipping append." >&2
        continue
    fi

    echo "Found metrics: $LATEST_METRICS"

    echo "Extracting mar_val_batches_classDice and appending to $RESULTS_FILE"
    python3 - <<PY
import json, os
metrics_path = r'''$LATEST_METRICS'''
with open(metrics_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

key = 'mar_val_batches_classDice'
val = data.get(key)
entry = {'support_idx': int(${SUPPORT_IDX}), key: val}

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


echo "Done."

exit 0
