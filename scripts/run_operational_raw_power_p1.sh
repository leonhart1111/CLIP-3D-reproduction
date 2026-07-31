#!/usr/bin/env bash
# Run the guarded, non-formal raw-power P1 operational experiment.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

MODE="${1:-}"
case "$MODE" in
    pilot|full) ;;
    *)
        echo "usage: $0 {pilot|full} [output-root]" >&2
        exit 2
        ;;
esac

ROOT="${2:-runs/operational_raw_power_p1/$MODE}"
test ! -e "$ROOT" || {
    echo "refusing existing output root: $ROOT" >&2
    exit 2
}
mkdir -p "$ROOT"

source scripts/env.sh

CONFIG="configs/experiments/clip3d_constrained_5p0_raw_power_p1_operational.json"
SOURCE_PROXY_REPORT="results/parameter_studies/raw_power_strict_20260730/proxy_train_16/calibration_report.json"
OPERATIONAL_REPORT="$ROOT/operational_proxy_report.json"
R1_ROOT="runs/architecture_sweep/r1/paper"

python -m workflow.analysis.evaluate_operational_proxy \
    --proxy-report "$SOURCE_PROXY_REPORT" \
    --config "$CONFIG" \
    --output "$OPERATIONAL_REPORT"

python - "$OPERATIONAL_REPORT" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    report = json.load(stream)
if report.get("recommendation", {}).get("accepted") is not True:
    raise SystemExit("refusing operational launch: policy recommendation is not accepted")
PY

sha256sum "$CONFIG" "$SOURCE_PROXY_REPORT" > "$ROOT/provenance.sha256"

if [[ "$MODE" == "pilot" ]]; then
    R1_POINT="$R1_ROOT/matmul/l1d_32kB/l2_512kB"
    python -m workflow.run_lifting_pipeline \
        --r1-dir "$R1_POINT" \
        --output-dir "$ROOT/fixed-bin" \
        --config "$CONFIG" \
        --layout-method fixed-bin \
        --run-r2
    python -m workflow.run_lifting_pipeline \
        --r1-dir "$R1_POINT" \
        --output-dir "$ROOT/clip3d" \
        --config "$CONFIG" \
        --layout-method clip3d \
        --run-r2
    exit 0
fi

mapfile -t STATUS_FILES < <(find "$R1_ROOT" -type f -name status.json | sort)
if [[ "${#STATUS_FILES[@]}" -ne 100 ]]; then
    echo "refusing full launch: all 100 R1 status files must be success (found ${#STATUS_FILES[@]})" >&2
    exit 2
fi
python - "${STATUS_FILES[@]}" <<'PY'
import json
import sys

failed = []
for path in sys.argv[1:]:
    try:
        with open(path, encoding="utf-8") as stream:
            state = json.load(stream).get("state")
    except (OSError, ValueError, TypeError) as error:
        failed.append(f"{path}: {error}")
        continue
    if state != "success":
        failed.append(f"{path}: state={state!r}")
if failed:
    print("refusing full launch: all 100 R1 status files must be success", file=sys.stderr)
    print("\n".join(failed), file=sys.stderr)
    raise SystemExit(2)
PY

python -m workflow.run_lifting_sweep \
    --r1-root "$R1_ROOT" \
    --output-root "$ROOT/fixed-bin" \
    --config "$CONFIG" \
    --layout-method fixed-bin \
    --jobs 1 \
    --run-r2
python -m workflow.run_lifting_sweep \
    --r1-root "$R1_ROOT" \
    --output-root "$ROOT/clip3d" \
    --config "$CONFIG" \
    --layout-method clip3d \
    --jobs 1 \
    --run-r2
