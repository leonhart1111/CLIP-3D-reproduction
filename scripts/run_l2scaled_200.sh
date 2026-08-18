#!/usr/bin/env bash
# Run the L2-scaled 200-point full flow: R1 (converged profiles) then
# fixed-bin + clip3d pipelines with R2 for the 100-architecture grid.
# Env overrides: PY, EXP, CFG, R1ROOT, OUTROOT, NPROC, R1_JOBS.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLIP_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$CLIP_ROOT"

PY="${PY:-$CLIP_ROOT/.venv/bin/python}"
EXP="${EXP:-configs/experiments/r1_short_convergence_l2scaled.json}"
CFG="${CFG:-configs/experiments/clip3d_constrained_5p0_raw_power_p1_lambda0020119_traffic_weighted_exploratory.json}"
R1ROOT="${R1ROOT:-runs/architecture_sweep/r1}"
OUTROOT="${OUTROOT:-runs/validation200_l2scaled_$(date +%Y%m%d_%H%M%S)}"
NPROC="${NPROC:-6}"
R1_JOBS="${R1_JOBS:-4}"

echo "== CLIP-3D L2-scaled 200-point full flow =="
echo "OUTROOT=$OUTROOT  NPROC=$NPROC  R1_JOBS=$R1_JOBS"

[ -x "$PY" ] || { echo "ERROR: python not executable: $PY"; exit 1; }
[ -f "$EXP" ] || { echo "ERROR: experiment config missing: $EXP"; exit 1; }
[ -f "$CFG" ] || { echo "ERROR: lifting config missing: $CFG"; exit 1; }
[ -f "$EXP" ] && echo "L2 grid: $(python3 -c "import json;print(json.load(open('$EXP'))['l2_sizes'])")"

mkdir -p "$OUTROOT/logs"
: > "$OUTROOT/progress.log"

# ---- Step 1: R1 with per-workload converged profiles ----
for w in fft cholesky stream stencil; do
  "$PY" scripts/run_r1_sweep.py --experiment "$EXP" --profile short_conv_10m \
    --workloads "$w" --output-root "$R1ROOT" --jobs "$R1_JOBS" --execute
done
"$PY" scripts/run_r1_sweep.py --experiment "$EXP" --profile short_conv_1m \
  --workloads matmul --output-root "$R1ROOT" --jobs "$R1_JOBS" --execute

# ---- Step 2: 200 pipelines (fixed-bin + clip3d, R2), resumable ----
declare -A PROF=( [fft]=short_conv_10m [cholesky]=short_conv_10m [matmul]=short_conv_1m [stream]=short_conv_10m [stencil]=short_conv_10m )
total=0
for w in fft cholesky matmul stream stencil; do
  for l1 in 16kB 32kB 64kB 128kB; do
    for l2 in 1MB 2MB 4MB 8MB 16MB; do
      r1="$R1ROOT/${PROF[$w]}/$w/l1d_$l1/l2_$l2"
      [ -f "$r1/r1_metadata.json" ] || { echo "ERROR: missing R1: $r1"; exit 1; }
      for m in fixed-bin clip3d; do
        total=$((total+1))
        out="$OUTROOT/$w/l1d_$l1/l2_$l2/$m"
        log="$OUTROOT/logs/${w}_${l1}_${l2}_${m}.log"
        if [ -f "$out/pipeline_summary.json" ]; then
          echo "skip $w/$l1/$l2/$m"
          continue
        fi
        ( "$PY" -m workflow.run_lifting_pipeline --r1-dir "$r1" --output-dir "$out" \
            --config "$CFG" --layout-method "$m" --run-r2 >"$log" 2>&1; \
          echo "$w/$l1/$l2/$m rc=$?" >> "$OUTROOT/progress.log" ) &
        while [ "$(jobs -rp | wc -l)" -ge "$NPROC" ]; do sleep 20; done
      done
    done
  done
done
wait

summaries=$(find "$OUTROOT" -name pipeline_summary.json | wc -l)
echo "ALL PIPELINES DONE: $summaries/$total summaries in $OUTROOT"
