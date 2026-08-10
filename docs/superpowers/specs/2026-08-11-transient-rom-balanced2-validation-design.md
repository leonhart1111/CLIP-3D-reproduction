# Transient ROM Balanced-2 Validation Design

## Goal

Provide one auditable, resumable entry point that compares the transient-ROM
extension with the completed steady baseline on two representative points:

- MATMUL, L1D 64 kB, L2 512 kB;
- STENCIL, L1D 64 kB, L2 512 kB.

MATMUL is a positive steady-result point and STENCIL is the negative steady
counterexample.  Together they test whether the transient model merely follows
the steady ranking or changes the physical conclusion.

## Scientific Scope

This remains an exploratory validation.  It uses the existing nonzero
`lambda_wire`, traffic-weighted delay, nearest integer-cycle partition search,
and transient-ROM configuration.  It is not paper-equivalent and must never be
called a formal CLIP-3D result.

The existing steady result at
`runs/discrete_partition_validation/balanced5_midcache_20260809/summary.csv`
is the comparison baseline.  Only its successful MATMUL and STENCIL rows may be
used.  The runner must not silently accept the provisional FFT or Cholesky rows.

## Considered Approaches

1. Hand-written commands only.  This changes no code but makes it easy to mix
   periodic R1, package, output, or baseline identities and provides no
   deterministic aggregate report.
2. A thin two-point orchestrator (selected).  It delegates all scientific work
   to the existing public transient-ROM pipeline and adds only point selection,
   phase control, status, and strict aggregation.
3. A generic arbitrary-size experiment scheduler.  This would duplicate the
   existing Balanced-50 scheduler machinery and is unnecessary for a two-point
   validation.

## Inputs

A tracked selection file records exactly the two workload/cache keys and their
order.  Runtime roots are supplied separately:

- canonical R1 root;
- 2 ms periodic-statistics R1 root;
- steady baseline CSV;
- transient-ROM scientific config;
- new output root.

For each point the runner derives, rather than accepts independently, the
canonical and periodic directories.  It validates workload, L1D, L2, source-R1
identity, successful status, and exactly 2 ms sampling before launching thermal
work.

## Two-Phase Flow

### Thermal phase

For each point, invoke the existing public pipeline with:

```text
--thermal-mode transient-rom
--transient-rom-r1-dir <matching periodic R1>
--transient-rom-calibrate
```

Do not request R2.  Each point creates an independent 12-call package, searches
the L2 layout without HotSpot inside the optimizer, and performs real paired
HotSpot frequency/PSS validation for fixed-bin and CLIP-3D.  A point is
`thermal_validated` only when both branch summaries report finite validated
frequencies and no failure.

### R2 phase

R2 is explicitly enabled by `--run-r2` and is permitted only after both points
have valid thermal evidence.  For each point, use a fresh R2 output directory
and the accepted package from the thermal phase.  This preserves the immutable
thermal checkpoint and follows the current pipeline contract, which does not
retrofit R2 into an already completed final-validation directory.

A point is `r2_validated` only when the existing pipeline publishes a real
paired comparison containing finite fixed-bin and CLIP-3D
`measured_bips2_trans` values.

## Resume and Failure Semantics

The runner writes one status per point plus a root status.  It validates a
completed checkpoint before skipping it; a state label or file existence alone
is insufficient.  It never deletes or overwrites a failed output.  A failed or
incompatible point must be rerun into a new output root.

Thermal failure blocks the entire R2 phase.  R2 failure preserves both thermal
checkpoints and the successful peer point.  No aggregate transient improvement
is published unless all evidence required by the selected phase is valid.

## Aggregate Report

The root report contains one row per point and separates evidence classes:

- steady fixed/CLIP temperature, frequency, IPC2, BIPS2, and percentage change;
- transient fixed/CLIP validated frequency;
- transient fixed/CLIP IPC2 and BIPS2 only after R2;
- transient percentage change only after R2;
- steady-versus-transient change in the CLIP-3D improvement;
- package and final HotSpot call counts;
- exact artifact paths and hashes.

The JSON report records `non_formal=true`, `paper_equivalent=false`, the exact
selection/config/baseline hashes, point counts, win/tie/loss counts, arithmetic
mean improvement, and completion state.  CSV is a deterministic presentation
of the same validated rows.  Thermal-only reports leave measured fields empty;
they do not substitute ROM predictions for R2 measurements.

## Files

- `configs/experiments/transient_rom_balanced2_selection.json`: exact two-point
  selection and exploratory classification.
- `workflow/experiments/transient_rom_balanced2.py`: preflight, phase
  orchestration, checkpoint validation, and aggregation.
- `tests/test_transient_rom_balanced2.py`: selection, preflight, phase gating,
  resume, and report regression tests.
- `docs/transient_rom_usage_zh.md`: direct commands and output interpretation.

## Acceptance

Automated tests must prove that wrong sampling/source/cache provenance is
rejected before pipeline execution; R2 never starts after any thermal failure;
predictions never populate measured fields; and complete synthetic paired
evidence produces deterministic JSON/CSV.

The first real smoke uses the already available MATMUL 32 kB/512 kB periodic R1
to validate the steady-initialization repair itself.  The Balanced-2 experiment
starts only after the new MATMUL and STENCIL 64 kB/512 kB periodic R1 directories
both report success.
