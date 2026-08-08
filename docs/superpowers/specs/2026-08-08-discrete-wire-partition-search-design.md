# Discrete Wire-Cycle Partition Search Design

Date: 2026-08-08

## Purpose and status

Add a small, optional layout-search mode that evaluates the core-to-L2 wire
penalty with exactly the same integer-cycle mapping later consumed by gem5 R2.
This closes the observed mismatch where the continuous optimizer accepted a
small thermal-proxy gain while crossing an R2 rounding boundary and increasing
the measured path by one full cycle.

The existing continuous L-BFGS-B mode and all current experiment outputs remain
unchanged. The new mode is an exploratory correction, not a retroactive change
to the running Balanced-50 experiment or a claim of paper equivalence.

## Goals

- Reuse existing R1, McPAT, CACTI, module, and communication-profile inputs.
- Use one identical `round_wire_cycles` policy in layout selection and R2.
- Search legal L2 positions by integer wire-cycle partition.
- Always include the fixed-bin layout as a candidate so a predicted regression
  is not selected merely because the fixed point was absent from the search.
- Keep tool calls unchanged: no HotSpot or gem5 invocation inside optimization.
- Make the smallest practical code and configuration change.
- Emit enough diagnostics to audit every integer-cycle partition and selection.

## Non-goals

- Do not modify or rerun R1.
- Do not change the active Balanced-50 configuration or its output directories.
- Do not replace the current non-formal `lambda_wire` calibration in this task.
- Do not solve the separate non-monotonic gem5 IPC-sensitivity problem.
- Do not change the current `nearest` rounding assumption in this task. A
  future `ceil` engineering sensitivity study remains separate.
- Do not expand the movable set beyond the single shared L2.
- Do not add repeated HotSpot or R2 validation calls during layout search.

## Mathematical definition

For a legal L2 position `p = (x, y, tier)`, retain the existing continuous
communication-weighted wire delay in cycles:

```text
d(p) = sum_i q_i * 0.69 * R * C * L_i(p)^2 * f0
```

Map it through the same configured function used by R2:

```text
k(p) = round_wire_cycles(d(p), wire_rounding)
```

The new primary score is the current equation-(15) approximation evaluated at
the integer cycle count:

```text
score(p) = -IPC1 * f_sus(T_proxy(p))
           + lambda_wire * IPC1 * k(p)
```

Continuous delay is not added with an arbitrary epsilon. It is used only as a
deterministic lexicographic tie-breaker after equal primary scores, followed by
coordinates and tier. This ensures a small continuous difference can never
override a one-cycle discrete penalty.

The legal placement space is partitioned by integer cycle:

```text
Omega_k = {p | k(p) = k}
```

The search records the best legal sampled representative of every non-empty
`Omega_k`, then chooses the globally lowest discrete score across those
representatives and the fixed-bin baseline.

## Search algorithm

Version 1 uses a deterministic bounded grid because only one rectangular L2
is movable. This is simpler and more reliable for a discontinuous objective
than applying L-BFGS-B directly to a step function.

1. Build the existing fixed-bin baseline and add it as an explicit candidate.
2. For every allowed L2 tier, sample an `N x N` grid over the legal lower-left
   coordinate bounds, including all four boundaries. Default `N` is 41.
3. Reject candidates with overlap or invalid geometry.
4. For every legal candidate, compute thermal proxy, sustainable frequency,
   continuous aggregated wire cycles, integer wire cycles, and discrete score.
5. Group candidates by integer wire-cycle value and retain the lowest-score
   representative per partition. Resolve exact score ties by continuous wire
   delay, tier, `y`, then `x`.
6. Choose the lowest-score representative across all partitions and the
   fixed-bin candidate using the same ordering.
7. Emit the selected physical layout. The existing pipeline performs its one
   final HotSpot validation and creates the R2 latency vector as before.

The grid resolution is configuration-controlled for reproducibility. Values
below 3 are rejected. The implementation does not add local continuous
refinement in version 1; that would complicate boundary handling without being
necessary for the current single-L2 feasibility fix.

## Configuration

Extend `layout_optimizer.wire_objective` with one value and one optional grid
setting:

```json
{
  "layout_optimizer": {
    "wire_objective": "discrete-partition",
    "partition_grid_steps": 41,
    "include_fixed_baseline": true
  },
  "delay": {
    "wire_rounding": "nearest",
    "wire_aggregation": "traffic-weighted"
  }
}
```

`include_fixed_baseline` must be true in `discrete-partition` mode. Existing
`continuous` and `r2-quantized` behavior remains unchanged for compatibility.
A new exploratory configuration enables the new mode; no existing experiment
configuration is edited.

## File responsibilities

### `workflow/floorplan/optimize_layout.py`

- Keep the existing continuous solver path unchanged.
- Add a small candidate-evaluation helper shared by reporting and selection.
- Add deterministic grid generation and integer-partition grouping.
- Include the fixed-bin candidate in discrete-partition mode.
- Use `round_wire_cycles` directly for the primary wire penalty.
- Emit selected origin (`fixed-bin` or `partition-grid`), partition summaries,
  legal/rejected counts, grid size, score components, and deterministic
  tie-break information in `optimizer_report.json`.

### `workflow/run_lifting_pipeline.py`

- Validate and pass `partition_grid_steps` and `include_fixed_baseline` only
  when the new mode is selected.
- Preserve every existing mode's defaults and output path behavior.

### `configs/experiments/`

- Add one non-formal exploratory configuration derived from the current
  traffic-weighted, non-zero-lambda configuration.
- Change only the new mode fields and experiment label/provenance.

### `tests/test_workflow.py` or the existing floorplan-focused test module

- Add focused unit and integration regressions without invoking real gem5,
  McPAT, CACTI, or HotSpot.

## Output contract

`optimizer_report.json` adds a discrete-search object containing:

- search mode and grid resolution;
- total, legal, and rejected candidate counts;
- fixed-bin candidate score and integer cycle;
- one best representative per observed integer-cycle partition;
- selected candidate origin, continuous cycle, integer cycle, proxy
  temperature, proxy frequency, and primary score;
- the configured aggregation and rounding policy.

Existing top-level selected fields remain populated so downstream consumers do
not need to change. `r2_latency.json` remains the authority for the final
integer path and must agree with the selected report value.

## Error handling

Fail before emitting an optimized layout when:

- `partition_grid_steps` is not an odd integer of at least 3;
- discrete mode disables the fixed-bin candidate;
- the configured aggregation lacks required communication weights;
- no legal candidate exists;
- candidate values are non-finite;
- the selected integer wire cycle differs when recomputed through the standard
  `derive_layout_delays` and `select_rounded_wire_cycles` path.

Odd grid sizes are required so the center is always sampled. Existing geometry
and communication-profile validation remains authoritative.

## Test strategy

1. **Rounding-boundary regression:** synthetic fixed delay `1.358` and movable
   delay `1.558` must map to one and two cycles under `nearest`; the discrete
   score must retain fixed-bin when the thermal gain is too small.
2. **Partition grouping:** multiple positions in one integer partition retain
   exactly one deterministic best representative.
3. **Baseline win:** fixed-bin can be selected and the emitted layout matches
   the baseline geometry.
4. **Optimized win:** a candidate with equal or lower integer latency and a
   better proxy score is selected.
5. **R2 identity:** the selected integer cycle equals the cycle generated by
   the normal latency-vector path.
6. **Determinism:** repeated searches over identical inputs produce identical
   selected layout and partition report.
7. **Validation:** even grid sizes, sizes below 3, disabled baseline, malformed
   values, and missing traffic weights fail clearly.
8. **Compatibility:** existing continuous-mode regression tests remain
   unchanged and pass.
9. **Real-point smoke without R2:** rerun only the lifting portion of
   Cholesky `128kB/1024kB` into a new output directory. Verify that the harmful
   `1 -> 2` wire-cycle selection is not chosen. Do not touch its existing R1 or
   Balanced-50 outputs.

## Acceptance criteria

- No existing R1, running experiment, or result file is modified.
- Existing continuous mode remains behaviorally compatible.
- The new mode evaluates the wire term with the exact same integer mapping as
  R2.
- Fixed-bin is always a candidate and can win.
- The known Cholesky `128kB/1024kB` boundary regression is prevented in a
  lifting-only smoke test.
- Optimization adds no HotSpot or gem5 calls; only the existing final HotSpot
  validation remains.
- Relevant unit, integration, and existing workflow tests pass.
- The new configuration remains explicitly non-formal and exploratory.

## Known residual limitation

This change fixes the continuous-to-integer selection inconsistency only. It
does not guarantee positive measured BIPS2 because the current shared
`lambda_wire` fit is non-formal and gem5 IPC is not strictly monotonic in the
tested latency sweep. Those issues require a separate per-workload IPC
sensitivity design.
