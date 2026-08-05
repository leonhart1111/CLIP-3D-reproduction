# Traffic-Weighted Core-to-L2 Wire Aggregation Design

Date: 2026-08-05

## Purpose and status

Add an optional profile-guided core-to-L2 wire aggregation mode to the
CLIP-3D reproduction workflow. The mode uses per-requestor shared-L2 demand
access counts already present in completed gem5 R1 `stats.txt` files to weight
the four physical core-to-L2 delays.

This mode is an explicitly labeled research extension. It does not replace the
paper-faithful arithmetic-mean mode and is not accepted as strict-P1 evidence.

## Goals

- Reuse completed R1 results; do not rerun or modify R1.
- Add `traffic-weighted` as a third `delay.wire_aggregation` choice alongside
  `mean` and `maximum`.
- Use the same aggregation in equation (15)'s optimizer wire term and in the
  scalar latency back-annotated to gem5 R2.
- Preserve raw counters, normalized weights, counter names, and source-file
  provenance in generated JSON.
- Fail explicitly when traffic-weighted mode lacks a complete, positive
  communication profile; never silently fall back to equal weights.
- Preserve all existing results and default behavior.

## Non-goals

- No changes to `configs/gem5/clip_r1.py` or the shared `L2XBar` topology.
- No per-core independent latency injection in gem5 R2.
- No DOE or IPC-sensitivity model.
- No changes to equations (13) or (14), McPAT, CACTI, HotSpot, transient
  simulation, or R1 sweep orchestration.
- No claim that traffic frequency equals causal IPC sensitivity.

## Mathematical definition

For each represented core `i`, read the shared-L2 demand-access count `A_i`
from the final gem5 R1 statistics section. Sum data and instruction requestor
counters when both exist:

```text
system.l2.demandAccesses::cpu<i>.data
system.l2.demandAccesses::cpu<i>.inst
```

Missing requestor subtypes contribute zero, but an available profile requires
every core to have at least one matching counter and the total across cores to
be positive. Normalize:

```text
q_i = A_i / sum_j(A_j)
```

The existing physical path calculation remains unchanged:

```text
L_i = smooth_abs(x_L2 - x_i) + smooth_abs(y_L2 - y_i)
tau_i = 0.69 * R * C * L_i^2 * f0
```

The new continuous aggregate is:

```text
tau_traffic = sum_i(q_i * tau_i)
```

When the configuration selects `traffic-weighted`, equation (15) becomes:

```text
loss = -IPC1 * f_sus(T_proxy)
       + lambda_wire * IPC1 * tau_traffic
       + overlap_penalty
```

For R2, `tau_traffic` is discretized by the configured existing rounding policy
and is inserted into the existing single shared-xbar latency. This is a
workload-weighted scalar approximation, not a per-core path timing model.

## Configuration

Reuse the existing field and add one allowed value:

```json
{
  "delay": {
    "wire_aggregation": "traffic-weighted",
    "wire_rounding": "nearest"
  }
}
```

No second counter-selection option is introduced. Version 1 always uses
shared-L2 demand accesses because every such demand request traverses the
core-to-L2 path, while L1 hits do not.

Existing configurations without `wire_aggregation` continue to default to
`mean`. A separate exploratory configuration will enable `traffic-weighted`.
Configuration validation must reject `traffic-weighted` when
`formal_validation.accepted` is true.

## Data flow and file responsibilities

### `workflow/floorplan/build_module_model.py`

- Parse the existing R1 statistics through `parse_gem5_stats`.
- Extract and validate per-core shared-L2 demand accesses when available.
- Store a `communication_profile` object in `modules.json` containing:
  - source R1 stats path;
  - instruction-window scope;
  - selected counter family;
  - exact matched counter names per core;
  - per-core raw counts;
  - normalized per-core weights;
  - total demand accesses.

The communication profile is produced for every newly lifted point, even when
the selected aggregation remains `mean`, so later extension runs can reuse the
same model and audit the source data. To preserve existing default behavior,
an incomplete profile is recorded as unavailable with explicit missing-counter
diagnostics when the selected mode is `mean` or `maximum`; it becomes a hard
error only when `traffic-weighted` is requested.

### `workflow/floorplan/layout_metrics.py`

- Keep `mean_wire_cycles` behavior unchanged for compatibility.
- Add a small aggregation helper that validates one finite, nonnegative weight
  for every represented core and verifies that weights sum to one within a
  numerical tolerance.
- Extend `derive_layout_delays` with optional communication weights.
- Always preserve current mean/minimum/maximum fields.
- When weights are supplied, additionally emit:
  - `traffic_weighted_wire_cycles_unrounded`;
  - `traffic_weighted_wire_cycles`;
  - per-core `communication_weight` and weighted contribution;
  - aggregation provenance.

### `workflow/floorplan/optimize_layout.py`

- Accept a `wire_aggregation` argument.
- For `mean`, preserve current behavior exactly.
- For `traffic-weighted`, read weights from the module model and use
  `tau_traffic` for every objective evaluation and candidate report.
- Preserve mean and traffic-weighted diagnostics side by side.
- Reject `maximum` as an optimizer objective unless existing behavior already
  explicitly supports it; this feature does not broaden conservative maximum
  mode.

### `workflow/r2/build_latency_vector.py`

- Allow `traffic-weighted` in `wire_aggregation` validation.
- Read the communication profile from `modules.json`.
- Derive both ordinary and traffic-weighted layout metrics.
- Select the rounded traffic-weighted cycle for
  `components_cycles.layout_wire` and the shared `xbar_forward_latency`.
- Record that R2 uses a scalar traffic-weighted approximation.

### `workflow/run_lifting_pipeline.py`

- Allow the new aggregation value.
- Pass the same aggregation choice to the optimizer and latency-vector builder.
- Reject accepted strict-P1 configurations that select the extension.
- Include traffic counts, weights, continuous cycles, rounded cycles, and the
  selected aggregation in `pipeline_summary.json`.

### Configuration and documentation

- Add a separate exploratory experiment configuration derived from the current
  non-formal raw-power/lambda profile.
- Do not change the default or strict formal configuration to enable this mode.
- Document that completed R1 outputs are sufficient and that the current gem5
  implementation still injects one scalar shared-xbar latency.

## Error handling

Traffic-weighted mode fails before layout optimization or R2 when any of these
conditions holds. The same conditions are recorded as an unavailable-profile
diagnostic, without failing, for legacy `mean` and `maximum` runs:

- a represented core has no shared-L2 demand-access counter;
- a counter is negative or non-finite;
- the total demand accesses are zero;
- weight keys do not match represented core IDs;
- weights do not normalize to one;
- an accepted strict-P1 configuration selects the extension.

Generated profiles may contain a zero count for a core only when that core has
an explicitly present zero-valued counter and at least one other core has
positive traffic. Such a core receives zero weight.

## Test strategy

### Unit and regression tests

- Four equal counts produce exactly the arithmetic mean.
- A synthetic dominant-core profile shifts the weighted aggregate toward that
  core's path delay.
- Raw counts normalize deterministically and preserve counter provenance.
- Data and instruction requestor counts are summed when both exist.
- Missing-core, negative, non-finite, and all-zero profiles fail explicitly.
- Existing `mean` and `maximum` outputs remain byte-for-byte compatible where
  practical and numerically identical otherwise.
- Optimizer and R2 select the same traffic-weighted aggregate.
- Accepted strict-P1 configuration validation rejects the extension.

### Existing real-R1 checks

- MATMUL `32kB/512kB`: near-equal traffic must reproduce mean aggregation and
  serve as a no-regression check.
- CHOLESKY `32kB/512kB`: the measured CPU0 share is about 40.66%; the run must
  preserve those observed counts and demonstrate a nonuniform weighted metric.
- Run the CHOLESKY lifting pipeline through real HotSpot without R2 first. This
  should finish in minutes and must not touch the active formal R1 or transient
  output directories.
- Run gem5 R2 only after the transient gem5 process finishes or when resource
  contention is otherwise acceptable. Reuse an R2 result only if the complete
  generated latency vector matches.

## Acceptance criteria

- Existing completed R1 directories are consumed unchanged.
- All repository tests pass.
- Default `mean` behavior has no regression.
- The exploratory configuration produces auditable traffic weights and uses
  one consistent weighted aggregate in the optimizer and R2 vector.
- MATMUL equal-traffic regression passes.
- CHOLESKY real-R1, real-HotSpot pilot completes and reports its nonuniform
  profile.
- No files beneath active `runs/` directories are modified by implementation
  or test setup except a new, uniquely named pilot output directory.
- Documentation labels the mode non-formal and records the shared-xbar scalar
  limitation.
