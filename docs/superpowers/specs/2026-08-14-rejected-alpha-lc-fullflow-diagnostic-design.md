# Rejected Alpha/Lc Full-Flow Diagnostic Design

## Objective

Run one non-formal MATMUL L1D 32 kB/L2 512 kB fixed-bin versus CLIP-3D
comparison using the completed unscaled parameter-identification contract.  The
experiment reuses canonical R1 and runs fresh downstream CACTI, McPAT, HotSpot,
and gem5 R2 work.  It must never present the rejected alpha/Lc fit as a formal
parameter promotion.

## Fair-comparison contract

Both layouts use one immutable configuration with grid 64, module-granularity
HotSpot input, compact traces, nine-decimal power traces, utilization 0.7,
ambient 25 C, `Rconv=1.042 K/W`, and all thermal resistance scales equal to
one.  Both use the same unscaled local CACTI/McPAT model and canonical R1.

The diagnostic CLIP-3D objective uses:

- `alpha=3.348558894986864`;
- `lc_die_side_ratio=0.3525311644629249`;
- `cross_tier_weight=0.8378797681280803`;
- `beta=0`, explicitly because beta was not identified under strict P1;
- the existing rejected exploratory `lambda_wire=0.0020119160767721133`;
- L2 restricted to tier 1;
- area-quadrature order 2 and discrete-partition wire search.

The fixed-bin branch is the control and does not use alpha/Lc to choose a
position.  Final temperature always comes from HotSpot, and BIPS2 always uses
fresh R2 results.

## Required plumbing

`workflow.floorplan.optimize_layout.optimize` accepts an optional positive
finite `lc_die_side_ratio`.  It converts the ratio to millimetres using the
current die side, forwards the value to every thermal-proxy evaluation, and
records both ratio and effective millimetres in `optimizer_report.json`.

`workflow.run_lifting_pipeline` validates and forwards the ratio from
`layout_optimizer`, and forwards the configured HotSpot input granularity,
compact-trace switch, and power-trace precision to all materialization paths.
Defaults remain backward compatible for historical configurations.

## Evidence and labeling

A new exploratory config records the rejected fit report path, SHA-256,
`accepted=false`, fitted fields, and the rejected lambda manifest.  Output
directories include `rejected_fit_diagnostic` and never overwrite historical
runs.  The comparison reports fixed/CLIP Tmax, sustainable frequency, rounded
R2 critical-path cycles, IPC2, BIPS2, and percentage changes.

## Verification

Tests first demonstrate that the optimizer previously ignored an explicit Lc
ratio and that the pipeline previously failed to forward module-level HotSpot
controls.  Focused workflow and alpha/Lc tests must pass before commands are
handed off.  No R1 file may be modified or regenerated.
