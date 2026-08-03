# Exploratory Lambda-Wire MATMUL Single-Point Design

## Objective

Test whether the CLIP-3D optimizer performs the intended joint thermal--wire
trade-off when given the locally measured, but formally rejected,
`lambda_wire = 0.0020119160767721133`.  The test reuses the completed canonical
MATMUL R1 result and executes one fresh CLIP-3D lifting/R2 point.  It must not
alter the canonical R1, the current operational configuration, or existing
pilot results.

This is an optimizer-feasibility experiment, not promotion of a formal shared
parameter and not a paper-equivalent result.

## Existing evidence and interpretation

The current operational configuration sets `lambda_wire` to zero.  In
`workflow/floorplan/optimize_layout.py`, the layout loss contains the term

```text
lambda_wire * IPC1 * wire_objective_cycles
```

so zero removes wire length from candidate scoring.  The optimizer still
optimizes the thermal proxy, and the selected layout's wire delay is still
computed, rounded into the R2 latency vector, and evaluated by gem5.  The
zero-lambda experiments therefore validate the thermal path and R2 plumbing,
but they do not validate joint thermal--wire optimization.

The candidate value comes from the FFT matched-R2 study at
`results/parameter_studies/raw_power_strict_20260730/r2_wire/fft/lambda_wire_report.json`.
That study measured
`lambda_wire = 0.0020119160767721133`, but its fit had
`R^2 = 0.7691549845761521`, one monotonicity violation, and no cross-workload
transfer validation.  The value is therefore suitable only for this explicitly
labelled exploratory run.

## Chosen approach

Create a separate configuration by copying
`configs/experiments/clip3d_constrained_5p0_raw_power_p1_operational.json` and
changing only fields needed to identify the experiment and its wire parameter:

- use the name `constrained_5p0_raw_power_p1_lambda0020119_exploratory`;
- set `layout_optimizer.lambda_wire` to
  `0.0020119160767721133`;
- retain `layout_optimizer.wire_objective = "continuous"`;
- state that the value is FFT-local, measured by matched gem5 R2, rejected for
  formal/shared promotion, and used only for optimizer feasibility;
- classify the configuration as operational, non-formal, and not
  paper-equivalent.

The source operational configuration will remain unchanged so the previous
zero-lambda result stays reproducible.

## Inputs, outputs, and data flow

The read-only R1 input is:

```text
runs/architecture_sweep/r1/paper/matmul/l1d_32kB/l2_512kB
```

The fresh experiment output is:

```text
runs/operational_raw_power_p1/lambda0020119_matmul_32kB_512kB_20260803/clip3d
```

The full pipeline will execute the existing sequence:

1. validate and read the completed MATMUL R1 metadata and gem5 statistics;
2. characterize caches with local CACTI data;
3. generate and run McPAT from the R1 counters without power multipliers;
4. construct the physical module/power model;
5. optimize L2 placement with the nonzero continuous wire penalty;
6. validate the selected layout with steady-state HotSpot;
7. compute the sustainable frequency;
8. calculate and integerize cache/TSV/wire latency for R2;
9. run a fresh gem5 R2 and calculate IPC2 and BIPS2.

The full R2 run will be performed even if the resulting integer latency vector
matches an existing result, because the user requested an end-to-end one-point
execution.  Identical vectors will be reported as evidence that cycle
quantization masked the continuous placement difference, not as proof that the
wire objective failed.

## Controls and comparison

The experiment will compare the new output with both existing controls under
`runs/operational_raw_power_p1/pilot_direct_20260731/`:

- `fixed-bin`;
- `clip3d`, whose recorded `lambda_wire` must equal zero.

Before comparison, the workflow inputs will be checked for the same workload,
L1D size, L2 size, R1 source, frequency/cooling assumptions, and relevant
non-wire optimizer parameters.  Any incompatible old control will be reported
instead of silently treated as matched evidence.

The comparison will include:

- selected L2 block coordinates and tier;
- Manhattan wire length and continuous wire-objective cycles;
- rounded wire cycles and the complete R2 latency vector;
- HotSpot maximum temperature and sustainable frequency;
- IPC1, IPC2, thermal BIPS1, and BIPS2;
- the optimizer objective terms and whether the nonzero wire term changed
  candidate ranking or the selected layout.

## Safety and failure handling

- Do not rerun, modify, delete, or overwrite canonical R1 artifacts.
- Do not overwrite either existing pilot or the operational configuration.
- Fail before the expensive R2 stage if required input/provenance files are
  absent or the exploratory configuration is not classified as non-formal.
- Preserve partial output and the relevant log when CACTI, McPAT, HotSpot, or
  gem5 fails so the failure can be diagnosed.
- Do not infer parameter validity from a favorable BIPS result.  The original
  FFT fit rejection remains unchanged.

## Verification and success criteria

Implementation verification will establish that:

1. the original operational configuration still contains `lambda_wire = 0`;
2. the new configuration contains the exact measured candidate and explicit
   rejected/non-formal provenance;
3. normal pipeline configuration validation accepts the new profile while no
   formal promotion path accepts it;
4. before/after hashes of the canonical R1 metadata and statistics remain
   unchanged;
5. the new full pipeline and fresh R2 finish successfully;
6. the generated `run_config.json`, optimizer report, latency JSON, HotSpot
   result, and R2 result all refer to the new run and are internally consistent;
7. a comparison report distinguishes continuous wire-objective effects from
   integer-cycle and statistical gem5 effects.

Success does not require a positive BIPS improvement.  The feasibility test is
successful if the exact nonzero coefficient is consumed by the optimizer, the
wire term is auditable in candidate scoring, and the complete physical/R2
consequences are measured without contaminating formal evidence.
