# Canonical R1 and Balanced-50 R2 Experiment Design

## 1. Objective

Run an auditable operational experiment using the already completed R1 data:

1. expose exactly the canonical 100 R1 architecture points without moving,
   rewriting, or rerunning any R1 point;
2. generate fixed-bin and CLIP-3D layout, HotSpot, and R2 latency-vector
   artifacts for all 100 canonical architectures;
3. predeclare 10 cache configurations per workload, producing 50 sampled
   architectures;
4. obtain complete fixed-bin and CLIP-3D BIPS2 values for all 50 sampled
   architectures by running fixed-bin R2 once, reusing it only when the full
   CLIP-3D latency vector is identical, and otherwise running a separate
   CLIP-3D R2;
5. report measured improvement without presenting the exploratory parameters
   as formally accepted or paper-equivalent.

The experiment reuses the completed R1 measurements. It does not run transient
thermal simulation and does not invoke the strict-P1 global-bound tool.

## 2. Scientific classification

The selected configuration is:

```text
configs/experiments/
clip3d_constrained_5p0_raw_power_p1_lambda0020119_traffic_weighted_exploratory.json
```

Its relevant values remain unchanged:

```text
lambda_wire       = 0.0020119160767721133
wire_aggregation  = traffic-weighted
alpha             = 1.5643788695171585
beta              = 0.0
cross_tier_weight = 0.995
```

Every launcher, manifest, and report must preserve the configuration's
`operational-exploratory-traffic-weighted`, `non_formal=true`,
`paper_equivalent=false`, and `shared_parameter_accepted=false`
classification. A larger sample demonstrates workflow stability and empirical
trends; it does not promote the parameters or reproduce the paper's reported
improvement by itself.

## 3. Canonical R1 catalogue

The canonical grid is defined by
`configs/experiments/r1_cache_sweep.json`:

```text
5 workloads × 4 L1D sizes × 5 L2 sizes = 100 architecture points
```

Canonical directories have exactly this relative form:

```text
<workload>/l1d_<configured-L1D>/l2_<configured-L2>
```

A shared catalogue utility will construct these 100 expected paths from the
experiment configuration instead of recursively treating every `status.json`
as an architecture. For each canonical path it will verify:

- `status.json` exists and records `state=success`;
- `r1_metadata.json` and `stats.txt` exist;
- metadata workload, L1D, and L2 values match the path;
- the architecture key `(workload, l1d_size, l2_size)` is unique.

Noncanonical status directories are never returned to a lifting sweep. They
remain visible in audit output with their path and exclusion reason. The known
directory

```text
stencil/l1d_32kB/l2_512kB.corrupt_duplicate_20260731T011500_CST
```

is therefore preserved on disk, reported as an excluded noncanonical path,
and never counted or executed.

The existing plan-only R1 command will regenerate a 100-job
`planned_jobs.json`. Before and after that command, SHA-256 manifests of every
canonical point's `status.json`, `r1_metadata.json`, and `stats.txt` will be
compared. Any changed canonical R1 artifact aborts the operation. No gem5 R1
command is launched.

`workflow.analysis.audit_r1` will report both canonical and excluded counts.
`complete=true` requires exactly 100 valid canonical points, a 100-job plan,
and one instruction-window scope. Explicitly reported noncanonical quarantine
directories do not enter those counts.

## 4. Layout-only phase for all 100 architectures

The existing lifting pipeline will run twice without `--run-r2`:

```text
100 fixed-bin points
100 CLIP-3D points
```

Each point must produce and retain at least:

```text
run_config.json
mcpat/mcpat.json
cacti/cacti_characterization.json
modules.json
hotspot/layout.json
hotspot/thermal_result.json
performance.json
r2_latency.json
pipeline_summary.json
```

The CLIP-3D point must additionally retain optimizer and layout-selection
evidence. Layout sweeps use the canonical R1 catalogue, so the duplicate
STENCIL directory cannot create a 101st job.

Both 100-point sweep reports must record the exact experiment-config path and
SHA-256, point count, method, success/failure count, and absence of R2. A
preflight validator must reject mixed configurations, incomplete point sets,
missing communication profiles, or any point whose runtime classification is
not non-formal exploratory.

## 5. Predeclared balanced-50 selection

The sample is independent of measured temperature, IPC, BIPS, and optimizer
outcome. Every workload uses the same checkerboard half-grid:

| L1D | Selected L2 sizes |
|---|---|
| 16 kB | 128, 512, 2048 kB |
| 32 kB | 256, 1024 kB |
| 64 kB | 128, 512, 2048 kB |
| 128 kB | 256, 1024 kB |

This selects 10 of the 20 cache combinations for each of FFT, CHOLESKY,
STREAM, MATMUL, and STENCIL: 50 sampled architectures in total. A tracked JSON
manifest will contain the complete ordered key list, selection rule, expected
counts, selected experiment configuration, and non-formal classification.

The selection validator must prove:

- exactly five configured workloads;
- exactly ten unique cache pairs per workload;
- exactly 50 unique architecture keys;
- every L1D and L2 level occurs in each workload's sample;
- every selected key exists in both complete 100-point layout roots.

## 6. Measured R2 and exact reuse policy

Each sampled architecture is processed as a pair:

1. run or resume fixed-bin gem5 R2 using the fixed-bin `r2_latency.json`;
2. attach its measured IPC2 and calculate fixed-bin BIPS2;
3. compare the fixed-bin and CLIP-3D latency vectors;
4. if the full `gem5_overrides` mappings are identical, attach the fixed-bin
   IPC2 to CLIP-3D through a recorded reuse artifact;
5. otherwise run or resume a separate CLIP-3D gem5 R2 and attach its IPC2;
6. mark the architecture complete only when both summaries contain real or
   exactly validated reused IPC2 and BIPS2.

Reuse requires all of the following:

- identical canonical architecture key and canonical R1 directory;
- identical R1 metadata identity fields and instruction-window scope;
- identical complete `gem5_overrides` mappings, not merely equal total cycles;
- identical experiment configuration SHA-256;
- a successful source R2 status and result;
- source result provenance that resolves to the fixed-bin latency vector.

The reuse artifact records source and target latency paths and SHA-256 values,
the exact overrides, source result and status paths, source IPC2, architecture
key, configuration SHA-256, and validation decision. A reused CLIP-3D summary
points `r2_source` to the fixed-bin result and records that no second gem5 run
occurred.

If `K` of the 50 architectures have different latency vectors, the number of
physical R2 simulations is:

```text
50 + K, where 0 <= K <= 50
```

This is intentionally not forced to exactly 50: doing so would leave
mismatched fixed-bin/CLIP-3D pairs without comparable measured BIPS2.

## 7. Resumption, concurrency, and failure handling

The sampled runner operates on architecture pairs with a configurable positive
worker count. Fixed-bin is processed before CLIP-3D inside each pair; different
pairs may run concurrently. The recommended initial concurrency is four.

Existing successful R2 results are reused only after the same provenance and
latency checks used for a new result. Failed or interrupted points retain their
local `status.json`; rerunning the same command resumes successful points and
retries only incomplete work unless an explicit rerun option is supplied.

The runner writes an atomic experiment-status document after every completed
pair. One pair failure does not delete other completed results. The process
returns nonzero if any selected pair is incomplete or invalid.

## 8. Result aggregation

The final machine-readable JSON and CSV contain one row per sampled
architecture with:

- workload, L1D, and L2;
- fixed-bin and CLIP-3D Tmax and sustainable frequency;
- fixed-bin and CLIP-3D wire cycles and full vector SHA-256;
- fixed-bin and CLIP-3D IPC2 and BIPS2;
- whether CLIP-3D reused fixed-bin R2;
- absolute and percentage BIPS2 difference;
- selected layout policy and any baseline-guard fallback.

Per-workload and aggregate summaries include sample count, arithmetic mean and
geometric-mean BIPS2 ratio, median percentage change, win/tie/loss counts, R2
reuse count, and separate-R2 count. A report is complete only with exactly 10
valid pairs per workload and 50 total pairs.

The report carries the exploratory parameter values and limitations. It must
not use missing IPC2 values, `IPC1 × f_sus` proxies, or mixed configurations.

## 9. Tests and acceptance criteria

Automated tests must cover:

- exact 100-point canonical discovery with a preserved corrupt duplicate;
- failure for missing, unsuccessful, metadata-mismatched, or duplicate
  canonical points;
- audit counts and explicit noncanonical exclusions;
- deterministic balanced-50 manifest validation;
- exact-vector reuse acceptance;
- rejection when any override, R1 identity, config hash, status, or result
  provenance differs;
- resumable fixed-first pair execution;
- 50-row aggregation and non-formal classification;
- rejection of proxy, missing, duplicate, or mixed-config rows.

Before launching the long experiment:

1. the full Python test suite passes;
2. Python compilation passes;
3. the regenerated R1 audit reports 100 canonical successes and one excluded
   corrupt duplicate;
4. canonical R1 artifact hashes are unchanged;
5. a one-pair dry run proves exact reuse and mismatch branching without
   launching gem5;
6. a single real sampled pair completes before raising concurrency.

The experiment is complete only after all 200 layout-only outputs and all 50
sampled architecture pairs pass their acceptance checks.
