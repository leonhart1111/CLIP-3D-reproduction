# McPAT-Native Short-ROI Thermal-Correction Design

## 1. Purpose

This change corrects five methodological problems in the CLIP-3D reproduction
workflow:

1. standalone CACTI must not be called because McPAT already embeds CACTI-P;
2. the four cores must remain decomposed into their fixed functional blocks so
   that the thermal model does not collapse them into four uniform heat sources;
3. HotSpot must map partially covered blocks by overlap area, model unused
   silicon, and run without SuperLU or vendor math acceleration;
4. gem5 must use a short, representative region instead of an assumed
   500-million-instruction measurement, with transient sampling derived from
   the resulting simulated duration; and
5. the analytic thermal proxy is a search heuristic, not a reportable
   temperature model or a parameter-identification acceptance gate.

The existing completed 500-million-instruction R1 and R2 artifacts are
historical evidence. They remain read-only and are never silently reclassified,
overwritten, or reused as short-protocol results.

## 2. Selected approach and rejected alternatives

The selected approach makes McPAT the single cache-model authority, exports the
cache metrics produced by McPAT's embedded CACTI-P, keeps fixed fine-grained
core blocks, lets HotSpot perform area-average block-to-grid mapping, introduces
a convergence-selected short gem5 protocol, and validates every selected
layout with real HotSpot and gem5 R2.

Two alternatives are rejected:

- Using McPAT XML latency constraints would remove standalone CACTI with a small
  change, but the resulting cache delay would be an input rather than a measured
  result from McPAT's internal cache model.
- Replacing the proxy with repeated HotSpot calls or requiring a newly accepted
  global parameter fit would increase cost and contradict the stated role of
  the proxy as a gradient heuristic.

## 3. McPAT as the cache-model authority

### 3.1 Machine-readable internal CACTI-P metrics

The repository will carry an auditable McPAT patch, applied to a clean upstream
McPAT source tree. At McPAT print level 5 it will emit one unambiguous record for
each core's L1I and L1D and for the shared L2. Each record contains:

- cache identity and core index where applicable;
- internal CACTI-P access time in seconds;
- internal CACTI-P cycle time in seconds;
- internal array height and width in millimetres; and
- the McPAT/CACTI-P version or build provenance needed to identify the model.

The patch exposes values that already exist in each cache object's
`ArrayST::local_result`; it does not run a second cache model and does not alter
McPAT's optimization objective. The parser rejects missing, duplicate,
non-finite, non-positive, or mismatched records.

### 3.2 Area, power, and geometry

Cache area and power are taken directly from the corresponding aggregate McPAT
blocks:

- area: McPAT `Area`;
- dynamic power: McPAT `Runtime Dynamic`;
- leakage: `Subthreshold Leakage + Gate Leakage`; and
- total power: dynamic plus both leakage components.

McPAT's printed cache block includes the main array and associated buffers, but
the embedded CACTI-P height and width describe the main array. Therefore the
raw CACTI-P dimensions are not used as aggregate block area. Their aspect ratio
is retained and scaled to the aggregate McPAT area:

\[
r = \frac{w_{\mathrm{array}}}{h_{\mathrm{array}}},\qquad
w_{\mathrm{block}}=\sqrt{A_{\mathrm{McPAT}}r},\qquad
h_{\mathrm{block}}=\frac{A_{\mathrm{McPAT}}}{w_{\mathrm{block}}}.
\]

This preserves both the authoritative aggregate area and a shape derived from
the same embedded cache model. The artifact records raw dimensions, normalized
block dimensions, and the transformation.

### 3.3 R2 latency

At nominal frequency \(f_0\), each cache access latency is converted to the
integer domain accepted by gem5:

\[
L_{\mathrm{cache}}=
\max\left(1,\left\lceil t_{\mathrm{access}}f_0\right\rceil\right),
\]

where seconds and hertz are used internally. The unrounded value, rounding
operation, integer result, source record, source-output hash, and McPAT binary
hash are retained in the latency artifact. L1 tag and data latency use the
matching L1 embedded result; L2 uses the shared-L2 embedded result. Existing
arbitration, TSV, pipeline, and layout-wire terms remain separate and retain
their current provenance.

### 3.4 Removal of standalone CACTI

The formal and exploratory execution paths will no longer:

- require `tools/src/cacti/cacti`;
- invoke `workflow.cacti.characterize_cache`;
- generate or require `cacti_characterization.json`;
- override McPAT cache area or geometry; or
- bind steady, sweep, R2, wire-sensitivity, or transient-ROM artifacts to an
  external CACTI path or hash.

Historical CACTI artifacts and diagnostic source files are not destructively
deleted. They are excluded from the new execution contract and clearly marked
as legacy evidence where documentation references them.

## 4. Fixed fine-grained physical module contract

Each of the four cores is represented by the following non-overlapping McPAT
functional blocks:

- instruction-fetch logic excluding L1I;
- rename logic;
- load/store logic excluding L1D;
- memory-management logic;
- execution logic;
- residual/other core logic when non-zero;
- L1I; and
- L1D.

The shared L2 and reported interconnect/NoC are additional modules. A normal
four-core point therefore contains approximately 34 physical modules; the
exact count may be smaller only when a mathematically zero residual block is
omitted.

Formal parsing requires four cores and the detailed functional headings. The
old aggregate `coreN_logic` fallback remains available only to explicitly
labelled parser fixtures or legacy diagnostics and is rejected by the pipeline.
For every core, subtraction of child cache blocks and residual construction
must conserve area, dynamic power, both leakage components, and total power
within recorded numerical tolerance.

Core functional-block coordinates and tiers are fixed after deterministic
baseline placement. They are not optimization variables. The shared L2 remains
the only movable module in this correction. The module-model artifact records
`movable=false` for all core blocks and `movable=true` only for L2.

## 5. Thermal-proxy semantics

The proxy consumes every fixed functional block plus the candidate L2. It must
never construct four aggregate core heat sources. The spatial coupling feature
is evaluated over each module's area-quadrature samples:

\[
H(\mathcal L)=
\max_{q\in Q_i}
\sum_j\sum_{p\in Q_j}
\frac{P_j}{|Q_j|}
w(z_i,z_j)
\frac{1}{\sqrt{1+(\lVert q-p\rVert/L_c)^2}}.
\]

The temperature-shaped heuristic remains

\[
\widehat T_{\max}=T_{\mathrm{amb}}+R_{\mathrm{conv}}P_{\mathrm{tot}}
+\alpha H(\mathcal L)+\beta P_{\mathrm{bottom}}.
\]

Its declared role is `search-heuristic`. Normal execution requires only:

- finite parameters and finite candidate scores;
- a positive characteristic length;
- legal candidate geometry; and
- non-degenerate spatial response when L2 has legal movement.

Normal execution does not require RMSE, Spearman, external-workload,
leave-one-workload-out, or promotion acceptance. The existing strict fitting
and promotion tools remain optional diagnostics and cannot relabel heuristic
output as measured data. Optimizer reports include the module count, evaluated
pair count or quadrature count, fixed/movable module sets, spatial response
range, parameter provenance, and the warning that proxy temperature is not
reportable.

Every selected layout is evaluated by real HotSpot. Reported temperature and
sustainable frequency always come from that validation. Reported BIPS2 always
uses measured gem5 R2 IPC and the HotSpot-validated frequency.

## 6. HotSpot overlap and solver contract

### 6.1 Geometry meanings

Three different forms of overlap are handled explicitly:

- Same-tier physical module overlap is illegal and is rejected before HotSpot.
- A module partially covering one or more HotSpot cells is legal and is mapped
  by area average.
- Modules on different tiers may have the same projected \(x,y\) region; this
  is legal and is required for vertical thermal coupling.

The active layers use native module rectangles. Unoccupied regions are tiled by
zero-power whitespace rectangles so that each silicon tier covers the complete
die without gaps or same-tier overlaps. HotSpot runs in grid, detailed-3D mode
with `grid_map_mode=avg`. Its occupancy mapping therefore implements

\[
P_c=\sum_i P_i\frac{A(b_i\cap c)}{A_i}.
\]

The generated floorplans use sufficient numeric precision to preserve the
validated geometry after serialization. Manifests record per-tier covered area,
whitespace area, same-tier overlap, coverage residual, cross-tier projected
overlap as an informational metric, and dynamic/leakage/total power residuals.

HotSpot output containing functional-block-overlap, invalid mapping, erroneous
b2gmap, non-finite temperature, or missing active-layer grid values is treated
as a failed run even if the process exits with status zero.

### 6.2 Disabling LU and vendor acceleration

HotSpot is rebuilt from a clean object tree with:

```text
SUPERLU=0 MATHACCEL=none
```

A build/provenance check verifies the requested flags, binary hash, and absence
of SuperLU symbols or dynamic linkage. The pipeline refuses a binary that fails
this guard. The existing power-of-two grid requirement of the non-SuperLU
detailed-3D solver is validated in configuration.

## 7. Short gem5 protocol and convergence selection

### 7.1 Isolation from historical results

The current 100M-warmup/500M-measurement profile and completed artifacts are
retained under their original protocol identity and labelled legacy. A new
short-protocol identity uses distinct output directories, metadata, hashes, and
cache keys. Neither protocol can reuse the other's R1 or R2 result.

### 7.2 Candidate experiment

The convergence experiment uses a fixed two-million-instruction warmup and
four all-core measurement targets:

- 1 million instructions;
- 2 million instructions;
- 5 million instructions; and
- 10 million instructions.

All four workload threads must satisfy the selected target. Each result records
the exact command, instruction scope, per-core instructions, simulated ROI
duration, IPC, McPAT module powers, and per-core shared-L2 traffic.

The ten-million-instruction point is the local reference. The shortest
candidate is selected only if that candidate and every larger candidate,
relative to the reference, satisfy the configured convergence limits for:

- aggregate IPC;
- total dynamic and leakage power;
- normalized per-module power distribution; and
- per-core normalized shared-L2 traffic.

Initial limits are 1% for IPC and 3% for each power/traffic metric. They are
configuration values and are always printed in the convergence report. If no
shorter point passes, 10 million is selected rather than claiming convergence.
The initial operational candidate remains 2M warmup plus 5M measurement, but
it is not promoted until its workload-specific convergence report passes.

### 7.3 Transient sampling interval

The short steady R1 provides the measured simulated ROI duration. Unless the
user explicitly supplies a recorded override, transient sampling uses

\[
\Delta t=\min(0.5\ \mathrm{ms},T_{\mathrm{ROI}}/50).
\]

This targets approximately 50 complete windows and guarantees a default below
2 ms. The final partial window retains its actual duration in the statistics
artifact and is padded only where the HotSpot trace interface requires a fixed
sampling interval. The requested interval, tick-rounded interval, actual
window durations, target count, actual count, and padding are recorded.

Generic 10 ms defaults, ROM-specific 2 ms defaults, and tests requiring exactly
2 ms are replaced by this single configuration-driven contract. Explicit
overrides remain supported but produce distinct provenance and cache identity.

## 8. Data flow

For one architecture point the corrected steady flow is:

```text
short gem5 R1
  -> stats and metadata
  -> McPAT XML conversion
  -> one patched McPAT run
  -> granular module area/power + embedded cache timing/shape
  -> fixed-module/L2-movable physical model
  -> proxy-guided L2 search over all module interactions
  -> one final HotSpot validation
  -> McPAT-native integer R2 latency vector
  -> short gem5 R2
  -> measured IPC2 x HotSpot-validated frequency = BIPS2
```

The optional transient flow reuses the selected short protocol, derives its
sampling interval from the R1 ROI, emits windowed McPAT power for the same
granular module contract, and runs HotSpot transient validation. It does not
reintroduce standalone CACTI.

## 9. Failure handling and compatibility

The corrected pipeline fails before expensive tools when it encounters:

- an unpatched McPAT binary or missing internal cache metrics;
- a core count other than four for this reproduction profile;
- aggregate-core fallback in a formal run;
- non-conserving module decomposition;
- illegal same-tier geometry or incomplete whitespace coverage;
- a HotSpot binary linked to SuperLU or built with unsupported acceleration;
- a non-power-of-two detailed-3D grid in the non-SuperLU path;
- cross-protocol R1/R2 artifact reuse; or
- a report attempting to present proxy temperature as measured temperature.

Partial output and tool logs are retained in the point directory. Historical
500M artifacts remain readable by legacy analysis but are never silently
upgraded to the new schema.

## 10. Verification and acceptance

Implementation is accepted only after the following evidence exists:

1. Unit tests parse valid McPAT internal-cache records and reject missing,
   duplicate, invalid, or mismatched records.
2. Cache area/power and normalized geometry come only from McPAT artifacts;
   standalone CACTI is neither checked nor executed by the new pipeline.
3. A real McPAT fixture produces four granular cores, no aggregate-core
   fallback, and conserving area/power totals.
4. The thermal proxy report proves that all fixed functional blocks and L2 are
   included in spatial coupling while only L2 is movable.
5. Serialized HotSpot floorplans cover each tier exactly, contain no same-tier
   overlap, conserve all power fields, and allow cross-tier projected overlap.
6. A real HotSpot smoke test with non-grid-aligned modules completes without
   overlap/mapping warnings and returns finite active-layer grid temperatures.
7. Tool checks prove `SUPERLU=0`, `MATHACCEL=none`, and no SuperLU linkage.
8. Short-profile tests prove protocol isolation, all-core stopping, convergence
   reporting, and selection fallback to 10M when shorter points do not pass.
9. Transient tests accept a derived sub-2-ms interval, reject stale hard-coded
   interval identities, preserve the final partial duration, and record padding.
10. One real end-to-end smoke point emits McPAT-native area, power, cache timing,
    layout, HotSpot temperature, integer R2 latency, IPC2, and BIPS2 without a
    standalone CACTI artifact or invocation.
