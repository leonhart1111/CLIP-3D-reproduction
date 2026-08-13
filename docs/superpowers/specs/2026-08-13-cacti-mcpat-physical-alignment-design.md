# CACTI/McPAT Physical Alignment and Unscaled-Area Design

Date: 2026-08-13

## Purpose

Repair the original steady-state CLIP-3D reproduction so that cache delay and
geometry are measured by the local CACTI executable using the architecture
actually simulated by gem5, McPAT and CACTI describe the same cache
organization, and no area is multiplied merely to reproduce the paper's
nominal 150 mm² die.

The paper's Table II remains a read-only comparison. Its numbers must never be
used as pipeline inputs or as targets for reverse-tuning CACTI.

## Source-of-truth contract

The R1 metadata is the architectural source of truth. The following fields
must be propagated into a shared cache-organization contract:

- technology node;
- cache level and capacity;
- cache-line size;
- associativity;
- number of banks;
- input/output bus width;
- access mode and ECC policy;
- modeled operating temperature;
- device/cell and interconnect assumptions.

L1I, L1D and L2 receive separate records. L1I may share an organization with
L1D only when their contract fields are equal; it must not be silently aliased
by level name.

McPAT consumes the same structural contract for power modeling. Its internally
reported cache area is retained only for diagnostics. The authoritative cache
geometry and access delay used by floorplanning and R2 come from the local
CACTI characterization artifact.

## Physical-area policy

- L1I, L1D and L2 area, width and height come directly from local CACTI.
- Non-cache logic and interconnect area comes directly from McPAT.
- No global area factor or dimension factor is applied.
- Power is never rescaled.
- Each module records its authoritative area source and, for caches, the
  referenced CACTI record/config identity.
- HotSpot die dimensions continue to be derived from the unscaled module
  geometry and the configured floorplan utilization.

The old `area_reference_mm2`, `area_reference_raw_mm2`,
`area_reference_basis`, `DEFAULT_AREA_SCALE`, `--area-scale`,
`area_before_global_scale_mm2`, and `area_calibration` concepts are removed
from executable configurations, production APIs and newly generated reports.
Historical result directories are not rewritten.

## Cache-delay policy

At clock period \(t_{clk}=1/f_0\), a CACTI access time is converted to gem5
latency by

\[
L_{cache}=\max\left(1,\left\lceil\frac{t_{access}}{t_{clk}}\right\rceil\right).
\]

A small floating-point tolerance may be applied only to prevent an exact
integer such as `3.0000000000000004` from becoming four cycles. It must not
change genuine non-integer values: 2.02 becomes 3 and 6.28 becomes 7.

The same characterization JSON record and its provenance identity must drive
both module geometry and R2 cache latency. A mismatch is a hard error rather
than a warning.

## Local Table-II-equivalent measurement

The project will provide a reproducible characterization command for:

- L1: 16, 32, 64 and 128 kB;
- L2: 128, 256, 512, 1024 and 2048 kB;
- nominal frequency: 2 GHz;
- technology: 45 nm.

Generated JSON and CSV/Markdown reports include access time, unrounded and
ceiling-rounded cycles, dimensions, area, organization parameters, CACTI Git
revision, executable SHA-256, base-config SHA-256, generated-config SHA-256 and
raw-output location. The locally measured values are authoritative. Paper
Table II may appear in a separate comparison report, clearly labeled as
external reference data and never imported by the workflow.

Generated cache configurations and raw CACTI output remain part of the
provenance package so every table row can be independently reproduced.

## Validation and failure handling

The pipeline rejects:

- a CACTI record whose size, line size, associativity, bank count, bus width,
  technology, temperature or policy differs from the requested contract;
- duplicate or missing L1I/L1D/L2 records;
- non-positive dimensions or area;
- a characterization artifact using a non-ceiling latency policy;
- a geometry/latency artifact identity mismatch;
- any new experiment configuration that attempts to request area scaling.

Validation errors state the mismatched field, expected value and observed
value.

## Compatibility and rerun boundary

Existing R1 outputs are preserved and do not need to be rerun. Existing result
directories are also preserved as historical, but their downstream physical
results are obsolete because the geometry and cache-cycle contract changed.

For a corrected result, rerun all stages after R1: McPAT mapping/execution,
local CACTI characterization, module construction, layout, HotSpot,
sustainable-frequency calculation, R2 latency-vector generation and R2 gem5.
New schema/provenance fields prevent corrected runs from silently consuming an
old scaled or structurally mismatched artifact.

## Testing strategy

Tests are added before implementation and must demonstrate:

1. 2.02 and 6.28 cycles round upward, while an exact integer remains unchanged.
2. L1I, L1D and L2 configurations carry the requested organization separately.
3. Each McPAT cache organization agrees with its CACTI contract.
4. Cache areas and dimensions are used verbatim from CACTI.
5. Non-cache area is used verbatim from McPAT.
6. No global scale fields, CLI option or arithmetic remain in the production
   steady-state path or official experiment configurations.
7. R2 latency and floorplan geometry reference the same characterized records.
8. Every alignment mismatch is rejected with a targeted error.
9. The complete local Table-II-equivalent set is generated with provenance.
10. Existing steady-state workflow tests remain green after updating expected
    physically unscaled values.

## Acceptance criteria

The change is complete when the focused tests and the full applicable workflow
suite pass, the nine-row local characterization table is reproducibly
generated, a corrected one-point steady-state smoke run reaches R2 using an
existing R1 directory, and inspection of its artifacts proves that no area
scale was applied and that geometry and latency share one aligned CACTI source.
