# CLIP-3D English Presentation Design

## Objective

Create a new, exactly twenty-slide, fully English presentation based on
`docs/CLIP-3D汇报.pptx`. The new deck must reuse the source deck's dark navy
technical template, typography hierarchy, accent colors, spacing, and visual
language. It must not overwrite the source file.

The deck is method-first: concise English slide text explains the CLIP-3D
closed loop, its equations, and the reproduction methodology; measured
experiments then show which parts are operational, rejected, successful, or
still pending. A separate Chinese Markdown speaker script provides the detail
that is intentionally omitted from the slides.

## Deliverables

- `docs/CLIP-3D_Reproduction_Methodology_and_Experiments_EN.pptx`
- `docs/CLIP-3D_Reproduction_Speaker_Notes_ZH.md`

The PowerPoint is a modern, macro-free `.pptx` in the same 16:9 format as the
source deck. All visible slide text is English. The Chinese script is organized
one-to-one with the slide numbers and titles.

## Audience and Reporting Position

The primary audience is the supervisor and the original CLIP-3D authors. The
deck therefore distinguishes four evidence levels:

1. paper-stated equations or parameters;
2. locally measured results;
3. operational, non-formal candidates that failed strict promotion gates;
4. unfinished work shown only as an explicit placeholder.

No proxy BIPS, ROM prediction, failed point, or incomplete experiment may be
presented as a formally reproduced paper result. The paper's reported mean
BIPS gain is context, not a locally reproduced claim.

## Visual System

- Reuse the source theme, masters, page size, title treatment, and footer.
- Preserve the dark navy background with blue/cyan highlights.
- Use green only for passed/complete evidence, amber for operational caveats,
  red for rejected gates, gray for pending or invalid evidence, and purple for
  the transient-ROM research extension.
- Keep slide titles near 28--32 pt and body text near 18--20 pt, following the
  source deck. Avoid paragraphs on slides.
- Use at most four short bullets in a content block.
- Give every experiment a visible `MEASURED`, `REJECTED`, `NON-FORMAL`, or
  `PENDING` status badge.
- Equations are centered in high-contrast formula cards. Important terms are
  color-coded consistently: thermal terms in amber, frequency in cyan, IPC/BIPS
  in green, wire delay in purple.
- Charts use the same navy background and light foreground as the deck, with
  labels large enough for projection.

## Twenty-Slide Storyboard

### 1. CLIP-3D Reproduction: Methodology, Experiments, and Open Issues

Title slide. Subtitle: `A physically constrained, closed-loop evaluation of
3D IC performance`. Add `Reproduction Project · August 2026`, but no results.

### 2. Executive Summary

Four concise cards:

- toolchain: gem5, McPAT, CACTI, and HotSpot operational;
- canonical R1: 100/100 successful points, five workloads;
- steady end-to-end pipeline: operational but not a formal reproduction of the
  reported mean gain;
- transient-ROM extension: implemented, two-point measured validation pending.

### 3. Why Architecture-Only Exploration Fails

Explain the coupled problem: cache and core choices change IPC, power, area,
wire delay, temperature, sustainable frequency, and therefore BIPS. A compact
cause-and-effect diagram replaces literature-review text.

### 4. The CLIP-3D Closed Loop

Show one horizontal flow:

`R1 -> McPAT/CACTI -> floorplan -> HotSpot -> closed-form frequency -> R2 -> BIPS2`.

Mark the two gem5 passes and the single final HotSpot validation. Explain that
closed-form evaluation avoids calling HotSpot for every candidate layout.

### 5. Reproduction Inputs, Tools, and R1

Condense the project/file-format material into one slide:

- versioned inputs: `benchmarks/`, `configs/`, `manifests/`;
- implementation: `workflow/`, `scripts/`, `tests/`, `tools/`;
- evidence: `runs/`, `results/`, `docs/`;
- R1 outputs: `stats.txt`, `r1_metadata.json`, and IPC/activity counts.

Include a small workload table for FFT, CHOLESKY, STREAM, MATMUL, and STENCIL.

### 6. Power, Area, and Cache Characterization

Explain the data boundary:

- gem5 statistics -> McPAT XML -> dynamic/leakage power and module area;
- cache configuration -> CACTI -> cache geometry, access time, energy;
- McPAT power is raw output with no result-oriented scaling.

Show the principal identities

`P_total = P_dynamic + P_leakage`

and a compact latency-to-cycle mapping for CACTI.

### 7. Floorplanning Variables and Physical Constraints

Show cores and L2 as rectangles on two tiers. Define the decision vector

`p = {(x_i, y_i, z_i)}`

and the legal constraints: die boundary, non-overlap, allowed tiers, module
area, TSV/arbitration, and utilization. State that the current strict-P1 space
moves only the shared L2; this limits the physical improvement ceiling.

### 8. From Layout and Power to Temperature

Explain the steady thermal relation

`Delta T = R_theta(p) P`

and why HotSpot is used as the final physical validator. Visually separate the
fast analytical proxy used in search from the real HotSpot checkpoint used for
reported results.

### 9. Deriving Sustainable Frequency

Present the derivation in three steps:

`P(f) = P_leak + (f/f0) P_dyn`

`Tmax(f)-Tamb = [gamma + (1-gamma) f/f0] [Tmax(f0)-Tamb]`

`f_sus = clip(f0/(1-gamma) * [(Tsafe-Tamb)/(Tmax(f0)-Tamb)-gamma], fmin, f0)`

The slide defines every symbol. The script explains the fixed-voltage and
uniform-leakage-ratio assumptions and why `Tmax(f0)` is required.

### 10. From IPC1 to Measured BIPS2

Use a two-column equation chain:

`BIPS1 = IPC1 * f_sus`

`BIPS2 = IPC2 * f_sus`

Explain that IPC1 ranks candidates cheaply, while IPC2 comes from gem5 R2 with
the selected layout's integer latency vector. Only BIPS2 is the final steady
score.

### 11. Thermal Proxy Objective

Present the locally implemented proxy in conceptual form:

`T_hat(p) = T_base + alpha H(p) + beta P_bottom + w_cross H_cross(p)`

and the optimization score

`J(p) = -IPC1 * f_hat_sus(p) + lambda_wire * W(p) + P_legal(p)`.

Explain the physical meaning of `alpha`, `beta`, and `w_cross`; explicitly note
that beta is unidentifiable when all legal L2 candidates remain on one tier.

### 12. Wire Delay, Integer Cycles, and R2

Show

`t_wire = 0.69 r c L^2`

`c_wire = round_policy(t_wire * f0)`

and the critical-path latency composition. Explain why layout improvement only
changes IPC when a candidate crosses an integer cycle boundary. Present the
current nonzero exploratory value
`lambda_wire = 0.0020119160767721133` as workflow validation evidence, not an
accepted cross-workload parameter.

### 13. Experiment 1: Parameter Identification Method

Describe measured HotSpot fitting:

- position samples generate real HotSpot temperatures;
- fit absolute temperature and within-workload spatial residuals;
- grid-search the non-smooth cross-tier weight;
- validate on held-out positions, leave-one-workload-out groups, and an
  independent STREAM target grid.

Show the composite fitting score and explain Spearman as rank agreement.

### 14. Experiment 1: Parameter Results

Show candidate values:

- `alpha = 1.5643788695`;
- `beta = 0`, fixed and unidentifiable under strict P1;
- `w_cross = 0.995`;
- `lambda_wire = 0.0020119160767721133`, exploratory only.

Use a bar chart for spatial Spearman values: validation `0.657`, STREAM external
`0.500`, LOO FFT `0.642`, LOO MATMUL `0.554`, and LOO STENCIL `0.286`, with a
red horizontal acceptance line at `0.8`. The conclusion badge is `REJECTED FOR
FORMAL PROMOTION`.

### 15. Experiment 2: End-to-End Validation Method

Define the balanced five-point pilot, one 64 kB L1D / 512 kB L2 point per
workload. Show the exact evidence chain: canonical R1, fixed-bin and CLIP-3D
layout, real HotSpot, sustainable frequency, layout-specific R2, and measured
BIPS2. Explain the signed gain formula.

### 16. Experiment 2: End-to-End Results

Use a signed BIPS2 improvement chart:

- STREAM: `+1.636%`, success;
- MATMUL: `+0.181%`, success;
- STENCIL: `-0.953%`, success;
- FFT: `+0.437%`, failed experiment state;
- CHOLESKY: `+0.236%`, failed experiment state.

Failed points are gray hatched markers and excluded from accepted aggregate
claims. The slide states that small gains are consistent with a restricted
only-L2-movable design space and discrete latency plateaus.

### 17. Transient Thermal Formulation

Explain why one steady `Tmax` cannot represent a time-varying workload. Show

`C d(theta)/dt + G theta = P(t)`

and the exact window recurrence

`theta_(k+1) = A_k(s) theta_k + B_k(s) H(p)[l_k + s d_k]`,

with `Delta t_k(s) = Delta t_k0 / s`. The script derives the time-stretching
effect and distinguishes leakage energy from dynamic energy.

### 18. Experiment 3: Transient HotSpot Validation

Explain the shared 2 ms periodic R1, 239 windows, windowed McPAT, and dual-layout
HotSpot trace. Use a combined power/temperature plot and a compact comparison:

- fixed trace peak: `116.261707 C`;
- CLIP-3D trace peak: `116.267229 C`;
- layout difference: `+0.005522 C` for CLIP-3D;
- power-to-temperature peak lag: `0.166 s`;
- total simulated ROI: approximately `0.478 s`.

Include a small 10 ms versus 2 ms table to show that 10 ms rounding hid timing
and sub-0.01 C detail. Mark the result `MEASURED, NON-FORMAL`.

### 19. Experiment 4: Transient ROM Closed Loop

Show the optional extension:

- eight PRBS training HotSpot calls;
- two matched average-power steady initializations;
- two holdout transient calls;
- zero HotSpot calls inside the layout optimizer;
- final fixed-bin and CLIP-3D real-HotSpot/R2 validation.

Show the reduced recurrence and `BIPS2_trans = IPC2 * f_sus,trans`. Reserve a
two-column MATMUL/STENCIL result panel marked `PENDING PERIODIC R1` and `NO
MEASURED BIPS2_TRANS YET`; do not invent values.

### 20. Conclusions and Questions

Conclude:

- the 100-point R1 and complete software chain are operational;
- the strict parameter fit is rejected, so the current coefficients remain
  non-formal;
- the five-point pilot proves workflow feasibility, not the paper's reported
  mean gain;
- 2 ms transient HotSpot exposes time lag and fine temperature variation;
- transient-ROM two-point evidence is pending.

End with three questions for the authors/supervisor: exact McPAT mapping and
technology settings; unpublished proxy and wire parameters; R1 acceleration
and reported runtime reconciliation.

## Chinese Speaker Script

The Markdown script contains one section per slide:

```text
## Slide NN — English slide title
### Purpose
### Suggested narration
### Formula and variable explanation
### Evidence and caveats
### Transition to the next slide
```

The narration is detailed Chinese prose suitable for direct rehearsal. Formula
slides explain every symbol and assumption. Experiment slides explain design,
inputs, outputs, numerical interpretation, failure semantics, and what must not
be claimed. The script may be substantially longer than the slide text.

## Evidence Sources

- Source format/template: `docs/CLIP-3D汇报.pptx`
- Canonical R1: `runs/architecture_sweep/r1/paper/summary.csv`
- Parameter fit:
  `results/parameter_studies/raw_power_strict_20260730/proxy_train_16/calibration_report.json`
- Five-point end-to-end pilot:
  `runs/discrete_partition_validation/balanced5_midcache_20260809/summary.csv`
- 2 ms transient validation:
  `runs/transient_validation/matmul_32kB_512kB_lambda0020119_2ms_precision6_20260806_030743/transient/comparison/`
- 10 ms comparison:
  `runs/transient_validation/matmul_32kB_512kB_lambda0020119_10ms_20260804/comparison/`
- Transient-ROM method and status: `docs/transient_rom_usage_zh.md` and
  `docs/transient_sustainable_frequency_zh.md`

All chart values are read from these files rather than copied from the old PPT.

## Validation and Acceptance

The deliverables are accepted only if:

1. the source PPT remains byte-for-byte unchanged;
2. the new PPT opens as a valid OOXML package and contains exactly twenty
   slides;
3. all visible slide text is English;
4. slide titles, body hierarchy, background, accent colors, and spacing visibly
   match the source template;
5. each experiment has at least one dedicated slide;
6. all formulas are legible at normal presentation scale;
7. measured, rejected, failed, non-formal, and pending evidence are visually
   distinguishable;
8. all reported values trace to the evidence sources above;
9. the Chinese Markdown script contains exactly twenty corresponding sections;
10. no unfinished transient-ROM result is represented by a fabricated number.
