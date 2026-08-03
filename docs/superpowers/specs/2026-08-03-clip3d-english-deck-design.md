# CLIP-3D English Reproduction Deck Design

## Objective

Create a new, fully English presentation from
`/home/zyjiang/Agenticflow/CLIP3D_reproduction_report_20260802/CLIP3D.pptx`.
Preserve the source deck and its 21-slide technical-audit narrative while
updating two scientific status areas:

1. the 10 ms transient-thermal extension must reflect its current implemented,
   audited, but not yet physically executed state;
2. the pilot experiment must use the exploratory nonzero
   `lambda_wire = 0.0020119160767721133` setup, with result fields deliberately
   left as editable placeholders until the running R2 completes.

The deck is for advisor and CLIP-3D-author discussion.  It must not imply that
the local parameter set passed formal acceptance or reproduced the paper's
reported BIPS improvement.

## Chosen approach

Rebuild the deck through a translated copy of the existing Python generator.
Do not edit text directly inside the source PPTX: English sentence lengths
differ from Chinese, so every text box must be reflowed and visually checked.

Create the independent output directory:

```text
/home/zyjiang/Agenticflow/CLIP3D_reproduction_report_20260803_en/
```

The directory will contain:

- `CLIP3D_English.pptx`: final editable 16:9 deck;
- `generate_ppt_en.py`: reproducible English generator;
- `report_data_en.json`: evidence and explicit result placeholders;
- `previews/slide-XX.png`: one deterministic preview per slide;
- `contact_sheet.png`: full-deck visual inspection sheet;
- `test_generate_ppt_en.py`: content and layout regressions.

No source file below `CLIP3D_reproduction_report_20260802/` will be modified.

## Narrative and visual design

Retain the existing 21-slide order:

1. title;
2. executive status;
3. paper motivation;
4. closed-loop method;
5. sustainable-frequency formula;
6. project map;
7. workloads;
8. configurations and parameter status;
9. provenance;
10. runs versus results;
11. scripts, tests, and documentation;
12. workflow package map;
13. call graph and artifacts;
14. R1 sweep;
15. McPAT/CACTI consistency;
16. floorplanning and thermal flow;
17. R2 latency feedback;
18. nonzero-lambda single-point template;
19. parameter identification and acceptance;
20. blockers and transient status;
21. questions and next decisions.

Keep the dark navy background and the existing color semantics: cyan for data
flow, green for completed evidence, amber for limitations, and red for rejected
or invalid claims.  Use an installed English font such as DejaVu Sans.  Formula
notation and file paths may use DejaVu Sans Mono.

Every slide footer must state `operational / non-formal`.  The generated PPTX
must contain no CJK characters in visible slide text.

## Transient-thermal update

Slides 11, 12, and 20 will consistently report:

- the optional 10 ms transient path is implemented on branch
  `feature/matmul-transient-validation`, commit `9f927d6`;
- its shared-power dual-layout design uses one dedicated periodic-statistics R1,
  per-window McPAT power, and two layout-specific HotSpot transient branches;
- the final unit-test evidence is 97 passing tests with 2 expected external-
  HotSpot skips;
- no real gem5/McPAT/HotSpot transient experiment has been launched;
- the branch is not merged into `main`;
- a generic config-parent output-alias guard remains parked, but the prescribed
  output below `runs/transient_validation/` is outside that unsafe relation;
- the extension is operational, non-formal, and outside the paper's main
  steady-state reproduction path.

The deck must not state that transient temperatures or layout differences were
measured, because no physical transient result exists yet.

## Nonzero-lambda pilot update

Slides 2, 8, 17, 18, 19, and 20 will replace the zero-lambda pilot narrative
with the exploratory MATMUL 32kB L1D / 512kB L2 run using:

```text
lambda_wire = 0.0020119160767721133
wire_objective = continuous
```

The coefficient is fixed text, not a placeholder.  It comes from the FFT-local
matched-R2 report.  The deck must show its rejection evidence:

- `R^2 = 0.7691549845761521`, below the 0.95 local acceptance threshold;
- one monotonicity violation;
- no cross-workload transfer validation.

Therefore the run has no formal/shared-parameter acceptance value.  Its valid
purpose is to demonstrate that a nonzero wire term is consumed by the optimizer
and that layout, HotSpot, frequency, integer latency, R2 IPC, and BIPS artifacts
can be generated end to end.

Slide 18 will be an editable comparison template.  All final-result cells,
including `Tmax`, `f_sus`, continuous and rounded wire cycles, critical-path
cycles, `IPC2`, `BIPS2`, runtime, and percentage improvement, will display
`[Pending final run]` or an equivalent visibly empty placeholder.  No running
intermediate number will be presented as a final result.  The slide will cite
the future source:

```text
runs/operational_raw_power_p1/
lambda0020119_matmul_32kB_512kB_20260803/clip3d/pipeline_summary.json
```

## Evidence classification

The English deck must distinguish four evidence classes:

- paper-reported facts;
- locally measured and completed evidence;
- implemented but not physically executed extensions;
- pending result placeholders.

Statements such as `strict reproduction completed`, `parameter accepted`, or
`paper-equivalent` are forbidden.  The main status remains: the software loop
is operational, while strict scientific reproduction is not established.

## Verification

The generated deliverable must pass:

1. 21-slide count and 16:9 dimensions;
2. ZIP/PPTX integrity and successful `python-pptx` reopen;
3. no visible slide shape outside page bounds;
4. no visible CJK character in any slide text;
5. required transient state, commit, test count, nonzero lambda, rejection
   evidence, placeholder language, and non-formal classification are present;
6. forbidden claims and obsolete statements such as `lambda_wire = 0` and
   `transient not implemented` are absent from the relevant narrative;
7. all 21 preview images are generated and the contact sheet is visually
   inspected for overlaps, clipping, unreadable type, and broken diagrams.
