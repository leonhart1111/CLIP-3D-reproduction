# CLIP-3D English Template Deck Design

## Goal

Create a polished, fully English version of the 21-slide CLIP-3D reproduction
progress deck. The output must be an edited copy of
`CLIP3D_reproduction_report_20260802/CLIP-3D汇报.pptx`, using that file only as
the visual-style template. The content, evidence, numerical values, ordering,
and conclusions come from `CLIP3D_reproduction_progress_zh.pptx`.

Create a separate Chinese Markdown speaker script for the final English deck.

## Output Files

- `CLIP3D_reproduction_report_20260802/CLIP-3D汇报.pptx`
- `CLIP3D_reproduction_report_20260802/CLIP-3D汇报_中文讲稿.md`

The template presentation is intentionally overwritten only after a working
copy has been generated and validated. Existing unrelated project files are
not modified.

## Content Scope

The final deck retains all 21 source slides in the same narrative order:

1. Title
2. Executive summary
3. Paper motivation and reproduction objective
4. CLIP-3D closed loop
5. Closed-form sustainable frequency
6. Project directory map
7. Benchmarks and reproduction boundaries
8. Configuration and validation state
9. Tool and artifact provenance
10. Runs and results
11. Scripts, tests, and documentation
12. Workflow package responsibilities
13. Main call chain and file interfaces
14. R1 progress and stopping semantics
15. McPAT and CACTI consistency issue
16. Floorplan, thermal grid, and frequency
17. R2 latency back-annotation
18. MATMUL pilot result
19. Parameter identification and rejection state
20. Required corrections and reproduction gaps
21. Questions and next steps

No result may be strengthened during translation. In particular,
`operational/non-formal`, `accepted=false`, the rejected FFT wire fit, the
McPAT/CACTI mismatch, and all disclosed workload assumptions remain explicit.

## Construction Strategy

1. Open the template presentation as the destination package so its slide
   masters, layouts, theme, embedded visual assets, and document properties are
   retained.
2. Import or recreate the 21 source slides in the destination, preserving the
   original content grouping and information architecture.
3. Remove every original business-content slide from the template. No template
   wording or unrelated diagram remains in the final presentation.
4. Translate every visible source string into concise technical English,
   including titles, body text, tables, labels, callouts, legends, conclusions,
   and source lines.
5. Restyle imported shapes to the template's visual system. Explicit styling
   inherited from the source deck must be replaced where it conflicts with the
   template.

## Visual System

- Preserve the template's 16:9 page geometry, master/theme relationship, Arial
  typography, blue-gray palette, restrained accent colors, title hierarchy,
  footer conventions, and shape language.
- Reuse suitable template backgrounds and decorative assets without retaining
  any original template subject matter.
- Prefer clear technical diagrams, compact tables, and aligned cards over dense
  paragraphs.
- Use consistent slide margins, grid alignment, card padding, corner radii,
  stroke weights, and page numbering across all 21 slides.
- Maintain a readable hierarchy at presentation distance. Body copy should not
  be reduced merely to force a direct Chinese-to-English translation into the
  original box; wording and layout are adjusted first.

## Translation Rules

- Translate for technical presentation quality, not word-for-word literalism.
- Preserve formulas, parameter names, paths, tool names, units, and numerical
  precision.
- Use consistent terminology throughout, including `sustainable frequency`,
  `latency back-annotation`, `reproduction boundary`, `formal artifact`,
  `operational/non-formal`, and `monotonicity violation`.
- The PPT contains English only. Chinese is confined to the speaker-script
  Markdown file.

## Chinese Speaker Script

The Markdown script contains one numbered section per final slide. Each section
provides natural spoken Chinese rather than a literal translation of the slide.
It explains the intended takeaway, important evidence, caveats, and transitions
to the next slide. Slide numbers and English slide titles match the final deck.

## Iterative Visual Review

Visual review is iterative rather than a single export check:

1. Generate the first complete deck.
2. Render all slides using the available rendering path. If an office renderer
   is unavailable, use the project's deterministic preview mechanism and OOXML
   geometry checks.
3. Inspect a contact sheet for global consistency.
4. Inspect dense and high-risk slides individually, especially slides 4, 5, 7,
   13, 15, 18, 19, 20, and 21.
5. Correct clipping, overflow, overlaps, inconsistent alignment, weak contrast,
   awkward line breaks, undersized type, and unbalanced whitespace.
6. Repeat rendering and inspection until no material readability defect remains.

## Validation

The final acceptance checks are:

- exactly 21 slides;
- template master/theme retained;
- no original template business content remains;
- no visible Chinese text remains in the PPT;
- all key numbers, formulas, statuses, and conclusions match the source deck;
- no shape extends materially beyond slide bounds;
- no text is clipped or hidden behind another object;
- consistent typography, colors, margins, titles, footers, and page numbers;
- the Chinese Markdown script has exactly 21 corresponding slide sections;
- the PPTX opens successfully as a valid Office Open XML package.

## Constraints and Fallbacks

The current host does not provide LibreOffice. The implementation must not
claim visual completion based only on successful PPTX serialization. It will
use the existing Python/Pillow preview infrastructure where possible, inspect
OOXML geometry and text metrics, and perform repeated preview-based correction.
If a renderer becomes available, its rendered slides take precedence for the
final visual review.
