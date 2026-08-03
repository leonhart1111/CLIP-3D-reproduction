# CLIP-3D English Template Deck Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the template deck's subject matter with the 21-slide reproduction report, translate every visible slide to English, apply the template's visual system, and deliver a matching Chinese speaker script.

**Architecture:** Extend the existing deterministic Python/Pillow deck generator so one content model drives both editable PPTX output and rendered preview PNGs. Load the destination from the template presentation to retain its package-level theme and masters, remove its original slides, then build the translated report slides with template-derived style tokens. Validate structure and text automatically, followed by repeated contact-sheet and individual-slide visual review.

**Tech Stack:** Python 3.12, vendored `python-pptx`, Pillow, ZIP/XML inspection, `unzip`, ImageMagick where available.

## Global Constraints

- The output PPTX is `/home/zyjiang/Agenticflow/CLIP3D_reproduction_report_20260802/CLIP-3D汇报.pptx`.
- The output script is `/home/zyjiang/Agenticflow/CLIP3D_reproduction_report_20260802/CLIP-3D汇报_中文讲稿.md`.
- Preserve the source deck's 21-slide order, numbers, formulas, caveats, and acceptance states.
- No visible Chinese text may remain in the PPTX; Chinese is confined to the Markdown script.
- Retain the template presentation's masters, layouts, theme, and appropriate decorative assets, but no original template business content.
- Perform at least two complete preview-and-correction rounds and individual inspection of slides 4, 5, 7, 13, 15, 18, 19, 20, and 21.
- Do not claim rendered visual verification from LibreOffice because it is unavailable on this host.

---

### Task 1: Add Template-Aware Deck Tests and Content Model

**Files:**
- Modify: `../CLIP3D_reproduction_report_20260802/test_generate_ppt.py`
- Modify: `../CLIP3D_reproduction_report_20260802/generate_ppt.py`
- Read: `../CLIP3D_reproduction_report_20260802/report_data.json`

**Interfaces:**
- Consumes: template path and existing `report_data.json`.
- Produces: `SLIDES_EN: list[dict]`, `TEMPLATE_PATH: Path`, and validation helpers used by later tasks.

- [ ] Add failing tests that require the template path to exist, require 21 English slide titles, reject CJK characters from visible PPT copy, and assert preservation of critical strings such as `0.0020119`, `R² = 0.769`, `accepted = false`, `84/100`, and `0.029%`.
- [ ] Run `cd CLIP3D_reproduction_report_20260802 && python test_generate_ppt.py` and confirm the new assertions fail.
- [ ] Add the complete 21-slide English content model, using concise technical English and retaining every result qualification.
- [ ] Run the test again and confirm the content assertions pass.
- [ ] Commit only the content-model and test changes with `git commit -m "test: define English CLIP-3D deck content"` if the report directory is tracked by the active repository; otherwise record the successful test output without committing untracked user material.

### Task 2: Rebuild the Report Inside the Template Package

**Files:**
- Modify: `../CLIP3D_reproduction_report_20260802/generate_ppt.py`
- Test: `../CLIP3D_reproduction_report_20260802/test_generate_ppt.py`
- Input: `../CLIP3D_reproduction_report_20260802/CLIP-3D汇报.pptx`

**Interfaces:**
- Consumes: `SLIDES_EN`, template masters/layouts/theme, and `report_data.json`.
- Produces: `build_template_deck(template_path: Path, output_path: Path) -> Path`.

- [ ] Add a failing test that copies the template to a temporary path, builds the report, and asserts exactly 21 slides, the original theme name remains present, and known template subject strings are absent.
- [ ] Run the focused test and confirm it fails before implementation.
- [ ] Implement safe template loading, original-slide removal, report-slide creation, and atomic replacement of the destination file.
- [ ] Preserve the source report's information architecture while applying template-derived Arial typography, blue-gray palette, title treatment, footer, page numbering, card geometry, and accents.
- [ ] Run the focused test and confirm it passes.

### Task 3: Translate and Restyle All 21 Slides

**Files:**
- Modify: `../CLIP3D_reproduction_report_20260802/generate_ppt.py`
- Modify: `../CLIP3D_reproduction_report_20260802/report_data.json` only if an existing label needs a language-neutral key.
- Test: `../CLIP3D_reproduction_report_20260802/test_generate_ppt.py`

**Interfaces:**
- Consumes: template-aware `Canvas`, style tokens, `SLIDES_EN`, and existing report metrics.
- Produces: editable English shapes and deterministic preview images for slides 1 through 21.

- [ ] Convert the global `base`, `source`, `card`, `table`, chart, and arrow primitives to the template visual system.
- [ ] Rewrite slide functions 1–7 with English text and template styling; generate previews and inspect title hierarchy, table widths, and benchmark caveats.
- [ ] Rewrite slide functions 8–14; generate previews and inspect parameter formatting, directory labels, workflow arrows, and R1 status readability.
- [ ] Rewrite slide functions 15–21; generate previews and inspect McPAT/CACTI comparison, formulas, pilot metrics, rejection evidence, priority table, and final questions.
- [ ] Run all generator tests and correct any content mismatch or boundary failure.

### Task 4: Produce the Chinese Speaker Script

**Files:**
- Create: `../CLIP3D_reproduction_report_20260802/CLIP-3D汇报_中文讲稿.md`
- Read: `../CLIP3D_reproduction_report_20260802/CLIP3D_reproduction_progress_zh_notes.md`

**Interfaces:**
- Consumes: final English slide titles and the existing Chinese notes.
- Produces: 21 numbered Chinese sections aligned one-to-one with the final deck.

- [ ] Write one section per slide with the English title, natural Chinese speaking text, the principal evidence, caveat, and transition.
- [ ] Update explanations for the benchmark consistency column, `lambda_wire = 0`, path plus SHA-256 artifacts, workflow-test scope/results, McPAT/CACTI mismatch impact, and rejected FFT wire fit.
- [ ] Check that the script contains exactly 21 slide headings and no obsolete template content.

### Task 5: First Full Visual Review and Correction

**Files:**
- Modify: `../CLIP3D_reproduction_report_20260802/generate_ppt.py`
- Generate: `../CLIP3D_reproduction_report_20260802/previews_en/slide-01.png` through `slide-21.png`
- Generate: `../CLIP3D_reproduction_report_20260802/contact_sheet_en.png`

**Interfaces:**
- Consumes: deterministic preview renderer from Task 3.
- Produces: first corrected complete deck and preview set.

- [ ] Generate the PPTX, all 21 preview PNGs, and the contact sheet.
- [ ] Inspect the contact sheet for global rhythm, density, alignment, color consistency, and slide-to-slide hierarchy.
- [ ] Inspect slides 4, 5, 7, 13, 15, 18, 19, 20, and 21 at full resolution.
- [ ] Correct every observed overflow, collision, weak contrast, awkward wrap, small label, misalignment, and unbalanced whitespace.
- [ ] Regenerate the complete preview set and PPTX.

### Task 6: Second Full Visual Review and Final Validation

**Files:**
- Modify if needed: `../CLIP3D_reproduction_report_20260802/generate_ppt.py`
- Validate: `../CLIP3D_reproduction_report_20260802/CLIP-3D汇报.pptx`
- Validate: `../CLIP3D_reproduction_report_20260802/CLIP-3D汇报_中文讲稿.md`

**Interfaces:**
- Consumes: corrected deck, previews, source deck, and template.
- Produces: final deliverables and verification evidence.

- [ ] Repeat contact-sheet and high-risk-slide inspection independently of the first review notes.
- [ ] Apply final typography, spacing, wrapping, and alignment corrections and regenerate outputs.
- [ ] Run `python test_generate_ppt.py` and require all tests to pass.
- [ ] Inspect the PPTX ZIP package and require valid OOXML, exactly 21 slide XML parts, retained template theme/master parts, and no stale template slide relationships.
- [ ] Extract visible slide text and require no CJK characters, no template subject strings, and presence of all 21 English titles and critical numerical evidence.
- [ ] Run geometry validation and require no material shape overflow or invalid dimensions.
- [ ] Verify the Markdown script has exactly 21 matching sections.
- [ ] Record the final file sizes and SHA-256 digests for both deliverables before reporting completion.
