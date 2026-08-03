# CLIP-3D English Reproduction Deck Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate a visually verified, fully English 21-slide CLIP-3D reproduction deck with current transient-thermal status and an explicitly non-formal nonzero-lambda result template.

**Architecture:** Reuse the source deck's dual-target `Canvas` generator so one set of slide functions produces both editable PowerPoint shapes and deterministic PNG previews.  Copy the source generator and data into an independent English output directory, replace all visible content, add focused content/layout regressions, then generate and inspect the PPTX and contact sheet.

**Tech Stack:** Python 3, vendored `python-pptx`, Pillow, JSON, ZIP/PPTX XML, Poppler image inspection.

## Global Constraints

- Treat `/home/zyjiang/Agenticflow/CLIP3D_reproduction_report_20260802/` as read-only.
- Write only below `/home/zyjiang/Agenticflow/CLIP3D_reproduction_report_20260803_en/` plus this tracked plan.
- Preserve 21 slides, 16:9 dimensions, and the dark technical-audit visual language.
- All visible PowerPoint text must be English and contain no CJK characters.
- Every footer must state `operational / non-formal`.
- Use exactly `lambda_wire = 0.0020119160767721133`; final result metrics remain `[Pending final run]`.
- State that the lambda fit has `R^2 = 0.7691549845761521`, one monotonicity violation, and no cross-workload transfer validation.
- State that transient code/audit is complete at `9f927d6`, 97 tests passed with 2 expected skips, the real transient run was not launched, and the branch is not merged.
- Do not claim formal acceptance, paper equivalence, measured transient temperatures, or reproduction of the paper's reported BIPS improvement.

---

### Task 1: Establish English content and layout regressions

**Files:**
- Create: `/home/zyjiang/Agenticflow/CLIP3D_reproduction_report_20260803_en/test_generate_ppt_en.py`
- Reference: `/home/zyjiang/Agenticflow/CLIP3D_reproduction_report_20260802/test_generate_ppt.py`

**Interfaces:**
- Consumes: `generate_ppt_en.SLIDES`, `generate_ppt_en.Canvas`, and `report_data_en.json`.
- Produces: a unittest suite that rejects CJK text, obsolete status, missing placeholders, out-of-bounds shapes, or a non-21-slide deck.

- [ ] **Step 1: Create the output directory and failing test**

The test imports `generate_ppt_en`, renders every slide to an in-memory
`Presentation`, concatenates visible text, and asserts:

```python
self.assertEqual(len(generate_ppt_en.SLIDES), 21)
self.assertIsNone(re.search(r"[\u3400-\u9fff]", all_text))
self.assertIn("0.0020119160767721133", all_text)
self.assertIn("[Pending final run]", all_text)
self.assertIn("9f927d6", all_text)
self.assertIn("97 tests passed", all_text)
self.assertIn("real transient run has not been launched", all_text.lower())
self.assertIn("not merged", all_text.lower())
self.assertNotIn("lambda_wire = 0 ", all_text)
```

It also builds the deck to a temporary path, reopens it, asserts 21 slides and
16:9 dimensions, and checks `left >= 0`, `top >= 0`, `left + width <=
slide_width`, and `top + height <= slide_height` for every shape.

- [ ] **Step 2: Run the test and verify RED**

Run:

```bash
PYTHONPATH=/home/zyjiang/Agenticflow/CLIP3D_reproduction_report_20260802/.python \
python -m unittest -v test_generate_ppt_en
```

Expected: import failure because `generate_ppt_en.py` does not yet exist.

### Task 2: Implement the English generator and evidence data

**Files:**
- Create: `/home/zyjiang/Agenticflow/CLIP3D_reproduction_report_20260803_en/generate_ppt_en.py`
- Create: `/home/zyjiang/Agenticflow/CLIP3D_reproduction_report_20260803_en/report_data_en.json`
- Reference: `/home/zyjiang/Agenticflow/CLIP3D_reproduction_report_20260802/generate_ppt.py`
- Reference: `/home/zyjiang/Agenticflow/CLIP3D_reproduction_report_20260802/report_data.json`

**Interfaces:**
- Consumes: the existing reusable drawing helpers and verified source-deck structure.
- Produces: `render_slide(slide_fn, data)`, `generate_deck(output: Path)`, 21 English slide functions, and a CLI `main()` that writes PPTX/previews/contact sheet.

- [ ] **Step 1: Copy the generator mechanics and translate all visible content**

Retain the color palette, `Canvas`, cards, tables, footer, preview rendering,
contact-sheet assembly, and PPTX ZIP validation.  Change the font names to
`DejaVu Sans` and `DejaVu Sans Mono`, the output filename to
`CLIP3D_English.pptx`, and every visible string in `slide01` through `slide21`
to English.

- [ ] **Step 2: Encode current transient state and nonzero-lambda placeholders**

The JSON must contain:

```json
{
  "status_label": "operational / non-formal",
  "exploratory_lambda": {
    "value": 0.0020119160767721133,
    "r_squared": 0.7691549845761521,
    "monotonic_violations": 1,
    "cross_workload_transfer_validated": false,
    "result_placeholder": "[Pending final run]"
  },
  "transient": {
    "branch": "feature/matmul-transient-validation",
    "commit": "9f927d6",
    "tests_passed": 97,
    "expected_skips": 2,
    "real_run_launched": false,
    "merged_to_main": false
  }
}
```

Retain source-deck facts needed by other slides.  Slide 18 reads all final
metrics from `result_placeholder`, while slide 19 displays the fixed lambda
value and its rejected-fit evidence.

- [ ] **Step 3: Run focused tests and reach GREEN**

Run:

```bash
PYTHONPATH=/home/zyjiang/Agenticflow/CLIP3D_reproduction_report_20260802/.python \
python -m unittest -v test_generate_ppt_en
```

Expected: all content and in-memory layout tests pass.

### Task 3: Generate, render, and inspect the deliverable

**Files:**
- Create: `/home/zyjiang/Agenticflow/CLIP3D_reproduction_report_20260803_en/CLIP3D_English.pptx`
- Create: `/home/zyjiang/Agenticflow/CLIP3D_reproduction_report_20260803_en/previews/slide-01.png` through `slide-21.png`
- Create: `/home/zyjiang/Agenticflow/CLIP3D_reproduction_report_20260803_en/contact_sheet.png`

**Interfaces:**
- Consumes: the English generator and JSON.
- Produces: the editable PPTX and deterministic visual-inspection artifacts.

- [ ] **Step 1: Generate all artifacts**

Run:

```bash
PYTHONPATH=/home/zyjiang/Agenticflow/CLIP3D_reproduction_report_20260802/.python \
python generate_ppt_en.py
```

Expected: `CLIP3D_English.pptx`, 21 previews, and `contact_sheet.png`.

- [ ] **Step 2: Run machine verification**

Run the focused test again, `unzip -t CLIP3D_English.pptx`, count 21 slide XML
files, and extract all shape text to prove the CJK regex has no match.  Verify
the PPTX reopens successfully with vendored `python-pptx`.

- [ ] **Step 3: Inspect the contact sheet and representative full-size slides**

Check slides 1, 4, 13, 18, 20, and 21 at original preview resolution.  Reject
and adjust any overlap, clipping, tiny text, unbalanced empty space, or broken
connector.  Re-run generation and tests after every adjustment.

- [ ] **Step 4: Final verification**

Run:

```bash
PYTHONPATH=/home/zyjiang/Agenticflow/CLIP3D_reproduction_report_20260802/.python \
python -m unittest -v test_generate_ppt_en
unzip -t CLIP3D_English.pptx
```

Expected: all tests pass and ZIP integrity reports no errors.
