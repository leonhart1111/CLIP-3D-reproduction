# CLIP-3D English Presentation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an exactly twenty-slide, fully English CLIP-3D methodology and experiment deck in the source deck's visual language, plus a detailed one-to-one Chinese Markdown speaker script.

**Architecture:** A deterministic Python builder reads the existing `.pptx` as an OOXML template, preserves its theme/master/page geometry, clones compatible source layouts, and replaces slide content with generated text, formula cards, diagrams, tables, and evidence-backed charts. A separate data module owns all measured values and evidence states so failed, rejected, non-formal, and pending results cannot be accidentally presented as accepted results; package-level tests validate slide count, visible language, evidence labels, relationship integrity, and source-file immutability.

**Tech Stack:** Python 3 standard library (`zipfile`, `xml.etree.ElementTree`, `csv`, `json`, `hashlib`), Matplotlib 3.11, NumPy 2.4, Pillow 12, PowerPoint OOXML, Markdown, pytest.

## Global Constraints

- Produce exactly 20 slides in 16:9 without overwriting `docs/CLIP-3D汇报.pptx`.
- Reuse the source PPT theme, masters, title treatment, typography hierarchy, dark navy background, accent colors, and spacing.
- All visible slide text must be English; detailed narration must be Chinese and map one-to-one to slide numbers and titles.
- Keep slide body text concise, normally no more than four bullets per content block; titles remain approximately 28--32 pt and body text approximately 18--20 pt.
- Every experiment slide must show a visible `MEASURED`, `REJECTED`, `NON-FORMAL`, or `PENDING` badge as appropriate.
- Failed FFT and CHOLESKY end-to-end points must be gray/hatched and excluded from aggregate claims.
- The transient-ROM panel must say `PENDING PERIODIC R1` and `NO MEASURED BIPS2_TRANS YET`; no placeholder may imply a measured result.
- Equations and numerical results must match the approved design specification and cited local evidence files.
- Matplotlib must use a writable cache directory such as `MPLCONFIGDIR=/tmp/clip3d-ppt-mpl`.

---

### Task 1: Template Inspection and Evidence Model

**Files:**
- Create: `tools/presentation/clip3d_deck_data.py`
- Create: `tools/presentation/inspect_pptx.py`
- Create: `tests/test_clip3d_presentation.py`

**Interfaces:**
- Consumes: source PPT path plus the evidence files listed in the design specification.
- Produces: `load_deck_evidence(repo_root: Path) -> dict[str, object]`, `inspect_package(path: Path) -> dict[str, object]`, and immutable slide metadata used by the builder.

- [ ] **Step 1: Write failing evidence and template tests**

```python
def test_evidence_preserves_status_semantics(repo_root):
    evidence = load_deck_evidence(repo_root)
    assert evidence["r1"]["successful_points"] == 100
    assert evidence["parameter_fit"]["accepted"] is False
    assert evidence["pilot"]["fft"]["valid"] is False
    assert evidence["pilot"]["stream"]["gain_pct"] == pytest.approx(1.636449)
    assert evidence["transient_rom"]["status"] == "PENDING"

def test_source_template_is_widescreen_and_has_theme(source_pptx):
    report = inspect_package(source_pptx)
    assert report["slide_count"] == 18
    assert report["aspect_ratio"] == pytest.approx(16 / 9, rel=0.01)
    assert report["theme_count"] >= 1
```

- [ ] **Step 2: Run tests and confirm they fail because the presentation modules do not exist**

Run: `pytest -q tests/test_clip3d_presentation.py -k 'evidence or template'`

Expected: collection failure naming `tools.presentation.clip3d_deck_data` or `inspect_pptx`.

- [ ] **Step 3: Implement strict evidence loading and OOXML inspection**

```python
def load_deck_evidence(repo_root: Path) -> dict[str, object]:
    return {
        "r1": load_r1_summary(repo_root / "runs/architecture_sweep/r1/paper/summary.csv"),
        "parameter_fit": load_parameter_report(repo_root / "results/parameter_studies/raw_power_strict_20260730/proxy_train_16/calibration_report.json"),
        "pilot": load_pilot(repo_root / "runs/discrete_partition_validation/balanced5_midcache_20260809/summary.csv"),
        "transient": load_transient(repo_root),
        "transient_rom": {"status": "PENDING", "measured_bips2_trans": None},
    }

def inspect_package(path: Path) -> dict[str, object]:
    with zipfile.ZipFile(path) as package:
        presentation = ET.fromstring(package.read("ppt/presentation.xml"))
        # Return slide count, page dimensions, aspect ratio, themes, masters,
        # layouts, fonts/colors, and all visible text by slide.
```

Reject missing evidence fields rather than silently substituting values. Encode the approved validity map explicitly: FFT and CHOLESKY invalid; STREAM, MATMUL, and STENCIL valid.

- [ ] **Step 4: Run focused tests and inspect the source package report**

Run: `pytest -q tests/test_clip3d_presentation.py -k 'evidence or template'`

Expected: all selected tests pass and the source reports 18 slides, 16:9 geometry, at least one theme and one master.

- [ ] **Step 5: Commit the evidence boundary and inspector**

```bash
git add tools/presentation/clip3d_deck_data.py tools/presentation/inspect_pptx.py tests/test_clip3d_presentation.py
git commit -m "test: define CLIP-3D presentation evidence boundary"
```

### Task 2: Deterministic Charts and Formula Assets

**Files:**
- Create: `tools/presentation/render_clip3d_assets.py`
- Create: `docs/presentation_assets/clip3d_en/.gitkeep`
- Modify: `tests/test_clip3d_presentation.py`

**Interfaces:**
- Consumes: the dictionary from `load_deck_evidence(repo_root)`.
- Produces: `render_assets(evidence: dict[str, object], output_dir: Path) -> dict[str, Path]`, with stable keys `parameter_spearman`, `pilot_bips2`, `transient_timeseries`, `closed_loop`, `floorplan`, and formula-card keys used by slide assembly.

- [ ] **Step 1: Add failing asset tests**

```python
def test_rendered_assets_are_projection_ready(tmp_path, evidence):
    assets = render_assets(evidence, tmp_path)
    required = {"parameter_spearman", "pilot_bips2", "transient_timeseries", "closed_loop", "floorplan"}
    assert required <= assets.keys()
    for key in required:
        with Image.open(assets[key]) as image:
            assert image.width >= 1600
            assert image.height >= 800

def test_pilot_chart_marks_invalid_points(tmp_path, evidence):
    manifest = json.loads(render_assets(evidence, tmp_path)["manifest"].read_text())
    assert manifest["pilot_bips2"]["invalid"] == ["FFT", "CHOLESKY"]
```

- [ ] **Step 2: Run asset tests and verify failure**

Run: `MPLCONFIGDIR=/tmp/clip3d-ppt-mpl pytest -q tests/test_clip3d_presentation.py -k rendered_assets`

Expected: failure because `render_assets` is undefined.

- [ ] **Step 3: Implement charts, diagrams, and formula cards**

```python
PALETTE = {
    "navy": "#0B1020", "blue": "#4F8CFF", "cyan": "#35D4FF",
    "green": "#4ADE80", "amber": "#FFB454", "purple": "#A78BFA",
    "red": "#FF6B6B", "text": "#F4F7FC", "gray": "#8792A6",
}

def render_assets(evidence: dict[str, object], output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    paths["parameter_spearman"] = render_spearman_chart(evidence, output_dir)
    paths["pilot_bips2"] = render_pilot_chart(evidence, output_dir, hatch_invalid=True)
    paths["transient_timeseries"] = render_transient_chart(evidence, output_dir)
    paths.update(render_method_diagrams(output_dir))
    paths.update(render_formula_cards(output_dir))
    paths["manifest"] = write_asset_manifest(evidence, paths, output_dir)
    return paths
```

Use the deck palette, transparent or navy backgrounds, 18 pt-equivalent labels, six-decimal transient annotations, a red `0.8` Spearman threshold, and gray hatching for invalid pilot rows. Render equations through Matplotlib mathtext at high DPI so PowerPoint never substitutes mathematical glyphs.

- [ ] **Step 4: Run asset tests and manually inspect a contact sheet**

Run: `MPLCONFIGDIR=/tmp/clip3d-ppt-mpl pytest -q tests/test_clip3d_presentation.py -k rendered_assets`

Run: `MPLCONFIGDIR=/tmp/clip3d-ppt-mpl python -m tools.presentation.render_clip3d_assets --repo-root . --output-dir docs/presentation_assets/clip3d_en`

Expected: tests pass; the asset manifest records exact evidence values and validity states.

- [ ] **Step 5: Commit deterministic visual assets**

```bash
git add tools/presentation/render_clip3d_assets.py docs/presentation_assets/clip3d_en tests/test_clip3d_presentation.py
git commit -m "feat: render CLIP-3D presentation charts and equations"
```

### Task 3: Twenty-Slide OOXML Deck Assembly

**Files:**
- Create: `tools/presentation/pptx_ooxml.py`
- Create: `tools/presentation/build_clip3d_en.py`
- Create: `docs/CLIP-3D_Reproduction_Methodology_and_Experiments_EN.pptx`
- Modify: `tests/test_clip3d_presentation.py`

**Interfaces:**
- Consumes: `source_pptx: Path`, `output_pptx: Path`, the asset map from Task 2, and a `SlideSpec` sequence containing exactly 20 slide definitions.
- Produces: `build_deck(source_pptx: Path, output_pptx: Path, repo_root: Path) -> Path`; preserves source theme/master/media relationships while replacing the presentation slide list with exactly twenty generated slides.

- [ ] **Step 1: Add failing package and content-contract tests**

```python
def test_built_deck_contract(built_pptx):
    report = inspect_package(built_pptx)
    assert report["slide_count"] == 20
    assert report["aspect_ratio"] == pytest.approx(16 / 9, rel=0.01)
    assert report["broken_relationships"] == []
    assert report["titles"] == EXPECTED_TITLES
    visible = "\n".join(report["visible_text"])
    assert "PENDING PERIODIC R1" in visible
    assert "NO MEASURED BIPS2_TRANS YET" in visible
    assert "REJECTED FOR FORMAL PROMOTION" in visible
    assert not re.search(r"[\u4e00-\u9fff]", visible)

def test_builder_does_not_modify_source(source_pptx, tmp_path):
    before = sha256(source_pptx)
    build_deck(source_pptx, tmp_path / "out.pptx", repo_root())
    assert sha256(source_pptx) == before
```

- [ ] **Step 2: Run package tests and verify failure**

Run: `pytest -q tests/test_clip3d_presentation.py -k 'built_deck or builder'`

Expected: failure because the OOXML builder does not exist.

- [ ] **Step 3: Implement reusable OOXML primitives**

```python
@dataclass(frozen=True)
class SlideSpec:
    number: int
    title: str
    kind: str
    blocks: tuple[ContentBlock, ...]
    status: str | None = None

class PptxPackage:
    def clone_template_slide(self, template_slide_number: int) -> SlideEditor: ...
    def add_text(self, slide: SlideEditor, box: Box, text: str, style: TextStyle) -> None: ...
    def add_image(self, slide: SlideEditor, box: Box, image_path: Path) -> None: ...
    def add_badge(self, slide: SlideEditor, text: str, color: str) -> None: ...
    def write(self, output_path: Path) -> None: ...
```

Preserve `[Content_Types].xml`, theme, master, layout, notes-master, and page-size parts from the source. Generate unique slide IDs, relationship IDs, shape IDs, and media names. Reuse the source footer/title visual treatment and use source-compatible layouts rather than generating a foreign theme.

- [ ] **Step 4: Define the exact twenty-slide storyboard and build the deck**

```python
SLIDE_META = (
    (1, "CLIP-3D Reproduction: Methodology, Experiments, and Open Issues", "title", None),
    (2, "Executive Summary", "cards", None),
    (3, "Why Architecture-Only Exploration Fails", "causal_diagram", None),
    (4, "The CLIP-3D Closed Loop", "process", None),
    (5, "Reproduction Inputs, Tools, and R1", "inventory", None),
    (6, "Power, Area, and Cache Characterization", "method", None),
    (7, "Floorplanning Variables and Physical Constraints", "floorplan", None),
    (8, "From Layout and Power to Temperature", "method", None),
    (9, "Deriving Sustainable Frequency", "derivation", None),
    (10, "From IPC1 to Measured BIPS2", "equation_chain", None),
    (11, "Thermal Proxy Objective", "derivation", None),
    (12, "Wire Delay, Integer Cycles, and R2", "derivation", None),
    (13, "Experiment 1: Parameter Identification Method", "experiment_method", "MEASURED · NON-FORMAL"),
    (14, "Experiment 1: Parameter Results", "experiment_result", "REJECTED FOR FORMAL PROMOTION"),
    (15, "Experiment 2: End-to-End Validation Method", "experiment_method", "MEASURED"),
    (16, "Experiment 2: End-to-End Results", "experiment_result", "MEASURED · PARTIAL VALIDITY"),
    (17, "Transient Thermal Formulation", "derivation", None),
    (18, "Experiment 3: Transient HotSpot Validation", "experiment_result", "MEASURED · NON-FORMAL"),
    (19, "Experiment 4: Transient ROM Closed Loop", "experiment_pending", "PENDING"),
    (20, "Conclusions and Questions", "conclusion", None),
)

SLIDES = build_slide_specs(SLIDE_META, content_contract=DESIGN_SPEC_CONTENT)

def build_deck(source_pptx: Path, output_pptx: Path, repo_root: Path) -> Path:
    evidence = load_deck_evidence(repo_root)
    assets = render_assets(evidence, repo_root / "docs/presentation_assets/clip3d_en")
    assert tuple(item.title for item in SLIDES) == EXPECTED_TITLES
    package = PptxPackage.from_template(source_pptx)
    for slide in SLIDES:
        compose_slide(package, slide, assets, evidence)
    package.write(output_pptx)
    return output_pptx
```

Run: `MPLCONFIGDIR=/tmp/clip3d-ppt-mpl python -m tools.presentation.build_clip3d_en --source docs/CLIP-3D汇报.pptx --output docs/CLIP-3D_Reproduction_Methodology_and_Experiments_EN.pptx`

Expected: a macro-free `.pptx` with exactly 20 English slides and no broken package relationships.

- [ ] **Step 5: Run deck contract tests**

Run: `pytest -q tests/test_clip3d_presentation.py -k 'built_deck or builder'`

Expected: all selected tests pass, including source SHA-256 preservation.

- [ ] **Step 6: Commit deck builder and generated presentation**

```bash
git add tools/presentation/pptx_ooxml.py tools/presentation/build_clip3d_en.py tests/test_clip3d_presentation.py docs/CLIP-3D_Reproduction_Methodology_and_Experiments_EN.pptx
git commit -m "feat: build twenty-slide English CLIP-3D presentation"
```

### Task 4: Detailed Chinese Speaker Script

**Files:**
- Create: `tools/presentation/build_clip3d_notes.py`
- Create: `docs/CLIP-3D_Reproduction_Speaker_Notes_ZH.md`
- Modify: `tests/test_clip3d_presentation.py`

**Interfaces:**
- Consumes: the same 20 `SlideSpec` titles and evidence model used by the deck.
- Produces: `build_notes(output_path: Path, evidence: dict[str, object]) -> Path`, with exactly one section per slide and the five required subsections.

- [ ] **Step 1: Add failing notes contract test**

```python
def test_chinese_notes_match_deck(notes_path):
    text = notes_path.read_text(encoding="utf-8")
    sections = re.findall(r"^## Slide (\d{2}) — (.+)$", text, re.M)
    assert [title for _, title in sections] == EXPECTED_TITLES
    assert len(sections) == 20
    for heading in ("Purpose", "Suggested narration", "Formula and variable explanation", "Evidence and caveats", "Transition to the next slide"):
        assert text.count(f"### {heading}") == 20
```

- [ ] **Step 2: Run the notes test and verify failure**

Run: `pytest -q tests/test_clip3d_presentation.py -k chinese_notes`

Expected: failure because the notes file and generator do not exist.

- [ ] **Step 3: Implement detailed notes from the shared storyboard**

```python
def build_notes(output_path: Path, evidence: dict[str, object]) -> Path:
    sections = []
    for slide in SLIDES:
        note = NOTES_BY_SLIDE[slide.number]
        sections.append(render_note_section(slide, note, evidence))
    output_path.write_text("\n\n".join(sections) + "\n", encoding="utf-8")
    return output_path
```

For formula slides, define every symbol, derivation assumption, clipping rule, and integer-cycle consequence. For experiment slides, document inputs, outputs, measured numbers, acceptance semantics, and forbidden claims. For slide 19, explicitly explain why periodic R1 is the blocking evidence and why no `BIPS2_trans` number is displayed.

- [ ] **Step 4: Generate and validate the Chinese script**

Run: `python -m tools.presentation.build_clip3d_notes --repo-root . --output docs/CLIP-3D_Reproduction_Speaker_Notes_ZH.md`

Run: `pytest -q tests/test_clip3d_presentation.py -k chinese_notes`

Expected: exactly 20 ordered sections, each containing all five required subsections.

- [ ] **Step 5: Commit the speaker script**

```bash
git add tools/presentation/build_clip3d_notes.py docs/CLIP-3D_Reproduction_Speaker_Notes_ZH.md tests/test_clip3d_presentation.py
git commit -m "docs: add detailed Chinese CLIP-3D speaker notes"
```

### Task 5: Final Package Verification and Visual Audit

**Files:**
- Create: `tools/presentation/validate_clip3d_deck.py`
- Create: `docs/presentation_assets/clip3d_en/validation_report.json`
- Modify: `tests/test_clip3d_presentation.py`

**Interfaces:**
- Consumes: source PPT, generated PPT, generated notes, and asset manifest.
- Produces: `validate_delivery(source: Path, deck: Path, notes: Path) -> dict[str, object]` and a machine-readable validation report with `passed: true` only when every contract check succeeds.

- [ ] **Step 1: Add failing delivery validation test**

```python
def test_delivery_validation_passes(delivery_report):
    assert delivery_report["passed"] is True
    assert delivery_report["source_unchanged"] is True
    assert delivery_report["slide_count"] == 20
    assert delivery_report["english_visible_text"] is True
    assert delivery_report["relationship_errors"] == []
    assert delivery_report["notes_sections"] == 20
```

- [ ] **Step 2: Run the delivery test and verify failure**

Run: `pytest -q tests/test_clip3d_presentation.py -k delivery_validation`

Expected: failure because `validate_delivery` is undefined.

- [ ] **Step 3: Implement full package and semantic validation**

```python
def validate_delivery(source: Path, deck: Path, notes: Path) -> dict[str, object]:
    deck_report = inspect_package(deck)
    checks = {
        "source_unchanged": source.exists() and source != deck,
        "slide_count": deck_report["slide_count"],
        "english_visible_text": not contains_cjk(deck_report["visible_text"]),
        "relationship_errors": deck_report["broken_relationships"],
        "notes_sections": count_notes_sections(notes),
        "required_statuses": required_statuses_present(deck_report["visible_text"]),
        "evidence_values": compare_visible_values_to_manifest(deck_report),
    }
    checks["passed"] = all_contract_checks_pass(checks)
    return checks
```

Also check that slide titles and notes titles match exactly, all three experiment classes plus transient ROM are present, all charts exist inside the package, the deck is macro-free, and no accepted aggregate is computed from invalid points.

- [ ] **Step 4: Run complete tests and write the validation report**

Run: `MPLCONFIGDIR=/tmp/clip3d-ppt-mpl pytest -q tests/test_clip3d_presentation.py`

Run: `python -m tools.presentation.validate_clip3d_deck --source docs/CLIP-3D汇报.pptx --deck docs/CLIP-3D_Reproduction_Methodology_and_Experiments_EN.pptx --notes docs/CLIP-3D_Reproduction_Speaker_Notes_ZH.md --output docs/presentation_assets/clip3d_en/validation_report.json`

Expected: all presentation tests pass and the report contains `"passed": true`.

- [ ] **Step 5: Inspect all slides as rendered images when a renderer is available, otherwise inspect generated slide thumbnails/contact sheet**

Run: `python -m tools.presentation.validate_clip3d_deck --source docs/CLIP-3D汇报.pptx --deck docs/CLIP-3D_Reproduction_Methodology_and_Experiments_EN.pptx --notes docs/CLIP-3D_Reproduction_Speaker_Notes_ZH.md --output docs/presentation_assets/clip3d_en/validation_report.json --contact-sheet docs/presentation_assets/clip3d_en/contact_sheet.png`

Expected: 20 ordered thumbnails with no clipped titles, overlapping blocks, unreadable charts, or unexplained placeholders. If native PowerPoint rendering is unavailable, the report must state `native_rendering: unavailable` rather than claiming pixel-perfect rendering.

- [ ] **Step 6: Commit verification artifacts**

```bash
git add tools/presentation/validate_clip3d_deck.py tests/test_clip3d_presentation.py docs/presentation_assets/clip3d_en/validation_report.json
git commit -m "test: validate CLIP-3D presentation delivery"
```
