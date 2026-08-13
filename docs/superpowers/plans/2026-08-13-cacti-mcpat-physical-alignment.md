# CACTI/McPAT Physical Alignment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the steady-state pipeline use one auditable gem5-derived cache organization for McPAT and local CACTI, ceiling-rounded CACTI latency, and physically unscaled McPAT/CACTI areas.

**Architecture:** Add a small shared cache-contract module that converts R1 metadata and explicit physical-model controls into distinct L1I/L1D/L2 organizations. Both McPAT XML generation and standalone CACTI characterization consume this contract; floorplanning and R2 validate and consume the same characterization artifact. Remove all 150 mm² scaling from production APIs/configurations, then generate a provenance-complete local Table-II-equivalent report.

**Tech Stack:** Python 3.12, `unittest`, local CACTI 6.5 executable, McPAT 1.3, JSON/CSV/Markdown artifacts, gem5 R1 metadata.

## Global Constraints

- Existing R1 outputs and historical run directories must not be modified.
- Paper Table II values are comparison-only and must never be workflow inputs.
- Cache geometry and access latency come only from the local CACTI run.
- Non-cache area comes directly from McPAT; no area or dimension scaling is allowed.
- The official default cache contract is 45 nm, one bank per gem5 cache object, 64-byte lines, 512-bit L2 path width, normal access mode, ECC enabled, and the configured McPAT operating temperature.
- Cache cycles use `max(1, ceil(access_time_ns / clock_period_ns))` with tolerance only at floating-point integer boundaries.
- Existing unrelated dirty-worktree files must be preserved.

---

### Task 1: Shared cache organization and correct local CACTI characterization

**Files:**
- Create: `workflow/cache_contract.py`
- Modify: `workflow/cacti/characterize_cache.py`
- Modify: `configs/gem5/clip_r1.py`
- Test: `tests/test_cache_alignment.py`

**Interfaces:**
- Produces: `build_cache_contract(metadata: dict, *, technology_nm: int, temperature_k: int, device_type: int, interconnect_projection_type: int) -> dict`.
- Produces: `cache_access_cycles(access_time_ns: float, frequency_ghz: float) -> int`.
- Changes: `characterize(..., contracts: list[dict] | None = None, ...) -> dict` records each distinct `l1i`, `l1d`, and `l2` organization plus artifact provenance.

- [ ] Write tests proving 2.02→3, 6.28→7 and exact 3.0→3; distinct L1I/L1D records; and generated CACTI directives for size, line, associativity, one bank, width, 45 nm, 320 K, normal access and ECC.
- [ ] Run `python -m unittest tests.test_cache_alignment -v` and verify failures are caused by the missing contract/ceiling behavior.
- [ ] Add explicit R1 metadata for L1/L2 bank counts and path widths while retaining backward-compatible defaults for already completed R1 outputs.
- [ ] Implement the contract and use it to generate one isolated CACTI config per cache level and size.
- [ ] Add SHA-256 identities for executable, base config, generated config and raw output, plus CACTI Git revision where available.
- [ ] Rerun `python -m unittest tests.test_cache_alignment -v` and verify PASS.
- [ ] Commit only Task 1 files with `git commit -m "fix: align local CACTI cache characterization"`.

### Task 2: Make McPAT consume and report the same cache contract

**Files:**
- Modify: `workflow/mcpat/gem5_to_mcpat.py`
- Modify: `workflow/run_lifting_pipeline.py`
- Test: `tests/test_cache_alignment.py`

**Interfaces:**
- Consumes: `build_cache_contract(...)` from Task 1.
- Produces: `mapping_report.json.cache_contract` with the exact organizations encoded in McPAT XML.
- Changes: `convert(..., settings=...)` validates McPAT cache encodings against the shared contract.

- [ ] Add failing XML-inspection tests requiring L1I/L1D/L2 capacity, line size, associativity, bank count and output width to equal their contract records.
- [ ] Run the focused tests and verify the legacy L2 `8,...,32` encoding causes the expected failure.
- [ ] Replace independent McPAT cache constants with values from the shared contract while retaining the documented McPAT-only synthetic throughput/latency constraints.
- [ ] Pass cache physical settings from the experiment configuration into both McPAT conversion and CACTI characterization.
- [ ] Run a real local McPAT smoke using an existing R1 point; if the aligned organization is rejected, stop with the complete McPAT diagnostic rather than weakening the contract.
- [ ] Rerun focused tests and verify PASS.
- [ ] Commit Task 2 files with `git commit -m "fix: share cache organization with McPAT"`.

### Task 3: Remove all global area scaling from the executable workflow

**Files:**
- Modify: `workflow/floorplan/build_module_model.py`
- Modify: `workflow/run_lifting_pipeline.py`
- Modify: `configs/experiments/*.json`
- Modify: `tests/test_workflow.py`
- Modify: `tests/test_cache_alignment.py`

**Interfaces:**
- Changes: `apply_physical_areas(modules, metadata, cacti) -> list[dict]` uses CACTI cache geometry verbatim and McPAT non-cache area verbatim.
- Changes: `build_model(..., require_communication_profile=False) -> dict` has no scale argument or calibration report.

- [ ] Add failing tests asserting exact CACTI cache areas/dimensions, exact McPAT non-cache areas, no scale-related output keys, and rejection of scale controls in official configurations.
- [ ] Run focused tests and verify failure against the current 3.278× scaling path.
- [ ] Remove scaling constants, arithmetic, CLI option, pipeline config reads and summary fields; rename retained diagnostic McPAT cache area to `mcpat_reported_area_mm2`.
- [ ] Remove `area_reference_mm2`, `area_reference_raw_mm2` and `area_reference_basis` from every official experiment configuration and update provenance language that claimed a 150 mm² calibration.
- [ ] Update existing workflow assertions to the unscaled physical values.
- [ ] Run `python -m unittest tests.test_cache_alignment tests.test_workflow -v` and verify PASS.
- [ ] Commit Task 3 files with `git commit -m "fix: preserve physical McPAT and CACTI areas"`.

### Task 4: Enforce one characterization identity for geometry and R2 delay

**Files:**
- Modify: `workflow/floorplan/build_module_model.py`
- Modify: `workflow/r2/build_latency_vector.py`
- Modify: `workflow/run_lifting_pipeline.py`
- Test: `tests/test_cache_alignment.py`
- Test: `tests/test_workflow.py`

**Interfaces:**
- Produces: cache module fields `cacti_record_id` and `cacti_characterization_sha256`.
- Produces: matching R2 provenance fields for each selected cache record.
- Produces: `validate_characterization(cacti: dict, expected_contract: dict) -> None` with field-specific errors.

- [ ] Add failing tests for missing, duplicate, wrong-size, wrong-associativity, wrong-bank, wrong-width, wrong-temperature, non-ceiling and artifact-identity mismatches.
- [ ] Run focused tests and verify each new test fails for its intended missing validation.
- [ ] Implement contract validation and stable per-record/artifact identities.
- [ ] Make module construction and R2 vector construction reject mismatches and record identical selected-record identities.
- [ ] Rerun focused tests and verify PASS.
- [ ] Commit Task 4 files with `git commit -m "fix: enforce CACTI artifact identity"`.

### Task 5: Reproduce the local Table-II-equivalent measurements and document reruns

**Files:**
- Create: `scripts/characterize_local_table_ii.py`
- Create: `data/cacti/local_45nm_table_ii_equivalent.json`
- Create: `data/cacti/local_45nm_table_ii_equivalent.csv`
- Create: `docs/local_cacti_table_ii_zh.md`
- Modify: `workflow/README.md`
- Test: `tests/test_cache_alignment.py`

**Interfaces:**
- Produces: a nine-row JSON/CSV dataset for L1 16–128 kB and L2 128 kB–2 MB at 2 GHz.
- Consumes: Task 1 characterization and provenance fields; does not embed paper values in executable code.

- [ ] Add a failing test that invokes the report writer on fixture records and checks row count, ordering, ceiling cycles and provenance columns.
- [ ] Run the report test and verify it fails because the script/report API is absent.
- [ ] Implement the report writer and Chinese methodology document, including the exact reproduction command and the R1-preserving downstream rerun boundary.
- [ ] Run the local CACTI executable to generate the nine authoritative measurements under `data/cacti/`.
- [ ] Run `python -m unittest tests.test_cache_alignment tests.test_workflow -v` and verify PASS.
- [ ] Run the complete applicable suite with `python -m unittest discover -s tests -v`; record any pre-existing unrelated transient failures separately and do not alter their dirty files.
- [ ] Run one existing-R1 steady-state smoke through R2 in a new output directory and inspect that cache record IDs match and no scale fields exist.
- [ ] Commit Task 5 files with `git commit -m "docs: publish local CACTI characterization"`.

### Task 6: Final audit

**Files:**
- Inspect: all files changed in Tasks 1–5

**Interfaces:**
- Produces: evidence that the approved design and acceptance criteria are met.

- [ ] Run `rg -n "DEFAULT_AREA_SCALE|DEFAULT_REFERENCE_RAW_AREA|area_reference_mm2|area_reference_raw_mm2|area_reference_basis|area_before_global_scale_mm2|area_calibration|--area-scale" workflow configs tests workflow/README.md docs/local_cacti_table_ii_zh.md` and verify no production/configuration scaling remnants remain.
- [ ] Run fresh focused tests, full applicable tests, local table generation and steady-state smoke verification.
- [ ] Inspect `git diff --check`, `git status --short`, generated provenance hashes and the smoke `pipeline_summary.json`.
- [ ] Invoke `superpowers:finishing-a-development-branch` and present integration choices without touching unrelated main-worktree changes.
