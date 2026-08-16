# McPAT-Native Short-ROI Thermal Corrections Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax for tracking.

**Goal:** Replace the standalone-CACTI execution contract with McPAT-native cache metrics, enforce fixed fine-grained thermal modules and audited non-SuperLU HotSpot mapping, and add a convergence-selected short gem5/transient protocol.

**Architecture:** A patched McPAT emits machine-readable metrics from its embedded CACTI-P model; one reusable McPAT runner turns those metrics and detailed power blocks into the sole physical-model artifact consumed by layout and R2. HotSpot receives native module rectangles plus zero-power whitespace and validates area-overlap mapping. A protocol identity isolates legacy 500M runs from new all-core short candidates, while transient sampling is derived from measured ROI duration.

**Tech Stack:** Python 3 standard library, gem5 Python configuration, McPAT 1.3/C++, embedded CACTI-P 6.5, HotSpot C detailed-3D grid model, unittest, JSON/CSV artifacts, POSIX shell, git patches.

## Global Constraints

- Existing completed 100M-warmup/500M-measurement R1/R2 artifacts remain read-only legacy evidence.
- Standalone CACTI source and historical evidence may remain for diagnostics, but no corrected execution path may require or invoke it.
- Cache area and power come from aggregate McPAT output; embedded CACTI-P supplies only timing and raw array aspect ratio.
- Exactly four detailed cores are required in the corrected reproduction profile; aggregate coreN_logic fallback is forbidden there.
- Core functional blocks are fixed; only the one shared L2 is movable.
- Thermal proxy role is search-heuristic; it is never a source of reportable temperature, frequency, or BIPS.
- Same-tier physical overlap is illegal; cross-tier projected overlap is legal; partial block-to-grid coverage uses area averaging.
- HotSpot must be clean-built with SUPERLU=0 and MATHACCEL=none.
- New short candidates use 2,000,000 warmup instructions, 1M/2M/5M/10M all-core measurement targets, and distinct protocol identities.
- Default transient sampling is min(0.5 ms, measured ROI duration / 50); explicit overrides have distinct provenance.
- Every change follows red-green-refactor, preserves partial tool logs on failure, and commits only scoped files.

## File and responsibility map

- patches/mcpat/0001-emit-embedded-cacti-p-metrics.patch: auditable vendor-source change that exposes existing internal cache results.
- scripts/build_mcpat.sh: idempotently applies the McPAT patch, clean-builds, and records binary provenance.
- workflow/mcpat/cache_metrics.py: parses/validates embedded CACTI-P records and McPAT patch capability.
- workflow/mcpat/run_mcpat.py: owns XML conversion, one McPAT invocation, strict parsing, and provenance.
- workflow/mcpat/parse_mcpat.py: extracts detailed functional blocks and binds cache records.
- workflow/floorplan/build_module_model.py: builds one McPAT-authoritative physical model and mobility contract.
- workflow/r2/build_latency_vector.py: converts embedded access seconds to integer gem5 cycles.
- workflow/floorplan/hotspot_audit.py: re-parses emitted floorplans/traces and audits geometry and power.
- workflow/thermal/hotspot_validation.py: shared steady/transient HotSpot log and grid-output validation.
- workflow/thermal/hotspot_toolchain.py: verifies non-SuperLU/non-vendor-math build provenance.
- workflow/r1_protocol.py: canonical protocol object and stable identity.
- workflow/analysis/select_short_roi.py: evaluates 1M/2M/5M candidates against 10M.
- workflow/transient/sampling.py: derives and identifies the transient sampling policy.
- tests/test_mcpat_native_cache.py: McPAT patch, parser, model, R2, and pipeline contract tests.
- tests/test_hotspot_contract.py: geometry, serialization, execution-warning, and build-provenance tests.
- tests/test_short_roi.py: short protocol, convergence, and cache-isolation tests.

---

### Task 1: Emit embedded CACTI-P metrics from McPAT

**Files:**
- Create: patches/mcpat/0001-emit-embedded-cacti-p-metrics.patch
- Create: scripts/build_mcpat.sh
- Create: tests/test_mcpat_native_cache.py
- Modify: docs/DOWNLOAD_TOOLS.md

**Interfaces:**
- Produces marker CLIP_MCPAT_CACTI_P_V1 in patched McPAT output and binary.
- Produces tools/build/mcpat/build_provenance.json bound to the binary and patch hashes.

- [ ] **Step 1: Write failing patch-content and build-contract tests**

~~~python
class McPATPatchTests(unittest.TestCase):
    def test_patch_names_exact_internal_results(self):
        text = PATCH.read_text()
        self.assertIn("cores[i]->ifu->icache.caches->local_result", text)
        self.assertIn("cores[i]->lsu->dcache.caches->local_result", text)
        self.assertIn("l2array[i]->unicache.caches->local_result", text)
        self.assertIn("CLIP_MCPAT_CACTI_P_V1", text)
        self.assertIn("1e-3", text)  # cache_ht/cache_len um -> mm

    def test_build_script_requires_clean_rebuild(self):
        text = BUILD.read_text()
        self.assertIn("make -C tools/src/mcpat clean", text)
        self.assertIn("0001-emit-embedded-cacti-p-metrics.patch", text)
~~~

- [ ] **Step 2: Run tests and verify the missing files fail**

Run:

~~~bash
python -m unittest tests.test_mcpat_native_cache.McPATPatchTests -v
~~~

Expected: FAIL because the patch and build script do not exist.

- [ ] **Step 3: Add the McPAT patch**

Patch Processor::displayEnergy at print level 5, using the already-computed local_result values. Emit four L1I, four L1D, and every shared L2 record using 17-digit scientific notation:

~~~cpp
cout << "CLIP_MCPAT_CACTI_P_V1"
     << " cache=" << cache_name
     << " core=" << core_index
     << " access_time_s=" << result.access_time
     << " cycle_time_s=" << result.cycle_time
     << " height_mm=" << result.cache_ht * 1e-3
     << " width_mm=" << result.cache_len * 1e-3
     << " mcpat_version=1.3 model=embedded-cacti-p"
     << endl;
~~~

Save and restore ostream flags/precision. Do not alter CACTI-P inputs, cost functions, or cache construction.

- [ ] **Step 4: Add an idempotent clean-build script**

The script must reverse-dry-run the patch to detect an already patched tree, apply it only when needed, run make clean before make, verify the marker with strings, and write JSON provenance containing patch SHA-256, binary SHA-256, build commands, and timestamps. It must not modify tracked vendor source.

- [ ] **Step 5: Run tests, patch dry-run, and a bounded build**

Run:

~~~bash
python -m unittest tests.test_mcpat_native_cache.McPATPatchTests -v
patch -d tools/src/mcpat -p1 --dry-run < patches/mcpat/0001-emit-embedded-cacti-p-metrics.patch
scripts/build_mcpat.sh
strings tools/src/mcpat/mcpat | rg CLIP_MCPAT_CACTI_P_V1
~~~

Expected: tests PASS; patch either applies or is detected as already applied; marker is present.

- [ ] **Step 6: Commit**

~~~bash
git add patches/mcpat/0001-emit-embedded-cacti-p-metrics.patch scripts/build_mcpat.sh tests/test_mcpat_native_cache.py docs/DOWNLOAD_TOOLS.md
git commit -m "build: expose embedded CACTI-P metrics from McPAT"
~~~

### Task 2: Parse strict McPAT-native cache records

**Files:**
- Create: workflow/mcpat/cache_metrics.py
- Modify: workflow/mcpat/parse_mcpat.py
- Modify: tests/test_mcpat_native_cache.py
- Modify: tests/test_workflow.py

**Interfaces:**
- Produces parse_embedded_cacti_records(text: str, expected_core_count: int = 4) -> list[dict].
- Extends parse_mcpat_text with expected_core_count, require_granular_cores, and require_embedded_cacti keyword arguments.
- Extracts the current power-block body into _parse_mcpat_power_blocks(text: str) -> dict and adds require_exact_core_count(result: dict, expected: int) and require_detailed_functional_blocks(result: dict) validators used by the public parser.

- [ ] **Step 1: Write failing parser tests**

~~~python
def test_accepts_exact_four_core_record_set(self):
    records = parse_embedded_cacti_records(native_text(), expected_core_count=4)
    self.assertEqual(len(records), 9)
    self.assertEqual(
        {(r["cache"], r["core"]) for r in records},
        {*(("l1i", i) for i in range(4)),
         *(("l1d", i) for i in range(4)),
         ("l2", None)},
    )

def test_rejects_duplicate_nonfinite_and_missing_records(self):
    for broken in (duplicate_l1i(), nan_l2(), missing_core3_l1d()):
        with self.assertRaises(ValueError):
            parse_embedded_cacti_records(broken, expected_core_count=4)
~~~

Also test that formal parsing rejects aggregate core fallback and that legacy synthetic parsing works only with both strict flags false.

- [ ] **Step 2: Run parser tests and verify failure**

~~~bash
python -m unittest tests.test_mcpat_native_cache.EmbeddedCACTIParserTests -v
~~~

Expected: FAIL because the parser/API does not exist.

- [ ] **Step 3: Implement strict parsing and stable record identity**

Use a full-line regular expression for CLIP_MCPAT_CACTI_P_V1. Convert seconds and millimetres to finite positive floats. Require the exact set of records. Compute record_id as the SHA-256 of sorted canonical scientific fields; do not include filesystem paths.

Return records under mcpat.json key embedded_cacti_p and record:

~~~python
{
    "schema_version": 1,
    "authority": "McPAT 1.3 embedded CACTI-P",
    "records": records,
}
~~~

- [ ] **Step 4: Enforce detailed four-core output in formal parsing**

Add:

~~~python
def parse_mcpat_text(
    text: str,
    *,
    expected_core_count: int | None = None,
    require_granular_cores: bool = False,
    require_embedded_cacti: bool = False,
) -> dict:
    result = parse_existing_power_and_area_blocks(text)
    if expected_core_count is not None:
        require_exact_core_count(result, expected_core_count)
    if require_granular_cores:
        require_detailed_functional_blocks(result)
    if require_embedded_cacti:
        result["embedded_cacti_p"] = {
            "schema_version": 1,
            "authority": "McPAT 1.3 embedded CACTI-P",
            "records": parse_embedded_cacti_records(
                text, expected_core_count or 4
            ),
        }
    return result
~~~

When strict, reject any coreN_logic module, missing detailed heading, wrong core count, or missing embedded record.

- [ ] **Step 5: Run focused and parser regressions**

~~~bash
python -m unittest tests.test_mcpat_native_cache.EmbeddedCACTIParserTests tests.test_workflow.ParserTests -v
~~~

Expected: PASS.

- [ ] **Step 6: Commit**

~~~bash
git add workflow/mcpat/cache_metrics.py workflow/mcpat/parse_mcpat.py tests/test_mcpat_native_cache.py tests/test_workflow.py
git commit -m "feat: parse strict McPAT-native cache metrics"
~~~

### Task 3: Extract one reusable strict McPAT runner

**Files:**
- Create: workflow/mcpat/run_mcpat.py
- Modify: workflow/mcpat/gem5_to_mcpat.py
- Modify: tests/test_mcpat_native_cache.py
- Modify: tests/test_workflow.py

**Interfaces:**
- Produces run_mcpat(r1_dir: Path, output_dir: Path, settings: dict, executable: Path = DEFAULT_MCPAT) -> dict.
- convert no longer accepts cache_characterization; its cache XML values are labelled optimization_constraints.

- [ ] **Step 1: Write a failing mocked-runner test**

~~~python
def test_runner_invokes_mcpat_once_and_writes_native_artifact(self):
    with patch("workflow.mcpat.run_mcpat.subprocess.run",
               return_value=CompletedProcess([], 0, stdout=native_text())) as run:
        result = run_mcpat(self.r1, self.out, {}, self.binary)
    self.assertEqual(run.call_count, 1)
    self.assertEqual(result["embedded_cacti_p"]["authority"],
                     "McPAT 1.3 embedded CACTI-P")
    self.assertTrue((self.out / "input.xml").is_file())
    self.assertTrue((self.out / "mcpat.out").is_file())
    self.assertTrue((self.out / "mcpat.json").is_file())
~~~

Add a test proving XML cache latency fields are reported only as optimization constraints, never as measured output.

- [ ] **Step 2: Run and verify failure**

~~~bash
python -m unittest tests.test_mcpat_native_cache.McPATRunnerTests -v
~~~

- [ ] **Step 3: Implement the runner**

The runner performs: validate patched binary -> convert R1 to XML -> invoke McPAT once at print level 5 -> always save combined output -> strict parse -> attach command, XML/mapping/output/binary/patch hashes -> write mcpat.json. A non-zero return, missing version header, missing marker, or strict parse error retains logs and raises.

- [ ] **Step 4: Remove standalone characterization from XML conversion**

Remove cache_characterization and cacti_characterization_id arguments/fields. Build cache sizes and organizations from R1 metadata. Preserve the existing 10/10 throughput/latency values only as McPAT optimization constraints and label them explicitly in mapping_report.json.

- [ ] **Step 5: Run tests**

~~~bash
python -m unittest tests.test_mcpat_native_cache.McPATRunnerTests tests.test_workflow.ParserTests -v
~~~

- [ ] **Step 6: Commit**

~~~bash
git add workflow/mcpat/run_mcpat.py workflow/mcpat/gem5_to_mcpat.py tests/test_mcpat_native_cache.py tests/test_workflow.py
git commit -m "refactor: centralize strict McPAT execution"
~~~

### Task 4: Build McPAT-authoritative granular modules

**Files:**
- Modify: workflow/floorplan/build_module_model.py
- Modify: workflow/cache_contract.py
- Modify: tests/test_mcpat_native_cache.py
- Modify: tests/test_cache_alignment.py
- Modify: tests/test_workflow.py

**Interfaces:**
- Produces apply_mcpat_cache_geometry(modules: list[dict], cache_records: list[dict]) -> list[dict].
- Changes build_model(r1_dir, mcpat_json, output, require_communication_profile=False, require_granular_cores=True) -> dict.
- Adds validate_mobility_contract(modules, expected_cores=4) -> dict.

- [ ] **Step 1: Write failing geometry and mobility tests**

~~~python
def test_preserves_mcpat_area_and_embedded_aspect_ratio(self):
    result = apply_mcpat_cache_geometry([cache_module(area=6.0)], [record(4.0, 2.0)])
    block = result[0]
    self.assertAlmostEqual(block["preferred_width_mm"] *
                           block["preferred_height_mm"], 6.0)
    self.assertAlmostEqual(block["preferred_width_mm"] /
                           block["preferred_height_mm"], 2.0)
    self.assertEqual(block["area_source"], "McPAT aggregate Area")

def test_contract_has_one_movable_l2_and_fixed_granular_cores(self):
    contract = validate_mobility_contract(granular_34_modules())
    self.assertEqual(contract["movable_names"], ["shared_l2"])
    self.assertEqual(len(contract["fixed_names"]), 33)
~~~

Test per-core L1 record matching, shared-L2 matching, zero dimensions, missing records, four-core requirement, and aggregate fallback rejection.

- [ ] **Step 2: Run and verify failure**

~~~bash
python -m unittest tests.test_mcpat_native_cache.McPATGeometryTests -v
~~~

- [ ] **Step 3: Replace external area overrides**

For cache modules use:

~~~python
ratio = record["width_mm"] / record["height_mm"]
width = math.sqrt(module["area_mm2"] * ratio)
height = module["area_mm2"] / width
~~~

Record raw_array_dimensions_mm, normalized_block_dimensions_mm, aspect_ratio, formula, and record_id. Remove source_cacti, cacti_characterization_id, and mcpat_reported_area_mm2 from corrected artifacts.

- [ ] **Step 4: Add explicit mobility and conservation metadata**

Set movable=true only on shared_l2. Validate required kinds per core and power/area conservation after cache subtraction. Store cache_authority, mobility_contract, module_schema, and McPAT hashes in modules.json.

- [ ] **Step 5: Run tests**

~~~bash
python -m unittest tests.test_mcpat_native_cache.McPATGeometryTests tests.test_cache_alignment tests.test_workflow.GridTests -v
~~~

- [ ] **Step 6: Commit**

~~~bash
git add workflow/floorplan/build_module_model.py workflow/cache_contract.py tests/test_mcpat_native_cache.py tests/test_cache_alignment.py tests/test_workflow.py
git commit -m "fix: build granular geometry from McPAT-native records"
~~~

### Task 5: Build R2 latency from embedded access time

**Files:**
- Modify: workflow/r2/build_latency_vector.py
- Modify: workflow/r2/run_wire_sensitivity.py
- Modify: tests/test_mcpat_native_cache.py
- Modify: tests/test_workflow.py

**Interfaces:**
- Produces access_cycles(access_time_s: float, frequency_hz: float) -> tuple[float, int].
- Changes build_vector(modules, output, tsv_hops=None, wire_cycles=None, layout_path=None, wire_rounding="nearest", cycles_per_tsv=2, l1_pipeline_cycles=1, wire_aggregation="mean") -> dict.

- [ ] **Step 1: Write failing conversion and provenance tests**

~~~python
def test_access_seconds_are_ceiled_at_nominal_frequency(self):
    raw, cycles = access_cycles(1.01e-9, 2.0e9)
    self.assertAlmostEqual(raw, 2.02)
    self.assertEqual(cycles, 3)

def test_exact_integer_boundary_does_not_add_a_cycle(self):
    self.assertEqual(access_cycles(1.0e-9, 2.0e9)[1], 2)
~~~

Also assert rejection of divergent per-core same-level L1 timing, and retention of separate arbitration/TSV/pipeline/wire terms.

- [ ] **Step 2: Run and verify failure**

~~~bash
python -m unittest tests.test_mcpat_native_cache.McPATLatencyTests -v
~~~

- [ ] **Step 3: Implement McPAT-native vector construction**

Read validated cache records embedded in modules.json. Require all four L1I records to agree and all four L1D records to agree. Convert at f0_hz. Rename component fields to l1i_mcpat_cacti_p, l1d_mcpat_cacti_p, and l2_mcpat_cacti_p. Store seconds, raw cycles, ceil policy, integer cycles, record IDs, output hash, binary hash, and authority under mcpat_cacti_p_provenance.

- [ ] **Step 4: Migrate wire-sensitivity caller**

Remove the cacti_path parameter and artifact checks. Preserve layout-derived wire-cycle behavior unchanged.

- [ ] **Step 5: Run tests**

~~~bash
python -m unittest tests.test_mcpat_native_cache.McPATLatencyTests tests.test_workflow.GridTests -v
~~~

- [ ] **Step 6: Commit**

~~~bash
git add workflow/r2/build_latency_vector.py workflow/r2/run_wire_sensitivity.py tests/test_mcpat_native_cache.py tests/test_workflow.py
git commit -m "fix: derive R2 cache cycles from embedded CACTI-P"
~~~

### Task 6: Remove standalone CACTI from steady and batch execution

**Files:**
- Modify: workflow/run_lifting_pipeline.py
- Modify: workflow/run_lifting_sweep.py
- Modify: workflow/experiments/balanced50.py
- Modify: workflow/r2/run_paired_sweep.py
- Modify: tests/test_mcpat_native_cache.py
- Modify: tests/test_workflow.py
- Modify: tests/test_balanced_experiment.py

**Interfaces:**
- The steady pipeline calls run_mcpat and build_model with no standalone cache artifact.
- Completion checks require the McPAT-native schema and hashes.

- [ ] **Step 1: Write a failing no-CACTI pipeline test**

~~~python
def test_corrected_pipeline_never_checks_or_invokes_standalone_cacti(self):
    with patch("workflow.run_lifting_pipeline.run_mcpat",
               return_value=native_mcpat_artifact()), \
         patch("workflow.run_lifting_pipeline.build_model",
               return_value=granular_model()), \
         patch("workflow.run_lifting_pipeline.subprocess.run") as process:
        summary = run_pipeline(self.r1, self.out, self.config)
    self.assertNotIn("cacti", summary["artifacts"])
    self.assertNotIn("cacti", summary["artifact_sha256"])
    self.assertFalse(any("tools/src/cacti" in str(call) for call in process.mock_calls))
~~~

Add completion tests proving an old directory with cacti_characterization.json but without native mcpat.json is incomplete.

- [ ] **Step 2: Run and verify failure**

~~~bash
python -m unittest tests.test_mcpat_native_cache.SteadyPipelineTests -v
~~~

- [ ] **Step 3: Reorder the steady pipeline**

Use this exact stage order:

~~~text
validate config/tools
run_mcpat once
build_model(r1, mcpat.json)
optimize/materialize/HotSpot
build_vector(modules.json)
optional gem5 R2
summary
~~~

Delete characterize imports/calls, cacti tools, cacti stage timing, cacti artifact paths/hashes, and cache-characterization identities. Add cache_authority and McPAT native provenance to summary.

- [ ] **Step 4: Migrate batch completion and paired-sweep validation**

Replace standalone-CACTI required artifacts with mcpat/mcpat.json and validate its embedded_cacti_p schema, binary hash, output hash, and strict module model. Do not let historical output satisfy the new completion predicate.

- [ ] **Step 5: Run focused regressions**

~~~bash
python -m unittest tests.test_mcpat_native_cache.SteadyPipelineTests tests.test_workflow tests.test_balanced_experiment -v
~~~

- [ ] **Step 6: Commit**

~~~bash
git add workflow/run_lifting_pipeline.py workflow/run_lifting_sweep.py workflow/experiments/balanced50.py workflow/r2/run_paired_sweep.py tests/test_mcpat_native_cache.py tests/test_workflow.py tests/test_balanced_experiment.py
git commit -m "refactor: remove standalone CACTI from steady execution"
~~~

### Task 7: Migrate transient ROM and remaining R2 paths from CACTI

**Files:**
- Modify: workflow/transient/rom/run_pipeline.py
- Modify: workflow/transient/rom/paired_validation.py
- Modify: workflow/r2/attachment_validation.py
- Modify: workflow/r2/reuse_result.py
- Modify: tests/test_transient_rom.py
- Modify: tests/test_transient_rom_discrete_parity.py
- Modify: tests/test_balanced_experiment.py

**Interfaces:**
- ROM preflight and final R2 use modules.json embedded McPAT records and hashes.
- No corrected artifact identity contains an external CACTI path/hash.

- [ ] **Step 1: Write failing ROM preflight tests**

~~~python
def test_rom_preflight_accepts_native_mcpat_without_cacti_file(self):
    self.cacti.unlink()
    summary = self.run_with_native_modules()
    self.assertEqual(summary["cache_authority"],
                     "McPAT 1.3 embedded CACTI-P")

def test_rom_rejects_replaced_mcpat_native_record(self):
    replace_l2_record_id(self.modules)
    with self.assertRaisesRegex(ValueError, "McPAT.*cache"):
        self.run_with_native_modules()
~~~

- [ ] **Step 2: Run and verify old CACTI assumptions fail**

~~~bash
python -m unittest tests.test_transient_rom.ROMPipelineTests tests.test_transient_rom_discrete_parity -v
~~~

- [ ] **Step 3: Replace ROM preflight identities**

Remove steady_preflight/cacti/cacti_characterization.json checks. Bind ROM package, layout, final validation, and R2 vector to modules.json cache authority, embedded record IDs, McPAT output hash, and McPAT binary hash.

- [ ] **Step 4: Replace attachment/reuse identities**

Require matching McPAT-native cache provenance in physical-coherence and R2 reuse checks. Keep latency-vector equality as the final reuse gate.

- [ ] **Step 5: Run regressions**

~~~bash
python -m unittest tests.test_transient_rom tests.test_transient_rom_discrete_parity tests.test_balanced_experiment -v
~~~

- [ ] **Step 6: Commit**

~~~bash
git add workflow/transient/rom/run_pipeline.py workflow/transient/rom/paired_validation.py workflow/r2/attachment_validation.py workflow/r2/reuse_result.py tests/test_transient_rom.py tests/test_transient_rom_discrete_parity.py tests/test_balanced_experiment.py
git commit -m "refactor: bind transient and R2 paths to McPAT cache data"
~~~

### Task 8: Enforce the fixed granular thermal-proxy contract

**Files:**
- Modify: workflow/floorplan/optimize_layout.py
- Modify: workflow/run_lifting_pipeline.py
- Modify: tests/test_workflow.py
- Modify: tests/test_alpha_lc_identification.py

**Interfaces:**
- Produces proxy_temperature_components(modules: list[dict], side: float, ambient: float, r_convec: float, alpha: float, beta: float, cross_tier_weight: float, spatial_model: str = "area-quadrature", quadrature_order: int = 2, lc_mm: float | None = None) -> dict while retaining the scalar wrapper with the same inputs.
- Optimizer report declares role search-heuristic and explicit fixed/movable sets.

- [ ] **Step 1: Write failing all-module and fixed-coordinate tests**

~~~python
def test_granular_proxy_uses_all_34_modules(self):
    report = optimize_model(granular_34_modules())
    self.assertEqual(report["thermal_proxy"]["module_count"], 34)
    self.assertEqual(report["thermal_proxy"]["fixed_module_count"], 33)
    self.assertEqual(report["thermal_proxy"]["movable_names"], ["shared_l2"])

def test_optimizer_changes_only_shared_l2(self):
    before, after = baseline_and_selected()
    for name in fixed_names(before):
        self.assertEqual(physical_tuple(before[name]), physical_tuple(after[name]))
~~~

Also test aggregate core rejection, non-finite components, and a degenerate spatial-response failure when multiple legal L2 positions exist.

- [ ] **Step 2: Run and verify failure**

~~~bash
python -m unittest \
  tests.test_workflow.GridTests.test_granular_proxy_uses_all_34_modules \
  tests.test_workflow.GridTests.test_optimizer_changes_only_shared_l2 -v
~~~

- [ ] **Step 3: Add component-level proxy evaluation**

~~~python
def proxy_temperature_components(
    modules, side, ambient, r_convec, alpha, beta,
    cross_tier_weight, spatial_model="area-quadrature",
    quadrature_order=2, lc_mm=None,
):
    total = sum(module["total_power_w"] for module in modules)
    bottom = sum(
        module["total_power_w"] for module in modules
        if module["tier"] == 0
    )
    spatial = spatial_coupling(
        modules, side, cross_tier_weight, spatial_model,
        quadrature_order, lc_mm,
    )
    proxy = ambient + r_convec * total + alpha * spatial + beta * bottom
    points_per_module = quadrature_order ** 2
    return {
        "role": "search-heuristic",
        "total_power_w": total,
        "bottom_power_w": bottom,
        "spatial_coupling_w": spatial,
        "proxy_temperature_c": proxy,
        "module_count": len(modules),
        "module_names": [module["name"] for module in modules],
        "module_pair_count": len(modules) ** 2,
        "quadrature_sample_pair_count": (
            len(modules) * points_per_module
        ) ** 2,
    }
~~~

Use every supplied module. Do not group modules by core. Preserve the current area-quadrature kernel and L2-only candidate construction.

- [ ] **Step 4: Enforce fixed-module invariance and heuristic semantics**

Compare baseline and selected modules by name; reject changes to any fixed module tier, coordinates, dimensions, area, or power. Record spatial-response range and warning text that proxy temperature is non-reportable. Normal heuristic mode checks finite/non-degenerate behavior but does not call parameter-promotion acceptance.

- [ ] **Step 5: Run regressions**

~~~bash
python -m unittest tests.test_workflow.GridTests tests.test_alpha_lc_identification.AlphaLcKernelTests -v
~~~

- [ ] **Step 6: Commit**

~~~bash
git add workflow/floorplan/optimize_layout.py workflow/run_lifting_pipeline.py tests/test_workflow.py tests/test_alpha_lc_identification.py
git commit -m "feat: enforce fixed granular thermal proxy modules"
~~~

### Task 9: Serialize and audit native HotSpot module overlap

**Files:**
- Create: workflow/floorplan/hotspot_audit.py
- Create: tests/test_hotspot_contract.py
- Modify: workflow/floorplan/generate_hotspot_inputs.py
- Modify: tests/test_alpha_lc_identification.py
- Modify: tests/test_workflow.py

**Interfaces:**
- Produces parse_floorplan(path: Path, tier: int) -> list[dict], parse_single_sample_ptrace(path: Path) -> dict[str, float], and audit_serialized_hotspot_inputs(layout: dict, tier_inputs: list[dict], floorplan_paths: dict[int, Path], ptrace_paths: dict[str, Path], rel_tol: float = 1e-12, abs_tol: float = 1e-12) -> dict.
- Corrected materialization requires input_granularity=module.

- [ ] **Step 1: Write failing serialization tests**

~~~python
def test_fractional_native_rectangles_tile_each_tier_after_serialization(self):
    manifest = materialize(self.modules, self.out, input_granularity="module")
    self.assertEqual(manifest["geometry_checks"]["same_tier_overlap_mm2"], 0.0)
    self.assertAlmostEqual(
        manifest["geometry_checks"]["maximum_abs_coverage_residual_mm2"], 0.0,
        places=12,
    )
    self.assertIn("core0_exec", (self.out / "bottom.flp").read_text())

def test_same_tier_overlap_fails_but_cross_tier_projection_passes(self):
    with self.assertRaisesRegex(ValueError, "same-tier"):
        materialize(overlapping_same_tier(), self.out)
    materialize(overlapping_cross_tier(), self.out)
~~~

Also corrupt one serialized coordinate and one trace value to prove the post-write audit catches them.

- [ ] **Step 2: Run and verify failure**

~~~bash
python -m unittest tests.test_hotspot_contract.HotSpotSerializationTests -v
~~~

- [ ] **Step 3: Strengthen geometry and whitespace generation**

Reject duplicate names, non-finite coordinates, non-positive dimensions, out-of-die blocks, and same-tier overlap. Keep cross-tier projection legal. Tile every uncovered rectangle with unique zero-power ws_tN_K blocks. Record functional, whitespace, covered, and residual area by tier.

- [ ] **Step 4: Increase serialization precision and audit bytes**

Write .flp coordinates with 17 significant digits. Re-read .flp and all three .ptrace files and verify complete die coverage, zero same-tier overlap, functional area, zero whitespace power, dynamic/leakage/total conservation, total=dynamic+leakage, native names, unique names, and grid_map_mode avg. Record functional-only cross-tier projected overlap for information.

- [ ] **Step 5: Run regressions**

~~~bash
python -m unittest tests.test_hotspot_contract.HotSpotSerializationTests tests.test_alpha_lc_identification.GridConvergenceTests tests.test_workflow.GridTests -v
~~~

- [ ] **Step 6: Commit**

~~~bash
git add workflow/floorplan/hotspot_audit.py workflow/floorplan/generate_hotspot_inputs.py tests/test_hotspot_contract.py tests/test_alpha_lc_identification.py tests/test_workflow.py
git commit -m "fix: serialize audited native HotSpot floorplans"
~~~

### Task 10: Reject warning-bearing or incomplete HotSpot output

**Files:**
- Create: workflow/thermal/hotspot_validation.py
- Modify: workflow/thermal/run_hotspot.py
- Modify: workflow/transient/run_hotspot_steady.py
- Modify: workflow/transient/run_hotspot_transient.py
- Modify: tests/test_hotspot_contract.py
- Modify: tests/test_transient.py

**Interfaces:**
- Produces forbidden_hotspot_diagnostics(text) -> list[str].
- Produces validate_hotspot_log(text) -> dict.
- Produces validate_active_grid_samples(samples, active_layers, grid_rows, grid_cols) -> dict.

- [ ] **Step 1: Write failing execution-validation tests**

~~~python
def test_zero_return_with_overlap_warning_is_failure(self):
    with fake_hotspot(returncode=0, log="overlap of functional blocks?"):
        with self.assertRaisesRegex(RuntimeError, "overlap"):
            run_hotspot(self.case)

def test_missing_active_grid_cell_is_failure(self):
    write_grid_output(active_layers={1: 1024, 3: 1023})
    with self.assertRaisesRegex(ValueError, "active layer 3"):
        run_hotspot(self.case)
~~~

Cover erroneous b2gmap, invalid mapping/floorplan, NaN/Inf, duplicate indices, missing layer, and truncated layer.

- [ ] **Step 2: Run and verify failure**

~~~bash
python -m unittest tests.test_hotspot_contract.HotSpotExecutionValidationTests -v
~~~

- [ ] **Step 3: Add shared log validation**

Reject diagnostics case-insensitively:

~~~python
FORBIDDEN = (
    "overlap of functional blocks",
    "erroneous b2gmap",
    "invalid floorplan",
    "invalid mapping",
    "unknown mapping mode",
    "negative overlap",
)
~~~

Require finite temperatures and exactly rows*cols unique indices for each active layer.

- [ ] **Step 4: Integrate all HotSpot launch paths**

Call the shared validator after saving logs but before writing a successful result in steady, transient-initialization, and transient-solve paths.

- [ ] **Step 5: Run regressions**

~~~bash
python -m unittest tests.test_hotspot_contract.HotSpotExecutionValidationTests tests.test_transient.HotSpotSteadyInitializationTests -v
~~~

- [ ] **Step 6: Commit**

~~~bash
git add workflow/thermal/hotspot_validation.py workflow/thermal/run_hotspot.py workflow/transient/run_hotspot_steady.py workflow/transient/run_hotspot_transient.py tests/test_hotspot_contract.py tests/test_transient.py
git commit -m "fix: reject invalid HotSpot mapping output"
~~~

### Task 11: Require a clean non-SuperLU HotSpot build

**Files:**
- Create: workflow/thermal/hotspot_toolchain.py
- Create: scripts/build_hotspot.sh
- Modify: scripts/check_tools.sh
- Modify: workflow/run_lifting_pipeline.py
- Modify: workflow/thermal/identify_alpha_lc.py
- Modify: workflow/transient/run_transient_pipeline.py
- Modify: tests/test_hotspot_contract.py
- Modify: tests/test_workflow.py

**Interfaces:**
- Produces inspect_hotspot_binary(binary) -> dict.
- Produces record_hotspot_build(binary, build_log, output, requested_superlu="0", requested_mathaccel="none") -> dict.
- Produces verify_hotspot_build(binary, provenance) -> dict.
- Produces validate_non_superlu_grid_size(value) -> int.

- [ ] **Step 1: Write failing build-provenance tests**

~~~python
def test_superlu_dependency_or_symbol_is_rejected(self):
    for evidence in (dependency("libsuperlu.so"), undefined_symbol("dgssv_")):
        with self.assertRaisesRegex(ValueError, "SuperLU"):
            inspect_evidence(evidence)

def test_non_power_of_two_grid_is_rejected(self):
    for value in (True, 1, 48, 96):
        with self.assertRaises(ValueError):
            validate_non_superlu_grid_size(value)
    self.assertEqual(validate_non_superlu_grid_size(32), 32)
~~~

Also test stale binary hash, missing compile macros, and vendor BLAS/MKL/ACML/Accelerate/sunperf symbols.

- [ ] **Step 2: Run and verify failure**

~~~bash
python -m unittest tests.test_hotspot_contract.HotSpotToolchainTests -v
~~~

- [ ] **Step 3: Add serialized clean-build script**

scripts/build_hotspot.sh must acquire a named flock, refuse to clean while a HotSpot process is using the shared tools/src tree, then execute:

~~~bash
make -C tools/src/hotspot clean
make -C tools/src/hotspot SUPERLU=0 MATHACCEL=none hotspot
~~~

Capture the full log. Write ignored tools/build/hotspot/build_provenance.json containing requested flags, observed -DSUPERLU=0 and -DMATHACCEL=0 macros, commands, log hash, binary hash, readelf/ldd dependencies, and nm undefined symbols.

- [ ] **Step 4: Add verifier and early pipeline guard**

Reject any superlu or vendor-math match. Cache verification by resolved path, size, mtime, and hash. Validate power-of-two grids during config validation. Run the guard before output-directory creation or expensive McPAT work in steady, identification, and transient entry points.

- [ ] **Step 5: Build only when safe and verify**

First run:

~~~bash
pgrep -af 'tools/src/hotspot/hotspot'
~~~

If no active process uses the binary:

~~~bash
scripts/build_hotspot.sh
python -m workflow.thermal.hotspot_toolchain verify
scripts/check_tools.sh
python -m unittest tests.test_hotspot_contract.HotSpotToolchainTests tests.test_workflow.FormalGuardTests -v
~~~

If a process is active, do not clean; verify the current already non-SuperLU binary and defer rebuild until the process exits.

- [ ] **Step 6: Commit**

~~~bash
git add workflow/thermal/hotspot_toolchain.py scripts/build_hotspot.sh scripts/check_tools.sh workflow/run_lifting_pipeline.py workflow/thermal/identify_alpha_lc.py workflow/transient/run_transient_pipeline.py tests/test_hotspot_contract.py tests/test_workflow.py
git commit -m "build: require dependency-free HotSpot provenance"
~~~

### Task 12: Define isolated short R1 protocols

**Files:**
- Create: workflow/r1_protocol.py
- Create: configs/experiments/r1_short_convergence.json
- Create: tests/test_short_roi.py
- Modify: scripts/run_r1_sweep.py
- Modify: configs/gem5/clip_r1.py

**Interfaces:**
- Produces R1Protocol dataclass, canonical_protocol(value), protocol_id(value), require_protocol(metadata), and classify_legacy_protocol(metadata).
- Adds short_conv_1m, short_conv_2m, short_conv_5m, and short_conv_10m profiles.

- [ ] **Step 1: Write failing canonical-identity and runner tests**

~~~python
def test_short_profiles_are_all_core_and_isolated(self):
    experiment = read_json(SHORT_CONFIG)
    expected = {
        "short_conv_1m": 1_000_000,
        "short_conv_2m": 2_000_000,
        "short_conv_5m": 5_000_000,
        "short_conv_10m": 10_000_000,
    }
    for name, target in expected.items():
        profile = experiment["profiles"][name]
        self.assertEqual(profile["warmup_insts"], 2_000_000)
        self.assertEqual(profile["measure_insts"], target)
        self.assertEqual(profile["instruction_window_scope"], "all-cores")

def test_successful_output_with_wrong_protocol_is_not_reused(self):
    write_success(self.output, protocol_id="old")
    self.assertFalse(reusable(job(protocol_id="new"), self.output))
~~~

- [ ] **Step 2: Run and verify failure**

~~~bash
python -m unittest tests.test_short_roi.ShortProfileProtocolTests -v
~~~

- [ ] **Step 3: Implement canonical protocol identity**

Canonical JSON includes family, profile, warmup_insts, measure_insts, instruction_window_scope, workload options, gem5 config hash, and benchmark binary hash. Hash sorted JSON with SHA-256. Missing identity may be classified read-only as legacy, but corrected short execution requires it.

- [ ] **Step 4: Make profile selection data-driven and bind outputs**

Remove hardcoded argparse choices. Load the selected profile from JSON and pass protocol fields/ID into clip_r1.py. Record canonical protocol and ID in metadata, command, plan, and status. Reuse successful output only when all these identities and instruction targets match. All-core acceptance remains instructions >= target for every core.

- [ ] **Step 5: Run tests**

~~~bash
python -m unittest tests.test_short_roi.ShortProfileProtocolTests -v
~~~

- [ ] **Step 6: Commit**

~~~bash
git add workflow/r1_protocol.py configs/experiments/r1_short_convergence.json scripts/run_r1_sweep.py configs/gem5/clip_r1.py tests/test_short_roi.py
git commit -m "feat: define isolated short R1 protocols"
~~~

### Task 13: Select the shortest converged R1 measurement

**Files:**
- Create: workflow/analysis/select_short_roi.py
- Modify: tests/test_short_roi.py
- Modify: workflow/mcpat/run_mcpat.py

**Interfaces:**
- Produces ConvergenceLimits and collect_candidate, compare_candidate, select_measurement_target, and write_convergence_report.
- Reuses strict McPAT runner to generate granular modules without running HotSpot.

- [ ] **Step 1: Write failing convergence-selection tests**

~~~python
def test_selects_shortest_candidate_only_when_it_and_all_larger_pass(self):
    comparisons = {
        1_000_000: passed(),
        2_000_000: passed(),
        5_000_000: passed(),
        10_000_000: reference(),
    }
    self.assertEqual(select_measurement_target(comparisons), 1_000_000)
    comparisons[2_000_000] = failed("module_distribution")
    self.assertEqual(select_measurement_target(comparisons), 5_000_000)

def test_falls_back_to_10m_when_5m_fails(self):
    self.assertEqual(select_measurement_target(with_failed_5m()), 10_000_000)
~~~

Also test wrong protocol/scope, missing L2 counters, non-finite power, mismatched module names, and inconsistent simSeconds versus simTicks/simFreq.

- [ ] **Step 2: Run and verify failure**

~~~bash
python -m unittest tests.test_short_roi.ConvergenceSelectionTests -v
~~~

- [ ] **Step 3: Implement candidate collection**

Require exact protocol family, 2M warmup, all-core scope, four cores >= target, positive ROI duration, matching architecture/cache/workload command, identical tool hashes, finite aggregate IPC and power, granular module names, and available per-core shared-L2 demand counts.

- [ ] **Step 4: Implement metrics and selection**

Use relative absolute error for aggregate IPC, dynamic power, and leakage power. Use total-variation distance for normalized module total-power and per-core L2-traffic distributions:

~~~python
tv = 0.5 * sum(abs(candidate[name] - reference[name]) for name in names)
~~~

Default limits are 1% IPC and 3% for total power, module distribution, and traffic distribution. Compare 1M/2M/5M to 10M; select a target only if it and every larger candidate pass. Report raw values, deltas, thresholds, pass flags, larger-candidate gate, selected protocol, and input hashes.

- [ ] **Step 5: Run tests**

~~~bash
python -m unittest tests.test_short_roi.ConvergenceSelectionTests -v
~~~

- [ ] **Step 6: Commit**

~~~bash
git add workflow/analysis/select_short_roi.py workflow/mcpat/run_mcpat.py tests/test_short_roi.py
git commit -m "feat: select converged short R1 windows"
~~~

### Task 14: Bind R1/R2/transient reuse to protocol identity

**Files:**
- Modify: workflow/r1_catalog.py
- Modify: workflow/floorplan/build_module_model.py
- Modify: workflow/r2/run_r2.py
- Modify: workflow/r2/reuse_result.py
- Modify: workflow/r2/attachment_validation.py
- Modify: workflow/transient/run_transient_r1.py
- Modify: workflow/transient/run_transient_pipeline.py
- Modify: tests/test_short_roi.py
- Modify: tests/test_balanced_experiment.py
- Modify: tests/test_transient.py

**Interfaces:**
- Threads r1_protocol and r1_protocol_id through modules, R2 status/results, catalogue, reuse, attachment, and transient-source matching.

- [ ] **Step 1: Write failing cross-protocol reuse tests**

~~~python
def test_identical_latency_with_different_r1_protocol_cannot_reuse(self):
    fixed = r2_fixture(protocol_id="short-5m")
    candidate = r2_fixture(protocol_id="legacy-500m")
    with self.assertRaisesRegex(ValueError, "protocol"):
        validate_reuse(fixed, candidate, self.r1)
~~~

Duplicate this principle for catalogue acceptance, local R2 attachment, paired-layout reuse, and transient R1 reuse.

- [ ] **Step 2: Run and verify failure**

~~~bash
python -m unittest tests.test_short_roi.ProtocolIsolationTests -v
~~~

- [ ] **Step 3: Add identity validation everywhere**

New short/formal execution rejects missing protocol identity. Legacy readers may classify missing identity without modifying old artifacts. Validate profile warmup, target, scope, workload options, command, binary/config hashes, and protocol ID against the selected experiment profile.

- [ ] **Step 4: Run focused and existing reuse tests**

~~~bash
python -m unittest tests.test_short_roi.ProtocolIsolationTests tests.test_balanced_experiment tests.test_transient -v
~~~

- [ ] **Step 5: Commit**

~~~bash
git add workflow/r1_catalog.py workflow/floorplan/build_module_model.py workflow/r2/run_r2.py workflow/r2/reuse_result.py workflow/r2/attachment_validation.py workflow/transient/run_transient_r1.py workflow/transient/run_transient_pipeline.py tests/test_short_roi.py tests/test_balanced_experiment.py tests/test_transient.py
git commit -m "fix: isolate R1 and R2 artifacts by protocol"
~~~

### Task 15: Derive transient sampling from measured ROI

**Files:**
- Create: workflow/transient/sampling.py
- Modify: configs/gem5/clip_r1_transient.py
- Modify: workflow/transient/run_transient_r1.py
- Modify: workflow/transient/stats_windows.py
- Modify: workflow/transient/run_transient_pipeline.py
- Modify: workflow/transient/run_dual_layout_validation.py
- Modify: workflow/run_lifting_pipeline.py
- Modify: tests/test_short_roi.py
- Modify: tests/test_transient.py

**Interfaces:**
- Produces measured_roi_duration(r1_dir) -> dict.
- Produces resolve_sampling_policy(r1_dir, override_ms=None, max_interval_ms=0.5, target_window_count=50) -> dict.
- Public sample_ms defaults become None; None derives, numeric values are explicit overrides.

- [ ] **Step 1: Write failing sampling-policy tests**

~~~python
def test_derives_sub_2ms_interval_from_roi(self):
    policy = resolve_sampling_policy(self.r1_with_roi_ms(4.8))
    self.assertEqual(policy["mode"], "roi-derived")
    self.assertAlmostEqual(policy["requested_interval_ms"], 0.096)
    self.assertEqual(policy["target_window_count"], 50)

def test_explicit_equal_number_has_different_identity(self):
    derived = resolve_sampling_policy(self.r1_with_roi_ms(25.0))
    explicit = resolve_sampling_policy(self.r1_with_roi_ms(25.0), override_ms=0.5)
    self.assertNotEqual(derived["policy_id"], explicit["policy_id"])
~~~

Test the 0.5ms cap, positive finite validation, tick rounding, source stats hash, and final partial-window duration/padding.

- [ ] **Step 2: Run and verify failure**

~~~bash
python -m unittest tests.test_transient.TransientSamplingPolicyTests tests.test_short_roi.TransientDerivationTests -v
~~~

- [ ] **Step 3: Implement measured ROI and policy identity**

Cross-check simSeconds with simTicks/simFreq from the post-reset R1 stats. Derive:

~~~python
requested_ms = min(max_interval_ms,
                   roi_duration_s * 1000.0 / target_window_count)
~~~

Record mode, protocol ID, stats hash, ROI seconds, formula, cap, target count, requested interval, and stable policy ID.

- [ ] **Step 4: Remove hardcoded 10ms and thread the policy**

Make clip_r1_transient.py require an internal numeric --sample-ms. Callers resolve it first. completed() compares policy ID, not just numeric interval. Record requested and tick-rounded intervals, target/actual window counts, each actual duration, and padded duration while preserving existing final-partial behavior.

- [ ] **Step 5: Run regressions**

~~~bash
python -m unittest tests.test_transient.TransientSamplingPolicyTests tests.test_short_roi.TransientDerivationTests tests.test_transient -v
~~~

- [ ] **Step 6: Commit**

~~~bash
git add workflow/transient/sampling.py configs/gem5/clip_r1_transient.py workflow/transient/run_transient_r1.py workflow/transient/stats_windows.py workflow/transient/run_transient_pipeline.py workflow/transient/run_dual_layout_validation.py workflow/run_lifting_pipeline.py tests/test_short_roi.py tests/test_transient.py
git commit -m "feat: derive transient sampling from short ROI"
~~~

### Task 16: Make transient ROM sampling policy-driven

**Files:**
- Modify: workflow/transient/rom/contracts.py
- Modify: workflow/transient/rom/run_pipeline.py
- Modify: workflow/transient/rom/pod_state_space.py
- Modify: workflow/experiments/transient_rom_balanced2.py
- Modify: configs/experiments/clip3d_transient_rom_exploratory.json
- Modify: configs/experiments/clip3d_transient_rom_lambda0020119_traffic_weighted_discrete_partition_exploratory.json
- Modify: configs/experiments/transient_rom_balanced2_selection.json
- Modify: tests/test_transient_rom.py
- Modify: tests/test_transient_rom_balanced2.py
- Modify: tests/test_transient_five_state.py

**Interfaces:**
- Changes parse_settings(config, sample_interval_ms, sampling_policy_id) -> ROMSettings.
- ROM identity includes sampling-policy ID and power-trace identity.

- [ ] **Step 1: Write failing non-2ms ROM tests**

~~~python
def test_rom_accepts_canonical_derived_interval(self):
    settings = parse_settings(self.config, sample_interval_ms=0.096,
                              sampling_policy_id="policy-a")
    self.assertAlmostEqual(settings.sample_interval_ms, 0.096)

def test_rom_package_rejects_same_interval_from_different_policy(self):
    with self.assertRaisesRegex(ValueError, "sampling policy"):
        reuse_package(existing_policy="a", requested_policy="b")
~~~

- [ ] **Step 2: Run and verify fixed-2ms assumptions fail**

~~~bash
python -m unittest tests.test_transient_rom_balanced2 tests.test_transient_five_state -v
~~~

- [ ] **Step 3: Inject resolved policy into ROM settings**

Remove the 2.0ms default and exact-equality constant. Resolve sampling from canonical R1 before ROM settings. Include policy ID, actual interval, source protocol, and trace identity in calibration/package/final-validation contracts.

- [ ] **Step 4: Migrate Balanced-2 selection/configs**

Replace fixed sample_interval_ms: 2.0 with a canonical roi-derived policy declaration. Allow per-workload intervals while requiring the stored policy ID to match each point.

- [ ] **Step 5: Run regressions**

~~~bash
python -m unittest tests.test_transient_rom tests.test_transient_rom_balanced2 tests.test_transient_five_state -v
~~~

- [ ] **Step 6: Commit**

~~~bash
git add workflow/transient/rom/contracts.py workflow/transient/rom/run_pipeline.py workflow/transient/rom/pod_state_space.py workflow/experiments/transient_rom_balanced2.py configs/experiments/clip3d_transient_rom_exploratory.json configs/experiments/clip3d_transient_rom_lambda0020119_traffic_weighted_discrete_partition_exploratory.json configs/experiments/transient_rom_balanced2_selection.json tests/test_transient_rom.py tests/test_transient_rom_balanced2.py tests/test_transient_five_state.py
git commit -m "fix: make transient ROM sampling policy-driven"
~~~

### Task 17: Correct configurations, documentation, and tool contracts

**Files:**
- Modify: configs/experiments/clip3d_pipeline.json
- Modify: relevant configs/experiments/clip3d_constrained_*.json
- Modify: workflow/README.md
- Modify: configs/gem5/README.md
- Modify: docs/clip3d_pipeline_zh.md
- Modify: docs/local_cacti_table_ii_zh.md
- Modify: manifests/tool_versions.tsv generation in scripts/check_tools.sh
- Modify: tests/test_workflow.py

**Interfaces:**
- Corrected configs declare proxy role search-heuristic, physical input granularity module, and non-SuperLU power-of-two grid.
- Historical standalone CACTI documentation is labelled legacy diagnostic.

- [ ] **Step 1: Write failing configuration-contract tests**

~~~python
def test_corrected_configs_use_native_modules_and_heuristic_proxy(self):
    for path in corrected_configs():
        config = read_json(path)
        self.assertEqual(config["physical"]["input_granularity"], "module")
        self.assertEqual(config["layout_optimizer"]["proxy_role"],
                         "search-heuristic")
        self.assertFalse(config["formal_validation"]["accepted"])
        self.assertNotIn("cacti", config)
~~~

- [ ] **Step 2: Run and verify failure**

~~~bash
python -m unittest tests.test_workflow.ConfigurationContractTests -v
~~~

- [ ] **Step 3: Migrate corrected configs and docs**

Remove active cacti sections, declare McPAT cache authority, module input, heuristic parameters/provenance, and sampling-policy defaults. Explain that standalone CACTI data/scripts remain only for historical diagnostics and are not used by corrected runs. Update tool manifest generation so the corrected required set is gem5, McPAT, and HotSpot while CACTI is optional diagnostic.

- [ ] **Step 4: Scan for forbidden active dependencies**

~~~bash
rg -n 'characterize_cache|cacti_characterization|tools/src/cacti|source_cacti|cacti_provenance|l1[di]_cacti|l2_cacti' \
  workflow/run_lifting_pipeline.py workflow/run_lifting_sweep.py workflow/r2 \
  workflow/transient/rom workflow/experiments
~~~

Expected: no active corrected-path matches; any legacy parser/helper match is explicitly labelled.

- [ ] **Step 5: Run tests and commit**

~~~bash
python -m unittest tests.test_workflow.ConfigurationContractTests -v
git add configs/experiments workflow/README.md configs/gem5/README.md docs/clip3d_pipeline_zh.md docs/local_cacti_table_ii_zh.md scripts/check_tools.sh manifests/tool_versions.tsv tests/test_workflow.py
git commit -m "docs: document corrected McPAT-native protocol"
~~~

### Task 18: Run integrated regressions and real tool smoke tests

**Files:**
- Modify only if a verified defect is found in earlier scoped files.
- Test: tests/test_mcpat_native_cache.py
- Test: tests/test_hotspot_contract.py
- Test: tests/test_short_roi.py

**Interfaces:**
- Produces one end-to-end corrected smoke directory and verification notes.

- [ ] **Step 1: Run syntax and focused suites**

~~~bash
python -m compileall workflow scripts configs/gem5
python -m unittest tests.test_mcpat_native_cache tests.test_hotspot_contract tests.test_short_roi -v
~~~

- [ ] **Step 2: Run affected regression suites**

~~~bash
python -m unittest \
  tests.test_cache_alignment \
  tests.test_workflow \
  tests.test_balanced_experiment \
  tests.test_transient \
  tests.test_transient_rom \
  tests.test_transient_rom_balanced2 \
  tests.test_transient_five_state \
  tests.test_alpha_lc_identification -v
~~~

- [ ] **Step 3: Run the full suite**

~~~bash
python -m unittest discover -s tests -v
git diff --check
~~~

- [ ] **Step 4: Run real McPAT and HotSpot contract smoke tests**

Only when shared vendor binaries are not in use:

~~~bash
CLIP_RUN_TOOL_SMOKE=1 python -m unittest \
  tests.test_mcpat_native_cache.RealMcPATNativeSmoke \
  tests.test_hotspot_contract.RealHotSpotOverlapSmoke -v
~~~

Assert nine cache records, four granular cores, approximately 34 modules, one movable L2, serialized area/power conservation, no forbidden HotSpot diagnostics, finite temperatures, and 1024 samples on each active 32x32 layer.

- [ ] **Step 5: Run one corrected end-to-end point without R2**

Use an existing R1 only as an explicit compatibility smoke; do not relabel it as short:

~~~bash
python -m workflow.run_lifting_pipeline \
  --r1-dir runs/architecture_sweep/r1/paper/matmul/l1d_32kB/l2_512kB \
  --output-dir runs/smoke/mcpat_native_matmul_32kB_512kB \
  --config configs/experiments/clip3d_pipeline.json \
  --layout-method clip3d
~~~

Verify the output has no cacti directory/artifact, has McPAT-native timing and geometry, and final Tmax comes from HotSpot.

- [ ] **Step 6: Generate a short convergence plan without launching long work**

~~~bash
for profile in short_conv_1m short_conv_2m short_conv_5m short_conv_10m; do
  python scripts/run_r1_sweep.py \
    --experiment configs/experiments/r1_short_convergence.json \
    --profile "$profile" \
    --workloads matmul \
    --l1d-sizes 32kB \
    --l2-sizes 512kB \
    --output-root runs/short_r1_convergence
done
~~~

Inspect planned_jobs.json and confirm all-core scope, distinct protocol IDs/directories, 2M warmup, and the four targets.

- [ ] **Step 7: Commit verification-only fixes, if any**

If no code changed, do not create an empty commit. If verified defects were fixed, rerun the failed and full suites, then commit only those fixes with:

~~~bash
git commit -m "test: verify corrected CLIP-3D toolchain"
~~~

## Completion evidence

Before claiming completion, retain and report:

- branch/worktree and final commit;
- exact test counts and any intentionally skipped real-tool tests;
- McPAT patch/binary/output hashes;
- HotSpot build flags, log/binary hashes, and dependency/symbol audit;
- corrected smoke output path;
- proof that no standalone CACTI artifact or process was used;
- granular module/mobility counts;
- serialized geometry and power residuals;
- HotSpot Tmax source;
- short-protocol plan IDs; and
- remaining required long-running convergence experiments, if not executed.
