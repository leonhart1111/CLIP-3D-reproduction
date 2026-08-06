# Canonical R1 and Balanced-50 R2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run an auditable experiment that uses exactly the existing 100 canonical R1 points, generates fixed-bin and CLIP-3D physical artifacts for all 100 points, and obtains paired measured BIPS2 for a predeclared 50-point sample.

**Architecture:** A shared canonical-R1 catalogue owns architecture discovery and provenance, so audit and lifting cannot disagree about what constitutes a point. A separate Balanced-50 experiment layer validates the two complete layout roots, schedules fixed-first R2 pairs, performs exact-vector reuse only after full provenance validation, and produces a paired report without proxy values. Existing single-point pipeline and gem5 execution functions remain the scientific engines.

**Tech Stack:** Python 3 standard library, `unittest`, existing gem5/McPAT/CACTI/HotSpot executables, JSON/CSV artifacts, `concurrent.futures.ProcessPoolExecutor`.

## Global Constraints

- Never execute, rewrite, move, or delete any canonical R1 point.
- The canonical grid comes only from `configs/experiments/r1_cache_sweep.json`: 5 workloads × 4 L1D sizes × 5 L2 sizes = 100 points.
- Preserve and report `stencil/l1d_32kB/l2_512kB.corrupt_duplicate_20260731T011500_CST`; never count it as canonical.
- Use `configs/experiments/clip3d_constrained_5p0_raw_power_p1_lambda0020119_traffic_weighted_exploratory.json` unchanged.
- Preserve `lambda_wire=0.0020119160767721133`, `wire_aggregation=traffic-weighted`, `alpha=1.5643788695171585`, `beta=0.0`, and `cross_tier_weight=0.995`.
- Preserve `mode=operational-exploratory-traffic-weighted`, `non_formal=true`, `paper_equivalent=false`, and `shared_parameter_accepted=false` in every experiment artifact.
- The Balanced-50 sample is fixed before reading outcomes and contains exactly 10 cache pairs per workload.
- CLIP-3D may reuse fixed-bin IPC2 only when the complete `gem5_overrides` mapping and all provenance checks match exactly.
- If `K` selected CLIP-3D vectors differ from fixed-bin, run exactly `50 + K` physical R2 simulations.
- Final reports require 50 real or strictly validated reused IPC2/BIPS2 pairs; no `IPC1 × f_sus` proxy is allowed.
- Do not run transient thermal simulation or the strict-P1 global-bound tool in this experiment.

## File Structure

- Create `workflow/r1_catalog.py`: canonical grid construction, point validation, noncanonical exclusion reporting, and immutable-artifact hashing.
- Modify `workflow/common.py`: expose one shared streaming SHA-256 helper.
- Modify `workflow/analysis/audit_r1.py`: consume the canonical catalogue and report canonical/excluded counts separately.
- Create `workflow/analysis/refresh_r1_plan.py`: regenerate only the 100-job plan and prove canonical R1 hashes did not change.
- Modify `workflow/run_lifting_sweep.py`: discover canonical points only and emit config/classification/provenance in sweep status.
- Create `workflow/experiments/__init__.py`: experiment package marker.
- Create `workflow/experiments/balanced50.py`: selection-manifest and complete-layout-root validation.
- Create `configs/experiments/balanced50_traffic_weighted.json`: explicit ordered list of all 50 selected architecture keys.
- Modify `workflow/r2/run_r2.py`: reject stale successful result caches that do not match the requested R1 and latency vector.
- Create `workflow/r2/reuse_result.py`: validate and attach exact fixed-to-CLIP R2 reuse with a dedicated provenance artifact.
- Create `workflow/r2/run_paired_sweep.py`: fixed-first, concurrent, resumable 50-pair scheduler.
- Create `workflow/analysis/summarize_paired_sweep.py`: strict paired CSV/JSON aggregation.
- Create `tests/test_balanced_experiment.py`: focused regression suite for catalogue, selection, reuse, resumption, and aggregation.
- Create `docs/balanced50_experiment_zh.md`: exact operational commands, output locations, resumption behavior, and non-formal interpretation.

---

### Task 1: Canonical R1 Catalogue and Artifact Identity

**Files:**
- Create: `workflow/r1_catalog.py`
- Modify: `workflow/common.py`
- Create: `tests/test_balanced_experiment.py`

**Interfaces:**
- Consumes: R1 root, `r1_cache_sweep.json`, profile name.
- Produces: `sha256_file(path: Path) -> str`, `ArchitectureKey`, `build_catalogue(root: Path, experiment_path: Path, profile: str = "paper", validate_counters: bool = True) -> dict`, `canonical_directories(catalogue: dict) -> list[Path]`, and `snapshot_canonical_artifacts(catalogue: dict) -> dict[str, str]`.

- [ ] **Step 1: Add a focused R1 fixture and failing canonical-discovery tests**

  In `tests/test_balanced_experiment.py`, define a compact two-workload grid fixture with two L1D values and one L2 value. Each canonical directory contains successful `status.json`, matching `r1_metadata.json`, and a final gem5 statistics section with four positive core instruction/cycle counters. Add an extra sibling directory named `l2_128kB.corrupt_duplicate_fixture` containing another successful status.

  ```python
  class CanonicalCatalogueTests(unittest.TestCase):
      def test_catalogue_returns_only_expected_paths_and_reports_extra_status(self):
          root, experiment = self.make_grid_fixture()
          catalogue = build_catalogue(root, experiment, profile="paper")
          self.assertEqual(catalogue["expected_count"], 4)
          self.assertEqual(catalogue["valid_count"], 4)
          self.assertEqual(len(catalogue["canonical_records"]), 4)
          self.assertEqual(len(catalogue["excluded_noncanonical"]), 1)
          self.assertIn("corrupt_duplicate_fixture",
                        catalogue["excluded_noncanonical"][0]["relative_directory"])

      def test_catalogue_rejects_metadata_that_disagrees_with_canonical_path(self):
          root, experiment = self.make_grid_fixture()
          metadata_path = root / "fft/l1d_16kB/l2_128kB/r1_metadata.json"
          metadata = read_json(metadata_path)
          metadata["l1d_size"] = "32kB"
          write_json(metadata_path, metadata)
          catalogue = build_catalogue(root, experiment, profile="paper")
          record = next(item for item in catalogue["canonical_records"]
                        if item["key"]["workload"] == "fft"
                        and item["key"]["l1d_size"] == "16kB")
          self.assertFalse(record["valid"])
          self.assertIn("metadata l1d_size", " ".join(record["errors"]))
  ```

- [ ] **Step 2: Run the focused tests and verify the import fails**

  Run:

  ```bash
  python -m unittest tests.test_balanced_experiment.CanonicalCatalogueTests -v
  ```

  Expected: FAIL because `workflow.r1_catalog` does not exist.

- [ ] **Step 3: Add the shared SHA-256 helper and canonical catalogue**

  Add to `workflow/common.py`:

  ```python
  def sha256_file(path: Path | str) -> str:
      digest = hashlib.sha256()
      with Path(path).open("rb") as stream:
          for chunk in iter(lambda: stream.read(1024 * 1024), b""):
              digest.update(chunk)
      return digest.hexdigest()
  ```

  In `workflow/r1_catalog.py`, define an ordered frozen key and derive exact paths from the grid, never from recursive discovery:

  ```python
  @dataclass(frozen=True, order=True)
  class ArchitectureKey:
      workload: str
      l1d_size: str
      l2_size: str

      def relative_path(self) -> Path:
          return Path(self.workload) / f"l1d_{self.l1d_size}" / f"l2_{self.l2_size}"

  def expected_keys(experiment: dict) -> list[ArchitectureKey]:
      return [ArchitectureKey(workload, l1d, l2)
              for workload, l1d, l2 in itertools.product(
                  experiment["workloads"], experiment["l1d_sizes"],
                  experiment["l2_sizes"])]
  ```

  `build_catalogue` must validate success state, required files, path/metadata workload and cache sizes, profile instruction scope (with existing `cpu0` default compatibility), per-core positive counters, unique keys, and exact expected count. It may recursively scan `status.json` only to populate `excluded_noncanonical`; that scan must never create canonical records. `snapshot_canonical_artifacts` hashes `status.json`, `r1_metadata.json`, and `stats.txt` for every valid canonical record using keys such as `fft/l1d_16kB/l2_128kB/stats.txt`.

- [ ] **Step 4: Run catalogue tests**

  Run:

  ```bash
  python -m unittest tests.test_balanced_experiment.CanonicalCatalogueTests -v
  ```

  Expected: all catalogue tests PASS.

- [ ] **Step 5: Commit the catalogue**

  ```bash
  git add workflow/common.py workflow/r1_catalog.py tests/test_balanced_experiment.py
  git commit -m "feat: add canonical R1 catalogue"
  ```

---

### Task 2: Canonical Audit and Hash-Protected Plan Refresh

**Files:**
- Modify: `workflow/analysis/audit_r1.py`
- Create: `workflow/analysis/refresh_r1_plan.py`
- Modify: `tests/test_balanced_experiment.py`

**Interfaces:**
- Consumes: Task 1 `build_catalogue` and `snapshot_canonical_artifacts`.
- Produces: `audit(root, output, experiment_path, profile="paper", expected_points=100) -> dict` and `refresh(root, experiment_path, profile, output) -> dict`.

- [ ] **Step 1: Write failing audit and mutation-guard tests**

  Add tests that assert the duplicate is excluded without making the audit incomplete, that a one-job `planned_jobs.json` makes the audit incomplete, and that `refresh` reports identical before/after hashes. Patch `subprocess.run` in the refresh success test so it writes a 4-job plan without touching point artifacts; add a second patched case that mutates one fixture `stats.txt` and assert `RuntimeError` contains `canonical R1 artifacts changed`.

  ```python
  def test_audit_counts_canonical_and_excluded_separately(self):
      result = audit(root, output, experiment, "paper", expected_points=4)
      self.assertEqual(result["canonical_status_count"], 4)
      self.assertEqual(result["excluded_noncanonical_count"], 1)
      self.assertTrue(result["complete"])

  def test_refresh_aborts_if_any_canonical_hash_changes(self):
      with patch("workflow.analysis.refresh_r1_plan.subprocess.run",
                 side_effect=mutate_stats_and_return_success):
          with self.assertRaisesRegex(RuntimeError,
                                      "canonical R1 artifacts changed"):
              refresh(root, experiment, "paper", output)
  ```

- [ ] **Step 2: Run tests and verify old recursive audit behavior fails**

  Run:

  ```bash
  python -m unittest tests.test_balanced_experiment.CanonicalAuditTests -v
  ```

  Expected: FAIL because the current audit counts the extra status as a 5th point and the refresh module is absent.

- [ ] **Step 3: Convert audit to the shared catalogue**

  Replace unrestricted point construction in `audit_r1.py` with `build_catalogue`. Report these explicit fields:

  ```python
  {
      "schema_version": 2,
      "canonical_status_count": len(catalogue["canonical_records"]),
      "valid_success_count": catalogue["valid_count"],
      "excluded_noncanonical_count": len(catalogue["excluded_noncanonical"]),
      "excluded_noncanonical": catalogue["excluded_noncanonical"],
      "planned_points": planned_count,
      "instruction_window_scopes": scopes,
      "complete": planned_count == expected_points
                  and catalogue["complete"]
                  and len(scopes) == 1,
      "records": catalogue["canonical_records"],
  }
  ```

  Add CLI options `--experiment` and `--profile`; retain `--expected-points` and `--require-complete`.

- [ ] **Step 4: Implement plan-only refresh with immutable-artifact proof**

  `refresh_r1_plan.py` must:

  1. build and require a valid complete catalogue;
  2. write `canonical_r1_before.sha256.json`;
  3. invoke `scripts/run_r1_sweep.py --experiment ... --profile paper --output-root <root.parent>`, deliberately omitting `--execute`;
  4. rebuild the catalogue and write `canonical_r1_after.sha256.json`;
  5. compare dictionaries exactly and raise on any change;
  6. require a 100-job `planned_jobs.json` and a complete canonical audit;
  7. write `refresh_report.json` including the command, both manifest paths, plan count, and `canonical_artifacts_unchanged=true`.

- [ ] **Step 5: Run audit/refresh tests and the existing workflow suite**

  Run:

  ```bash
  python -m unittest tests.test_balanced_experiment.CanonicalAuditTests -v
  python -m unittest tests.test_workflow -v
  ```

  Expected: both commands PASS.

- [ ] **Step 6: Commit canonical audit and refresh**

  ```bash
  git add workflow/analysis/audit_r1.py workflow/analysis/refresh_r1_plan.py tests/test_balanced_experiment.py
  git commit -m "fix: audit only canonical R1 points"
  ```

---

### Task 3: Canonical Layout Sweep Discovery and Provenance

**Files:**
- Modify: `workflow/run_lifting_sweep.py`
- Modify: `tests/test_balanced_experiment.py`

**Interfaces:**
- Consumes: `build_catalogue`, experiment config, and existing `run_pipeline`.
- Produces: `discover(root, experiment_path, profile="paper", workloads=None, require_complete=True) -> tuple[list[Path], dict]` and schema-3 `sweep_status.json`.

- [ ] **Step 1: Write failing lifting discovery and report tests**

  Add tests proving `discover` returns the four fixture canonical points, not the extra duplicate; filtering one workload returns exactly two points; and `require_complete=True` rejects a missing canonical point. Patch `one_job` for the CLI/report test and assert the report includes `config_sha256`, exact classification, `canonical_expected`, `canonical_selected`, `excluded_noncanonical`, and `run_r2=false`.

  ```python
  points, catalogue = discover(root, experiment, "paper", None, True)
  self.assertEqual(len(points), 4)
  self.assertEqual(len(catalogue["excluded_noncanonical"]), 1)
  self.assertNotIn("corrupt_duplicate", "\n".join(map(str, points)))
  ```

- [ ] **Step 2: Run focused tests and observe the duplicate failure**

  Run:

  ```bash
  python -m unittest tests.test_balanced_experiment.CanonicalLiftingTests -v
  ```

  Expected: FAIL because current `discover` recursively accepts the duplicate metadata directory.

- [ ] **Step 3: Use canonical discovery and strengthen sweep status**

  Add CLI options:

  ```text
  --r1-experiment configs/experiments/r1_cache_sweep.json
  --r1-profile paper
  --allow-incomplete-canonical
  ```

  Default behavior requires all 100 canonical points before workload filtering. The report must record the canonical catalogue summary, resolved config path and SHA-256, experiment classification, selected workload count, method, executed/skipped/failed counts, `run_r2`, and `contains_r2`. For layout-only runs, `contains_r2` must be false and every successful/skipped point must have `ipc2=null` and `bips2=null`.

  Strengthen `completed` with a `required_artifacts` tuple covering `run_config.json`, `mcpat/mcpat.json`, `cacti/cacti_characterization.json`, `modules.json`, `hotspot/layout.json`, `hotspot/thermal_result.json`, `performance.json`, `r2_latency.json`, and `pipeline_summary.json`; CLIP-3D additionally requires `optimizer_report.json` and `layout_selection.json`.

- [ ] **Step 4: Run focused and full existing tests**

  Run:

  ```bash
  python -m unittest tests.test_balanced_experiment.CanonicalLiftingTests -v
  python -m unittest tests.test_workflow -v
  ```

  Expected: PASS.

- [ ] **Step 5: Commit canonical lifting**

  ```bash
  git add workflow/run_lifting_sweep.py tests/test_balanced_experiment.py
  git commit -m "fix: constrain lifting sweeps to canonical R1 grid"
  ```

---

### Task 4: Predeclared Balanced-50 Manifest and Layout Preflight

**Files:**
- Create: `workflow/experiments/__init__.py`
- Create: `workflow/experiments/balanced50.py`
- Create: `configs/experiments/balanced50_traffic_weighted.json`
- Modify: `tests/test_balanced_experiment.py`

**Interfaces:**
- Consumes: two complete canonical layout roots, canonical grid config, and exploratory experiment config.
- Produces: `load_selection(path: Path) -> dict`, `selection_keys(manifest: dict) -> list[ArchitectureKey]`, `validate_selection(manifest: dict, grid: dict) -> dict`, and `validate_layout_roots(fixed_root: Path, clip_root: Path, keys: list[ArchitectureKey], config_path: Path) -> dict`.

- [ ] **Step 1: Write failing deterministic-selection tests**

  Assert the tracked manifest has five workloads, ten unique pairs per workload, 50 unique keys, every configured L1D and L2 level within every workload, and this exact per-workload order:

  ```python
  expected_pairs = [
      ("16kB", "128kB"), ("16kB", "512kB"), ("16kB", "2048kB"),
      ("32kB", "256kB"), ("32kB", "1024kB"),
      ("64kB", "128kB"), ("64kB", "512kB"), ("64kB", "2048kB"),
      ("128kB", "256kB"), ("128kB", "1024kB"),
  ]
  ```

  Add layout-root fixture tests that reject a missing point, mismatched layout method, changed embedded run config, wrong experiment classification, missing traffic communication profile, or a non-null R2 result in the layout-only preflight.

- [ ] **Step 2: Run selection tests and verify missing module/manifest failures**

  Run:

  ```bash
  python -m unittest tests.test_balanced_experiment.BalancedSelectionTests -v
  ```

  Expected: FAIL because the manifest and validator do not exist.

- [ ] **Step 3: Add all 50 explicit ordered keys to the tracked manifest**

  The JSON contains schema version, selection name, resolved-by-validator config references, expected counts, selection rule, exact exploratory classification, and a `points` array. Construct the array by listing all ten pairs for `fft`, then `cholesky`, `stream`, `matmul`, and `stencil`; do not compute the tracked sample from results at runtime.

- [ ] **Step 4: Implement selection and layout-root validation**

  `validate_layout_roots` verifies every selected point and also scans each root to prove it represents the same complete 100-point canonical key set. It verifies the embedded `run_config.json["config"]` equals the selected experiment config, the SHA-256 matches, required files exist, fixed/CLIP methods are correct, classification is non-formal, communication weights are present, and layout-only summaries contain no IPC2/BIPS2. Return a JSON-serializable preflight report with 100/100 root counts and 50 selected pairs.

- [ ] **Step 5: Run selection/preflight tests**

  Run:

  ```bash
  python -m unittest tests.test_balanced_experiment.BalancedSelectionTests -v
  ```

  Expected: PASS.

- [ ] **Step 6: Commit the predeclared sample**

  ```bash
  git add workflow/experiments configs/experiments/balanced50_traffic_weighted.json tests/test_balanced_experiment.py
  git commit -m "feat: define balanced 50-point experiment"
  ```

---

### Task 5: Strict R2 Cache Validation and Exact Reuse Attachment

**Files:**
- Modify: `workflow/r2/run_r2.py`
- Create: `workflow/r2/reuse_result.py`
- Modify: `tests/test_balanced_experiment.py`

**Interfaces:**
- Consumes: fixed and CLIP point directories, the canonical R1 directory, and selected experiment config.
- Produces: `validate_local_result(r1_dir: Path, latency_path: Path, output_dir: Path) -> dict`, `validate_reuse(fixed_point: Path, clip_point: Path, r1_dir: Path, config_path: Path) -> dict`, and `attach_reused_result(...) -> dict`.

- [ ] **Step 1: Write failing stale-cache and reuse-provenance tests**

  Add tests for all acceptance and rejection branches:

  - local successful R2 result points at the requested latency path and matching R1 identity: accept;
  - result points at another latency path: reject instead of silently resuming;
  - one `gem5_overrides` field differs: reuse rejects;
  - `gem5_args` differ while overrides match: reject malformed vector;
  - architecture key, canonical R1 identity, instruction scope, config hash, source status, or source result provenance differs: reject;
  - exact match: write `r2_reuse.json`, attach source IPC2 to CLIP, recompute BIPS2 using CLIP sustainable frequency, and preserve `r2_source` as the fixed result path.

  ```python
  decision = validate_reuse(fixed, clip, r1, config)
  self.assertTrue(decision["accepted"])
  summary = attach_reused_result(fixed, clip, r1, config)
  self.assertEqual(summary["ipc2"], 3.25)
  self.assertEqual(summary["bips2"], 3.25 * 1.1)
  self.assertEqual(read_json(clip / "r2_reuse.json")["decision"], "accepted")
  ```

- [ ] **Step 2: Run strict reuse tests and verify failures**

  Run:

  ```bash
  python -m unittest tests.test_balanced_experiment.StrictR2ReuseTests -v
  ```

  Expected: FAIL because current R2 resume checks only status and current pipeline reuse does not verify full provenance.

- [ ] **Step 3: Make local R2 resumption provenance-safe**

  Before returning an existing result in `run_r2.run`, verify:

  ```python
  Path(result["latency_vector"]).resolve() == latency_path.resolve()
  result["r1_directory"] == str(r1_dir.resolve())
  result["r1_metadata_sha256"] == sha256_file(r1_dir / "r1_metadata.json")
  result["r1_stats_sha256"] == sha256_file(r1_dir / "stats.txt")
  result["latency_sha256"] == sha256_file(latency_path)
  ```

  Record these fields in every new schema-3 `r2_result.json` and status. If an existing cache is incompatible, raise a clear `ValueError` requiring `--rerun`; never overwrite a successful directory implicitly.

- [ ] **Step 4: Implement exact fixed-to-CLIP reuse**

  Validate complete `gem5_overrides` equality and verify each vector's `gem5_args` is the canonical argument rendering of those overrides. Compare canonical architecture keys, R1 directory, R1 metadata/stats hashes, instruction-window fields, exact embedded configs, and experiment config SHA-256. The reuse artifact records source/target vector paths and hashes, overrides, source result/status paths and hashes, source IPC2, canonical key, R1 identity, config identity, and `decision="accepted"`.

  Recompute target `performance.json` via the existing sustainable-frequency evaluator with reused IPC2, then update target `pipeline_summary.json` with `ipc2`, `bips2`, `r2_source`, `r2_reused=true`, `r2_reuse_artifact`, and no fake gem5 elapsed time.

- [ ] **Step 5: Run reuse tests and existing workflow tests**

  Run:

  ```bash
  python -m unittest tests.test_balanced_experiment.StrictR2ReuseTests -v
  python -m unittest tests.test_workflow -v
  ```

  Expected: PASS.

- [ ] **Step 6: Commit strict R2 provenance**

  ```bash
  git add workflow/r2/run_r2.py workflow/r2/reuse_result.py tests/test_balanced_experiment.py
  git commit -m "fix: validate R2 provenance before reuse"
  ```

---

### Task 6: Resumable Fixed-First Paired R2 Scheduler

**Files:**
- Create: `workflow/r2/run_paired_sweep.py`
- Modify: `tests/test_balanced_experiment.py`

**Interfaces:**
- Consumes: Task 4 preflight, Task 5 local/reuse validation, existing `run_r2.run` and `attach_result.attach`.
- Produces: `run_pair(key, r1_root, fixed_root, clip_root, config_path, rerun=False) -> dict` and `run_sweep(..., jobs: int = 1, rerun: bool = False) -> dict`.

- [ ] **Step 1: Write failing pair-order, branch, and resume tests**

  Patch the expensive gem5 call. Assert:

  - fixed `run_r2` and local attach happen before any CLIP decision;
  - equal overrides call `attach_reused_result` and make one physical-run record;
  - unequal overrides call CLIP `run_r2` and local attach, making two physical-run records;
  - a compatible completed pair is skipped on the second invocation;
  - an interrupted fixed success plus incomplete CLIP resumes at the CLIP decision;
  - one pair failure is retained while other pairs complete;
  - non-positive `--jobs` is rejected;
  - `--limit 1` deterministically chooses only the first manifest entry for a real-run smoke test;
  - process exit is nonzero when any of 50 pairs remains incomplete.

  ```python
  result = run_pair(key, r1_root, fixed_root, clip_root, config)
  self.assertEqual(result["state"], "success")
  self.assertEqual(result["physical_r2_runs"], 1)
  self.assertTrue(result["clip3d_reused_fixed_r2"])
  self.assertEqual(events, ["fixed-run", "fixed-attach", "reuse-attach"])
  ```

- [ ] **Step 2: Run scheduler tests and verify missing module failure**

  Run:

  ```bash
  python -m unittest tests.test_balanced_experiment.PairedR2RunnerTests -v
  ```

  Expected: FAIL because the paired scheduler does not exist.

- [ ] **Step 3: Implement atomic per-pair state and fixed-first execution**

  Each pair writes `<status_root>/<workload>/l1d_<L1D>/l2_<L2>/pair_status.json` with `running`, then `success` or `failed`. A successful record includes vector hashes, whether reuse occurred, physical run count, IPC2/BIPS2 values, and artifact paths. `run_pair` first validates or produces fixed R2, then compares/validates CLIP. Compatible existing results must pass the same validators as newly generated results.

  `run_sweep` validates the complete layout roots before creating jobs, uses `ProcessPoolExecutor(max_workers=jobs)` plus `as_completed` for independent architecture pairs, and atomically rewrites experiment `status.json` after each returned pair. The final status records exactly 50 selected pairs, completed/failed counts, reuse count, separate-CLIP count `K`, and physical run count `50 + K` when complete. A positive smoke-test limit records `limited_run=true` and can succeed for the selected subset, but can never mark the full experiment complete.

- [ ] **Step 4: Add the CLI**

  Required arguments are `--r1-root`, `--fixed-root`, `--clip-root`, `--selection`, `--config`, and `--status-root`; optional controls are `--jobs`, `--limit`, and `--rerun`. Print progress after each pair. A normal run returns nonzero unless all 50 pairs are valid; a limited smoke run returns zero only when every pair in its deterministic subset is valid and still records `complete=false` for the full experiment.

- [ ] **Step 5: Run scheduler and full focused tests**

  Run:

  ```bash
  python -m unittest tests.test_balanced_experiment.PairedR2RunnerTests -v
  python -m unittest tests.test_balanced_experiment -v
  ```

  Expected: PASS.

- [ ] **Step 6: Commit paired scheduling**

  ```bash
  git add workflow/r2/run_paired_sweep.py tests/test_balanced_experiment.py
  git commit -m "feat: run resumable paired R2 sweep"
  ```

---

### Task 7: Strict Paired Aggregation

**Files:**
- Create: `workflow/analysis/summarize_paired_sweep.py`
- Modify: `tests/test_balanced_experiment.py`

**Interfaces:**
- Consumes: 50 successful pair statuses and fixed/CLIP pipeline summaries.
- Produces: `summarize(fixed_root: Path, clip_root: Path, selection_path: Path, config_path: Path, csv_path: Path, json_path: Path) -> dict`.

- [ ] **Step 1: Write failing paired-report tests**

  Build small five-workload fixtures and assert rejection of missing IPC2/BIPS2, proxy markers, duplicate keys, mixed config hashes, wrong classification, or an unvalidated reused row. For a valid 50-row fixture assert:

  ```python
  self.assertEqual(result["point_count"], 50)
  self.assertEqual(set(result["workloads"]),
                   {"fft", "cholesky", "stream", "matmul", "stencil"})
  self.assertTrue(all(item["n"] == 10
                      for item in result["workloads"].values()))
  self.assertEqual(result["score_definition"],
                   "paired measured BIPS2=IPC2*f_sus; exact validated reuse allowed")
  self.assertTrue(result["complete"])
  ```

  Verify CSV temperature columns retain six fractional digits and report columns include both methods' Tmax, frequency, wire cycles, vector hash, IPC2, BIPS2, reuse flag, absolute difference, and percentage difference.

- [ ] **Step 2: Run aggregation tests and verify missing module failure**

  Run:

  ```bash
  python -m unittest tests.test_balanced_experiment.PairedAggregationTests -v
  ```

  Expected: FAIL because the paired summarizer does not exist.

- [ ] **Step 3: Implement strict row loading and statistics**

  Validate each row against the selection and config before computing:

  ```python
  ratio = clip_bips2 / fixed_bips2
  percent_change = 100.0 * (ratio - 1.0)
  geometric_mean_ratio = math.exp(sum(math.log(value) for value in ratios) /
                                  len(ratios))
  ```

  Per workload and aggregate output includes arithmetic mean ratio, geometric mean ratio, median percentage change, win/tie/loss counts using exact numeric comparison, fixed-to-CLIP reuse count, and separate-CLIP R2 count. Include parameter values and the complete non-formal classification at top level.

- [ ] **Step 4: Run aggregation and all automated tests**

  Run:

  ```bash
  python -m unittest tests.test_balanced_experiment.PairedAggregationTests -v
  python -m unittest discover -s tests -v
  ```

  Expected: PASS.

- [ ] **Step 5: Commit strict paired reporting**

  ```bash
  git add workflow/analysis/summarize_paired_sweep.py tests/test_balanced_experiment.py
  git commit -m "feat: summarize paired balanced R2 results"
  ```

---

### Task 8: Operational Documentation, Verification, and Experiment Launch

**Files:**
- Create: `docs/balanced50_experiment_zh.md`
- Generated but not committed: `results/operational_balanced50_traffic_weighted/*`
- Generated but not committed: `runs/operational_balanced50_traffic_weighted/*`

**Interfaces:**
- Consumes: all preceding commands and artifacts.
- Produces: reproducible commands, immutable R1 evidence, 200 layout-only outputs, 50 complete pairs, and final paired CSV/JSON.

- [ ] **Step 1: Document exact roots, commands, resumption, and interpretation**

  Use these stable paths:

  ```bash
  R1=runs/architecture_sweep/r1/paper
  CFG=configs/experiments/clip3d_constrained_5p0_raw_power_p1_lambda0020119_traffic_weighted_exploratory.json
  SELECT=configs/experiments/balanced50_traffic_weighted.json
  ROOT=runs/operational_balanced50_traffic_weighted
  FIXED=$ROOT/fixed_bin
  CLIP3D=$ROOT/clip3d
  STATUS=$ROOT/paired_r2_status
  RESULTS=results/operational_balanced50_traffic_weighted
  ```

  State explicitly that rerunning the same commands resumes validated outputs, `--rerun` is opt-in, four workers are the initial recommendation, and this is non-formal exploratory evidence.

- [ ] **Step 2: Run complete code verification before touching experiment outputs**

  Run:

  ```bash
  source scripts/env.sh
  python -m unittest discover -s tests -v
  python -m compileall -q workflow scripts tests
  git diff --check
  ```

  Expected: every test passes, compilation exits zero, and `git diff --check` prints nothing.

- [ ] **Step 3: Refresh the 100-job R1 plan without executing R1**

  Run:

  ```bash
  mkdir -p results/operational_balanced50_traffic_weighted/r1_audit
  python -m workflow.analysis.refresh_r1_plan \
    --root runs/architecture_sweep/r1/paper \
    --experiment configs/experiments/r1_cache_sweep.json \
    --profile paper \
    --output results/operational_balanced50_traffic_weighted/r1_audit
  python -m workflow.analysis.audit_r1 \
    --root runs/architecture_sweep/r1/paper \
    --experiment configs/experiments/r1_cache_sweep.json \
    --profile paper \
    --output results/operational_balanced50_traffic_weighted/r1_audit/audit.json \
    --expected-points 100 \
    --require-complete
  ```

  Expected: 100 canonical valid successes, one excluded noncanonical duplicate, 100 planned jobs, one instruction scope, and identical before/after canonical hashes.

- [ ] **Step 4: Generate/resume 100 fixed-bin layout-only points**

  Run:

  ```bash
  python -m workflow.run_lifting_sweep \
    --r1-root "$R1" \
    --r1-experiment configs/experiments/r1_cache_sweep.json \
    --r1-profile paper \
    --output-root "$FIXED" \
    --config "$CFG" \
    --layout-method fixed-bin \
    --jobs 4
  ```

  Expected: `discovered=100`, no failed point, and no R2 value in any summary.

- [ ] **Step 5: Generate/resume 100 CLIP-3D layout-only points**

  Run:

  ```bash
  python -m workflow.run_lifting_sweep \
    --r1-root "$R1" \
    --r1-experiment configs/experiments/r1_cache_sweep.json \
    --r1-profile paper \
    --output-root "$CLIP3D" \
    --config "$CFG" \
    --layout-method clip3d \
    --jobs 4
  ```

  Expected: `discovered=100`, no failed point, optimizer/layout-selection evidence at every point, and no R2 value in any summary.

- [ ] **Step 6: Run preflight and one real selected pair at concurrency one**

  Use the paired runner's smoke-only `--limit 1`, which always selects the first manifest entry and records the limit in status. Run:

  ```bash
  python -m workflow.r2.run_paired_sweep \
    --r1-root "$R1" --fixed-root "$FIXED" --clip-root "$CLIP3D" \
    --selection "$SELECT" --config "$CFG" --status-root "$STATUS" \
    --jobs 1 --limit 1
  ```

  Expected: one pair has valid fixed and CLIP BIPS2, using reuse only if full overrides match.

- [ ] **Step 7: Run/resume all 50 selected pairs with four workers**

  Run:

  ```bash
  python -m workflow.r2.run_paired_sweep \
    --r1-root "$R1" --fixed-root "$FIXED" --clip-root "$CLIP3D" \
    --selection "$SELECT" --config "$CFG" --status-root "$STATUS" \
    --jobs 4
  ```

  Expected on completion: 50 successful pairs, `K` recorded separate CLIP runs, and `physical_r2_runs=50+K`.

- [ ] **Step 8: Produce and verify the paired report**

  Run:

  ```bash
  mkdir -p "$RESULTS"
  python -m workflow.analysis.summarize_paired_sweep \
    --fixed-root "$FIXED" --clip-root "$CLIP3D" \
    --selection "$SELECT" --config "$CFG" \
    --csv "$RESULTS/paired_results.csv" \
    --output "$RESULTS/paired_summary.json"
  ```

  Expected: 50 rows, 10 per workload, no proxy values, valid reuse evidence for every reused row, and `complete=true`.

- [ ] **Step 9: Commit operational documentation and final code-state evidence**

  ```bash
  git add docs/balanced50_experiment_zh.md
  git commit -m "docs: add balanced 50-point runbook"
  git status --short
  ```

  Do not add generated `runs/` or `results/` artifacts unless the repository's existing tracking policy explicitly includes a small final summary.
