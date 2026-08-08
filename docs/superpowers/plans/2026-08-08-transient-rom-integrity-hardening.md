# Transient-ROM Integrity Hardening Implementation Plan

> For agentic workers: REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

Goal: Ensure optional transient-rom results are reproducible from current inputs and stored evidence, without changing the default steady Eq. (13) flow.

Architecture: Introduce shared evidence helpers for classifications, regular-file topology, hashes and trace manifests. Use them at package reuse, deterministic ROM-cache replay and final HotSpot replay. A canonical frequency/PSS contract and independent analytic fixtures eliminate different numerical horizons and circular tests.

Tech Stack: Python 3.12, unittest, NumPy, SciPy, JSON/SHA-256; all external tools mocked in tests.

## Global Constraints

- Do not modify workflow/thermal/sustainable_frequency.py, workflow/floorplan/optimize_layout.py, source R1 directories, existing R2 results or live experiments.
- thermal-mode steady remains default and retains its Eq. (13) behavior, artifacts and tests.
- Every ROM artifact uses thermal_mode="transient-rom", non_formal=true, paper_equivalent=false.
- Calibration remains exactly 8 training plus 2 holdout HotSpot calls; optimizer makes zero HotSpot/subprocess calls.
- Dynamic power uses s=f/f0; leakage is fixed; duration uses 1/s.
- Frequency grids are finite, positive, in [fmin_ghz,f0_ghz], sorted, and contain both endpoints.
- ROM and HotSpot evaluate all pss_period_repeats; they judge PSS by final full-grid delta and peak by final-period inclusive peak.
- Reuse accepts only regular non-symlink descendants of declared roots; output directories are disjoint from packages and read-only inputs.
- bips2_trans is present only after trace-bound final HotSpot validation and R2. Invalid evidence records rom_final_validation_failed/validation_contract_error and never starts R2.
- Use PYTHONPATH=. /home/zyjiang/Agenticflow/CLIP/.venv/bin/python in this linked worktree. Tests never invoke HotSpot, gem5, McPAT, CACTI, R1 or R2.
- Use apply_patch; each task commits only its own files.

---

### Task 1: Make classification, source inventory and path topology strict

Files:

- Create: workflow/transient/rom/evidence.py
- Modify: workflow/transient/rom/contracts.py
- Modify: workflow/transient/rom/materialize_calibration.py
- Modify: workflow/transient/rom/run_pipeline.py
- Modify: workflow/transient/rom/calibrate_rom.py
- Modify: tests/test_transient_rom.py

Interfaces:

    ROM_CLASSIFICATION: dict[str, object]
    require_rom_classification(value: object, context: str) -> dict
    require_regular_descendant(root: Path, relative: str, context: str) -> Path
    require_new_output_root(path: Path, forbidden_roots: list[Path]) -> Path
    sha256_identity(path: Path) -> str

- [ ] Step 1: Write failing topology tests.

    def test_reuse_rejects_contradictory_classification_and_symlink_tree(self):
        self.rewrite_and_rehash(self.package / "rom_acceptance.json",
                                thermal_mode="steady", non_formal=False)
        with self.assertRaisesRegex(ValueError, "classification"):
            require_accepted_package(self.package, self.identity())
        self.replace_case_directory_with_symlink("training_tier0_corner0")
        with self.assertRaisesRegex(ValueError, "symlink"):
            require_package_calibration_evidence(self.package, self.identity(), self.settings())

    def test_reuse_rejects_missing_packaged_source_power(self):
        self.remove_packaged_source("source_power_windows.json")
        with self.assertRaisesRegex(ValueError, "source power"):
            require_package_calibration_evidence(self.package, self.identity(), self.settings())

- [ ] Step 2: Run the new tests.

Run: PYTHONPATH=. /home/zyjiang/Agenticflow/CLIP/.venv/bin/python -m unittest tests.test_transient_rom.ROMContractTests -v

Expected: RED because classification, missing source and a directory symlink are accepted.

- [ ] Step 3: Implement minimum shared helpers.

    ROM_CLASSIFICATION = {"thermal_mode": "transient-rom",
                          "non_formal": True, "paper_equivalent": False}

    def require_regular_descendant(root: Path, relative: str, context: str) -> Path:
        candidate = Path(relative)
        if candidate.is_absolute() or ".." in candidate.parts:
            raise ValueError(f"{context} must be package-relative")
        current = Path(root)
        for part in candidate.parts:
            current /= part
            if current.is_symlink():
                raise ValueError(f"{context} contains a symlink")
        if not current.is_file() or current.is_symlink():
            raise ValueError(f"{context} must be a regular file")
        return current

Require this classification on acceptance, manifest, reports and all inventory-listed ROM JSON. Copy/inventory modules, source power and config as required regular files; eliminate stale absolute source references.

- [ ] Step 4: Verify GREEN.

Run: PYTHONPATH=. /home/zyjiang/Agenticflow/CLIP/.venv/bin/python -m unittest tests.test_transient_rom.ROMContractTests tests.test_transient_rom.ROMCalibrationCaseTests -v

Expected: all tests pass and no artifact can escape through a symlink.

- [ ] Step 5: Commit.

    git add workflow/transient/rom/evidence.py workflow/transient/rom/contracts.py workflow/transient/rom/materialize_calibration.py tests/test_transient_rom.py
    git commit -m "fix: harden transient ROM artifact topology"

### Task 2: Bind calibration power/trace evidence and geometry to current inputs

Files:

- Modify: workflow/transient/generate_hotspot_trace.py
- Modify: workflow/transient/rom/materialize_calibration.py
- Modify: workflow/transient/rom/contracts.py
- Modify: workflow/transient/rom/calibration_design.py
- Modify: workflow/transient/rom/optimize_layout.py
- Modify: workflow/transient/rom/calibrate_rom.py
- Modify: tests/test_transient.py, tests/test_transient_rom.py

Interfaces:

    trace_input_identity(modules: Path, layout: Path, power_windows: Path,
                         config: Path, frequency_scale: float) -> dict
    validate_materialized_trace(case_dir: Path, expected_identity: dict,
                               require_temperature_trace: bool) -> dict
    require_design_matches_modules(design: dict, modules: dict) -> None

- [ ] Step 1: Write failing trace, design and interpolation tests.

    def test_emitted_trace_scales_dynamic_only(self):
        trace = materialize_trace(self.modules_path, self.layout_path,
                                  self.power_windows_path, self.output,
                                  self.config, frequency_scale=0.5, period_repeats=2)
        self.assertEqual(self.dynamic_rows(trace), self.source_dynamic_rows() * 0.5)
        self.assertEqual(self.leakage_rows(trace), self.source_leakage_rows())

    def test_reuse_rejects_rehashed_malformed_training_trace(self):
        self.write_one_cell_training_trace_and_refresh_all_internal_hashes()
        with self.assertRaisesRegex(ValueError, "training.*trace"):
            require_package_calibration_evidence(self.package, self.identity(), self.settings())

    def test_optimizer_rejects_shifted_fixed_module_or_domain_coordinate(self):
        self.shift_fixed_core_or_bilinear_corner_and_refresh_internal_hashes()
        with self.assertRaisesRegex(ValueError, "base layout|anchor"):
            optimize_transient_layout(self.modules_path, self.package, self.output,
                                      self.config_path, self.power_windows_path,
                                      hotspot=self.hotspot)

- [ ] Step 2: Run RED.

Run: PYTHONPATH=. /home/zyjiang/Agenticflow/CLIP/.venv/bin/python -m unittest tests.test_transient.TransientTraceTests tests.test_transient_rom.ROMOptimizerTests -v

Expected: component scaling, malformed training trace and modified package geometry are insufficiently checked.

- [ ] Step 3: Record and replay canonical trace evidence.

Add modules/layout/power/config hashes, power-trace identity/hash, grid names/count, scale and duration transform to each trace manifest. Recompute expected PRBS windows and emitted power rows, then parse temperature traces before package reuse. Derive baseline layout from current modules; require every non-L2 design module and every domain corner/Delaunay coordinate to equal its named anchor. Validate standalone calibration input before case creation.

- [ ] Step 4: Verify GREEN.

Run: PYTHONPATH=. /home/zyjiang/Agenticflow/CLIP/.venv/bin/python -m unittest tests.test_transient.TransientTraceTests tests.test_transient_rom.CalibrationDesignTests tests.test_transient_rom.ROMCalibrationCaseTests tests.test_transient_rom.ROMOptimizerTests -v

Expected: any changed input component, trace shape/grid, fixed module or domain coordinate is rejected before fitting/optimization.

- [ ] Step 5: Commit.

    git add workflow/transient/generate_hotspot_trace.py workflow/transient/rom/materialize_calibration.py workflow/transient/rom/contracts.py workflow/transient/rom/calibration_design.py workflow/transient/rom/optimize_layout.py workflow/transient/rom/calibrate_rom.py tests/test_transient.py tests/test_transient_rom.py
    git commit -m "fix: bind transient ROM calibration evidence to inputs"

### Task 3: Use one fixed-horizon PSS and endpoint-bounded frequency contract

Files:

- Modify: workflow/transient/rom/layout_rom.py
- Modify: workflow/transient/rom/optimize_layout.py
- Modify: workflow/transient/rom/run_pipeline.py
- Modify: tests/test_transient_rom.py

Interfaces:

    canonical_frequency_grid(config: dict) -> list[float]
    evaluate_layout_rom(model: StateSpaceModel, design: dict, power_windows: dict,
                        layout: dict, frequency_ghz: float, settings: ROMSettings,
                        config: dict) -> dict

- [ ] Step 1: Write failing numerical tests.

    def test_rom_does_not_accept_an_early_tolerance_dip(self):
        result = evaluate_layout_rom(self.non_normal_model(), self.design(),
                                     self.power_windows(), self.layout(), 2.0,
                                     self.settings(), self.config())
        self.assertEqual(result["periods_evaluated"], self.settings().pss_period_repeats)
        self.assertFalse(result["converged"])

    def test_frequency_grid_requires_fmin_and_f0_inside_bounds(self):
        for values in ([1.2], [0.8, 2.0], [1.0, 2.2], [1.1, 1.9]):
            with self.assertRaisesRegex(ValueError, "frequency grid"):
                canonical_frequency_grid(self.config_with_grid(values))

- [ ] Step 2: Run RED.

Run: PYTHONPATH=. /home/zyjiang/Agenticflow/CLIP/.venv/bin/python -m unittest tests.test_transient_rom.ROMEvaluationTests tests.test_transient_rom.ROMOptimizerTests -v

Expected: ROM stops early and arbitrary positive grids are accepted.

- [ ] Step 3: Implement canonical numerical contract.

Parse finite fmin_ghz/f0_ghz; canonicalize grid values, reject outside values and require both endpoints. Use helper from optimizer, pipeline and final validation. Remove ROM early break, propagate every configured period, decide convergence from final full-grid delta, and take final-period inclusive peak.

- [ ] Step 4: Verify GREEN.

Run: PYTHONPATH=. /home/zyjiang/Agenticflow/CLIP/.venv/bin/python -m unittest tests.test_transient_rom.ROMEvaluationTests tests.test_transient_rom.ROMPipelineTests tests.test_transient_rom.ROMOptimizerTests -v

Expected: early dips do not certify PSS; all searches share endpoint-bounded grid.

- [ ] Step 5: Commit.

    git add workflow/transient/rom/layout_rom.py workflow/transient/rom/optimize_layout.py workflow/transient/rom/run_pipeline.py tests/test_transient_rom.py
    git commit -m "fix: align transient ROM PSS and frequency semantics"

### Task 4: Recompute cached optimization before it can reach R2

Files:

- Modify: workflow/transient/rom/optimize_layout.py
- Modify: workflow/transient/rom/run_pipeline.py
- Modify: tests/test_transient_rom.py

Interfaces:

    replay_transient_optimization(modules_path: Path, package_dir: Path,
                                 config_path: Path, power_windows_path: Path,
                                 hotspot: Path) -> dict
    validate_cached_optimization(cached: dict, replayed: dict) -> None

- [ ] Step 1: Write failing cache-forgery test.

    def test_rerun_r2_rejects_cached_non_l2_move_grid_and_prediction_forgery(self):
        self.write_cached_optimization(self.layout_with_shifted_core(), [100.0], 999.0)
        with self.assertRaisesRegex(ValueError, "cached ROM optimization"):
            run_transient_rom_pipeline(self.source_r1, self.steady, self.output,
                                       self.config_path, self.transient_r1,
                                       calibrate=False, execute_r2=True, rerun_r2=True)
        self.run_r2.assert_not_called()

- [ ] Step 2: Run RED.

Run: PYTHONPATH=. /home/zyjiang/Agenticflow/CLIP/.venv/bin/python -m unittest tests.test_transient_rom.ROMPipelineTests -v

Expected: internally consistent cached report is trusted and can call R2.

- [ ] Step 3: Extract pure deterministic replay.

Use one pure lattice/refinement search for fresh optimization and cache validation. Replay from current modules/config/power/HotSpot identity and package; compare canonical JSON for candidates, selected layout, wire metrics, frequency grid, ROM temperature and predicted BIPS. Reject output roots inside package/input roots before writes. Replay makes zero HotSpot/subprocess calls.

- [ ] Step 4: Verify GREEN and no-tool invariant.

Run:

    PYTHONPATH=. /home/zyjiang/Agenticflow/CLIP/.venv/bin/python -m unittest tests.test_transient_rom.ROMOptimizerTests tests.test_transient_rom.ROMPipelineTests -v
    rg -n "run_hotspot|subprocess" workflow/transient/rom/optimize_layout.py

Expected: tests pass; rg prints no execution call in optimizer code.

- [ ] Step 5: Commit.

    git add workflow/transient/rom/optimize_layout.py workflow/transient/rom/run_pipeline.py tests/test_transient_rom.py
    git commit -m "fix: replay cached transient ROM optimization"

### Task 5: Bind final HotSpot trace cases to selected scientific inputs

Files:

- Modify: workflow/transient/verify_sustainable_frequency.py
- Modify: workflow/transient/rom/evidence.py
- Modify: workflow/transient/rom/run_pipeline.py
- Modify: tests/test_transient_rom.py

Interfaces:

    final_trace_identity(modules: Path, layout: Path, power_windows: Path,
                         config: Path, hotspot: Path, frequency_ghz: float) -> dict
    validate_final_search_evidence(result: dict, final_dir: Path,
                                   expected_identity: dict, settings: ROMSettings,
                                   config: dict) -> dict

- [ ] Step 1: Write failing final-trace provenance tests.

    def test_final_trace_from_another_layout_or_power_source_suppresses_r2(self):
        self.write_valid_final_cases(source_layout=self.other_layout)
        summary, _, r2 = self.run_pipeline_with_trace_artifacts()
        self.assertEqual(summary["final_validation_classification"], "validation_contract_error")
        r2.assert_not_called()

    def test_final_trace_rejects_changed_config_or_hotspot_identity(self):
        self.mutate_final_case_manifest("config_sha256")
        summary, _, r2 = self.run_pipeline_with_trace_artifacts()
        self.assertIsNone(summary["bips2_trans"])
        r2.assert_not_called()

- [ ] Step 2: Run RED.

Run: PYTHONPATH=. /home/zyjiang/Agenticflow/CLIP/.venv/bin/python -m unittest tests.test_transient_rom.ROMPipelineTests -v

Expected: another layout/workload can produce structurally valid trace evidence that reaches R2.

- [ ] Step 3: Persist/replay trace input identity.

Per frequency, persist identities for modules, selected layout, power, config and HotSpot plus scaled trace hashes/grid/scale. Derive expected identity before final replay; require exact match, parse real trace and replay canonical PSS/frequency search. Every mismatch writes existing failure summary as validation_contract_error; true all-unsafe/PSS-nonconverged classifications do not change.

- [ ] Step 4: Verify GREEN.

Run: PYTHONPATH=. /home/zyjiang/Agenticflow/CLIP/.venv/bin/python -m unittest tests.test_transient_rom.ROMPipelineTests tests.test_transient.TransientTraceTests -v

Expected: selected inputs mandatory, invalid trace makes no R2 call, valid thermal classes stay distinct.

- [ ] Step 5: Commit.

    git add workflow/transient/verify_sustainable_frequency.py workflow/transient/rom/evidence.py workflow/transient/rom/run_pipeline.py tests/test_transient_rom.py
    git commit -m "fix: bind final transient ROM traces to selected inputs"

### Task 6: Replace circular positives with independent numerical oracles

Files:

- Modify: tests/test_transient_rom.py
- Modify: tests/test_transient.py
- Modify: docs/transient_rom_usage_zh.md

Interfaces:

    analytic_periodic_trace(a_d: numpy.ndarray, b_d: numpy.ndarray,
                            inputs: list[numpy.ndarray], periods: int) -> list[numpy.ndarray]
    independent_rom_package_fixture(root: Path) -> dict

- [ ] Step 1: Write failing oracle tests.

    def test_fit_recovers_known_stable_state_space_from_independent_traces(self):
        fixture = independent_rom_package_fixture(self.tempdir)
        model, _ = fit_state_space(fixture["training_cases"], self.settings(), package_dir=fixture["root"])
        self.assertLess(numpy.linalg.norm(model.a_d - fixture["known_a_d"]), 1e-6)

    def test_independent_eight_plus_two_fixture_passes_gate_then_optimizer(self):
        fixture = independent_rom_package_fixture(self.tempdir)
        report = validate_calibration_holdouts(fixture["model"], fixture["design"],
                                               fixture["holdout_cases"], self.settings(),
                                               fixture["config"], output_dir=fixture["root"],
                                               identity=fixture["identity"], publish=False)
        self.assertTrue(report["accepted"])
        result = optimize_transient_layout(fixture["modules_path"], fixture["root"],
                                           fixture["optimizer_output"], fixture["config_path"],
                                           fixture["power_windows_path"],
                                           hotspot=fixture["hotspot_path"])
        self.assertEqual(result["hotspot_calls_inside_optimizer"], 0)

- [ ] Step 2: Run RED.

Run: PYTHONPATH=. /home/zyjiang/Agenticflow/CLIP/.venv/bin/python -m unittest tests.test_transient_rom.StateSpaceTests tests.test_transient_rom.ROMEvaluationTests -v

Expected: no positive fixture derives trace values independently of production ROM/fitting code.

- [ ] Step 3: Implement independent test-only source.

Build a two-cell stable recurrence directly with x[k+1]=A_d@x[k]+B_d@u[k]; generate eight anchors and two holdouts without production evaluator/fitter/search helpers. Write traces first, derive expected PSS/search second, retain mutations for model, power, trace, layout, identity/cache, and assert dynamic/leakage components individually.

- [ ] Step 4: Verify all relevant tests.

Run: PYTHONPATH=. /home/zyjiang/Agenticflow/CLIP/.venv/bin/python -m unittest tests.test_transient_rom tests.test_transient tests.test_workflow -v

Expected: PASS with only environment-dependent HotSpot tests skipped.

- [ ] Step 5: Commit.

    git add tests/test_transient_rom.py tests/test_transient.py docs/transient_rom_usage_zh.md
    git commit -m "test: add independent transient ROM numerical oracles"

### Task 7: Document integrity conditions and run completion gate

Files:

- Modify: docs/transient_rom_usage_zh.md
- Modify: docs/superpowers/specs/2026-08-08-transient-rom-integrity-hardening-design.md
- Modify: tests/test_transient_rom.py

- [ ] Step 1: Write failing guide assertion.

    def test_rom_guide_declares_trace_bound_replay_and_nonformal_scope(self):
        guide = Path("docs/transient_rom_usage_zh.md").read_text(encoding="utf-8")
        self.assertIn("trace-bound", guide)
        self.assertIn("non_formal=true", guide)
        self.assertIn("8+2", guide)

- [ ] Step 2: Run RED.

Run: PYTHONPATH=. /home/zyjiang/Agenticflow/CLIP/.venv/bin/python -m unittest tests.test_transient_rom.TransientROMDocumentationTests -v

Expected: guide lacks cache replay and trace-input binding.

- [ ] Step 3: Document verified behavior only.

Describe package topology, trace-bound cache/final replay, fixed-horizon PSS, endpoint-bounded frequencies, audit failure categories, zero extra HotSpot calls during replay and non-formal scope. Update design spec with implementation locations/tests.

- [ ] Step 4: Final verification.

    PYTHONPATH=. /home/zyjiang/Agenticflow/CLIP/.venv/bin/python -m unittest tests.test_transient_rom tests.test_transient tests.test_workflow -v
    PYTHONPATH=. /home/zyjiang/Agenticflow/CLIP/.venv/bin/python -m compileall workflow/transient/rom workflow/transient
    git diff --check
    git diff --name-only 451e75c..HEAD -- workflow/thermal/sustainable_frequency.py workflow/floorplan/optimize_layout.py

Expected: relevant tests and compile/diff checks pass; steady-path diff prints no file. Run full discovery separately and report the existing missing lambda_wire_report.json fixture without creating data.

- [ ] Step 5: Commit.

    git add docs/transient_rom_usage_zh.md docs/superpowers/specs/2026-08-08-transient-rom-integrity-hardening-design.md tests/test_transient_rom.py
    git commit -m "docs: record transient ROM integrity evidence"
