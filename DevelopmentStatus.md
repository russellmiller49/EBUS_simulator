# DevelopmentStatus.md

## Current Development State

This document is a point-in-time summary of what the standalone linear EBUS simulator can do today, what is partially complete, and what still needs to be built to reach the current v1 target.

The project is focused on:
- linear / convex CP-EBUS only
- preset-driven rendering from curated contacts and targets
- standalone local execution outside 3D Slicer

The project is not targeting:
- radial EBUS
- full bronchoscopy navigation
- web deployment
- a Slicer runtime module

---

## Overall Status

The repo has moved past the initial scaffold stage.

The current state is:
- geometry and preset pose generation are implemented and test-covered
- the original renderer is now an explicit `localizer` / QA renderer
- a first `physics` renderer exists and can render repo presets
- the physics renderer now supports tunable artifacts, debug-map export, basic evaluation summaries, and first-class consistency metrics
- shared render-state preparation is now explicit in `render_state.py`, with transform/sampling helpers in `transforms.py` and drawing helpers in `annotations.py`
- renderer tuning profiles are now externalized in `configs/render_profiles.yaml` and validated through `render_profiles.py`
- CI smoke coverage exists for validation, pose generation, and localizer rendering
- the test suite is now split into fast `tests/unit` coverage and slower dataset-backed `tests/integration` coverage
- the current desktop preset browser now exists behind the optional `ui` dependency, with queued rendering and a structured reviewer/teaching inspector
- a cross-preset `analyze-render-consistency` workflow now exists for localizer/physics divergence summaries

In practical terms, the repo is already useful for:
- loading and validating the dataset
- generating reproducible preset poses
- exporting localizer review renders
- exporting early physics-style CP-EBUS renders for inspection
- generating structured review bundles with localizer, physics, eval-summary, and rubric outputs

It is not yet at the intended end state for:
- polished desktop review workflow
- calibration against real CP-EBUS reference material
- a mature physics image model

---

## Current Focus

The active development focus is:
- hardening the desktop preset browser around responsiveness and reviewer ergonomics
- reducing avoidable cross-preset render instability where the evidence supports a narrow fix
- reducing the remaining physics sparsity gap on weak presets with metadata-visible, geometry-aware guardrails
- moving calibration/tuning iteration onto named profiles instead of repeated source edits
- manually validating the PySide6 browser against the checked-in dataset
- keeping the current review defaults stable while the desktop workflow comes online
- using first-class consistency metrics to separate anatomy-driven differences from normalization artifacts
- keeping the renderer split intact rather than reopening scaffold or geometry phases

---

## Implemented Now

### Repo and data plumbing
- Python packaging and editable install flow
- `make bootstrap`
- repo-relative / portable manifest roots
- support for `${REPO_ROOT}` and `${DATA_ROOT}`
- manifest-driven dataset loading
- NIfTI, VTP, and `.mrk.json` loading

### Validation and geometry
- `validate-case`
- `generate-poses`
- centerline graph abstraction
- preset pose generation from curated contact and target markups
- centerline-based shaft-axis resolution
- airway-contact refinement
- station-specific handling for multiple approaches such as station 7 `lms` and `rms`

### Rendering
- explicit render-engine dispatch
- first-class `PreparedRenderState` / `RenderContext` preparation shared by localizer and physics
- `localizer` renderer for geometry / QA review
- `physics` renderer for first-pass CP-EBUS-like B-mode output
- clean and debug render modes
- per-render JSON sidecars
- batch rendering through `render-all-presets`
- deterministic seeds recorded in metadata
- baseline and named physics tuning profiles with explicit CLI override precedence

### Physics renderer capabilities
- label-first acoustic property mapping from CT plus masks
- ray-domain sector sampling derived from the current `DevicePose`
- attenuation and log compression
- guarded blended high-percentile normalization when the physics upper tail is spike-dominated
- sparse-case target/anatomy support and wall-guardrail activation for very empty or wall-dominant sectors
- tunable speckle, reverberation, and distal shadow controls
- baseline and named physics tuning profiles covering normalization, sparse-support guardrails, artifact defaults, and related magic numbers
- optional debug-map export for boundary, transmission, shadow, reverberation, speckle, target focus, precompression, and compressed signal maps
- basic region-level evaluation summaries in metadata
- first-class per-render consistency metrics covering target position, occupancy, brightness, normalization stats, and sparse-support activation metadata
- active physics profile name, source path, effective settings, and explicit overrides recorded in metadata
- wall-eval fallback derived from the visible lumen shell when direct airway-wall samples are too sparse for review summaries

### Review and automation
- `review-presets` physics-aware batch review exports with deterministic preset/approach folders
- `compare-review-bundles` before/after summaries for calibration passes across review bundle runs
- `analyze-render-consistency` summaries for cross-preset localizer/physics divergence analysis
- JSON, CSV, and Markdown review indexes
- per-entry review sheets and a shared rubric template
- configurable geometry and physics auto-flag thresholds for review bundles
- default wall-contrast auto-flagging set to a conservative floor, with CLI support to disable it for experiments
- CI smoke workflow for validation, pose generation, and localizer rendering
- repo-root smoke targets such as `make render-smoke`, `make physics-smoke`, and `make ci-smoke`
- a fast local developer path through `make test-fast` / `pytest tests/unit -q`, while keeping dataset-backed integration coverage intact

### Desktop app
- `launch-app` CLI wired to the current PySide6 preset browser
- reusable non-Qt preset-browser render session for testing and future UI expansion
- 2D EBUS pane driven by the selected render engine
- 3D context pane derived from the existing localizer diagnostic/context path
- queued background rendering so the UI no longer blocks during preset changes
- structured inspector with preset, pose, anatomy-in-fan, review/eval, and render-setting sections
- inspector surfacing for occupancy, target-prominence, brightness, and normalization metrics
- inspector surfacing for consistency bucket and sparse-support activation details when they apply
- live warning and auto-flag surfacing inside the inspector using current render metadata plus derived review metrics
- screenshot export from the current browser state with state-aware default filenames

---

## Partially Complete

### Physics realism
The physics path is functional, but still intentionally simple.

What exists:
- plausible air-boundary brightening
- darker vessel regions
- target-focused hypoechoic shaping
- seeded artifact control

What is still missing:
- stronger calibration against reference CP-EBUS images
- more realistic reverberation / comet-tail behavior
- better tissue-specific texture modeling
- more formal physics-oriented review metrics

### Review workflow
The repo can already generate structured physics-aware review bundles, but calibration use of those bundles is still early.

Current state:
- localizer and physics renders are bundled per preset/approach
- evaluation summaries are extracted into reviewer-facing JSON artifacts
- localizer and physics consistency metrics are now stored as first-class metadata and bundled into review entries
- airway-wall eval stats now fall back to a lumen-adjacent shell when direct wall samples are too sparse
- physics debug maps can be bundled on request
- deterministic JSON, CSV, and Markdown indexes are generated for batch review
- before/after JSON, CSV, and Markdown comparison artifacts can now be generated from two review bundle summaries
- geometry and physics auto-flag thresholds can now be tuned from the review CLI for calibration passes, including opting out of the default wall threshold

Still missing:
- direct incorporation of real CP-EBUS reference imagery into the review loop
- more mature reviewer thresholds and scoring conventions
- calibration iteration informed by expert feedback rather than synthetic-only inspection

### Renderer code structure
The engine split is in place, and the shared layer is now more explicit.

Current state:
- `render_state.py` owns shared manifest/pose/device preparation and reusable `PreparedRenderState`
- `transforms.py` owns voxel/world/probe sampling and fan/plane mapping helpers
- `annotations.py` owns contour, marker, legend, and label drawing helpers
- `render_profiles.py` owns typed render-tuning profiles and YAML validation/loading
- `localizer_renderer.py` and `physics_renderer.py` consume the prepared state instead of rebuilding the request preamble ad hoc
- `rendering.py` is thinner and focused on orchestration, shared consistency helpers, and the remaining shared context/render helpers

Still missing:
- further reduction of the remaining orchestration/context helper weight in `rendering.py`
- cleaner separation of shared context-panel construction from engine-specific synthesis where it is still mixed
- broader profile-driven tuning beyond the current physics baseline/non-default examples

---

## Remaining Work

### Near-term remaining work
- continue tuning the physics renderer for better vessel, airway-wall, and node appearance
- expand named render profiles as calibration evidence accumulates, rather than editing constants in source
- use the bundled review outputs and consistency summaries to refine thresholds and reviewer ergonomics, now that wall metrics are no longer mostly null
- continue reducing the remaining shared helper weight in `rendering.py` now that render-state prep is explicit

### Major remaining milestone: polished desktop workflow
The main remaining milestone is turning the current preset browser into a polished desktop workflow.

Planned scope:
- `launch-app` CLI
- PySide6 desktop app
- preset selector
- approach selector
- localizer / physics engine toggle
- depth, angle, roll, gain, and attenuation controls
- overlay toggles
- 2D EBUS pane
- 3D context pane
- screenshot export

This is the largest remaining v1 deliverable.

---

## Known Limitations Right Now

- The desktop app currently requires the optional `ui` dependency (`PySide6`) rather than the default bootstrap path.
- The desktop UI now has queued rendering and a structured inspector, but it still needs broader manual interaction testing and packaged distribution setup.
- The full suite is still materially slower than the fast unit slice because the integration half intentionally exercises the checked-in dataset, rendering stack, review flow, and app session path.
- The physics renderer is still an inspectable first-pass model, not a mature ultrasound simulation.
- The physics path currently renders only the 2D sector view; it does not have a dedicated physics-specific 3D context panel.
- The review bundle is implemented, but its rubric and eval summaries are still lightweight rather than expert-calibrated.
- Some dataset validations still emit warnings rather than a fully clean result, including raw airway mesh metadata issues and a small number of borderline presets.

---

## Verified Commands

These commands are part of the current usable surface area:

- `make bootstrap`
- `make test-fast`
- `make test-integration`
- `make test`
- `.venv/bin/python -m pytest tests/unit -q`
- `.venv/bin/python -m pytest tests/integration -q`
- `validate-case configs/3d_slicer_files.yaml`
- `generate-poses configs/3d_slicer_files.yaml --report-json reports/pose_report.json`
- `render-preset configs/3d_slicer_files.yaml station_4r_node_b --output reports/renders/station_4r_node_b.png`
- `render-preset configs/3d_slicer_files.yaml station_4r_node_b --engine physics --mode clean --virtual-ebus false --debug-map-dir reports/renders/station_4r_node_b_debug_maps --output reports/renders/station_4r_node_b_physics.png`
- `render-preset configs/3d_slicer_files.yaml station_7_node_a --approach lms --engine physics --mode clean --virtual-ebus false --physics-profile sparse_support_boost --output reports/renders/station_7_node_a_lms_sparse_support_boost.png`
- `render-all-presets configs/3d_slicer_files.yaml --output-dir reports/renders/all_debug --mode debug`
- `review-presets configs/3d_slicer_files.yaml --output-dir reports/preset_review`
- `review-presets configs/3d_slicer_files.yaml --output-dir reports/preset_review --preset-id station_4r_node_b --preset-id station_7_node_a --physics-debug-maps`
- `review-presets configs/3d_slicer_files.yaml --output-dir reports/preset_review --preset-id station_4r_node_b --preset-id station_7_node_a --physics-debug-maps --physics-speckle-strength 0.22 --physics-reverberation-strength 0.28 --physics-shadow-strength 0.47 --warn-min-target-contrast 0.00 --warn-max-vessel-contrast -0.01 --width 64 --height 64`
- `analyze-render-consistency configs/3d_slicer_files.yaml --output-dir reports/consistency --physics-profile sparse_support_boost --width 64 --height 64`
- `analyze-render-consistency configs/3d_slicer_files.yaml --output-dir reports/consistency --width 64 --height 64`
- `compare-review-bundles reports/preset_review_20260316/review_summary.json reports/preset_review_stabilized/review_summary.json --output-dir reports/preset_review_stabilized`
- `launch-app configs/3d_slicer_files.yaml`

Note:
- `make test` re-enters `bootstrap.sh`; if the environment is already provisioned, `.venv/bin/python -m pytest -q` is the most direct rerun path.

---

## Latest Validation Snapshot

Latest verified run snapshot from `2026-03-18`:
- `make test-fast` -> `47 passed in 0.94s`
- `make test-integration` -> `26 passed in 1025.94s (0:17:05)`
- `.venv/bin/python -m pytest -q` -> `73 passed in 1034.21s (0:17:14)`
- `.venv/bin/render-preset configs/3d_slicer_files.yaml station_4r_node_b --output reports/_render_profiles/localizer_baseline.png --width 64 --height 64` -> succeeded with `engine: localizer`
- `.venv/bin/render-preset configs/3d_slicer_files.yaml station_4r_node_b --engine physics --mode clean --virtual-ebus false --simulated-ebus true --output reports/_render_profiles/physics_baseline.png --width 64 --height 64` -> succeeded with `physics_profile: baseline` and no explicit profile overrides
- `.venv/bin/render-preset configs/3d_slicer_files.yaml station_7_node_a --approach lms --engine physics --mode clean --virtual-ebus false --simulated-ebus true --physics-profile sparse_support_boost --speckle-strength 0.22 --output reports/_render_profiles/physics_sparse_support_boost.png --width 64 --height 64` -> succeeded with `physics_profile: sparse_support_boost` and `physics_profile_overrides: {'speckle_strength': 0.22}`
- `.venv/bin/launch-app --help` -> succeeded and printed the current CLI usage
- `QT_QPA_PLATFORM=offscreen .venv/bin/python - <<'PY' ... launch_app('configs/3d_slicer_files.yaml', width=64, height=64, close_after_ms=300000, close_on_first_render=True) ... PY` -> exited `0`
- `.venv/bin/render-preset configs/3d_slicer_files.yaml station_4r_node_b --output reports/_render_state_refactor/localizer_smoke.png --width 64 --height 64` -> succeeded with `engine: localizer`
- `.venv/bin/render-preset configs/3d_slicer_files.yaml station_4r_node_b --engine physics --mode clean --virtual-ebus false --simulated-ebus true --output reports/_render_state_refactor/physics_smoke.png --width 64 --height 64` -> succeeded with `engine: physics`
- `.venv/bin/review-presets configs/3d_slicer_files.yaml --output-dir reports/_review_render_state_smoke --preset-id station_4r_node_b --width 64 --height 64` -> succeeded with `review_count: 1` and emitted review summary/index artifacts
- `.venv/bin/python -m pip install -e '.[dev,ui]'` -> succeeded and refreshed the editable install plus console scripts
- `.venv/bin/analyze-render-consistency configs/3d_slicer_files.yaml --output-dir reports/_consistency_signal_support_all --width 64 --height 64` -> succeeded with `analysis_count: 16`, `support_logic_activations: 2`, `sparse_sector_cases: 7`, and representative improvements on `station_7_node_a/lms` and `station_11ri_node_a/default`
- `.venv/bin/python - <<'PY' ... compare_consistency_summaries(...) ... PY` -> matched `16` entries and reported `2` improvements each for empty-sector fraction, non-background occupancy, target contrast, and occupancy-gap trend
- `QT_QPA_PLATFORM=offscreen .venv/bin/python - <<'PY' ... launch_app('configs/3d_slicer_files.yaml', width=64, height=64, close_after_ms=300000, close_on_first_render=True) ... PY` -> exited `0` with the current app launch path
- `.venv/bin/review-presets configs/3d_slicer_files.yaml --output-dir reports/_review_calibration_all --physics-speckle-strength 0.22 --physics-reverberation-strength 0.28 --physics-shadow-strength 0.47 --warn-min-target-contrast 0.00 --warn-max-vessel-contrast -0.01 --width 128 --height 128` -> succeeded with `review_count: 16`, `flagged_count: 8`, `wall_present_count: 16`, `vessel_present_count: 15`, and broad-run wall contrast ranging from `0.0389` to `0.9648`
- `.venv/bin/review-presets configs/3d_slicer_files.yaml --output-dir reports/_review_calibration_default_wall --physics-speckle-strength 0.22 --physics-reverberation-strength 0.28 --physics-shadow-strength 0.47 --warn-min-target-contrast 0.00 --warn-max-vessel-contrast -0.01 --width 128 --height 128` -> succeeded with `review_count: 16`, `flagged_count: 8`, and `wall_contrast_vs_sector_min: 0.02`
- `.venv/bin/review-presets configs/3d_slicer_files.yaml --output-dir reports/_review_wall_optout_smoke --preset-id station_4r_node_b --physics-speckle-strength 0.22 --physics-reverberation-strength 0.28 --physics-shadow-strength 0.47 --warn-min-wall-contrast off --width 64 --height 64` -> succeeded with `review_count: 1`, `flagged_count: 1`, and `wall_contrast_vs_sector_min: None` in the bundle thresholds
- `.venv/bin/compare-review-bundles reports/preset_review_20260316/review_summary.json reports/preset_review_stabilized/review_summary.json --output-dir reports/_comparison_smoke` -> succeeded with `matched_entry_count: 16`, `before_flagged_count: 8`, `after_flagged_count: 4`, `resolved_flagged_count: 4`, and emitted `before_after_summary.{json,csv,md}`
- `.venv/bin/compare-review-bundles reports/_review_calibration_all/review_summary.json reports/_review_calibration_default_wall/review_summary.json --output-dir reports/_review_calibration_default_wall` -> succeeded with `matched_entry_count: 16`, `before_flagged_count: 8`, `after_flagged_count: 8`, `resolved_flagged_count: 0`, and `regressed_flagged_count: 0`
- `.venv/bin/analyze-render-consistency --help` -> succeeded and printed the consistency-analysis CLI usage
- `.venv/bin/analyze-render-consistency configs/3d_slicer_files.yaml --output-dir reports/_consistency_all --width 64 --height 64` -> succeeded with `analysis_count: 16`, emitted `consistency_summary.{json,md}` plus `consistency_entries.csv`, and identified `station_7_node_a / lms` as the most divergent current preset
- `.venv/bin/launch-app --help` -> succeeded
- `QT_QPA_PLATFORM=offscreen .venv/bin/python - <<'PY' ... launch_app('configs/3d_slicer_files.yaml', width=64, height=64, close_after_ms=300000, close_on_first_render=True) ... PY` -> succeeded with exit code `0` after the first completed browser render
- `.venv/bin/review-presets configs/3d_slicer_files.yaml --output-dir reports/_review_smoke_wall_eval --preset-id station_4r_node_b --preset-id station_7_node_a --physics-debug-maps --physics-speckle-strength 0.22 --physics-reverberation-strength 0.28 --physics-shadow-strength 0.47 --warn-min-target-contrast 0.00 --warn-max-vessel-contrast -0.01 --width 64 --height 64` -> succeeded with `review_count: 3`, `flagged_count: 2`, and non-null wall eval stats in the bundle entries

The review smoke bundle shape remains:
- deterministic review indexes
- per-entry localizer, physics, `eval_summary.json`, `review_sheet.md`, and optional `debug_maps/`
- distinct station 7 `lms` and `rms` review folders
- geometry and physics auto-flag reasons alongside non-null wall eval stats in the current smoke run

---

## Next Session Start Point

If work resumes in a fresh session, the highest-value next step is:

1. run `launch-app` against the checked-in dataset for a longer manual desktop validation pass
2. exercise preset switching, screenshot export, and inspector usefulness now that queued rendering is in place
3. add a few navigation/ergonomic affordances such as preset search, favorites, or recent renders if manual use shows friction

---

## Suggested Next Implementation Order

If development continues along the current roadmap, the next sensible order is:

1. use the current review bundles to tune physics appearance and reviewer thresholds
2. tighten the rubric and summary surfaces if real reviewer feedback shows gaps
3. continue polishing the PySide6 desktop preset browser

That keeps the work aligned with the current plan without jumping into unrelated scope such as scoring, free navigation, or radial EBUS support.
