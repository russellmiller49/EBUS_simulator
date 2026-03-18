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
- the physics renderer now supports tunable artifacts, debug-map export, and basic evaluation summaries
- CI smoke coverage exists for validation, pose generation, and localizer rendering
- a first desktop preset-browser slice now exists behind the optional `ui` dependency, with queued rendering and reviewer-facing summary text

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
- manually validating the PySide6 browser against the checked-in dataset
- keeping the current review defaults stable while the desktop workflow comes online
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
- `localizer` renderer for geometry / QA review
- `physics` renderer for first-pass CP-EBUS-like B-mode output
- clean and debug render modes
- per-render JSON sidecars
- batch rendering through `render-all-presets`
- deterministic seeds recorded in metadata

### Physics renderer capabilities
- label-first acoustic property mapping from CT plus masks
- ray-domain sector sampling derived from the current `DevicePose`
- attenuation and log compression
- tunable speckle, reverberation, and distal shadow controls
- optional debug-map export for boundary, transmission, shadow, reverberation, speckle, target focus, and precompression signal maps
- basic region-level evaluation summaries in metadata
- wall-eval fallback derived from the visible lumen shell when direct airway-wall samples are too sparse for review summaries

### Review and automation
- `review-presets` physics-aware batch review exports with deterministic preset/approach folders
- `compare-review-bundles` before/after summaries for calibration passes across review bundle runs
- JSON, CSV, and Markdown review indexes
- per-entry review sheets and a shared rubric template
- configurable geometry and physics auto-flag thresholds for review bundles
- default wall-contrast auto-flagging set to a conservative floor, with CLI support to disable it for experiments
- CI smoke workflow for validation, pose generation, and localizer rendering
- repo-root smoke targets such as `make render-smoke`, `make physics-smoke`, and `make ci-smoke`

### Desktop app
- `launch-app` CLI wired to a first PySide6 preset-browser slice
- reusable non-Qt preset-browser render session for testing and future UI expansion
- 2D EBUS pane driven by the selected render engine
- 3D context pane derived from the existing localizer diagnostic/context path
- queued background rendering so the UI no longer blocks during preset changes
- reviewer-facing summary panel with pose, overlay, eval, and sidecar metadata
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
The engine split is in place, but the shared rendering layer is still heavier than ideal.

Current state:
- `localizer_renderer.py` and `physics_renderer.py` exist
- `rendering.py` is acting as the shared orchestration layer

Still missing:
- further reduction of shared coupling
- cleaner separation of common render-state preparation from engine-specific image synthesis

---

## Remaining Work

### Near-term remaining work
- continue tuning the physics renderer for better vessel, airway-wall, and node appearance
- use the bundled review outputs to refine thresholds and reviewer ergonomics, now that wall metrics are no longer mostly null
- decide whether more render-state preparation should move out of `rendering.py`

### Major remaining milestone: desktop app
The main remaining milestone is turning the first preset-browser slice into a polished desktop workflow.

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
- The desktop UI now has queued rendering and reviewer summary text, but it still needs broader manual interaction testing and packaged distribution setup.
- The full `pytest -q` suite was not rerun after the latest desktop UI pass, so the current verified test evidence is the targeted app and review subsets plus the offscreen Qt smoke.
- The physics renderer is still an inspectable first-pass model, not a mature ultrasound simulation.
- The physics path currently renders only the 2D sector view; it does not have a dedicated physics-specific 3D context panel.
- The review bundle is implemented, but its rubric and eval summaries are still lightweight rather than expert-calibrated.
- Some dataset validations still emit warnings rather than a fully clean result, including raw airway mesh metadata issues and a small number of borderline presets.

---

## Verified Commands

These commands are part of the current usable surface area:

- `make bootstrap`
- `make test`
- `validate-case configs/3d_slicer_files.yaml`
- `generate-poses configs/3d_slicer_files.yaml --report-json reports/pose_report.json`
- `render-preset configs/3d_slicer_files.yaml station_4r_node_b --output reports/renders/station_4r_node_b.png`
- `render-preset configs/3d_slicer_files.yaml station_4r_node_b --engine physics --mode clean --virtual-ebus false --debug-map-dir reports/renders/station_4r_node_b_debug_maps --output reports/renders/station_4r_node_b_physics.png`
- `render-all-presets configs/3d_slicer_files.yaml --output-dir reports/renders/all_debug --mode debug`
- `review-presets configs/3d_slicer_files.yaml --output-dir reports/preset_review`
- `review-presets configs/3d_slicer_files.yaml --output-dir reports/preset_review --preset-id station_4r_node_b --preset-id station_7_node_a --physics-debug-maps`
- `review-presets configs/3d_slicer_files.yaml --output-dir reports/preset_review --preset-id station_4r_node_b --preset-id station_7_node_a --physics-debug-maps --physics-speckle-strength 0.22 --physics-reverberation-strength 0.28 --physics-shadow-strength 0.47 --warn-min-target-contrast 0.00 --warn-max-vessel-contrast -0.01 --width 64 --height 64`
- `compare-review-bundles reports/preset_review_20260316/review_summary.json reports/preset_review_stabilized/review_summary.json --output-dir reports/preset_review_stabilized`
- `launch-app configs/3d_slicer_files.yaml`

Note:
- `make test` re-enters `bootstrap.sh`; if the environment is already provisioned, `.venv/bin/python -m pytest -q` is the most direct rerun path.

---

## Latest Validation Snapshot

Latest verified run snapshot from `2026-03-17`:
- `.venv/bin/python -m pytest tests/test_app.py -q` -> `6 passed in 70.99s (0:01:10)`
- `.venv/bin/python -m pytest tests/test_review.py -q` -> `6 passed in 238.11s (0:03:58)`
- `.venv/bin/python -m pip install -e '.[ui]'` -> succeeded and installed `PySide6 6.10.2`
- `.venv/bin/review-presets configs/3d_slicer_files.yaml --output-dir reports/_review_calibration_all --physics-speckle-strength 0.22 --physics-reverberation-strength 0.28 --physics-shadow-strength 0.47 --warn-min-target-contrast 0.00 --warn-max-vessel-contrast -0.01 --width 128 --height 128` -> succeeded with `review_count: 16`, `flagged_count: 8`, `wall_present_count: 16`, `vessel_present_count: 15`, and broad-run wall contrast ranging from `0.0389` to `0.9648`
- `.venv/bin/review-presets configs/3d_slicer_files.yaml --output-dir reports/_review_calibration_default_wall --physics-speckle-strength 0.22 --physics-reverberation-strength 0.28 --physics-shadow-strength 0.47 --warn-min-target-contrast 0.00 --warn-max-vessel-contrast -0.01 --width 128 --height 128` -> succeeded with `review_count: 16`, `flagged_count: 8`, and `wall_contrast_vs_sector_min: 0.02`
- `.venv/bin/review-presets configs/3d_slicer_files.yaml --output-dir reports/_review_wall_optout_smoke --preset-id station_4r_node_b --physics-speckle-strength 0.22 --physics-reverberation-strength 0.28 --physics-shadow-strength 0.47 --warn-min-wall-contrast off --width 64 --height 64` -> succeeded with `review_count: 1`, `flagged_count: 1`, and `wall_contrast_vs_sector_min: None` in the bundle thresholds
- `.venv/bin/compare-review-bundles reports/preset_review_20260316/review_summary.json reports/preset_review_stabilized/review_summary.json --output-dir reports/_comparison_smoke` -> succeeded with `matched_entry_count: 16`, `before_flagged_count: 8`, `after_flagged_count: 4`, `resolved_flagged_count: 4`, and emitted `before_after_summary.{json,csv,md}`
- `.venv/bin/compare-review-bundles reports/_review_calibration_all/review_summary.json reports/_review_calibration_default_wall/review_summary.json --output-dir reports/_review_calibration_default_wall` -> succeeded with `matched_entry_count: 16`, `before_flagged_count: 8`, `after_flagged_count: 8`, `resolved_flagged_count: 0`, and `regressed_flagged_count: 0`
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

1. install the `ui` extra and run `launch-app` against the checked-in dataset for manual desktop validation
2. do a longer manual desktop pass around preset switching, screenshot export, and summary-panel usefulness now that queued rendering is in place
3. add a few navigation/ergonomic affordances such as preset search, favorites, or recent renders if manual use shows friction

---

## Suggested Next Implementation Order

If development continues along the current roadmap, the next sensible order is:

1. use the current review bundles to tune physics appearance and reviewer thresholds
2. tighten the rubric and summary surfaces if real reviewer feedback shows gaps
3. continue polishing the PySide6 desktop preset browser

That keeps the work aligned with the current plan without jumping into unrelated scope such as scoring, free navigation, or radial EBUS support.
