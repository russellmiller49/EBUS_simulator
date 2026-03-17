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
- the desktop app does not exist yet

In practical terms, the repo is already useful for:
- loading and validating the dataset
- generating reproducible preset poses
- exporting localizer review renders
- exporting early physics-style CP-EBUS renders for inspection

It is not yet at the intended end state for:
- desktop review workflow
- polished review bundles for real-vs-synthetic comparison
- a mature physics image model

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

### Review and automation
- `review-presets` batch review exports
- CI smoke workflow for validation, pose generation, and localizer rendering
- repo-root smoke targets such as `make render-smoke`, `make physics-smoke`, and `make ci-smoke`

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
The repo can already generate review renders, but the newer physics diagnostics are not fully integrated into the review layer yet.

Current state:
- debug maps can be exported per physics render
- evaluation summaries are written into physics metadata

Still missing:
- structured review folders that automatically bundle debug maps
- direct review consumption of the new physics `eval_summary`
- a lightweight written rubric for human comparison

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
- integrate physics debug maps and evaluation summaries into `review-presets`
- add a simple human review rubric document or report output
- continue tuning the physics renderer for better vessel, airway-wall, and node appearance
- decide whether more render-state preparation should move out of `rendering.py`

### Major remaining milestone: desktop app
The main unbuilt milestone is the desktop preset browser.

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

- There is no desktop UI yet.
- The physics renderer is still an inspectable first-pass model, not a mature ultrasound simulation.
- The physics path currently renders only the 2D sector view; it does not have a dedicated physics-specific 3D context panel.
- `review-presets` remains more localizer-centric than physics-centric.
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

---

## Suggested Next Implementation Order

If development continues along the current roadmap, the next sensible order is:

1. extend `review-presets` so it can collect physics debug maps and evaluation summaries
2. add a simple review rubric / comparison output for human inspection
3. start the PySide6 desktop preset browser

That keeps the work aligned with the current plan without jumping into unrelated scope such as scoring, free navigation, or radial EBUS support.
