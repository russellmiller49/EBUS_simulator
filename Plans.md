# PLANS.md

## Execution plan: standalone linear EBUS simulator

This plan supersedes the original scaffold-first roadmap.
The repo already contains the core geometry/localizer scaffold, so the remaining work should focus on **hardening, separating renderer responsibilities, adding a physics image engine, and then exposing the simulator through a desktop UI**.

Each bounded pass should still end with:
- a brief implementation summary
- exact files changed
- exact commands run
- validation or test evidence
- known limitations

---

## Current baseline already implemented
These earlier milestones are already in the repo and should not be reopened unless a bug or refactor requires it:
- Python project scaffold and packaging
- manifest-driven case loading
- NIfTI, VTP, and `.mrk.json` loaders
- `validate-case` CLI with machine-readable reporting
- centerline graph and preset pose generation
- `generate-poses` CLI
- direct probe-centered CT/localizer rendering
- `render-preset` CLI with clean/debug modes and metadata sidecars
- overlay controls for airway, target, station, and vessels
- cutaway/context rendering support
- `render-all-presets` batch exports
- `review-presets` review summaries and comparison bundles

Important interpretation:
- the current renderer is useful and should be preserved
- it is best described as a **localizer / QA renderer**, not the final CP-EBUS physics renderer

---

## Current gaps
The main remaining gaps are:
- `review-presets` is still more localizer-centric than physics-centric
- physics debug maps and eval summaries are not yet bundled into review exports
- the repo guidance is stale in places
- rendering responsibilities are still somewhat concentrated in `rendering.py`
- there is no desktop preset browser yet

These gaps define the active roadmap.

---

## Overall objective
Preserve the existing geometry-first CP-EBUS scaffold and evolve it into a portable, inspectable simulator with:
- reliable preset-driven geometry
- an explicit localizer renderer
- a working first-pass physics-based CP-EBUS renderer
- calibration/review hooks
- a local desktop UI for preset browsing and screenshot export

The next active milestone is the **physics-aware review / calibration layer**.

---

## Phase A — repo hardening and renderer split
### Status
Complete.

### Goal
Make the repo portable and explicit about the fact that the current renderer is the **localizer** engine.

### Implement
- portable manifest roots
- support for repo-relative paths
- support for `${REPO_ROOT}` and `${DATA_ROOT}`
- clearer manifest-root error handling
- `RenderEngine`
- `RenderRequest`
- `RenderResult`
- expanded render metadata with engine/version/seed/view-kind fields
- extraction of the current renderer into `localizer_renderer.py`
- thin engine dispatch/orchestration in `rendering.py`
- CI smoke tests for validation, pose generation, and localizer rendering

### Files expected to change
- `README.md`
- `pyproject.toml`
- `configs/3D_slicer_files.yaml`
- `src/ebus_simulator/models.py`
- `src/ebus_simulator/manifest.py`
- `src/ebus_simulator/rendering.py`
- `src/ebus_simulator/render_cli.py`
- `src/ebus_simulator/render_all_cli.py`

### Files expected to add
- `src/ebus_simulator/render_engines.py`
- `src/ebus_simulator/localizer_renderer.py`
- `.github/workflows/ci.yml`
- tests covering manifest portability, engine dispatch, and CLI smoke behavior

### Acceptance criteria
- a fresh clone can run `validate-case` without editing a hard-coded absolute path
- current `render-preset` behavior is preserved by default
- `render-preset ... --engine localizer` works after the split lands
- metadata records engine information
- station 7 `lms` and `rms` outputs remain distinct and reproducible

### Explicitly out of scope
- desktop UI work
- scoring / trainer mode
- radial EBUS
- any large rewrite of the geometry scaffold

---

## Phase B — physics renderer core
### Status
Complete as a first bounded slice.

### Goal
Generate the first believable CP-EBUS-like B-mode images from the existing pose/device logic.

### Implement
- label-first acoustic property mapping
- `AcousticTissueProfile`
- `AcousticVolume`
- `PhysicsRenderConfig`
- sector scanline geometry derived from current `DevicePose`
- reflection, diffuse backscatter, and attenuation accumulation
- TGC and log compression
- deterministic seed handling
- `render-preset ... --engine physics`

### Files expected to add
- `src/ebus_simulator/acoustic_properties.py`
- `src/ebus_simulator/physics_renderer.py`
- tests for acoustic mapping and physics-renderer phantoms

### Files expected to change
- `src/ebus_simulator/models.py`
- `src/ebus_simulator/rendering.py`
- `src/ebus_simulator/render_cli.py`
- `src/ebus_simulator/render_all_cli.py`
- `README.md`

### Acceptance criteria
- phantom tests show a bright air boundary and distal suppression
- vessel lumen is darker than surrounding wall/soft tissue
- node region is plausibly hypoechoic
- fixed seeds produce reproducible outputs
- repo presets render through the physics engine with PNG + JSON output
- station 7 `lms` and `rms` still produce distinct views

### Explicitly out of scope
- GAN-first image synthesis
- advanced Doppler or full ultrasound-console emulation
- desktop app work during the same pass

---

## Phase C — artifacts, debug maps, and evaluation hooks
### Status
Partially complete.

### Goal
Improve realism while keeping the renderer inspectable and tunable.

### Implement
- seeded speckle
- reverberation / comet-tail behavior near strong air interfaces
- simple distal shadowing
- optional debug-map exports
- histogram and region-contrast summaries
- review folder layout for real-vs-synthetic comparison
- a simple human review rubric

### Files expected to add
- `src/ebus_simulator/artifacts.py`
- `src/ebus_simulator/eval.py`
- tests for artifact determinism and evaluation helpers

### Acceptance criteria
- artifacts can be toggled and tuned
- seeded outputs remain reproducible
- debug maps can be exported per render
- the review workflow stays repeatable instead of ad hoc

### Remaining work in Phase C
- integrate physics debug maps into `review-presets`
- expose `eval_summary` in reviewer-friendly bundles
- add a lightweight human review rubric / sheet
- tighten docs so they match the current repo state

### Explicitly out of scope
- changing geometry logic to chase cosmetics
- free-navigation simulator work

---

## Phase D — desktop preset browser
### Goal
Expose the simulator through a stable local desktop app for review and teaching.

### Implement
- `launch-app` CLI
- PySide6 preset browser
- preset selector
- approach selector
- localizer/physics engine toggle
- depth/angle/roll/gain/attenuation controls
- overlay toggles
- 2D EBUS pane
- 3D context pane
- screenshot export

### Files expected to add
- `src/ebus_simulator/app.py`
- `src/ebus_simulator/ui/` package
- app smoke tests

### Acceptance criteria
- the app launches locally from CLI
- switching presets updates both 2D and 3D views
- switching station 7 from `lms` to `rms` changes the approach correctly
- screenshots export successfully

### Explicitly out of scope
- unrestricted free navigation
- scoring / quiz logic in the same pass

---

## Deferred after the active roadmap
These were part of earlier brainstorming but are **not** on the current critical path:
- study mode
- quiz mode
- hide / reveal target
- session logging
- scoring engine

They can return only after the geometry, renderer split, physics path, and desktop UI are stable.

---

## Out of scope for v1
Do not implement these unless explicitly requested:
- radial EBUS
- full procedural bronchoscopy navigation
- live scope animation through the airway tree
- Slicer runtime module
- web deployment
- DICOM-first ingest path
- learned image synthesis as the primary renderer
- Doppler
- automated staging recommendations

---

## Dataset-specific notes
### Active v1 presets
The current manifest already supports these as first-class presets:
- `station_2l_node_a`
- `station_2r_node_a`
- `station_4l_node_a`
- `station_4l_node_b`
- `station_4l_node_c`
- `station_4r_node_a`
- `station_4r_node_b`
- `station_4r_node_c`
- `station_7_node_a` with approach `lms`
- `station_7_node_a` with approach `rms`
- `station_10r_node_a`
- `station_10r_node_b`
- `station_11l_node_a`
- `station_11ri_node_a`
- `station_11rs_node_a`
- `station_11rs_node_b`

### Context-only masks
Other station masks and anatomy masks without complete curated presets may be shown as context overlays only.

---

## Command baseline
Current commands expected to work from repo root:
- `validate-case configs/3D_slicer_files.yaml`
- `generate-poses configs/3D_slicer_files.yaml`
- `render-preset configs/3D_slicer_files.yaml station_4r_node_b --output reports/renders/station_4r_node_b.png`
- `render-preset configs/3D_slicer_files.yaml station_7_node_a --approach lms --output reports/renders/station_7_node_a_lms.png`
- `render-preset configs/3D_slicer_files.yaml station_7_node_a --approach rms --output reports/renders/station_7_node_a_rms.png`
- `render-all-presets configs/3D_slicer_files.yaml --output-dir reports/renders/all_debug`
- `review-presets configs/3D_slicer_files.yaml --output-dir reports/preset_review`
- `pytest -q`

Future command to add:
- `launch-app configs/3D_slicer_files.yaml`

---

## Execution rule
At the start of every bounded pass:
1. restate the exact phase goal
2. list the files you expect to create or edit
3. state what is explicitly out of scope for the pass

At the end of every bounded pass:
1. summarize what was completed
2. list exact files changed
3. show exact commands run
4. show validation evidence
5. state known limitations and the next recommended pass
