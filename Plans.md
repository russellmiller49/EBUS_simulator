# PLANS.md

## Execution plan: standalone linear CP-EBUS simulator

This plan supersedes the original scaffold-first roadmap.
The repo already contains the core geometry/localizer scaffold, a first-pass physics renderer, review-bundle workflow, and a working desktop preset browser. Remaining work should focus on hardening the linked teaching workflow, refining renderer responsibilities, improving reviewer ergonomics, and preserving the preset-driven architecture.

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
- direct probe-centered localizer rendering
- `render-preset` CLI with clean/debug modes and metadata sidecars
- explicit localizer / physics engine split
- overlay controls for airway, target, station, and vessels
- cutaway/context rendering support
- `render-all-presets` batch exports
- `review-presets` physics-aware review bundles with localizer and physics outputs, eval summaries, debug maps, and rubric sheets
- `compare-review-bundles`
- `launch-app` desktop browser with queued rendering, 2D EBUS, 3D context, reviewer summary, and screenshot export

Important interpretation:
- the current localizer renderer is useful and should be preserved
- it is best described as a localizer / QA renderer, not the final CP-EBUS image model
- the current app is useful and should be preserved
- it should now evolve from a preset browser into a clearer linked multi-view teaching workflow

---

## Current gaps
The main remaining gaps are:
- the desktop UI still explains itself too weakly during use
- the current reviewer-facing summary should become a clearer structured inspector
- the 3D context / fan / image relationship is present but still under-explained
- the physics-review workflow is usable but still early in calibration against real CP-EBUS reference material
- rendering responsibilities are still somewhat concentrated in shared layers
- broader packaging and manual desktop validation are still incomplete

These gaps define the active roadmap.

---

## Overall objective
Preserve the existing geometry-first CP-EBUS scaffold and evolve it into a portable, inspectable, linked multi-view teaching simulator with:
- reliable preset-driven geometry
- an explicit localizer renderer
- a working first-pass physics-based CP-EBUS renderer
- calibration/review hooks
- a local desktop UI that synchronizes anatomy context, fan/pose understanding, and simulated image output

Version 1 remains preset-driven.
Version 1 does not require unrestricted airway navigation.

The next active milestone is the desktop linked-workflow refinement layer.

---

## Product direction
Treat the intended end product as a linked multi-view CP-EBUS simulator.

A single shared probe pose/state should conceptually drive:
- external 3D anatomy/context
- fan/localizer bridge representation
- simulated CP-EBUS image

The educational goal is to teach:
- where the scope/probe is contacting the airway
- what surrounding anatomy is near the active sector
- why a given pose/orientation produces a given EBUS image

The project should avoid drifting into a disconnected “static preset image browser” design.

---

## Phase A — repo hardening and renderer split
### Status
Complete.

### Goal
Make the repo portable and explicit about the fact that the original renderer is the localizer engine.

### Implemented
- portable manifest roots
- support for repo-relative paths
- support for `${REPO_ROOT}` and `${DATA_ROOT}`
- clearer manifest-root error handling
- `RenderEngine`
- `RenderRequest`
- `RenderResult`
- expanded render metadata with engine/version/seed/view-kind fields
- extraction of the original renderer into `localizer_renderer.py`
- engine dispatch/orchestration in `rendering.py`
- CI smoke tests for validation, pose generation, and localizer rendering

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
- large rewrites of the geometry scaffold

---

## Phase B — physics renderer core
### Status
Complete as a first bounded slice.

### Goal
Generate the first believable CP-EBUS-like B-mode images from the existing pose/device logic.

### Implemented
- label-first acoustic property mapping
- `AcousticTissueProfile`
- `AcousticVolume`
- `PhysicsRenderConfig`
- sector scanline geometry derived from current `DevicePose`
- reflection, diffuse backscatter, and attenuation accumulation
- TGC and log compression
- deterministic seed handling
- `render-preset ... --engine physics`

### Acceptance criteria
- phantom tests show a bright air boundary and distal suppression
- vessel lumen is darker than surrounding wall/soft tissue
- node region is plausibly hypoechoic
- fixed seeds produce reproducible outputs
- repo presets render through the physics engine with PNG + JSON output
- station 7 `lms` and `rms` still produce distinct views

### Explicitly out of scope
- GAN-first image synthesis
- advanced Doppler or full console emulation
- unrelated desktop app work during the same pass

---

## Phase C — artifacts, debug maps, evaluation hooks, and review bundles
### Status
Complete as a first bounded slice; calibration refinement remains.

### Goal
Improve realism while keeping the renderer inspectable and tunable.

### Implemented
- seeded speckle
- reverberation / comet-tail behavior near strong air interfaces
- simple distal shadowing
- optional debug-map exports
- histogram and region-contrast summaries
- review folder layout and rubric workflow
- deterministic review-bundle and comparison summaries

### Acceptance criteria
- artifacts can be toggled and tuned
- seeded outputs remain reproducible
- debug maps can be exported per render
- the review workflow stays repeatable instead of ad hoc

### Remaining work in Phase C
- continue tuning physics appearance against better reference material
- refine reviewer thresholds and summary surfaces as expert feedback arrives
- continue using deterministic before/after bundle summaries for calibration
- keep docs and smoke examples synchronized with the review workflow

### Explicitly out of scope
- changing geometry logic to chase cosmetics
- unrestricted free navigation

---

## Phase D — linked multi-view desktop CP-EBUS simulator
### Status
In progress.

### Current browser baseline already implemented
- preset and approach selectors
- localizer / physics engine toggle
- depth, angle, roll, gain, and attenuation controls
- overlay toggles
- 2D EBUS pane
- 3D context pane
- queued background rendering
- reviewer-facing metadata summary
- screenshot export

### Goal
Expose the simulator through a stable local desktop app that synchronizes external anatomy context, active render state, and simulated CP-EBUS output from one shared preset/pose state.

This phase is not just about launching a browser.
It is about turning the current desktop surface into a clearer teaching workstation.

### Core principles for Phase D
- preserve preset-driven v1 scope
- preserve one shared active state across visible panes
- improve clarity of pose/anatomy/image relationships
- improve reviewer ergonomics inside the app before adding convenience features like favorites/search
- keep the UI responsive under queued rendering

### Implement
- `launch-app` CLI
- PySide6 desktop app
- preset selector
- approach selector
- localizer/physics engine toggle
- depth/angle/roll/gain/attenuation controls
- overlay toggles
- 2D EBUS pane
- 3D context pane
- queued background rendering
- structured in-window metadata inspector
- screenshot export

### Required in-window inspector content
The app should surface, directly in the window:
- preset id
- station label
- target/node label
- approach
- engine
- current pose values
- current render settings
- target/wall/vessel presence when available
- review/eval values such as target/vessel/wall contrast when available
- warning / auto-flag reasons
- seed / reproducibility metadata where useful

### Files expected to exist or evolve in this phase
- `src/ebus_simulator/app.py`
- `src/ebus_simulator/ui/` package
- app smoke tests
- app metadata formatting / inspector helper surfaces

### Acceptance criteria
- the app launches locally from CLI
- switching presets updates both 2D and 3D views
- switching station 7 from `lms` to `rms` changes the approach correctly
- control changes do not block the UI thread during longer renders
- reviewer-facing pose/eval metadata is visible without opening sidecar JSON manually
- the metadata surface is structured enough to explain the active preset and render state
- screenshots export successfully

### Next preferred order inside Phase D
1. improve in-window structured metadata / inspector surface
2. improve pose/fan/context clarity in the 3D teaching workflow
3. add light ergonomic aids such as favorites/search only if manual usage shows real friction
4. expand export polish only after the live app surface is clearer

### Explicitly out of scope
- unrestricted free navigation
- live scope animation through the airway tree
- scoring / quiz logic in the same pass
- radial EBUS
- web deployment

---

## Phase E — post-v1 expansion
### Status
Deferred.

### Potential future work after the active roadmap
These can return only after the geometry, localizer, physics path, review loop, and linked desktop UI are stable:
- constrained micro-navigation between curated contact regions
- smoother continuous pose manipulation
- guided navigation mode
- favorites/search/history if not already added
- richer export bundles and comparison sheets from the live app
- study mode
- quiz mode
- hide/reveal target
- session logging
- scoring engine

These are not on the current critical path.

---

## Out of scope for v1
Do not implement these unless explicitly requested:
- radial EBUS
- unrestricted full procedural bronchoscopy navigation
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
- `validate-case configs/3d_slicer_files.yaml`
- `generate-poses configs/3d_slicer_files.yaml`
- `render-preset configs/3d_slicer_files.yaml station_4r_node_b --output reports/renders/station_4r_node_b.png`
- `render-preset configs/3d_slicer_files.yaml station_7_node_a --approach lms --output reports/renders/station_7_node_a_lms.png`
- `render-preset configs/3d_slicer_files.yaml station_7_node_a --approach rms --output reports/renders/station_7_node_a_rms.png`
- `render-all-presets configs/3d_slicer_files.yaml --output-dir reports/renders/all_debug`
- `review-presets configs/3d_slicer_files.yaml --output-dir reports/preset_review`
- `review-presets configs/3d_slicer_files.yaml --output-dir reports/preset_review --preset-id station_4r_node_b --preset-id station_7_node_a --physics-debug-maps`
- `compare-review-bundles reports/preset_review_20260316/review_summary.json reports/preset_review_stabilized/review_summary.json --output-dir reports/preset_review_stabilized`
- `launch-app configs/3d_slicer_files.yaml`
- `pytest -q`

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