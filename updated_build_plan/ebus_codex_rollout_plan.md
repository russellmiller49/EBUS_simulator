# EBUS Simulator — Codex Rollout Plan, Repo Context, and Owner Checklist

## Purpose
This is a repo-specific handoff for implementing the next wave of improvements in `EBUS_simulator`.

It is designed for two audiences:
1. **Codex** implementing the code changes
2. **You** handling any asset, labeling, or validation work that cannot be done purely in code

---

## 1. Current repo state in plain English

The current repo is not a throwaway prototype. It already has the most important v1 ingredients:
- manifest-driven case loading
- a curated preset list for CP-EBUS targets
- airway centerline and airway-network support
- contact-refined device pose generation
- a constrained device model for a `bf_uc180f`-like probe
- a diagnostic rendering workflow with metadata sidecars

What it does **not** have yet is a real CP-EBUS image engine. The current renderer is best understood as a **CT localizer / QA renderer** that is already useful for geometry review.

That is good news: the best path is not a rewrite. The best path is:
- keep the geometry core
- split renderer responsibilities
- add a new physics renderer
- calibrate realism against real EBUS later

---

## 2. Design decision to lock in now

### Decision
For v1, this project is:

**A CP-EBUS / convex-probe training simulator with a geometry-first core and a physics-first image engine.**

### Consequences
Do not do these yet:
- radial EBUS
- full free-navigation bronchoscopy
- procedural scoring
- learned image synthesis as the primary renderer
- Slicer runtime integration
- web app deployment

### What this protects
This keeps the scope aligned with the current repo, dataset, and the attached research reports.

---

## 3. What Codex should know before touching code

### Clinical / product scope
- This is for **linear / convex-probe EBUS** only.
- The most important educational relationship is:
  - airway wall contact
  - node location
  - vessel adjacency
- Presets are a feature, not a limitation.

### Data truths
- Current case is manifest-driven.
- Contacts and targets from `.mrk.json` are authoritative.
- Current active presets include 2L, 2R, 4L, 4R, 7, 10R, 11L, 11Ri, and 11Rs.
- Station 7 must keep distinct LMS and RMS approaches.

### Technical truths
- Current device model is effectively `bf_uc180f` only.
- Current render defaults still disable speckle and edge enhancement.
- Existing airway meshes already support contact refinement and display.
- The manifest root is currently non-portable and should be fixed first.

### Architecture truths
- `models.py`, `manifest.py`, `centerline.py`, `poses.py`, `device.py`, and `validation.py` are worth preserving.
- `rendering.py` is the file most likely to become a maintenance bottleneck if it is not split.

### Research-guided truths
- The strongest next step is a **fast physics renderer**, not a GAN-first renderer.
- A future learned model, if used, should refine texture and artifacts on top of a geometry-correct base image.
- The realism gap in EBUS is dominated by speckle, reverberation/comet-tail behavior, shadowing, gain/compression behavior, and overall console look.

---

## 4. Codex implementation plan

## Phase A — repo hardening and renderer split

### Goal
Make the repo portable and explicit about the fact that the current renderer is a localizer engine.

### Files to edit
- `README.md`
- `pyproject.toml`
- `configs/3d_slicer_files.yaml`
- `src/ebus_simulator/models.py`
- `src/ebus_simulator/manifest.py`
- `src/ebus_simulator/rendering.py`
- `src/ebus_simulator/render_cli.py`
- `src/ebus_simulator/render_all_cli.py`

### Files to add
- `src/ebus_simulator/render_engines.py`
- `src/ebus_simulator/localizer_renderer.py`
- `.github/workflows/ci.yml`
- `tests/test_manifest_portability.py`
- `tests/test_render_engine_dispatch.py`
- `tests/test_render_cli_smoke.py`

### New types to add
- `RenderEngine`
- `RenderRequest`
- `RenderResult`
- expanded `RenderMetadata` with:
  - `engine`
  - `engine_version`
  - `seed`
  - `view_kind`

### Concrete tasks
1. Make manifest roots portable.
   - Support repo-relative paths.
   - Support `${REPO_ROOT}` and `${DATA_ROOT}`.
2. Keep current CLI behavior working.
3. Extract current rendering logic into `localizer_renderer.py`.
4. Make `rendering.py` dispatch to an engine instead of owning everything.
5. Rename conceptual outputs in docs and metadata so the current renderer is not overstated as final ultrasound.
6. Add CI smoke tests.

### Acceptance criteria
- a fresh clone works without editing an absolute local path
- `render-preset ... --engine localizer` works
- old behavior is preserved by default
- JSON metadata includes engine info
- station 7 LMS/RMS output remains distinct and reproducible

---

## Phase B — physics renderer v1

### Goal
Generate the first believable CP-EBUS-like B-mode images from the existing pose engine.

### Files to add
- `src/ebus_simulator/acoustic_properties.py`
- `src/ebus_simulator/physics_renderer.py`
- `src/ebus_simulator/artifacts.py`
- `src/ebus_simulator/eval.py`
- `tests/test_acoustic_mapping.py`
- `tests/test_physics_renderer_phantom.py`
- `tests/test_artifacts.py`
- `tests/support/phantoms.py`

### Files to edit
- `src/ebus_simulator/models.py`
- `src/ebus_simulator/rendering.py`
- `src/ebus_simulator/render_cli.py`
- `src/ebus_simulator/render_all_cli.py`
- `README.md`

### New classes
- `AcousticTissueProfile`
- `AcousticVolume`
- `PhysicsRenderConfig`
- `ScanlineGeometry`
- `PhysicsRenderDebug`
- `PhysicsRenderMetadata`

### Concrete tasks
1. Add label-first acoustic property mapping.
   - air / lumen
   - airway wall / cartilage-like interface
   - vessel lumen
   - lymph node target region
   - generic soft tissue
   - HU fallback for unlabeled voxels
2. Build sector scanline geometry from current `DevicePose`.
3. Implement reflection + backscatter + attenuation accumulation.
4. Add TGC and log compression.
5. Add seeded artifact stage:
   - speckle
   - reverberation/comet-tail near strong air interfaces
   - simple shadowing
6. Save optional debug maps.
7. Keep the output format and metadata reproducible.

### Acceptance criteria
- phantom tests show expected bright air boundary and distal suppression
- vessel lumen is darker than surrounding wall/soft tissue
- node region is plausibly hypoechoic
- seeded speckle is deterministic
- `render-preset ... --engine physics` writes PNG + JSON + optional debug images
- station 7 LMS/RMS give distinct outputs

---

## Phase C — validation hooks and reality checks

### Goal
Stop development from drifting into "pretty but wrong."

### Tasks
1. Add simple quantitative hooks:
   - region contrast checks
   - histogram summaries
   - near-field intensity checks
2. Add a human review rubric:
   - airway wall plausibility
   - vessel relationship
   - node shape/border plausibility
   - overall station believability
3. Create a review folder layout for real-vs-synthetic comparison.

### Deliverable
A repeatable tuning loop for appearance and geometry.

---

## Phase D — desktop trainer UI

### Goal
Expose the simulator in a stable way for review and teaching.

### Files to add
- `src/ebus_simulator/app.py`
- `src/ebus_simulator/ui/` package
- `tests/test_app_smoke.py`

### Features
- preset selector
- approach selector
- physics/localizer toggle
- gain/depth/angle/roll/attenuation controls
- 2D pane
- 3D context pane
- screenshot export

### Note
Do not build full free-navigation first.
Keep the UI as a preset browser.

---

## 5. Suggested PR slicing for Codex

### PR 1
Portable manifest roots + environment-variable support

### PR 2
Renderer engine abstraction + localizer extraction

### PR 3
Metadata cleanup + CLI engine flag + CI smoke tests

### PR 4
Acoustic property system + phantom helpers

### PR 5
Physics scanline renderer core

### PR 6
Artifact post-processing + debug outputs

### PR 7
Evaluation hooks + review folder layout

### PR 8
PySide6 preset browser

---

## 6. What **you** need to do outside the code

This is the non-code checklist.

## A. Do you need new meshes right now?
### Short answer
**No, not for Pass 1 or the first physics renderer.**

You already have enough to start because the current repo includes:
- airway lumen mask
- airway solid mask
- raw airway endoluminal mesh
- smoothed airway display mesh
- centerline VTPs
- target/contact markups
- station masks
- overlay masks for major vessels/organs

### Optional mesh work that would help later
These are helpful, not mandatory.

1. **Explicit cutaway display mesh**
   - useful for a cleaner 3D context panel
   - not required because the code model already supports a cutaway mesh fallback path

2. **Airway wall / cartilage surface mesh**
   - useful for 3D educational display and more explicit near-field interface visualization
   - not required for the first physics renderer if masks are good

3. **Surface meshes for key vessels**
   - especially aorta, pulmonary artery, azygous, SVC, left atrium
   - useful for fast 3D context rendering
   - not required for physics, because masks are enough

4. **Per-node surface meshes**
   - helpful for cleaner 3D teaching views and screenshots
   - optional

### Priority
If you build only one new mesh, make it an **explicit airway cutaway mesh** for the desktop UI.

---

## B. The highest-value non-code tasks for you

### 1. Curate a small real CP-EBUS reference set
This is the single most valuable thing you can do after the geometry scaffold.

Target stations first:
- 4R
- 4L
- 7
- 10R
- 11Rs

Ideal dataset contents:
- still frames or short clips
- known station
- approximate approach / orientation
- expert note on what should be visible
- note whether vessel adjacency is expected
- note whether node is benign-appearing vs malignant-appearing if known

### 2. Build a clinician review rubric
Keep it simple and repeatable:
- airway wall believable: yes / no
- node believable: yes / no
- vessel relationship believable: yes / no
- overall station plausible: 1–5
- comments

### 3. Review and clean current markups
Manually spot-check every active preset:
- target point is centered in the intended node
- contact point is truly on the airway wall
- station 7 LMS and RMS contacts are both valid
- no target/contact is drifting out of bounds due to coordinate mismatch

### 4. Add 1–2 more cases later
Do not do this before the first physics renderer works.
Once it does, the next leverage point is more than one case.

---

## C. If you add a new case later, export this exact asset set

Required:
- CT volume as NIfTI
- airway lumen mask
- airway solid mask
- airway centerline VTP
- airway network VTP
- raw airway endoluminal mesh VTP
- smoothed airway display mesh VTP
- station masks
- target markups `.mrk.json`
- contact markups `.mrk.json`

Strongly recommended:
- overlay masks for:
  - aorta
  - pulmonary artery
  - azygous
  - SVC
  - left atrium
  - esophagus
- optional cutaway airway mesh

Optional but useful:
- node surface meshes
- vessel surface meshes
- screenshot set from Slicer for QA

---

## D. What you do **not** need to build yet
- radial EBUS geometry
- a needle model
- haptics
- Doppler
- a full scoring engine
- wave-equation simulation assets
- surface meshes for every structure in the chest

---

## 7. Recommended "owner" order of operations

1. Let Codex complete Pass 1.
2. While that is happening, curate a small real CP-EBUS frame set.
3. Let Codex complete the first physics renderer.
4. Then do a blinded review of synthetic vs real examples.
5. Only after that, decide whether you need:
   - appearance refinement
   - more cases
   - explicit cutaway / vessel meshes
   - UI polish

---

## 8. My recommendation on meshes specifically

### Build now
- nothing mandatory

### Nice to build next
- explicit airway cutaway mesh

### Build later only if UI review demands it
- key vessel surfaces
- per-node surfaces
- cartilage / airway wall surface

### Why
The physics renderer depends more on **good masks and markups** than on extra meshes. The current raw and smoothed airway meshes are already enough for contact refinement and display work.
