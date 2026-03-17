# EBUS Simulator — Issue-Ready GitHub Tickets

Use these as issue drafts or milestone planning notes.

---

## 1. Make case manifests portable
**Title**
Portable manifest roots and environment-variable support

**Why**
The current manifest uses a machine-specific absolute path. This makes fresh clones and CI brittle.

**Tasks**
- support repo-relative paths
- support `${REPO_ROOT}` and `${DATA_ROOT}` in manifest root resolution
- improve error messages for missing roots
- update README examples
- add tests

**Definition of done**
- a clean clone can run `validate-case` without editing a hard-coded absolute path
- tests cover relative and env-var roots
- failure messages are clear

**Depends on**
None

**Suggested labels**
`infra`, `manifest`, `good first PR`

---

## 2. Add renderer engine abstraction
**Title**
Introduce render engine abstraction and standard request/result types

**Why**
The repo needs a clean seam between the existing localizer renderer and the future physics renderer.

**Tasks**
- add `RenderEngine`
- add `RenderRequest`
- add `RenderResult`
- update metadata model with engine/version/seed fields
- refactor shared render entry points to dispatch by engine

**Definition of done**
- `render-preset` can select an engine
- metadata records engine info
- current output behavior is preserved

**Depends on**
Ticket 1 recommended but not strictly required

**Suggested labels**
`architecture`, `rendering`

---

## 3. Extract current renderer into a localizer module
**Title**
Move current CT/debug renderer into `localizer_renderer.py`

**Why**
The current renderer already works as a geometry localizer and QA tool. It should be preserved but separated from future ultrasound synthesis.

**Tasks**
- move current rendering logic into `localizer_renderer.py`
- keep `rendering.py` as thin dispatch/orchestration
- keep diagnostic panel support intact
- preserve overlays and metadata

**Definition of done**
- generated images match prior behavior within expected tolerance
- CLI still works
- code size and responsibility of `rendering.py` decrease materially

**Depends on**
Ticket 2

**Suggested labels**
`rendering`, `refactor`

---

## 4. Add CI smoke tests for validation and rendering
**Title**
Continuous integration smoke tests for manifest, pose generation, and rendering

**Why**
The repo needs a safety net before larger renderer changes.

**Tasks**
- add GitHub Actions workflow
- run `pytest -q`
- add smoke tests for:
  - `validate-case`
  - `generate-poses`
  - `render-preset --engine localizer`
- archive artifacts on CI failure if practical

**Definition of done**
- PRs automatically run tests
- failures are easy to interpret

**Depends on**
Tickets 1–3 preferred

**Suggested labels**
`ci`, `testing`

---

## 5. Implement label-first acoustic property mapping
**Title**
Add acoustic tissue property mapping for CP-EBUS physics rendering

**Why**
Physics rendering needs explicit tissue behaviors, not just CT windowing.

**Tasks**
- add `AcousticTissueProfile`
- add `AcousticVolume`
- map labels for:
  - air / lumen
  - airway wall
  - vessels
  - node region
  - soft tissue
- add HU fallback for unlabeled tissue
- add tests

**Definition of done**
- acoustic maps are reproducible and test-covered
- labels override HU fallback where both exist

**Depends on**
Ticket 2

**Suggested labels**
`physics`, `rendering`, `data-model`

---

## 6. Build a minimal scanline CP-EBUS physics renderer
**Title**
Implement first-pass CP-EBUS scanline physics renderer

**Why**
This is the first step from CT-localizer output to true ultrasound-like rendering.

**Tasks**
- add scanline geometry from `DevicePose`
- accumulate reflection/backscatter/attenuation
- add TGC and log compression
- write PNG + JSON via `render-preset --engine physics`
- support deterministic seeds

**Definition of done**
- synthetic phantom tests pass
- repo case renders through the physics engine
- outputs are reproducible with fixed seeds

**Depends on**
Ticket 5

**Suggested labels**
`physics`, `core`

---

## 7. Add artifact post-processing
**Title**
Add seeded speckle, reverberation, and shadowing to physics renders

**Why**
The realism gap is dominated by artifact behavior, not just geometry.

**Tasks**
- add speckle module
- add reverberation/comet-tail behavior near strong air interfaces
- add simple distal shadowing
- expose config knobs
- add tests for determinism and toggles

**Definition of done**
- artifacts can be toggled and tuned
- seeded outputs are reproducible
- debug output can separate base image vs artifact stage

**Depends on**
Ticket 6

**Suggested labels**
`physics`, `appearance`

---

## 8. Add debug-map exports and evaluation hooks
**Title**
Export physics debug maps and add basic evaluation hooks

**Why**
The simulator must stay inspectable and tunable.

**Tasks**
- export reflection / attenuation / shadow / base-image debug maps
- add histogram and region-contrast summaries
- define placeholder hooks for real-vs-synthetic review later

**Definition of done**
- debug assets can be written per render
- evaluation helpers run on synthetic phantoms and repo case outputs

**Depends on**
Tickets 6–7

**Suggested labels**
`evaluation`, `debugging`

---

## 9. Create a real CP-EBUS reference review pack
**Title**
Add review rubric and data layout for real-vs-synthetic CP-EBUS comparison

**Why**
Visual plausibility needs clinician review, not only engineering metrics.

**Tasks**
- add docs for organizing real reference frames/clips
- define a simple review rubric
- create expected folder layout in `reports/` or `reference_data/`
- add example review template

**Definition of done**
- review process is documented and repeatable
- at least one synthetic-vs-real comparison session can be run consistently

**Depends on**
Ticket 8 preferred

**Suggested labels**
`validation`, `clinical-review`, `docs`

---

## 10. Build a PySide6 preset browser
**Title**
Create first desktop preset browser for CP-EBUS simulator

**Why**
The simulator becomes much more usable once presets can be browsed interactively.

**Tasks**
- add `launch-app` CLI
- build PySide6 app with:
  - preset selector
  - approach selector
  - 2D pane
  - 3D context pane
  - gain/depth/angle/roll controls
  - overlay toggles
  - screenshot export
- support engine switch between localizer and physics

**Definition of done**
- app launches from CLI
- switching presets updates views consistently
- screenshots export correctly

**Depends on**
Tickets 2–8

**Suggested labels**
`ui`, `desktop`

---

## 11. Add explicit cutaway mesh support in config and UI
**Title**
Support optional explicit cutaway airway mesh for 3D context view

**Why**
The data model already anticipates a cutaway mesh, but the current config does not explicitly supply one.

**Tasks**
- add config example for `cutaway_display_mesh`
- use the cutaway mesh when available
- fall back gracefully to current smoothed mesh
- document export guidance

**Definition of done**
- cutaway mesh is optional
- UI and renderer use it when present
- fallback path remains stable

**Depends on**
Ticket 10 for biggest value, but may land earlier

**Suggested labels**
`ui`, `3d`, `data`

---

## 12. Add multi-case onboarding documentation
**Title**
Document exact asset requirements for adding a new EBUS case

**Why**
The project will eventually need more than one case, and onboarding should be predictable.

**Tasks**
- document required files for a new case
- document optional files
- include naming and manifest examples
- include checklist for target/contact validation

**Definition of done**
- a collaborator can prepare a second case without reverse engineering the repo

**Depends on**
Ticket 1 recommended

**Suggested labels**
`docs`, `dataset`
