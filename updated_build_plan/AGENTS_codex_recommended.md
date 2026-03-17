# AGENTS.md — Recommended Codex Instructions for `EBUS_simulator`

Use this as a Codex-facing instruction file or paste the key parts into a Codex session.

## Project goal
Build a **high-quality CP-EBUS / convex-probe EBUS simulator** from exported imaging assets.

This project is **not** for:
- radial EBUS
- full bronchoscopy navigation
- Slicer module development
- web deployment
- procedural scoring in v1

## What the current repo already has
Treat the existing repo as a **strong geometry scaffold**.

Current strengths:
- manifest-driven case loading
- CT + mask + VTP + `.mrk.json` IO
- centerline graph abstraction
- preset-driven targets and contacts
- station 7 with separate `lms` and `rms` approaches
- CP-EBUS device pose construction
- airway-wall contact refinement against voxel data and raw mesh
- CLI rendering with metadata sidecars
- overlay and diagnostic panel support

Do **not** rewrite these systems from scratch unless there is a hard blocker.
Prefer extending and reorganizing them.

## Important current repo truths
- The project is already **CP-EBUS only**.
- The current device model supports only `bf_uc180f`.
- The current device assumptions are approximately:
  - sector angle: `60°`
  - displayed range: `40 mm`
  - probe origin offset: `6 mm`
  - video axis offset: `20°`
- The checked-in manifest still uses an **absolute local dataset path**.
- Current render defaults still have:
  - `use_speckle: false`
  - `use_edge_enhancement: false`
- Existing airway meshes already include:
  - raw endoluminal mesh
  - smoothed display mesh
- `airway_cutaway_display_mesh` exists in the data model, but the current config does not explicitly provide one.

## Non-negotiable clinical / geometry rules
1. Keep the project **CP-EBUS / convex-probe only** for v1.
2. Preserve the **preset-driven workflow**.
3. Preserve the curated **contact** and **target** markups as authoritative.
4. Preserve the existing **contact-anchored, target-directed** pose logic.
5. The image apex / near field must remain anchored to the airway wall contact point.
6. The airway wall must appear in the near field.
7. The node must appear in a believable orientation relative to airway and vessels.
8. Do not add radial EBUS.
9. Do not rename the original source assets.
10. Do not replace curated markups with inferred positions.

## Architecture direction
Treat the current renderer as a **localizer / QA renderer**, not the final ultrasound renderer.

The codebase should evolve toward two explicit rendering engines:
- `localizer` = existing CT/debug renderer, cleaned up and preserved
- `physics` = new CP-EBUS B-mode synthesis renderer

Do **not** keep overloading one giant renderer module forever.

## High-level plan
### Pass 1 — harden and split
Goal:
- keep current behavior intact
- make the repo portable
- separate the localizer renderer from the future physics renderer

### Pass 2 — add a physics CP-EBUS renderer
Goal:
- keep current pose/device logic
- add acoustic property mapping and scanline rendering
- add seeded artifacts: speckle, reverberation/comet-tail, shadowing
- produce the first genuinely ultrasound-like CP-EBUS frames

### Pass 3 — validation and calibration
Goal:
- compare synthetic frames against curated real CP-EBUS examples
- tune appearance only after the geometry and physics look right

### Pass 4 — desktop trainer UI
Goal:
- stable preset browser
- 2D physics EBUS pane
- 3D context pane
- screenshot export

## Required implementation style
- Work in **small, mergeable PRs**.
- Do not broaden scope silently.
- At the end of each phase:
  - summarize what changed
  - list exact files changed
  - list commands to run
  - list test results
  - state known limitations honestly
- Prefer a narrow working implementation over a large speculative one.

## File-level direction
### Keep and reuse
- `src/ebus_simulator/models.py`
- `src/ebus_simulator/manifest.py`
- `src/ebus_simulator/centerline.py`
- `src/ebus_simulator/poses.py`
- `src/ebus_simulator/device.py`
- `src/ebus_simulator/validation.py`

### Refactor / split
- `src/ebus_simulator/rendering.py`
- `src/ebus_simulator/render_cli.py`
- `src/ebus_simulator/render_all_cli.py`

### Add
- `src/ebus_simulator/render_engines.py`
- `src/ebus_simulator/localizer_renderer.py`
- `src/ebus_simulator/acoustic_properties.py`
- `src/ebus_simulator/physics_renderer.py`
- `src/ebus_simulator/artifacts.py`
- `src/ebus_simulator/eval.py`
- `src/ebus_simulator/app.py`
- `.github/workflows/ci.yml`

## Data / coordinate rules
- Normalize all data into one canonical internal world frame.
- Be careful with `.mrk.json` coordinate systems.
- Do not hard-code blind RAS/LPS assumptions.
- Every major change must keep QA checks alive:
  - contact inside CT bounds
  - target inside CT bounds
  - contact near airway surface
  - target near station mask
  - contact projects to centerline
  - tangent is defined

## Physics renderer v1 guidance
Use a **fast, inspectable scanline model**.

Start with label-first tissue logic:
- airway lumen / air
- airway wall / cartilage-like boundary behavior
- blood vessels
- lymph node region
- generic soft tissue
- fallback HU-based mapping for unlabeled tissue

Implement:
- reflection at impedance changes
- diffuse backscatter
- cumulative attenuation
- TGC / log compression
- seeded speckle
- reverberation / comet-tail near strong air interfaces
- simple distal shadowing

Do **not** start with GAN-first rendering.
If a learned appearance model is added later, it should refine texture, not geometry.

## UI guidance
The first UI should be a **stable preset browser**, not a free-navigation simulator.

Desired controls:
- preset selector
- approach selector
- depth
- sector angle
- fine roll
- gain
- attenuation
- overlay toggles
- screenshot export

## Commands that should exist or remain working
From repo root:
- environment setup
- `validate-case ...`
- `generate-poses ...`
- `render-preset ...`
- `render-all-presets ...`
- later: `launch-app`
- `pytest -q`

## Definition of success for v1
The project is successful when:
- the dataset loads through a portable manifest
- all intended presets validate
- station 7 supports separate `lms` and `rms` approaches correctly
- the localizer renderer still works
- the physics renderer produces believable CP-EBUS sector images
- the near field is anchored to the airway contact point
- the desktop app switches presets reliably
- screenshots export consistently

## First PRs Codex should open
1. Manifest portability + render engine enum + metadata schema
2. Extract current renderer into `localizer_renderer.py`
3. Add physics-renderer interfaces and phantom tests
4. Add acoustic property mapping
5. Add scanline physics renderer
6. Add artifacts and debug maps
7. Add CI smoke tests
8. Add PySide6 preset browser

## Paste-in kickoff prompt for Codex
You are working inside `EBUS_simulator`.

Goal: preserve the existing CP-EBUS geometry scaffold, split the current renderer into a `localizer` engine, and implement a new physics-based CP-EBUS renderer in small, mergeable PRs.

Hard constraints:
- CP-EBUS / convex-probe only
- do not add radial EBUS
- do not rename source assets
- do not replace curated contacts/targets with inferred points
- do not use rectangular reslice + 2D fan mask as the primary final renderer
- keep station 7 LMS/RMS behavior intact
- keep QA and metadata sidecars strong
- preserve current CLI behavior where possible

Execution rules:
- make one narrow change set at a time
- add or update tests for each change
- after each phase, summarize files changed, commands to run, and limitations
- prefer reusable modules over a bigger `rendering.py`

First task:
1. add portable manifest root handling
2. add `RenderEngine` / `RenderRequest` / `RenderResult`
3. move the current renderer into `localizer_renderer.py`
4. keep `render-preset` working with `--engine localizer`
