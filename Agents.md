# AGENTS.md

## Project
Build and refine a standalone local **linear / convex-probe EBUS simulator** from exported imaging assets.

Treat the current repo as a **working geometry-first scaffold**, not a throwaway prototype. It already supports case loading, validation, preset pose generation, localizer-style rendering, cutaway/context views, and review exports. The next work should extend and reorganize that base rather than rebuilding it.

This project is for **linear / convex EBUS only**.
It is **not** for:
- radial EBUS
- 360-degree EBUS
- full bronchoscopy navigation
- Slicer module/runtime development
- web deployment
- procedural scoring in v1

The simulator should run **outside 3D Slicer**.
3D Slicer may still be used later for optional QA or cross-checking, not as the runtime application.

---

## Current repo state
What is already implemented and should be preserved unless there is a hard blocker:
- manifest-driven case loading
- CT, mask, VTP, and `.mrk.json` IO
- geometry validation with machine-readable reports
- centerline graph abstraction and tangent lookup
- preset-driven contact/target handling
- station 7 with distinct `lms` and `rms` approaches
- CP-EBUS device pose generation
- airway-wall contact refinement against voxel data and airway meshes
- direct probe-centered CT/localizer rendering
- clean/debug render modes, overlay controls, metadata sidecars, and diagnostic panels
- batch rendering, cutaway/context views, and physics-aware preset review workflows with eval summaries and rubric sheets

What is **not** implemented yet:
- polished calibration/tuning workflow on top of the existing review bundle
- PySide6 desktop UI
- a polished review-first desktop workflow

Do **not** reopen the scaffold / loader / validation / pose-generation phases from scratch. Reuse them.

---

## Core intent
The educational goal remains the same:
- correct geometry first
- correct airway-wall / node / vessel relationships
- stable preset-driven outputs
- a usable local desktop workflow

Current repo reality:
- the existing renderer is best treated as a **CT localizer / QA renderer**
- future ultrasound realism work should arrive as a **separate physics renderer**, not by overloading the current renderer indefinitely

---

## Data location
The working case currently used by the repo is the checked-in dataset copy referenced by:

`configs/3d_slicer_files.yaml`

Use the files there directly.
Do **not** rename source assets unless explicitly asked.
The manifest root is already portable and should remain so. The filenames should remain valid.

---

## Authoritative inputs
Treat these as authoritative unless the user says otherwise:

- `ct.nii.gz` -> source CT
- `centerlines/airway_centerline.vtp` -> primary centerline geometry / tangent source
- `centerlines/airway_network.vtp` -> airway branch graph
- `markups/*.mrk.json` -> authoritative contact / target points
- `masks/airway.nii.gz` -> airway lumen mask
- `masks/airway_solid.nii.gz` -> airway wall helper
- `masks/station_*.nii.gz` -> station-level region masks
- vessel / organ masks -> context overlays unless explicitly promoted to active render logic

The many `Network curve_1 (...).mrk.json` files remain secondary / fallback / debugging aids, not the primary source-of-truth geometry.

---

## Clinical / geometry model
The simulator must remain **contact-anchored** and **target-directed**.

Do **not** make the final render path:
- a rectangular oblique CT reslice first
- then a pasted 2D fan mask

Instead:
- contact markup = simulated probe contact on the airway wall
- target markup = intended lymph-node focus
- centerline tangent = shaft direction reference
- rendered sector = sampled in **probe coordinates**

Required pose logic for each preset:
- `contact_world` comes from the selected contact markup
- `target_world` comes from the target markup
- `shaft_axis` comes from the local centerline tangent near contact
- `depth_axis` is the target direction projected perpendicular to `shaft_axis`
- optional fine roll rotates around `shaft_axis`

The image apex / near field must originate at the **contact point**.
The airway wall should remain visible in the near field.
The node should appear in a believable direction relative to airway and vessels.

---

## Preset-driven workflow
Version 1 remains **preset-driven**, not free-navigation-first.

Each active preset corresponds to:
- one station
- one node
- one contact approach

Examples:
- `station_4r_node_b`
- `station_7_node_a` with approaches `lms` and `rms`

The curated contacts and targets are a feature, not a limitation.

---

## Active presets
The current dataset/manifest supports these as first-class presets:
- `station_2l_node_a`
- `station_2r_node_a`
- `station_4l_node_a`
- `station_4l_node_b`
- `station_4l_node_c`
- `station_4r_node_a`
- `station_4r_node_b`
- `station_4r_node_c`
- `station_7_node_a` with `lms`
- `station_7_node_a` with `rms`
- `station_10r_node_a`
- `station_10r_node_b`
- `station_11l_node_a`
- `station_11ri_node_a`
- `station_11rs_node_a`
- `station_11rs_node_b`

Other masks without matching curated preset pairs remain context overlays for now.

---

## Coordinate handling
Be careful with coordinate systems.

All imported data must normalize into one internal world coordinate system before geometry computations.
Do **not** hard-code blind RAS/LPS assumptions.
Instead:
- inspect metadata where available
- read markup coordinate-system metadata from `.mrk.json`
- validate registration with geometry checks

Every significant change must preserve geometry QA:
- contact near airway surface
- target near or inside station mask
- contact and target inside CT bounds
- contact projects onto centerline successfully
- centerline tangent is defined

---

## Current repo truths
- The project is already **CP-EBUS / convex-probe only**.
- The current device model is effectively **`bf_uc180f` only**.
- Current device defaults are approximately:
  - sector angle `60°`
  - displayed range `40 mm`
  - probe origin offset `6 mm`
  - video axis offset `20°`
- The checked-in manifest uses a **portable repo-relative dataset root**.
- Current render defaults still keep `use_speckle: false` and `use_edge_enhancement: false`.
- Existing airway assets already include:
  - raw endoluminal mesh
  - smoothed display mesh
- The data model supports cutaway display meshes, but the current config does not explicitly provide one.

---

## Architecture direction
Preserve and reuse:
- `src/ebus_simulator/models.py`
- `src/ebus_simulator/manifest.py`
- `src/ebus_simulator/centerline.py`
- `src/ebus_simulator/poses.py`
- `src/ebus_simulator/device.py`
- `src/ebus_simulator/validation.py`
- `src/ebus_simulator/cutaway.py`
- `src/ebus_simulator/review.py`

Refactor / split before adding major renderer complexity:
- `src/ebus_simulator/rendering.py`
- `src/ebus_simulator/render_cli.py`
- `src/ebus_simulator/render_all_cli.py`

Preferred additions for the next major passes:
- `src/ebus_simulator/render_engines.py`
- `src/ebus_simulator/localizer_renderer.py`
- `src/ebus_simulator/acoustic_properties.py`
- `src/ebus_simulator/physics_renderer.py`
- `src/ebus_simulator/artifacts.py`
- `src/ebus_simulator/eval.py`
- `src/ebus_simulator/app.py`
- `.github/workflows/ci.yml`

The current renderer should become the explicit **localizer** engine.
Any future ultrasound-like renderer should be a separate **physics** engine.

The next active milestone is:
- review / calibration refinement and parameter tuning
- reviewer-facing bundle iteration rather than first-pass bundling
- desktop UI only after the review loop is stable

---

## Next-pass priorities
Preferred order for remaining work:
1. review / calibration tuning using the current physics-aware bundles
2. small review/render coupling cleanup where it has clear value
3. PySide6 preset browser

Trainer/quiz layers are deferred until the above are stable.

---

## Rendering requirements
For the current localizer / geometry path:
- keep direct probe-centered sampling
- keep overlay and metadata support
- keep deterministic, preset-driven outputs

For the future physics path:
- use a fast, inspectable scanline model
- start with label-first tissue logic
- add reflection, backscatter, attenuation, TGC/log compression
- add seeded speckle, reverberation/comet-tail, and simple shadowing
- do **not** start with GAN-first rendering

Do not chase appearance ahead of geometry correctness.

---

## UI guidance
The first desktop UI should be a **stable preset browser**, not unrestricted free navigation.

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

The app should expose:
- a 2D EBUS pane
- a 3D context pane

---

## Commands
These commands should already work from repo root or remain working as the repo evolves:
- `validate-case configs/3d_slicer_files.yaml`
- `generate-poses configs/3d_slicer_files.yaml`
- `render-preset configs/3d_slicer_files.yaml station_4r_node_b --output reports/renders/station_4r_node_b.png`
- `render-all-presets configs/3d_slicer_files.yaml --output-dir reports/renders/all_debug`
- `review-presets configs/3d_slicer_files.yaml --output-dir reports/preset_review`
- later: `launch-app configs/3d_slicer_files.yaml`
- `pytest -q`

---

## Working style
- Work in small, mergeable changes.
- Do not silently broaden scope.
- Prefer extending existing modules over rewrites.
- At the end of each bounded pass:
  - summarize what changed
  - list exact files changed
  - list commands to run
  - list validation or test results
  - state known limitations honestly

---

## Definition of success
The project is successful when:
- the dataset loads through a portable manifest
- all intended presets validate
- station 7 `lms` and `rms` remain distinct and correct
- the localizer renderer stays reliable for geometry review
- the near field remains anchored to the airway contact point
- a physics renderer can produce believable CP-EBUS-like sector images
- reviewer bundles expose localizer, physics, eval summaries, and review sheets in a deterministic layout
- the desktop app can browse presets and export screenshots consistently
