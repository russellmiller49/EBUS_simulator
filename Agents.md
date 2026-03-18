# AGENTS.md

## Project
Build and refine a standalone local linear / convex-probe CP-EBUS simulator from exported imaging assets.

Treat the current repo as a working geometry-first simulator, not a throwaway prototype. It already supports case loading, validation, preset pose generation, localizer rendering, first-pass physics rendering, review exports, and a desktop preset browser. Extend and reorganize that base rather than rebuilding it.

This project is for linear / convex CP-EBUS only.
It is not for:
- radial EBUS
- 360-degree EBUS
- unrestricted full bronchoscopy navigation in v1
- Slicer runtime/module development
- web deployment
- scoring/quiz logic in v1

The simulator should run outside 3D Slicer.
3D Slicer may be used later for QA or cross-checking, not as the runtime application.

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
- direct probe-centered localizer rendering
- first-pass physics renderer with deterministic artifact controls
- clean/debug render modes, overlay controls, metadata sidecars, and diagnostic panels
- batch rendering, review bundles, eval summaries, and comparison workflow
- current `launch-app` desktop browser with queued rendering, reviewer-facing summary, 2D EBUS, 3D context, and screenshot export

What is not fully implemented yet:
- polished linked multi-view teaching workflow
- robust in-window reviewer ergonomics
- mature calibration against real CP-EBUS reference material
- unrestricted navigation
- a fully mature ultrasound image model

Do not reopen scaffold, loader, validation, pose-generation, or engine-split phases from scratch.

---

## Core intent
The product should be treated as a linked multi-view CP-EBUS teaching simulator.

The educational objective is to teach:
- station-level spatial understanding
- airway-wall / node / vessel relationships
- how a given contact and probe orientation produces a given EBUS view
- how external anatomy and fan geometry map to the ultrasound image

The simulator should not be treated as just a static preset screenshot generator.

Even in preset-driven v1, the app should conceptually revolve around one shared probe pose/state that drives:
- external 3D anatomy/context
- fan/localizer bridge visualization
- simulated CP-EBUS image

Version 1 may remain preset-driven, but architecture should avoid locking the project into “one preset = one disconnected render artifact.”

---

## Data location
The working case currently used by the repo is the checked-in dataset copy referenced by:

`configs/3d_slicer_files.yaml`

Use the files there directly.
Do not rename source assets unless explicitly asked.
Keep the manifest root portable.

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
The simulator must remain contact-anchored and target-directed.

Do not make the final render path:
- a rectangular oblique CT reslice first
- then a pasted 2D fan mask

Instead:
- contact markup = simulated probe contact on the airway wall
- target markup = intended lymph-node focus
- centerline tangent = shaft direction reference
- rendered sector = sampled in probe coordinates

Required pose logic for each preset:
- `contact_world` comes from the selected contact markup
- `target_world` comes from the target markup
- `shaft_axis` comes from the local centerline tangent near contact
- `depth_axis` is the target direction projected perpendicular to `shaft_axis`
- optional fine roll rotates around `shaft_axis`

The image apex / near field must originate at the contact point.
The airway wall should remain visible in the near field.
The node should appear in a believable direction relative to airway and vessels.

---

## Preset-driven workflow
Version 1 remains preset-driven, not free-navigation-first.

Each active preset corresponds to:
- one station
- one node
- one contact approach

Examples:
- `station_4r_node_b`
- `station_7_node_a` with approaches `lms` and `rms`

The curated contacts and targets are a feature, not a limitation.

Presets should be treated as initializing a shared pose/state that may later support small motion changes without redesigning the architecture.

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
Do not hard-code blind RAS/LPS assumptions.
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
- The project is already CP-EBUS / convex-probe only.
- The current device model is effectively `bf_uc180f` only.
- Current device defaults are approximately:
  - sector angle `60°`
  - displayed range `40 mm`
  - probe origin offset `6 mm`
  - video axis offset `20°`
- The checked-in manifest uses a portable repo-relative dataset root.
- Existing airway assets already include:
  - raw endoluminal mesh
  - smoothed display mesh
- The data model supports cutaway display meshes.
- The current app already includes 2D EBUS, 3D context, queued rendering, metadata summary, and screenshot export.

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
- existing localizer/physics engine split
- existing app launch and queued-rendering model

Refactor or extend cautiously before adding major complexity:
- `src/ebus_simulator/rendering.py`
- app metadata formatting / UI presentation surfaces
- shared render-state preparation only where clearly needed

Preferred additions for near-term passes:
- app-side metadata adapter / inspector helpers
- clearer pose/fan/localizer bridge visualization
- app ergonomics that reinforce the linked multi-view model

The current localizer renderer should remain the geometry/localizer engine.
Ultrasound-like appearance work should remain a separate physics engine.

Do not chase appearance ahead of geometry correctness.

---

## UI guidance
The first desktop UI is a stable preset-driven teaching browser, not unrestricted free navigation.

Required UI concept:
one shared active preset/pose state should drive the visible panes together.

The app should expose:
- a 2D EBUS pane
- a 3D context pane
- a reviewer/inspector panel that explains the active preset and render

Desired controls:
- preset selector
- approach selector
- localizer / physics engine toggle
- depth
- sector angle
- fine roll
- gain
- attenuation
- overlay toggles
- screenshot export

Desired inspector content:
- preset id
- station
- node/target label
- approach
- engine
- current pose values
- whether target/wall/vessel are present in the current render
- non-null eval values such as target/vessel/wall contrast where available
- warning/auto-flag reasons
- overlay state
- seed / reproducibility metadata when useful

Prefer improving in-window metadata clarity before adding:
- favorites
- preset search
- richer export bundles

Those can come later if manual usage shows real friction.

---

## Rendering requirements
For the current localizer / geometry path:
- keep direct probe-centered sampling
- keep overlay and metadata support
- keep deterministic, preset-driven outputs
- keep the 3D context path useful for anatomy teaching

For the physics path:
- use a fast, inspectable scanline-style model
- keep label-first tissue logic
- keep reflection, backscatter, attenuation, TGC/log compression
- keep seeded speckle, reverberation/comet-tail, and simple shadowing
- do not pivot to GAN-first rendering

A good feature is one that strengthens the learner’s understanding of:
pose, contact, fan geometry, and resulting image content.

---

## Scope protections
Do not broaden into these unless explicitly requested:
- unrestricted airway navigation
- full live scope animation through the airway tree
- radial EBUS
- web deployment
- scoring / quiz / study logic
- Doppler
- Slicer runtime work
- learned image synthesis as the primary renderer
- automated staging recommendations

---

## Commands
These commands should already work from repo root or remain working as the repo evolves:
- `validate-case configs/3d_slicer_files.yaml`
- `generate-poses configs/3d_slicer_files.yaml`
- `render-preset configs/3d_slicer_files.yaml station_4r_node_b --output reports/renders/station_4r_node_b.png`
- `render-all-presets configs/3d_slicer_files.yaml --output-dir reports/renders/all_debug`
- `review-presets configs/3d_slicer_files.yaml --output-dir reports/preset_review`
- `compare-review-bundles reports/preset_review_20260316/review_summary.json reports/preset_review_stabilized/review_summary.json --output-dir reports/preset_review_stabilized`
- `launch-app configs/3d_slicer_files.yaml`
- `pytest -q`

---

## Working style
- Work in small, mergeable changes.
- Do not silently broaden scope.
- Prefer extending existing modules over rewrites.
- Preserve current working commands and tests.
- Preserve station 7 `lms` and `rms` separation everywhere.
- At the end of each bounded pass:
  - summarize what changed
  - list exact files changed
  - list exact commands run
  - list validation or test results
  - state known limitations honestly
  - recommend the next bounded pass

---

## Definition of success
The project is successful when:
- the dataset loads through a portable manifest
- all intended presets validate
- station 7 `lms` and `rms` remain distinct and correct
- the localizer renderer stays reliable for geometry review
- the near field remains anchored to the airway contact point
- the physics renderer produces believable CP-EBUS-like sector images
- reviewer bundles expose localizer, physics, eval summaries, and review sheets in a deterministic layout
- the desktop app behaves like a linked multi-view CP-EBUS teaching simulator
- the app clearly explains the active preset, pose, anatomy-in-fan, and review flags without requiring sidecar JSON inspection
- screenshots export consistently