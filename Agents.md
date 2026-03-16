# AGENTS.md

## Project
Build a standalone local **linear EBUS simulator** from exported imaging assets.

This project is for **linear / convex EBUS only**.
It is **not** for radial EBUS, 360-degree EBUS, or general bronchoscopy navigation.

The simulator should use:
- CT volume
- airway masks
- vessel / organ masks
- station masks
- airway centerline / network
- curated contact and target markups

The simulator should run **outside 3D Slicer**.
3D Slicer may be used later only for optional QA / visual cross-checking, not as the runtime application.

---

## Core intent
The main goal is to create a realistic, educationally useful **linear EBUS view** for known lymph node targets.

Version 1 should prioritize:
1. correct geometry
2. correct relationship between airway wall, target node, and vessels
3. stable preset-driven rendering
4. a usable local desktop UI

Version 1 should **not** prioritize:
- full ultrasound physics
- free-navigation bronchoscopy
- radial EBUS
- procedural scoring engine
- web deployment
- Slicer module development

---

## Data location
Primary case data root:

`/Users/russellmiller/Downloads/3D_slicer_files`

Use the files there directly.
Do **not** rename user files unless explicitly asked.
Use a manifest/config file so current filenames remain valid.

---

## Authoritative inputs
Treat these as authoritative unless the user says otherwise:

- `ct.nii.gz` -> source CT
- `centerlines/airway_centerline.vtp` -> primary centerline geometry / tangent source
- `centerlines/airway_network.vtp` -> airway branch graph
- `markups/*.mrk.json` -> authoritative contact / target points
- `masks/airway.nii.gz` -> airway lumen mask
- `masks/airway_solid.nii.gz` -> airway solid mask / airway wall helper
- `masks/station_*.nii.gz` -> station-level region masks
- vessel / organ masks -> context overlays only unless explicitly promoted to active logic

The many `Network curve_1 (...).mrk.json` files are secondary / fallback / debugging aids, not primary source-of-truth geometry.

---

## Clinical / geometry model
### Important
The simulator must be **contact-anchored** and **target-directed**.

Do **not** build the image as:
- oblique CT slice first
- then apply a 2D fan mask as the main method

That earlier approach produced geometry that looked wrong.

Instead:
- contact markup = simulated probe contact location on airway wall
- target markup = intended node focus point
- centerline tangent = shaft direction reference
- rendered sector = sampled **directly in probe coordinates** from the CT volume

### Required pose logic
For each preset:
- `contact_world` comes from the selected contact markup
- `target_world` comes from the target markup
- `shaft_axis` comes from local centerline tangent near the contact
- `depth_axis` is target direction projected perpendicular to shaft axis
- optional fine roll rotates around shaft axis

The image apex / near field must originate at the **contact point**, not at the centerline midpoint and not at an arbitrary screen location.

The airway wall should appear in the near field.
The node should appear in a believable sector direction relative to the airway and vessels.

---

## Preset-driven workflow
Version 1 is **preset-driven**, not free-navigation-first.

Each active preset corresponds to:
- one station
- one node
- one contact approach

Examples:
- `station_4r_node_b`
- `station_7_node_a` with approaches `lms` and `rms`

This is intentional.
The curated contacts and targets are high-value inputs and should drive the build.

---

## Active v1 stations / nodes
Implement only presets that have both:
- a target markup
- at least one contact markup

Use the current dataset to support:
- 2L node a
- 2R node a
- 4L nodes a, b, c
- 4R nodes a, b, c
- 7 node a with two approaches: lms and rms
- 10R nodes a, b
- 11L node a
- 11Ri node a
- 11Rs nodes a, b

Treat other station masks without matching contact/target presets as context overlays only for now.

---

## Coordinate handling
Be extremely careful with coordinate systems.

All imported data must be normalized into one internal world coordinate system before any geometry computations.
Prefer one canonical internal world frame for the whole application.

Do not hard-code blind RAS/LPS assumptions.
Instead:
- inspect metadata where available
- read markup coordinate system from `.mrk.json`
- validate registration using geometry checks

Every build should include geometry QA:
- contact is near airway surface
- target is near or inside its station mask
- contact and target are inside CT bounds
- contact projects onto centerline successfully
- centerline tangent is defined at that location

---

## Architecture
Use a clean local Python project.

Recommended package layout:

- config / manifest loading
- volume / mask IO
- VTP IO
- MRK JSON IO
- centerline graph + tangent lookup
- preset / pose generation
- direct sector sampler
- overlays
- renderer
- local desktop UI
- tests
- CLI entry points

Preferred runtime:
- Python
- NumPy
- nibabel / SimpleITK as needed
- VTK / PyVista as needed
- PySide6 for desktop UI

Keep components modular.
The sampling engine must be reusable independently of the UI.

---

## Rendering requirements
### Required for v1
- direct probe-centered CT sampling
- linear EBUS sector output
- airway / target / vessel overlays
- adjustable depth
- adjustable sector angle
- adjustable gain / attenuation
- optional fine roll

### Not required for v1
- true acoustic simulation
- Doppler
- advanced posterior shadow physics
- procedural tool animation
- radial EBUS mode

Start simple:
- soft-tissue windowing
- contrast mapping
- optional mild attenuation
- optional mild speckle
- optional light edge enhancement

Do not chase cosmetics before geometry is correct.

---

## UI requirements
Build a local desktop UI with:
- preset selector
- contact approach selector
- depth control
- sector angle control
- roll fine-adjustment
- gain / attenuation controls
- overlay toggles
- 2D EBUS pane
- 3D context pane
- screenshot export

Default mode should be a stable preset browser, not unrestricted free navigation.

---

## Testing discipline
After each phase:
- stop
- summarize what was completed
- list exact files changed
- provide run commands
- provide validation results
- state known limitations honestly

Do not silently broaden scope.

For any phase, prefer a narrow working implementation over a large speculative one.

---

## Commands
Expected commands should work from repo root:

- environment setup
- case validation
- render single preset
- launch app
- run tests

Design CLI entry points such as:
- `validate-case`
- `render-preset`
- `launch-app`

---

## Constraints
- Do not convert this into a Slicer scripted module
- Do not add radial EBUS
- Do not rename source assets unless user explicitly asks
- Do not replace curated contact / target markups with inferred positions
- Do not use rectangular reslice + 2D wedge mask as the primary renderer
- Do not optimize prematurely for web deployment

---

## Definition of success for v1
The project is successful when:
- the dataset loads through a manifest
- all intended presets validate
- each preset produces a believable linear EBUS sector
- the near field is anchored to the airway contact point
- station 7 correctly supports separate LMS and RMS approaches
- the desktop app can switch presets and render consistently
- screenshots can be exported for review