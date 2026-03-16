# PLANS.md

## Project plan: standalone linear EBUS simulator

This document is the execution plan for the local linear EBUS simulator.

The build will proceed in bounded phases.
Each phase should end with:
- a brief implementation summary
- exact files changed
- exact commands run
- validation evidence
- known limitations

---

## Overall objective
Create a standalone local application that renders believable **linear EBUS** views from:
- CT
- centerline / network
- airway mask
- station masks
- vessel / organ masks
- curated target / contact markups

The simulator should be preset-driven and educationally useful for common mediastinal / hilar lymph node stations.

---

## Phase 0 — repo scaffold and manifest
### Goal
Create the project structure and a manifest/config that maps the current dataset files without renaming them.

### Deliverables
- Python project scaffold
- `configs/3D_slicer_files.yaml`
- base package structure
- README with setup instructions

### Acceptance criteria
- repository installs in a fresh virtual environment
- manifest parses successfully
- no source data files are renamed or modified

### Notes
Do not implement rendering yet.

---

## Phase 1 — data loaders and case validator
### Goal
Load all required inputs and validate geometry / registration assumptions.

### Implement
- NIfTI volume loader
- NIfTI mask loader
- VTP polydata loader
- MRK JSON loader
- manifest loader
- case validator CLI

### Validator output should include
- file existence
- parsed preset count
- station/contact/target mapping
- contact-to-airway distance
- target-to-station distance
- contact projection to centerline
- CT bounds checks
- warnings for suspicious registration / coordinate issues

### Acceptance criteria
- all intended v1 presets parse successfully
- validator produces machine-readable report
- suspicious presets are flagged clearly instead of ignored

### Stop after this phase
Do not build UI yet.

---

## Phase 2 — centerline graph and preset pose generation
### Goal
Generate a valid probe pose for every preset.

### Implement
- centerline graph abstraction
- nearest-point lookup on centerline / network
- local tangent estimation
- contact-anchored pose generation
- target-directed sector axis generation
- optional roll fine-adjustment
- preset pose report / debug export

### Required geometry model
For each preset:
- contact = apex / near-field anchor
- shaft axis = centerline tangent near contact
- target direction = projected perpendicular direction from contact to target
- optional roll = rotation around shaft axis

### Acceptance criteria
- every v1 preset yields a valid pose
- station 7 LMS and RMS yield different poses for the same target
- target is plausibly inside or near the default sector direction
- pose generation is deterministic and test-covered

### Stop after this phase
Do not build full desktop UI yet.

---

## Phase 3 — direct probe-centered sector renderer
### Goal
Render linear EBUS images correctly from the CT using direct sampling in probe coordinates.

### Implement
- output image grid in probe coordinates
- direct CT sampling for each sector pixel
- outside-sector pixels remain black because they are never sampled
- soft tissue intensity mapping
- optional attenuation
- optional mild speckle
- optional mild edge enhancement
- PNG export for rendered presets

### Important
Do **not** implement the renderer as:
- rectangular oblique reslice
- plus pasted 2D fan mask
as the main rendering path

### Acceptance criteria
- the sector apex is anchored to the contact point
- airway wall appears near the near field
- target appears in believable orientation relative to contact
- at least these presets render to image files:
  - station_4r_node_b
  - station_7_node_a via lms
  - station_7_node_a via rms
  - station_11rs_node_b

### Stop after this phase
Do not broaden to quiz logic yet.

---

## Phase 4 — overlay system
### Goal
Add meaningful educational overlays without obscuring the geometry.

### Implement
- airway overlay
- target marker
- station mask overlay
- optional vessel overlays
- simple legend / labeling support

### Default behavior
- minimal overlays on by default for debug mode
- clutter off by default for “clean view” mode

### Acceptance criteria
- overlays register correctly with rendered sector
- airway and target overlays are useful but not visually dominant
- vessel overlays can be toggled independently

---

## Phase 5 — desktop application
### Goal
Create a local desktop app for interacting with presets and viewing the simulator.

### Implement
- PySide6 application
- preset selector
- contact approach selector
- controls for:
  - depth
  - sector angle
  - roll fine-adjustment
  - gain
  - attenuation
- 2D EBUS pane
- 3D context pane
- screenshot export

### Acceptance criteria
- app launches locally from CLI
- switching presets updates both 2D and 3D views
- switching station 7 from LMS to RMS changes approach correctly
- screenshots export successfully

### Stop after this phase
Do not add scoring engine yet.

---

## Phase 6 — polish and QA tools
### Goal
Make the simulator easier to review, debug, and refine.

### Implement
- geometry debug overlays
- preset report export
- screenshot batch export
- optional side-by-side clean/debug views
- optional landmark visibility toggles

### Acceptance criteria
- a reviewer can inspect geometry for all v1 presets
- exported screenshots are organized and reproducible

---

## Phase 7 — trainer mode
### Goal
Layer educational interaction on top of the working simulator.

### Implement
- study mode
- quiz mode
- hide / reveal target
- hint / overlay toggles
- simple session logging

### Acceptance criteria
- the user can browse stations in study mode
- the user can test himself in quiz mode
- no core rendering regressions

---

## Out of scope for v1
Do not implement these unless explicitly requested:
- radial EBUS
- full procedural bronchoscopy navigation
- live scope animation through the airway tree
- web deployment
- DICOM-first ingest path
- Slicer runtime module
- Doppler
- advanced ultrasound physics
- automated staging recommendations

---

## Dataset-specific notes
### Active v1 presets
Implement these as first-class presets:
- station_2l_node_a
- station_2r_node_a
- station_4l_node_a
- station_4l_node_b
- station_4l_node_c
- station_4r_node_a
- station_4r_node_b
- station_4r_node_c
- station_7_node_a with approach lms
- station_7_node_a with approach rms
- station_10r_node_a
- station_10r_node_b
- station_11l_node_a
- station_11ri_node_a
- station_11rs_node_a
- station_11rs_node_b

### Context-only masks for now
Other station masks and anatomy masks without complete presets may be shown as context overlays only.

---

## Suggested CLI
Expected commands:

- `validate-case configs/3D_slicer_files.yaml`
- `render-preset configs/3D_slicer_files.yaml station_4r_node_b`
- `render-preset configs/3D_slicer_files.yaml station_7_node_a --approach lms`
- `render-preset configs/3D_slicer_files.yaml station_7_node_a --approach rms`
- `launch-app configs/3D_slicer_files.yaml`
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
5. state known limitations and next recommended pass