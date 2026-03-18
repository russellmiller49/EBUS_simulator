# EBUS Simulator

Geometry-first standalone local linear EBUS simulator with localizer, physics, and review workflows.

Current implemented capabilities:

- Python project scaffold
- Manifest-driven case loading
- portable manifest roots
- NIfTI, VTP, and Slicer MRK JSON loaders
- `validate-case` CLI
- Preset-by-preset geometry QA for the configured dataset
- Centerline graph abstraction and preset pose generation
- `generate-poses` CLI
- explicit render-engine split between `localizer` and `physics`
- direct probe-centered preset rendering
- `render-preset` CLI
- clean/debug render modes, station/vessel overlays, metadata sidecars, and batch render indexes
- `render-all-presets` CLI
- `review-presets` CLI for deterministic physics-aware review bundles with JSON/CSV/Markdown indexes and review sheets
- `compare-review-bundles` CLI for before/after calibration summaries across review bundle runs
- configurable geometry and physics auto-flag thresholds for review bundles
- conservative default wall-contrast auto-flagging with CLI override support
- first-pass physics CP-EBUS renderer with artifact controls, debug maps, and eval summaries
- desktop preset-browser slice with queued rendering, reviewer summary, 2D EBUS, and 3D context panes
- CI smoke workflow

Not implemented yet:

- review / calibration workflow still needs tuning against real CP-EBUS reference material
- a packaged Qt runtime setup and broader manual desktop validation

Next active milestone:

- desktop preset browser refinement and manual Qt validation
- review / calibration refinement on top of the current bundle workflow as reference feedback becomes available

## Dataset

The case manifest in this repository points at the checked-in `3D_slicer_files` dataset copy. No source assets are renamed or modified.

## Setup

Bootstrap the repo with one command:

```bash
make bootstrap
```

That creates `.venv`, upgrades `pip`, and installs the package in editable mode with dev dependencies.

If you prefer not to use `make`, the equivalent command is:

```bash
./scripts/bootstrap.sh
```

Manual setup remains available:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
```

To use the desktop app, install the UI extra:

```bash
python -m pip install -e '.[dev,ui]'
```

## Commands

The `make` targets below do not require manually activating the virtual environment.
For the raw CLI commands, either activate `.venv` first or prefix commands with `.venv/bin/`.

Run tests:

```bash
make test
```

`make test` runs the full suite, including dataset-backed rendering tests, so it can take materially longer than the small unit-test subset.

Validate the configured case:

```bash
validate-case configs/3d_slicer_files.yaml
```

Write a machine-readable JSON report:

```bash
validate-case configs/3d_slicer_files.yaml --report-json reports/validation_report.json
```

Generate preset poses and export the machine-readable pose report:

```bash
generate-poses configs/3d_slicer_files.yaml --report-json reports/pose_report.json
```

Render a single preset and emit the JSON sidecar metadata:

```bash
render-preset configs/3d_slicer_files.yaml station_4r_node_b --output reports/renders/station_4r_node_b.png
```

Render the physics engine with tunable artifact strengths and per-render debug maps:

```bash
render-preset configs/3d_slicer_files.yaml station_4r_node_b --engine physics --mode clean --virtual-ebus false --speckle-strength 0.22 --reverberation-strength 0.28 --shadow-strength 0.47 --debug-map-dir reports/renders/station_4r_node_b_debug_maps --output reports/renders/station_4r_node_b_physics.png
```

Render a CP-EBUS diagnostic multi-panel export with a refined wall contact, larger source oblique section, and labeled contour overlays:

```bash
render-preset configs/3d_slicer_files.yaml station_7_node_a --approach lms --mode debug --diagnostic-panel --device bf_uc180f --refine-contact true --virtual-ebus true --simulated-ebus true --overlay-airway-lumen true --overlay-airway-wall true --overlay-station true --overlay-target true --show-contact --show-frustum --show-legend --label-overlays --overlay-vessels azygous,pulmonary_artery,left_atrium --source-oblique-size-mm 51.79 --reference-fov-mm 100 --slice-thickness-mm 1.5 --output reports/renders/station_7_node_a_lms_panel.png
```

Render all preset/contact approaches in debug mode and write review indexes:

```bash
render-all-presets configs/3d_slicer_files.yaml --output-dir reports/renders/all_debug --mode debug --overlay-vessels aorta,azygous,superior_vena_cava
```

Generate a batch review bundle:

```bash
review-presets configs/3d_slicer_files.yaml --output-dir reports/preset_review
```

Generate a filtered physics-aware review bundle with debug maps and tuned auto-flag thresholds:

```bash
review-presets configs/3d_slicer_files.yaml --output-dir reports/preset_review --preset-id station_4r_node_b --preset-id station_7_node_a --physics-debug-maps --physics-speckle-strength 0.22 --physics-reverberation-strength 0.28 --physics-shadow-strength 0.47 --warn-min-target-contrast 0.00 --warn-max-vessel-contrast -0.01
```

Disable the default wall-contrast auto-flag threshold for a calibration run:

```bash
review-presets configs/3d_slicer_files.yaml --output-dir reports/preset_review --warn-min-wall-contrast off
```

Compare a baseline and tuned review bundle to generate before/after calibration summaries:

```bash
compare-review-bundles reports/preset_review_20260316/review_summary.json reports/preset_review_stabilized/review_summary.json --output-dir reports/preset_review_stabilized
```

Convenience targets:

```bash
make validate
make poses
make render-smoke
make physics-smoke
make review-smoke
make ci-smoke
make test
```

Launch the first desktop preset browser:

```bash
launch-app configs/3d_slicer_files.yaml
```
