# EBUS Simulator

Geometry-first scaffold for a standalone local linear EBUS simulator.

Current implemented capabilities:

- Python project scaffold
- Manifest-driven case loading
- NIfTI, VTP, and Slicer MRK JSON loaders
- `validate-case` CLI
- Preset-by-preset geometry QA for the configured dataset
- Centerline graph abstraction and preset pose generation
- `generate-poses` CLI
- direct probe-centered preset rendering
- `render-preset` CLI
- clean/debug render modes, station/vessel overlays, and batch render indexes
- `render-all-presets` CLI
- `review-presets` CLI for batch review exports

Not implemented yet:

- portable manifest roots
- render-engine split between localizer and physics modes
- desktop UI
- physics-based CP-EBUS renderer

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
