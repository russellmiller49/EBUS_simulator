# EBUS Simulator User Guide

This guide is a practical quick-start for running the repository day to day.

## 1) What this repo does

`EBUS_simulator` provides:
- Dataset validation
- Preset pose generation
- Localizer and physics rendering
- Review bundle generation and comparison
- A desktop app for browsing presets and render settings

Core dataset manifest used in examples:
- `configs/3d_slicer_files.yaml`

## 2) Setup

### Recommended setup (project-managed `.venv`)

```bash
make bootstrap
```

This creates `.venv`, installs package dependencies, and installs the project in editable mode with dev dependencies.

### Manual setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
```

To use the desktop app, also install UI dependencies:

```bash
python -m pip install -e '.[dev,ui]'
```

## 3) Important commands (most common)

### Test

```bash
make test
```

### Validate dataset configuration

```bash
make validate
```

Equivalent direct CLI:

```bash
validate-case configs/3d_slicer_files.yaml
```

### Generate poses

```bash
make poses
```

Equivalent direct CLI:

```bash
generate-poses configs/3d_slicer_files.yaml --report-json reports/pose_report.json
```

### Render one preset

```bash
render-preset configs/3d_slicer_files.yaml station_4r_node_b --output reports/renders/station_4r_node_b.png
```

### Physics render smoke test

```bash
make physics-smoke
```

### Render all presets (debug batch)

```bash
render-all-presets configs/3d_slicer_files.yaml --output-dir reports/renders/all_debug --mode debug --overlay-vessels aorta,azygous,superior_vena_cava
```

### Generate review bundle

```bash
review-presets configs/3d_slicer_files.yaml --output-dir reports/preset_review
```

### Compare two review bundles

```bash
compare-review-bundles reports/preset_review_20260316/review_summary.json reports/preset_review_stabilized/review_summary.json --output-dir reports/preset_review_stabilized
```

### Analyze render consistency

```bash
analyze-render-consistency configs/3d_slicer_files.yaml --output-dir reports/consistency --width 64 --height 64
```

### Launch desktop app

```bash
launch-app configs/3d_slicer_files.yaml
```

## 4) Useful `make` shortcuts

```bash
make validate
make poses
make render-smoke
make physics-smoke
make review-smoke
make ci-smoke
make test
```

## 5) Troubleshooting

### `launch-app: command not found`

Cause: package console scripts are not installed in the current environment.

Fix:

```bash
python -m pip install -e '.[ui]'
```

Then retry:

```bash
launch-app configs/3d_slicer_files.yaml
```

If your shell still cannot see the command immediately:

```bash
hash -r
```

### Using `requirements.txt` only

`requirements.txt` installs dependencies, but not this repo's console scripts.  
For CLI commands like `launch-app`, `validate-case`, `render-preset`, install the project itself:

```bash
python -m pip install -e '.[dev]'
```

or with UI extras:

```bash
python -m pip install -e '.[dev,ui]'
```

## 6) Output/report locations

Common output locations used by the commands:
- `reports/validation_report.json`
- `reports/pose_report.json`
- `reports/renders/`
- `reports/preset_review*/`
- `reports/consistency/`

## 7) Suggested first-run workflow

```bash
make bootstrap
make validate
make poses
make render-smoke
make physics-smoke
make review-smoke
```

Then launch and inspect:

```bash
launch-app configs/3d_slicer_files.yaml
```
