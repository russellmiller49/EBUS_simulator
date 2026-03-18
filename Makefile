VENV := .venv
PYTHON := $(VENV)/bin/python
PIP := $(PYTHON) -m pip
PYTEST := $(PYTHON) -m pytest
MANIFEST := configs/3d_slicer_files.yaml

.PHONY: bootstrap test test-fast test-integration validate poses render-smoke physics-smoke review-smoke ci-smoke clean-venv

bootstrap: $(VENV)/.bootstrap-stamp

$(VENV)/.bootstrap-stamp: scripts/bootstrap.sh pyproject.toml
	./scripts/bootstrap.sh

test: $(VENV)/.bootstrap-stamp
	$(PYTEST) -q

test-fast: $(VENV)/.bootstrap-stamp
	$(PYTEST) tests/unit -q

test-integration: $(VENV)/.bootstrap-stamp
	$(PYTEST) tests/integration -q

validate: $(VENV)/.bootstrap-stamp
	$(VENV)/bin/validate-case $(MANIFEST)

poses: $(VENV)/.bootstrap-stamp
	$(VENV)/bin/generate-poses $(MANIFEST)

render-smoke: $(VENV)/.bootstrap-stamp
	$(VENV)/bin/render-preset $(MANIFEST) station_4r_node_b --output reports/_smoke_station_4r_node_b_default.png

physics-smoke: $(VENV)/.bootstrap-stamp
	$(VENV)/bin/render-preset $(MANIFEST) station_4r_node_b --engine physics --mode clean --virtual-ebus false --output reports/_smoke_station_4r_node_b_physics.png

review-smoke: $(VENV)/.bootstrap-stamp
	$(VENV)/bin/review-presets $(MANIFEST) --output-dir reports/preset_review_smoke

ci-smoke: $(VENV)/.bootstrap-stamp
	mkdir -p reports
	$(VENV)/bin/validate-case $(MANIFEST) --report-json reports/ci_validation_report.json
	$(VENV)/bin/generate-poses $(MANIFEST) --report-json reports/ci_pose_report.json
	$(VENV)/bin/render-preset $(MANIFEST) station_4r_node_b --engine localizer --output reports/ci_station_4r_node_b.png

clean-venv:
	rm -rf $(VENV)
