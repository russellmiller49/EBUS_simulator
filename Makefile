VENV := .venv
PYTHON := $(VENV)/bin/python
PIP := $(PYTHON) -m pip
PYTEST := $(PYTHON) -m pytest
MANIFEST := configs/3d_slicer_files.yaml
REFERENCE_CONFIG := configs/video_references.yaml
WEB_DIR := web
WEB_CASE_DIR := reports/web_case

.PHONY: bootstrap test validate poses render-smoke physics-smoke reference-smoke review-smoke web-bootstrap web-export web-build web-launch web-test ci-smoke clean-venv

bootstrap: $(VENV)/.bootstrap-stamp

$(VENV)/.bootstrap-stamp: scripts/bootstrap.sh pyproject.toml
	./scripts/bootstrap.sh

test: $(VENV)/.bootstrap-stamp
	$(PYTEST) -q

validate: $(VENV)/.bootstrap-stamp
	$(VENV)/bin/validate-case $(MANIFEST)

poses: $(VENV)/.bootstrap-stamp
	$(VENV)/bin/generate-poses $(MANIFEST)

render-smoke: $(VENV)/.bootstrap-stamp
	$(VENV)/bin/render-preset $(MANIFEST) station_4r_node_b --output reports/_smoke_station_4r_node_b_default.png

physics-smoke: $(VENV)/.bootstrap-stamp
	$(VENV)/bin/render-preset $(MANIFEST) station_4r_node_b --engine physics --mode clean --virtual-ebus false --output reports/_smoke_station_4r_node_b_physics.png

reference-smoke: $(VENV)/.bootstrap-stamp
	$(VENV)/bin/build-reference-library $(REFERENCE_CONFIG) --output-dir reports/reference_library --frame-size-px 256

review-smoke: $(VENV)/.bootstrap-stamp
	$(VENV)/bin/review-presets $(MANIFEST) --output-dir reports/preset_review_smoke

web-bootstrap: $(VENV)/.bootstrap-stamp
	$(PIP) install -e '.[dev,web]'
	npm --prefix $(WEB_DIR) install

web-export: web-bootstrap
	$(VENV)/bin/export-web-case $(MANIFEST) --output-dir $(WEB_CASE_DIR)

web-build: web-bootstrap
	npm --prefix $(WEB_DIR) run build

web-launch: web-export web-build
	$(VENV)/bin/launch-web-app $(MANIFEST) --web-case $(WEB_CASE_DIR)

web-test: web-build
	npm --prefix $(WEB_DIR) run test:browser

ci-smoke: $(VENV)/.bootstrap-stamp
	mkdir -p reports
	$(VENV)/bin/validate-case $(MANIFEST) --report-json reports/ci_validation_report.json
	$(VENV)/bin/generate-poses $(MANIFEST) --report-json reports/ci_pose_report.json
	$(VENV)/bin/render-preset $(MANIFEST) station_4r_node_b --engine localizer --output reports/ci_station_4r_node_b.png

clean-venv:
	rm -rf $(VENV)
