from pathlib import Path

from ebus_simulator.validation import validate_case


REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO_ROOT / "configs" / "3d_slicer_files.yaml"


def test_validator_runs_for_manifest():
    report = validate_case(MANIFEST_PATH)
    assert report.case_id == "3D_slicer_files"
    assert report.preset_count == 15
    assert len(report.presets) == 15
