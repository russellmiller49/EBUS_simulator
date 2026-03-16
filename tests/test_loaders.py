from pathlib import Path

from ebus_simulator.io.mrkjson import load_mrk_json
from ebus_simulator.io.nifti import load_nifti
from ebus_simulator.io.vtp import load_vtp_polydata
from ebus_simulator.manifest import load_case_manifest


REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO_ROOT / "configs" / "3d_slicer_files.yaml"


def test_manifest_loads_presets():
    manifest = load_case_manifest(MANIFEST_PATH)
    assert manifest.case_id == "3D_slicer_files"
    assert len(manifest.presets) == 15
    assert manifest.root.name == "3D_slicer_files"


def test_mrk_loader_normalizes_to_lps():
    markup = load_mrk_json(REPO_ROOT / "3D_slicer_files" / "markups" / "station_7_node_a_contact_lms.mrk.json")
    assert len(markup.markups) == 1
    assert markup.markups[0].coordinate_system == "LPS"
    assert len(markup.markups[0].control_points) == 1


def test_nifti_loader_reads_shape_and_axes():
    ct = load_nifti(REPO_ROOT / "3D_slicer_files" / "ct.nii.gz", kind="ct", load_data=False)
    assert len(ct.shape) == 3
    assert len(ct.axis_codes_ras) == 3


def test_vtp_loader_reads_points_and_lines():
    polydata = load_vtp_polydata(REPO_ROOT / "3D_slicer_files" / "centerlines" / "airway_centerline.vtp")
    assert polydata.points_lps.shape[1] == 3
    assert len(polydata.lines) > 0
    assert polydata.source_space == "LPS"


def test_vtp_loader_reads_binary_mesh_polygons():
    polydata = load_vtp_polydata(REPO_ROOT / "3D_slicer_files" / "meshes" / "airway_endoluminal_surface_smoothed.vtp")
    assert polydata.points_lps.shape[1] == 3
    assert len(polydata.polygons) > 0
