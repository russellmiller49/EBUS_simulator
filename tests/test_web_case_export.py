from pathlib import Path
import json

from ebus_simulator.web_case_export import attach_clean_model_assets, attach_scope_model_asset, export_web_case


REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO_ROOT / "configs" / "3d_slicer_files.yaml"


def test_export_web_case_writes_manifest_and_assets(tmp_path):
    result = export_web_case(
        MANIFEST_PATH,
        output_dir=tmp_path / "web_case",
        max_mask_points=25,
        max_station_points=20,
    )

    manifest_path = Path(result.manifest_path)
    payload = json.loads(manifest_path.read_text())

    assert manifest_path.exists()
    assert payload["schema_version"] == 1
    assert payload["case_id"] == "3D_slicer_files"
    assert payload["navigation"]["mode"] == "guided_centerline"
    assert len(payload["presets"]) == 16
    assert result.airway_vertex_count > 0
    assert result.airway_triangle_count > 0
    assert result.vessel_asset_count > 0
    assert result.station_asset_count > 0
    assert result.clean_model_asset_count == 0
    assert payload["color_map"]["airway"] == "#22c7c9"
    assert payload["color_map"]["superior_vena_cava"] == "#2276c9"
    assert payload["color_map"]["pulmonary_artery"] == "#0b4f9f"
    assert payload["color_map"]["pulmonary_venous_system"] == "#ee7772"
    assert payload["color_map"]["aorta"] == "#d13f3f"
    assert payload["assets"]["scope_model"] is None

    airway_path = manifest_path.parent / payload["assets"]["airway_mesh"]
    centerline_path = manifest_path.parent / payload["assets"]["centerlines"]
    assert airway_path.exists()
    assert centerline_path.exists()

    airway = json.loads(airway_path.read_text())
    centerlines = json.loads(centerline_path.read_text())
    assert airway["vertex_count"] > 0
    assert airway["triangle_count"] > 0
    assert centerlines["primary_total_length_mm"] > 0
    assert centerlines["polylines"]


def test_export_web_case_preserves_station_7_approaches(tmp_path):
    result = export_web_case(
        MANIFEST_PATH,
        output_dir=tmp_path / "web_case",
        max_mask_points=10,
        max_station_points=10,
    )
    payload = json.loads(Path(result.manifest_path).read_text())
    station_7 = {
        preset["preset_key"]: preset
        for preset in payload["presets"]
        if preset["preset_id"] == "station_7_node_a"
    }

    assert sorted(station_7) == ["station_7_node_a::lms", "station_7_node_a::rms"]
    assert station_7["station_7_node_a::lms"]["approach"] == "lms"
    assert station_7["station_7_node_a::rms"]["approach"] == "rms"
    assert station_7["station_7_node_a::lms"]["contact"] != station_7["station_7_node_a::rms"]["contact"]
    assert station_7["station_7_node_a::lms"]["vessel_overlays"] != station_7["station_7_node_a::rms"]["vessel_overlays"]


def test_export_web_case_can_attach_clean_glb_models(tmp_path):
    clean_model_dir = tmp_path / "clean_models"
    clean_model_dir.mkdir()
    (clean_model_dir / "case_001.glb").write_bytes(b"glb placeholder")
    (clean_model_dir / "CT_segmentation_1.glb").write_bytes(b"another placeholder")

    result = export_web_case(
        MANIFEST_PATH,
        output_dir=tmp_path / "web_case",
        max_mask_points=10,
        max_station_points=10,
        clean_model_dir=clean_model_dir,
    )
    payload = json.loads(Path(result.manifest_path).read_text())

    clean_models = payload["assets"]["clean_models"]
    assert result.clean_model_asset_count == 2
    assert [asset["asset"] for asset in clean_models] == [
        "models/CT_segmentation_1.glb",
        "models/case_001.glb",
    ]
    assert sum(1 for asset in clean_models if asset["primary"]) == 1
    assert next(asset for asset in clean_models if asset["primary"])["asset"] == "models/case_001.glb"
    assert (Path(result.manifest_path).parent / "models" / "case_001.glb").exists()

    replacement_dir = tmp_path / "replacement_models"
    replacement_dir.mkdir()
    (replacement_dir / "replacement.glb").write_bytes(b"replacement placeholder")
    attached = attach_clean_model_assets(Path(result.manifest_path).parent, replacement_dir)
    updated = json.loads(Path(result.manifest_path).read_text())
    assert attached == 1
    assert updated["assets"]["clean_models"][0]["asset"] == "models/replacement.glb"


def test_export_web_case_can_attach_scope_glb_model(tmp_path):
    scope_model = tmp_path / "EBUS_tip.glb"
    scope_model.write_bytes(b"scope placeholder")

    result = export_web_case(
        MANIFEST_PATH,
        output_dir=tmp_path / "web_case",
        max_mask_points=10,
        max_station_points=10,
        scope_model_path=scope_model,
    )
    payload = json.loads(Path(result.manifest_path).read_text())
    scope_asset = payload["assets"]["scope_model"]

    assert scope_asset["asset"] == "models/device/EBUS_tip.glb"
    assert scope_asset["shaft_axis"] == "+x"
    assert scope_asset["depth_axis"] == "+y"
    assert scope_asset["lateral_axis"] == "-z"
    assert scope_asset["origin"] == "fan_apex_anchor_at_probe_contact"
    assert scope_asset["fan_apex_anchor"] == {"x": "center", "y": "max", "z": "center"}
    assert scope_asset["fan_apex_anchor_point"] == [-0.334, -0.055, 0.0]
    assert scope_asset["scale_mm_per_unit"] == 44.0
    assert scope_asset["lock_to_fan"] is True
    assert scope_asset["show_auxiliary_shaft"] is False
    assert (Path(result.manifest_path).parent / scope_asset["asset"]).exists()

    replacement = tmp_path / "replacement_scope.glb"
    replacement.write_bytes(b"replacement scope")
    attached = attach_scope_model_asset(Path(result.manifest_path).parent, replacement)
    updated = json.loads(Path(result.manifest_path).read_text())
    assert attached is True
    assert updated["assets"]["scope_model"]["asset"] == "models/device/replacement_scope.glb"
