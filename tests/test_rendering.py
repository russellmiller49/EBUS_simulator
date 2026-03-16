from pathlib import Path
import json

import numpy as np
from PIL import Image

from ebus_simulator.rendering import render_all_presets, render_preset


REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO_ROOT / "configs" / "3d_slicer_files.yaml"


def test_clean_mode_disables_overlays_by_default(tmp_path):
    output_path = tmp_path / "station_4r_node_b_clean.png"
    rendered = render_preset(
        MANIFEST_PATH,
        "station_4r_node_b",
        output_path=output_path,
        width=96,
        height=96,
        mode="clean",
    )

    image = np.asarray(Image.open(output_path))
    assert output_path.exists()
    assert output_path.with_suffix(".json").exists()
    assert image.shape == (96, 96, 3)
    assert rendered.metadata.mode == "clean"
    assert rendered.metadata.airway_overlay_enabled is False
    assert rendered.metadata.airway_lumen_overlay_enabled is False
    assert rendered.metadata.airway_wall_overlay_enabled is False
    assert rendered.metadata.target_overlay_enabled is False
    assert rendered.metadata.contact_overlay_enabled is False
    assert rendered.metadata.station_overlay_enabled is False
    assert rendered.metadata.vessel_overlay_names == []
    assert rendered.metadata.overlays_enabled == []


def test_debug_mode_enables_expected_contour_overlays_and_thin_slab(tmp_path):
    output_path = tmp_path / "station_4r_node_b_debug.png"
    rendered = render_preset(
        MANIFEST_PATH,
        "station_4r_node_b",
        output_path=output_path,
        width=96,
        height=96,
        mode="debug",
        vessel_overlay_names=["aorta", "superior_vena_cava"],
    )

    image = np.asarray(Image.open(output_path))
    assert np.any(image > 0)
    assert rendered.metadata.mode == "debug"
    assert rendered.metadata.airway_overlay_enabled is True
    assert rendered.metadata.airway_lumen_overlay_enabled is True
    assert rendered.metadata.airway_wall_overlay_enabled is True
    assert rendered.metadata.target_overlay_enabled is True
    assert rendered.metadata.contact_overlay_enabled is True
    assert rendered.metadata.station_overlay_enabled is True
    assert rendered.metadata.vessel_overlay_names == ["aorta", "superior_vena_cava"]
    assert rendered.metadata.slice_thickness_mm == 1.5
    assert rendered.metadata.voxel_refined_contact_to_airway_distance_mm is not None
    assert np.allclose(rendered.metadata.device_axes["nB"], rendered.metadata.pose_axes["shaft_axis"])
    assert "wall_normal" in rendered.metadata.device_axes
    assert rendered.metadata.pose_comparison["mesh_refinement_method"] is not None
    assert rendered.metadata.overlays_enabled == [
        "airway_lumen",
        "airway_wall",
        "station",
        "aorta",
        "superior_vena_cava",
        "target",
        "contact",
    ]


def test_diagnostic_panel_writes_panel_image_and_metadata(tmp_path):
    output_path = tmp_path / "station_11rs_node_b_panel.png"
    rendered = render_preset(
        MANIFEST_PATH,
        "station_11rs_node_b",
        output_path=output_path,
        width=96,
        height=96,
        mode="debug",
        diagnostic_panel=True,
        vessel_overlay_names=["azygous"],
    )

    image = np.asarray(Image.open(output_path))
    sidecar = json.loads(output_path.with_suffix(".json").read_text())

    assert image.shape == (192, 192, 3)
    assert rendered.metadata.diagnostic_panel_enabled is True
    assert rendered.metadata.image_size == [192, 192]
    assert rendered.metadata.device_model == "bf_uc180f"
    assert rendered.metadata.diagnostic_panel_layout == ["virtual_ebus", "simulated_ebus", "wide_ct_localizer", "context_3d"]
    assert rendered.metadata.display_plane == "nUS_nB_fan"
    assert rendered.metadata.reference_plane == "nUS_nB_with_lateral_thickness"
    assert sidecar["diagnostic_panel_enabled"] is True
    assert sidecar["diagnostic_panel_layout"] == ["virtual_ebus", "simulated_ebus", "wide_ct_localizer", "context_3d"]
    assert sidecar["original_contact_world"] != sidecar["refined_contact_world"]
    assert sidecar["voxel_refined_contact_world"] != []
    assert "pose_comparison" in sidecar
    assert sidecar["pose_comparison"]["mesh_refined_contact_world"] == sidecar["refined_contact_world"]
    assert sidecar["pose_comparison"]["final_nUS_world"] == sidecar["device_axes"]["nUS"]
    assert sidecar["display_plane"] == "nUS_nB_fan"
    assert sidecar["reference_plane"] == "nUS_nB_with_lateral_thickness"
    assert sidecar["cutaway_mode"] == "lateral"
    assert sidecar["cutaway_mesh_source"] == "smoothed"
    assert sidecar["show_full_airway"] is False
    assert len(sidecar["cutaway_normal"]) == 3
    assert "contact" in rendered.metadata.overlays_enabled
    assert "azygous" in rendered.metadata.vessel_overlay_names


def test_station_7_render_variants_differ(tmp_path):
    lms_output = tmp_path / "station_7_node_a_lms.png"
    rms_output = tmp_path / "station_7_node_a_rms.png"

    lms = render_preset(
        MANIFEST_PATH,
        "station_7_node_a",
        approach="lms",
        output_path=lms_output,
        width=96,
        height=96,
    )
    rms = render_preset(
        MANIFEST_PATH,
        "station_7_node_a",
        approach="rms",
        output_path=rms_output,
        width=96,
        height=96,
    )

    assert lms.metadata.approach == "lms"
    assert rms.metadata.approach == "rms"
    assert lms.metadata.pose_axes["shaft_axis"] != rms.metadata.pose_axes["shaft_axis"]
    assert lms.metadata.cutaway_side == "left"
    assert rms.metadata.cutaway_side == "right"
    assert lms.metadata.cutaway_normal != rms.metadata.cutaway_normal
    assert not np.array_equal(lms.image_rgb, rms.image_rgb)


def test_render_all_presets_writes_index_and_all_outputs(tmp_path):
    output_dir = tmp_path / "all_debug"
    index = render_all_presets(
        MANIFEST_PATH,
        output_dir=output_dir,
        width=48,
        height=48,
        mode="debug",
        vessel_overlay_names=["aorta"],
    )

    index_json = output_dir / "index.json"
    index_csv = output_dir / "index.csv"
    payload = json.loads(index_json.read_text())

    assert index.render_count == 16
    assert index_json.exists()
    assert index_csv.exists()
    assert payload["render_count"] == 16
    assert len(payload["renders"]) == 16
    assert all(Path(entry["output_image_path"]).exists() for entry in payload["renders"])
    assert all(Path(entry["sidecar_path"]).exists() for entry in payload["renders"])
