from pathlib import Path

import numpy as np
from PIL import Image

from ebus_simulator.acoustic_properties import AcousticField
from ebus_simulator.physics_renderer import PHYSICS_ENGINE_VERSION, simulate_bmode_from_acoustic_field
from ebus_simulator.rendering import render_preset


REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO_ROOT / "configs" / "3d_slicer_files.yaml"


def test_simulate_bmode_highlights_wall_and_suppresses_air():
    impedance = np.full((48, 24), 1.55, dtype=np.float32)
    scatter = np.full((48, 24), 0.26, dtype=np.float32)
    attenuation = np.full((48, 24), 0.45, dtype=np.float32)
    airway_wall = np.zeros((48, 24), dtype=bool)
    airway_lumen = np.zeros((48, 24), dtype=bool)
    vessel = np.zeros((48, 24), dtype=bool)
    station = np.zeros((48, 24), dtype=bool)
    target_focus = np.zeros((48, 24), dtype=np.float32)

    airway_wall[7:9, 8:16] = True
    impedance[7:9, 8:16] = 1.92
    scatter[7:9, 8:16] = 0.82

    airway_lumen[9:20, 8:16] = True
    impedance[9:20, 8:16] = 0.04
    scatter[9:20, 8:16] = 0.0
    attenuation[9:20, 8:16] = 1.85

    image = simulate_bmode_from_acoustic_field(
        AcousticField(
            impedance=impedance,
            scatter=scatter,
            attenuation=attenuation,
            airway_lumen_mask=airway_lumen,
            airway_wall_mask=airway_wall,
            vessel_mask=vessel,
            station_mask=station,
            target_focus=target_focus,
        ),
        depth_step_mm=0.8,
        gain=1.0,
        attenuation_scale=1.2,
        seed=7,
    )

    wall_band = float(image[7:9, 8:16].mean())
    lumen_band = float(image[11:18, 8:16].mean())
    soft_tissue = float(image[24:32, 2:10].mean())

    assert wall_band > lumen_band
    assert soft_tissue > lumen_band
    assert np.max(image) <= 1.0


def test_render_preset_supports_physics_engine(tmp_path):
    output_path = tmp_path / "station_4r_node_b_physics.png"
    debug_map_dir = tmp_path / "debug_maps"
    rendered = render_preset(
        MANIFEST_PATH,
        "station_4r_node_b",
        output_path=output_path,
        engine="physics",
        width=96,
        height=96,
        mode="clean",
        virtual_ebus=False,
        simulated_ebus=True,
        debug_map_dir=debug_map_dir,
        speckle_strength=0.22,
        reverberation_strength=0.28,
        shadow_strength=0.47,
    )

    image = np.asarray(Image.open(output_path))
    debug_map_paths = rendered.metadata.engine_diagnostics["debug_map_paths"]

    assert output_path.exists()
    assert output_path.with_suffix(".json").exists()
    assert image.shape == (96, 96, 3)
    assert rendered.metadata.engine == "physics"
    assert rendered.metadata.engine_version == PHYSICS_ENGINE_VERSION
    assert rendered.metadata.view_kind == "physics_bmode"
    assert rendered.metadata.virtual_ebus_enabled is False
    assert rendered.metadata.simulated_ebus_enabled is True
    assert rendered.metadata.display_plane == "nUS_nB_fan"
    assert rendered.metadata.engine_diagnostics["artifact_settings"] == {
        "speckle_strength": 0.22,
        "reverberation_strength": 0.28,
        "shadow_strength": 0.47,
    }
    assert rendered.metadata.consistency_metrics["normalization_method"].startswith("log_percentile_99.5")
    assert rendered.metadata.consistency_metrics["normalization_reference_percentile"] == 99.5
    assert rendered.metadata.consistency_metrics["normalization_reference_value"] is not None
    assert rendered.metadata.consistency_metrics["non_background_occupancy_fraction"] >= 0.0
    assert rendered.metadata.consistency_metrics["target_sector_coverage_fraction"] >= 0.0
    assert rendered.metadata.consistency_metrics["consistency_bucket"] is not None
    assert rendered.metadata.consistency_metrics["support_logic_active"] in {True, False}
    assert rendered.metadata.consistency_metrics["support_logic_mode"] is not None
    assert "normalization" in rendered.metadata.engine_diagnostics
    assert "support_logic" in rendered.metadata.engine_diagnostics
    assert "eval_summary" in rendered.metadata.engine_diagnostics
    assert rendered.metadata.engine_diagnostics["eval_summary"]["wall"]["pixel_count"] > 0
    assert rendered.metadata.engine_diagnostics["eval_summary"]["wall_contrast_vs_sector"] is not None
    assert "boundary_map" in debug_map_paths
    assert "support_map" in debug_map_paths
    assert all(Path(path).exists() for path in debug_map_paths.values())
    assert np.any(image > 0)


def test_sparse_case_support_activates_without_touching_target_prominent_control(tmp_path):
    sparse = render_preset(
        MANIFEST_PATH,
        "station_7_node_a",
        approach="lms",
        output_path=tmp_path / "station_7_node_a_lms_physics.png",
        engine="physics",
        width=64,
        height=64,
        mode="clean",
        virtual_ebus=False,
        simulated_ebus=True,
    )
    control = render_preset(
        MANIFEST_PATH,
        "station_2l_node_a",
        approach="default",
        output_path=tmp_path / "station_2l_node_a_physics.png",
        engine="physics",
        width=64,
        height=64,
        mode="clean",
        virtual_ebus=False,
        simulated_ebus=True,
    )

    sparse_metrics = sparse.metadata.consistency_metrics
    control_metrics = control.metadata.consistency_metrics

    assert sparse_metrics["pre_support_consistency_bucket"] == "sparse_empty_dominant"
    assert sparse_metrics["support_logic_active"] is True
    assert sparse_metrics["support_logic_mode"] == "sparse_target_support"
    assert float(sparse_metrics["target_region_contrast_vs_sector"]) > 0.0
    assert float(sparse_metrics["empty_sector_fraction"]) < 0.98

    assert control_metrics["pre_support_consistency_bucket"] == "target_prominent"
    assert control_metrics["support_logic_active"] is False
    assert control_metrics["support_logic_mode"] == "none"
    assert float(control_metrics["target_region_contrast_vs_sector"]) >= 0.08
