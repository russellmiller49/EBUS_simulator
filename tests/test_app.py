from dataclasses import replace
from pathlib import Path

import numpy as np

from ebus_simulator.app import (
    PresetBrowserSession,
    PresetBrowserState,
    PresetBrowserRender,
    build_browser_screenshot_name,
    build_render_summary_text,
    build_screenshot_strip,
    compute_target_offsets_mm,
    extract_context_tile,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO_ROOT / "configs" / "3d_slicer_files.yaml"


def test_extract_context_tile_uses_lower_right_quadrant():
    panel = np.arange(4 * 4 * 3, dtype=np.uint8).reshape((4, 4, 3))
    tile = extract_context_tile(panel)

    assert tile.shape == (2, 2, 3)
    assert np.array_equal(tile, panel[2:, 2:, :])


def test_build_screenshot_strip_concatenates_panes():
    sector = np.zeros((3, 2, 3), dtype=np.uint8)
    context = np.ones((3, 2, 3), dtype=np.uint8)

    screenshot = build_screenshot_strip(sector, context)

    assert screenshot.shape == (3, 4, 3)
    assert np.array_equal(screenshot[:, :2, :], sector)
    assert np.array_equal(screenshot[:, 2:, :], context)


def test_compute_target_offsets_mm_uses_pose_axes():
    metadata = {
        "contact_world": [1.0, 2.0, 3.0],
        "target_world": [1.0, 12.0, -1.0],
        "pose_axes": {
            "depth_axis": [0.0, 1.0, 0.0],
            "lateral_axis": [0.0, 0.0, 1.0],
        },
    }

    target_depth_mm, target_lateral_mm = compute_target_offsets_mm(metadata)

    assert target_depth_mm == 10.0
    assert target_lateral_mm == -4.0


def test_build_render_summary_text_reports_eval_and_sidecars():
    sector_metadata = {
        "preset_id": "station_4r_node_b",
        "approach": "default",
        "engine": "physics",
        "image_size": [128, 128],
        "max_depth_mm": 40.0,
        "sector_angle_deg": 60.0,
        "roll_deg": 5.0,
        "gain": 1.1,
        "attenuation": 0.2,
        "contact_world": [0.0, 0.0, 0.0],
        "target_world": [0.0, 15.0, -2.0],
        "pose_axes": {
            "depth_axis": [0.0, 1.0, 0.0],
            "lateral_axis": [0.0, 0.0, 1.0],
        },
        "contact_to_airway_distance_mm": 0.3,
        "centerline_projection_distance_mm": 1.5,
        "overlays_enabled": ["airway_lumen", "target"],
        "metadata_path": "/tmp/sector.json",
        "engine_diagnostics": {
            "eval_summary": {
                "target_contrast_vs_sector": 0.12,
                "wall_contrast_vs_sector": 0.45,
                "vessel_contrast_vs_sector": -0.05,
            },
            "artifact_settings": {
                "speckle_strength": 0.22,
                "reverberation_strength": 0.28,
                "shadow_strength": 0.47,
            },
        },
    }
    context_metadata = {
        "cutaway_side": "right",
        "overlays_enabled": ["airway_lumen", "airway_wall", "target", "contact"],
        "metadata_path": "/tmp/context.json",
    }

    summary = build_render_summary_text(sector_metadata, context_metadata)

    assert "Preset: station_4r_node_b / default" in summary
    assert "Pose: target depth 15.00 mm, lateral -2.00 mm, in sector True" in summary
    assert "Physics eval: target 0.1200, wall 0.4500, vessel -0.0500" in summary
    assert "Artifacts: speckle 0.22, reverberation 0.28, shadow 0.47" in summary
    assert "Sector sidecar: /tmp/sector.json" in summary
    assert "Context sidecar: /tmp/context.json" in summary


def test_build_browser_screenshot_name_uses_render_state():
    rendered = PresetBrowserRender(
        preset_id="station_7_node_a",
        approach="lms",
        engine="physics",
        state=PresetBrowserState(
            preset_id="station_7_node_a",
            approach="lms",
            engine="physics",
            max_depth_mm=42.5,
            sector_angle_deg=55.0,
            roll_deg=-7.5,
        ),
        sector_rgb=np.zeros((2, 2, 3), dtype=np.uint8),
        context_rgb=np.zeros((2, 2, 3), dtype=np.uint8),
        sector_metadata={},
        context_metadata={},
        sector_metadata_path="/tmp/sector.json",
        context_metadata_path="/tmp/context.json",
        summary_text="summary",
        warnings=[],
    )

    filename = build_browser_screenshot_name(rendered)

    assert filename == "station_7_node_a_lms_physics_depth42.5_angle55.0_roll-7.5_browser.png"


def test_preset_browser_session_renders_sector_and_context():
    session = PresetBrowserSession(MANIFEST_PATH, width=64, height=64)
    try:
        state = replace(
            session.default_state(),
            preset_id="station_4r_node_b",
            approach="default",
            engine="physics",
            overlay_vessels=True,
        )

        rendered = session.render(state)

        assert rendered.preset_id == "station_4r_node_b"
        assert rendered.approach == "default"
        assert rendered.engine == "physics"
        assert rendered.sector_rgb.shape == (64, 64, 3)
        assert rendered.context_rgb.shape == (64, 64, 3)
        assert Path(rendered.sector_metadata_path).exists()
        assert Path(rendered.context_metadata_path).exists()
        assert rendered.summary_text
        assert "Preset: station_4r_node_b / default" in rendered.summary_text
    finally:
        session.close()
