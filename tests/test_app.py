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
        "mode": "clean",
        "view_kind": "physics_bmode",
        "max_depth_mm": 40.0,
        "sector_angle_deg": 60.0,
        "roll_deg": 5.0,
        "gain": 1.1,
        "attenuation": 0.2,
        "seed": 17,
        "contact_world": [0.0, 0.0, 0.0],
        "target_world": [0.0, 15.0, -2.0],
        "pose_axes": {
            "depth_axis": [0.0, 1.0, 0.0],
            "lateral_axis": [0.0, 0.0, 1.0],
        },
        "refined_contact_to_airway_distance_mm": 0.3,
        "centerline_projection_distance_mm": 1.5,
        "overlays_enabled": ["airway_lumen", "target"],
        "visible_overlay_names": ["target"],
        "contact_refinement_method": "raw_mesh_projection_search",
        "pose_comparison": {
            "branch_hint": "network:0",
            "branch_hint_applied": True,
        },
        "preset_override_notes": "Review SVC/azygous relationship.",
        "consistency_metrics": {
            "target_sector_coverage_fraction": 0.12,
            "target_centerline_offset_fraction": 0.33,
            "near_field_wall_occupancy_fraction": 0.22,
            "non_background_occupancy_fraction": 0.64,
            "empty_sector_fraction": 0.36,
            "target_region_contrast_vs_sector": 0.11,
            "wall_region_contrast_vs_sector": 0.17,
            "sector_brightness_mean": 0.31,
            "near_field_brightness_mean": 0.41,
            "normalization_method": "log_percentile_99.5_blended_98.5",
            "normalization_reference_percentile": 99.5,
            "normalization_reference_value": 0.88,
        },
        "engine_diagnostics": {
            "eval_summary": {
                "sector": {"pixel_count": 25, "mean": 0.31},
                "target": {"pixel_count": 12, "mean": 0.18},
                "wall": {"pixel_count": 5, "mean": 0.45},
                "vessel": {"pixel_count": 8, "mean": 0.26},
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
        "engine": "localizer",
        "mode": "debug",
        "view_kind": "diagnostic_panel",
        "cutaway_side": "right",
        "cutaway_mode": "lateral",
        "overlays_enabled": ["airway_lumen", "airway_wall", "target", "contact"],
        "visible_overlay_names": ["airway_wall", "contact"],
    }
    review_metrics = {
        "target_in_sector": True,
        "nUS_delta_deg_from_voxel_baseline": 3.25,
        "contact_delta_mm_from_voxel_baseline": 0.8,
        "station_overlap_fraction_in_fan": 0.0142,
    }

    summary = build_render_summary_text(
        sector_metadata,
        context_metadata,
        station_label="4r",
        node_label="b",
        review_metrics=review_metrics,
        flag_reasons=["wall contrast 0.450 < 0.500"],
        warnings=["Flagged local pose optimization selected branch_shift_mm=4.0."],
        screenshot_name_hint="station_4r_node_b_default_physics_depth40.0_angle60.0_roll5.0_browser.png",
    )

    assert "Preset" in summary
    assert "- Station: Station 4R" in summary
    assert "- Target / Node: Node B" in summary
    assert "- Contact Refinement: Raw Mesh Projection Search" in summary
    assert "- Target Present: Yes (eval)" in summary
    assert "- Vessel Present: Yes (eval)" in summary
    assert "- Target Coverage: 0.1200" in summary
    assert "- Near-Field Wall Occupancy: 0.2200" in summary
    assert "- Target Contrast: 0.1200" in summary
    assert "- Target Region Contrast: 0.1100" in summary
    assert "- Normalization: log_percentile_99.5_blended_98.5" in summary
    assert "- nUS Delta vs Voxel Baseline: 3.25 deg" in summary
    assert "- 3D Context: Localizer / Debug (diagnostic_panel)" in summary
    assert "wall contrast 0.450 < 0.500" in summary
    assert "station_4r_node_b_default_physics_depth40.0_angle60.0_roll5.0_browser.png" in summary


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
        assert "Render Settings" in rendered.summary_text
        assert "- Station: Station 4R" in rendered.summary_text
        assert "- Normalization:" in rendered.summary_text
        assert rendered.inspector_sections
        assert rendered.screenshot_name_hint.endswith("_browser.png")
    finally:
        session.close()


def test_preset_browser_session_station_7_approaches_remain_distinct_in_inspector():
    session = PresetBrowserSession(MANIFEST_PATH, width=64, height=64)
    try:
        lms = session.render(
            replace(
                session.default_state(),
                preset_id="station_7_node_a",
                approach="lms",
                engine="physics",
            )
        )
        rms = session.render(
            replace(
                session.default_state(),
                preset_id="station_7_node_a",
                approach="rms",
                engine="physics",
            )
        )

        assert "- Station: Station 7" in lms.summary_text
        assert "- Approach: LMS" in lms.summary_text
        assert "- Station: Station 7" in rms.summary_text
        assert "- Approach: RMS" in rms.summary_text
        assert lms.summary_text != rms.summary_text
    finally:
        session.close()
