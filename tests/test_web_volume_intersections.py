from pathlib import Path

from ebus_simulator.rendering import build_render_context
from ebus_simulator.web_navigation import preset_navigation_entries
from ebus_simulator.web_volume_intersections import build_volume_sector_response


REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO_ROOT / "configs" / "3d_slicer_files.yaml"


def _context_and_presets():
    context = build_render_context(MANIFEST_PATH)
    presets = {entry.preset_key: entry for entry in preset_navigation_entries(context)}
    centerlines = {int(polyline.line_index): polyline for polyline in context.main_graph.polylines}
    return context, presets, centerlines


def test_volume_sector_intersections_are_deterministic_for_station_snap():
    context, presets, centerlines = _context_and_presets()
    preset = presets["station_7_node_a::lms"]

    first = build_volume_sector_response(
        context,
        centerlines_by_index=centerlines,
        preset=preset,
        line_index=preset.line_index,
        centerline_s_mm=preset.centerline_s_mm,
        roll_deg=0.0,
        max_depth_mm=40.0,
        sector_angle_deg=60.0,
        station_keys=["station_7"],
        vessel_keys=[],
        depth_samples=40,
        lateral_samples=40,
        slab_samples=5,
    )
    second = build_volume_sector_response(
        context,
        centerlines_by_index=centerlines,
        preset=preset,
        line_index=preset.line_index,
        centerline_s_mm=preset.centerline_s_mm,
        roll_deg=0.0,
        max_depth_mm=40.0,
        sector_angle_deg=60.0,
        station_keys=["station_7"],
        vessel_keys=[],
        depth_samples=40,
        lateral_samples=40,
        slab_samples=5,
    )

    assert first == second
    labels = first["sector"]["labels"]
    assert [label["id"] for label in labels] == ["station_7"]
    assert labels[0]["source"] == "volume_mask"
    assert labels[0]["hit_base_sample_count"] > 0


def test_free_navigation_does_not_report_vessels_outside_current_fan_volume():
    context, presets, centerlines = _context_and_presets()
    preset = presets["station_10r_node_b::default"]

    response = build_volume_sector_response(
        context,
        centerlines_by_index=centerlines,
        preset=preset,
        line_index=preset.line_index,
        centerline_s_mm=20.0,
        roll_deg=0.0,
        max_depth_mm=40.0,
        sector_angle_deg=60.0,
        station_keys=["station_10r"],
        vessel_keys=["pulmonary_artery", "azygous"],
        depth_samples=44,
        lateral_samples=44,
        slab_samples=5,
    )

    visible_ids = {label["id"] for label in response["sector"]["labels"]}
    assert "pulmonary_artery" not in visible_ids
    assert "azygous" not in visible_ids


def test_volume_sector_reports_shape_axes_for_long_axis_vessel_cut():
    context, presets, centerlines = _context_and_presets()
    preset = presets["station_10r_node_a::default"]

    response = build_volume_sector_response(
        context,
        centerlines_by_index=centerlines,
        preset=preset,
        line_index=preset.line_index,
        centerline_s_mm=112.0,
        roll_deg=0.0,
        max_depth_mm=40.0,
        sector_angle_deg=60.0,
        station_keys=[],
        vessel_keys=["superior_vena_cava"],
        depth_samples=48,
        lateral_samples=48,
        slab_samples=5,
    )

    labels = response["sector"]["labels"]
    assert [label["id"] for label in labels] == ["superior_vena_cava"]
    svc = labels[0]
    assert svc["major_axis_mm"] > svc["minor_axis_mm"]
    assert svc["aspect_ratio"] > 1.15
    assert len(svc["major_axis_vector_mm"]) == 2
    assert svc["contour_count"] > 0
    assert svc["contour_source"] == "surface_triangle_plane_intersection"
    assert svc["surface_source"] in {"marching_cubes", "voxel_boundary_faces"}
    assert svc["surface_triangle_count"] > 0
    assert max(len(contour) for contour in svc["contours_mm"]) >= 12
