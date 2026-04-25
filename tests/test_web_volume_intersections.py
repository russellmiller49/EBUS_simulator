from pathlib import Path
import math

import numpy as np

from ebus_simulator.models import VolumeData
from ebus_simulator.rendering import build_render_context
from ebus_simulator.web_navigation import WebNavigationPose, preset_navigation_entries
from ebus_simulator.web_volume_intersections import (
    MASK_HIT_OCCUPANCY_THRESHOLD,
    WebVolumeMaskSource,
    _intersection_from_hits,
    _sample_mask_occupancy,
    build_sector_sampling_grid,
    build_volume_sector_response,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO_ROOT / "configs" / "3d_slicer_files.yaml"


def _context_and_presets():
    context = build_render_context(MANIFEST_PATH)
    presets = {entry.preset_key: entry for entry in preset_navigation_entries(context)}
    centerlines = {int(polyline.line_index): polyline for polyline in context.main_graph.polylines}
    return context, presets, centerlines


def _identity_mask_volume(data: np.ndarray) -> VolumeData:
    affine = np.eye(4, dtype=np.float64)
    return VolumeData(
        path=Path("synthetic_mask.nii.gz"),
        kind="mask",
        shape=tuple(int(dim) for dim in data.shape),
        dtype=str(data.dtype),
        affine_ras=affine,
        affine_lps=affine,
        inverse_affine_lps=affine,
        voxel_sizes_mm=np.asarray([1.0, 1.0, 1.0], dtype=np.float64),
        axis_codes_ras=("R", "A", "S"),
        data=data,
    )


def _sphere_volume(*, center: tuple[float, float, float] = (30.0, 23.0, 24.0), radius: float = 5.5) -> VolumeData:
    indices = np.indices((64, 64, 64), dtype=np.float32)
    squared_distance = (
        (indices[0] - center[0]) ** 2
        + (indices[1] - center[1]) ** 2
        + (indices[2] - center[2]) ** 2
    )
    return _identity_mask_volume((squared_distance <= radius**2).astype(np.float32))


def _sphere_pose(*, plane_x: float = 30.0) -> WebNavigationPose:
    return WebNavigationPose(
        line_index=0,
        centerline_s_mm=0.0,
        position_lps=[plane_x, 5.0, 24.0],
        tangent_lps=[0.0, 0.0, 1.0],
        depth_axis_lps=[0.0, 1.0, 0.0],
        lateral_axis_lps=[-1.0, 0.0, 0.0],
        roll_deg=0.0,
    )


def _synthetic_intersection(volume: VolumeData, pose: WebNavigationPose):
    grid = build_sector_sampling_grid(
        pose,
        max_depth_mm=40.0,
        sector_angle_deg=60.0,
        depth_samples=96,
        lateral_samples=96,
        slab_half_thickness_mm=1.0,
        slab_samples=5,
    )
    sample_occupancy = _sample_mask_occupancy(volume, grid.points_lps)
    hits = sample_occupancy >= MASK_HIT_OCCUPANCY_THRESHOLD
    return _intersection_from_hits(
        WebVolumeMaskSource(
            key="synthetic_sphere",
            id="synthetic_sphere",
            kind="node",
            label="Synthetic sphere",
            color="#93c56f",
            mask_path=Path("synthetic_mask.nii.gz"),
        ),
        hits=hits,
        sample_occupancy=sample_occupancy,
        grid=grid,
        pose=pose,
        max_depth_mm=40.0,
        sector_angle_deg=60.0,
        surface_mesh=None,
        min_hit_base_samples=2,
    )


def test_mask_occupancy_uses_interpolation_at_voxel_boundaries():
    data = np.zeros((4, 4, 4), dtype=np.float32)
    data[1, 1, 1] = 1.0
    volume = _identity_mask_volume(data)

    occupancy = _sample_mask_occupancy(
        volume,
        np.asarray(
            [
                [1.0, 1.0, 1.0],
                [1.5, 1.0, 1.0],
                [8.0, 1.0, 1.0],
            ],
            dtype=np.float64,
        ),
    )

    assert occupancy.dtype == np.float32
    assert np.all((0.0 <= occupancy) & (occupancy <= 1.0))
    assert occupancy[0] == 1.0
    assert 0.0 < occupancy[1] < 1.0
    assert occupancy[2] == 0.0


def test_interpolated_occupancy_returns_closed_sphere_contour():
    intersection = _synthetic_intersection(_sphere_volume(), _sphere_pose())

    assert intersection is not None
    assert intersection.contour_source == "interpolated_mask_occupancy"
    assert intersection.contour_count > 0
    contour = max(intersection.contours_mm, key=len)
    assert math.dist(contour[0], contour[-1]) <= 1.5
    assert intersection.aspect_ratio < 1.45


def test_offset_sphere_slice_reports_smaller_or_no_contour_outside_slab():
    volume = _sphere_volume()
    central = _synthetic_intersection(volume, _sphere_pose())
    grazing = _synthetic_intersection(volume, _sphere_pose(plane_x=35.0))
    outside = _synthetic_intersection(volume, _sphere_pose(plane_x=39.0))

    assert central is not None
    assert grazing is not None
    assert grazing.major_axis_mm < central.major_axis_mm
    assert outside is None


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
    assert {
        "contours_mm",
        "contour_count",
        "contour_source",
        "depth_extent_mm",
        "lateral_extent_mm",
        "major_axis_mm",
        "minor_axis_mm",
        "aspect_ratio",
    } <= labels[0].keys()
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
