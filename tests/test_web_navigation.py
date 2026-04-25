from pathlib import Path

import numpy as np

from ebus_simulator.rendering import build_render_context
from ebus_simulator.web_navigation import navigation_pose_from_polyline, preset_navigation_entries


REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO_ROOT / "configs" / "3d_slicer_files.yaml"


def test_navigation_pose_axes_are_orthonormal():
    context = build_render_context(MANIFEST_PATH)
    polyline = context.main_graph.polylines[0]
    pose = navigation_pose_from_polyline(
        polyline,
        centerline_s_mm=polyline.total_length_mm * 0.35,
        roll_deg=12.0,
    )

    tangent = np.asarray(pose.tangent_lps)
    depth = np.asarray(pose.depth_axis_lps)
    lateral = np.asarray(pose.lateral_axis_lps)

    assert np.isclose(np.linalg.norm(tangent), 1.0)
    assert np.isclose(np.linalg.norm(depth), 1.0)
    assert np.isclose(np.linalg.norm(lateral), 1.0)
    assert abs(float(np.dot(tangent, depth))) < 1e-6
    assert abs(float(np.dot(tangent, lateral))) < 1e-6
    assert abs(float(np.dot(depth, lateral))) < 1e-6


def test_station_7_navigation_snaps_remain_distinct():
    context = build_render_context(MANIFEST_PATH)
    entries = {
        entry.preset_key: entry
        for entry in preset_navigation_entries(context)
        if entry.preset_id == "station_7_node_a"
    }

    assert sorted(entries) == ["station_7_node_a::lms", "station_7_node_a::rms"]
    lms = entries["station_7_node_a::lms"]
    rms = entries["station_7_node_a::rms"]

    assert lms.approach == "lms"
    assert rms.approach == "rms"
    assert (lms.line_index, round(lms.centerline_s_mm, 3)) != (rms.line_index, round(rms.centerline_s_mm, 3))
    assert not np.allclose(lms.contact_lps, rms.contact_lps)
    assert lms.vessel_overlays != rms.vessel_overlays


def test_preset_navigation_pose_uses_contact_and_exported_axes():
    context = build_render_context(MANIFEST_PATH)
    entries = {entry.preset_key: entry for entry in preset_navigation_entries(context)}
    preset = entries["station_7_node_a::rms"]
    polyline = next(polyline for polyline in context.main_graph.polylines if polyline.line_index == preset.line_index)

    pose = navigation_pose_from_polyline(
        polyline,
        centerline_s_mm=preset.centerline_s_mm,
        roll_deg=0.0,
        target_lps=np.asarray(preset.target_lps),
        contact_lps=np.asarray(preset.contact_lps),
        contact_centerline_s_mm=preset.centerline_s_mm,
        shaft_axis_lps=np.asarray(preset.shaft_axis_lps),
        depth_axis_lps=np.asarray(preset.depth_axis_lps),
    )

    assert np.allclose(pose.position_lps, preset.contact_lps)
    assert np.dot(np.asarray(pose.tangent_lps), np.asarray(preset.shaft_axis_lps)) > 0.99
    assert np.dot(np.asarray(pose.depth_axis_lps), np.asarray(preset.depth_axis_lps)) > 0.99
