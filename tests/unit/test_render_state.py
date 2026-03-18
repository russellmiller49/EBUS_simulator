from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from ebus_simulator.render_state import (
    _resolve_cutaway_config,
    _resolve_overlay_config,
    _resolve_pose,
    _resolve_slice_thickness_mm,
)


def _manifest_with_vessels() -> SimpleNamespace:
    return SimpleNamespace(
        overlay_masks={
            "aorta": Path("aorta.nii.gz"),
            "azygous": Path("azygous.nii.gz"),
        }
    )


def test_resolve_pose_requires_approach_when_multiple_exist():
    report = SimpleNamespace(
        poses=[
            SimpleNamespace(preset_id="station_7_node_a", contact_approach="lms"),
            SimpleNamespace(preset_id="station_7_node_a", contact_approach="rms"),
        ]
    )

    with pytest.raises(ValueError, match="multiple approaches"):
        _resolve_pose(report, preset_id="station_7_node_a", approach=None)


def test_resolve_pose_selects_requested_approach():
    lms_pose = SimpleNamespace(preset_id="station_7_node_a", contact_approach="lms")
    report = SimpleNamespace(poses=[lms_pose])

    assert _resolve_pose(report, preset_id="station_7_node_a", approach="lms") is lms_pose


def test_resolve_overlay_config_uses_clean_defaults():
    config = _resolve_overlay_config(
        _manifest_with_vessels(),
        mode="clean",
        airway_overlay=None,
        airway_lumen_overlay=None,
        airway_wall_overlay=None,
        target_overlay=None,
        contact_overlay=None,
        station_overlay=None,
        vessel_overlay_names=None,
        diagnostic_panel=False,
        virtual_ebus=False,
        simulated_ebus=True,
        show_legend=None,
        label_overlays=None,
        show_frustum=None,
        min_contour_area_px=20.0,
        min_contour_length_px=15.0,
        single_vessel_name=None,
    )

    assert config.mode == "clean"
    assert config.airway_lumen_enabled is False
    assert config.airway_wall_enabled is False
    assert config.target_enabled is False
    assert config.contact_enabled is False
    assert config.station_enabled is False
    assert config.vessel_names == []


def test_resolve_overlay_config_uses_debug_defaults_and_single_vessel_override():
    config = _resolve_overlay_config(
        _manifest_with_vessels(),
        mode="debug",
        airway_overlay=None,
        airway_lumen_overlay=None,
        airway_wall_overlay=None,
        target_overlay=None,
        contact_overlay=None,
        station_overlay=None,
        vessel_overlay_names=None,
        diagnostic_panel=True,
        virtual_ebus=True,
        simulated_ebus=True,
        show_legend=None,
        label_overlays=None,
        show_frustum=None,
        min_contour_area_px=20.0,
        min_contour_length_px=15.0,
        single_vessel_name="aorta",
        preset_default_vessel_names=["azygous"],
    )

    assert config.mode == "debug"
    assert config.airway_lumen_enabled is True
    assert config.airway_wall_enabled is True
    assert config.target_enabled is True
    assert config.contact_enabled is True
    assert config.station_enabled is True
    assert config.vessel_names == ["aorta"]


def test_resolve_overlay_config_rejects_unknown_vessel():
    with pytest.raises(ValueError, match="Unknown vessel overlay"):
        _resolve_overlay_config(
            _manifest_with_vessels(),
            mode="debug",
            airway_overlay=None,
            airway_lumen_overlay=None,
            airway_wall_overlay=None,
            target_overlay=None,
            contact_overlay=None,
            station_overlay=None,
            vessel_overlay_names=["svc"],
            diagnostic_panel=False,
            virtual_ebus=False,
            simulated_ebus=True,
            show_legend=None,
            label_overlays=None,
            show_frustum=None,
            min_contour_area_px=20.0,
            min_contour_length_px=15.0,
            single_vessel_name=None,
        )


def test_slice_thickness_and_cutaway_defaults_are_mode_aware():
    assert _resolve_slice_thickness_mm("debug", None) == 1.5
    assert _resolve_slice_thickness_mm("clean", None) == 4.0

    cutaway = _resolve_cutaway_config(
        cutaway_mode=None,
        cutaway_side=None,
        cutaway_depth_mm=None,
        cutaway_origin=None,
        show_full_airway=None,
        default_side="right",
    )

    assert cutaway.mode == "lateral"
    assert cutaway.side == "right"
    assert cutaway.depth_mm == 0.0
    assert cutaway.origin_mode == "contact"
    assert cutaway.show_full_airway is False
