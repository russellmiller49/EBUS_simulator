from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

from ebus_simulator.cutaway import build_display_cutaway
from ebus_simulator.device import build_device_pose
from ebus_simulator.manifest import resolve_preset_overrides
from ebus_simulator.render_engines import RenderEngine, RenderRequest, RenderResult
from ebus_simulator.rendering import (
    AIRWAY_LUMEN_COLOR,
    AIRWAY_WALL_COLOR,
    CONTACT_MARKER_COLOR,
    DEFAULT_SLAB_SAMPLES,
    FAN_BOUNDARY_COLOR,
    OPTIMIZATION_EPSILON,
    STATION_COLOR,
    TARGET_MARKER_COLOR,
    VESSEL_OVERLAY_PALETTE,
    OverlayLayer,
    RenderContext,
    RenderMetadata,
    RenderedPreset,
    SourceSection,
    _LOCAL_POSE_OPTIMIZATION_CACHE,
    _annotate_legend_and_labels,
    _apply_contour_overlay,
    _build_cp_context_snapshot,
    _build_sector_grid,
    compute_render_consistency_metrics,
    _compose_cp_diagnostic_panel,
    _draw_cross_marker,
    _extract_local_mask_points,
    _fan_target_row_col,
    _filter_mask_components,
    _get_mask_volume,
    _map_plane_to_fan,
    _optimize_flagged_pose_locally,
    _overlay_summary,
    _plane_point_to_pixel,
    _preprocess_source_section,
    _resolve_branch_shift_seed,
    _resolve_cutaway_config,
    _resolve_overlay_config,
    _resolve_pose,
    _resolve_preset_manifest,
    _resolve_slice_thickness_mm,
    _sample_contact_plane,
    _sample_plane,
    _window_ct,
    build_render_context,
)


LOCALIZER_ENGINE_VERSION = "localizer-v1"


def render_localizer_preset(
    request: RenderRequest,
    *,
    context: RenderContext | None = None,
) -> RenderResult:
    if request.engine is not RenderEngine.LOCALIZER:
        raise ValueError(f"render_localizer_preset expected engine=localizer, got {request.engine.value!r}.")

    manifest_path = request.manifest_path
    preset_id = request.preset_id
    approach = request.approach
    output_path = request.output_path
    metadata_path = request.metadata_path
    width = request.width
    height = request.height
    sector_angle_deg = request.sector_angle_deg
    max_depth_mm = request.max_depth_mm
    roll_deg = request.roll_deg
    mode = request.mode
    airway_overlay = request.airway_overlay
    airway_lumen_overlay = request.airway_lumen_overlay
    airway_wall_overlay = request.airway_wall_overlay
    target_overlay = request.target_overlay
    contact_overlay = request.contact_overlay
    station_overlay = request.station_overlay
    vessel_overlay_names = request.vessel_overlay_names
    slice_thickness_mm = request.slice_thickness_mm
    diagnostic_panel = request.diagnostic_panel
    device = request.device
    refine_contact = request.refine_contact
    virtual_ebus = request.virtual_ebus
    simulated_ebus = request.simulated_ebus
    reference_fov_mm = request.reference_fov_mm
    source_oblique_size_mm = request.source_oblique_size_mm
    single_vessel = request.single_vessel
    show_legend = request.show_legend
    label_overlays = request.label_overlays
    min_contour_area_px = request.min_contour_area_px
    min_contour_length_px = request.min_contour_length_px
    show_contact = request.show_contact
    show_frustum = request.show_frustum
    cutaway_mode = request.cutaway_mode
    cutaway_side = request.cutaway_side
    cutaway_depth_mm = request.cutaway_depth_mm
    cutaway_origin = request.cutaway_origin
    show_full_airway = request.show_full_airway
    cutaway_custom_origin_world = request.cutaway_custom_origin_world
    seed = request.seed

    render_context = build_render_context(manifest_path, roll_deg=roll_deg) if context is None else context
    manifest = render_context.manifest
    defaults = manifest.render_defaults
    pose = _resolve_pose(render_context.pose_report, preset_id=preset_id, approach=approach)
    preset_manifest = _resolve_preset_manifest(manifest, pose.preset_id)
    preset_overrides = resolve_preset_overrides(preset_manifest, approach=pose.contact_approach)

    overlay_config = _resolve_overlay_config(
        manifest,
        mode=mode,
        airway_overlay=airway_overlay,
        airway_lumen_overlay=airway_lumen_overlay,
        airway_wall_overlay=airway_wall_overlay,
        target_overlay=target_overlay,
        contact_overlay=(show_contact if contact_overlay is None else contact_overlay),
        station_overlay=station_overlay,
        vessel_overlay_names=vessel_overlay_names,
        diagnostic_panel=diagnostic_panel,
        virtual_ebus=virtual_ebus,
        simulated_ebus=simulated_ebus,
        show_legend=show_legend,
        label_overlays=label_overlays,
        show_frustum=show_frustum,
        min_contour_area_px=min_contour_area_px,
        min_contour_length_px=min_contour_length_px,
        single_vessel_name=single_vessel,
        preset_default_vessel_names=(None if preset_overrides is None else preset_overrides.vessel_overlays),
    )
    cutaway_config = _resolve_cutaway_config(
        cutaway_mode=cutaway_mode,
        cutaway_side=cutaway_side,
        cutaway_depth_mm=cutaway_depth_mm,
        cutaway_origin=cutaway_origin,
        show_full_airway=show_full_airway,
        default_side=(None if preset_overrides is None else preset_overrides.cutaway_side),
    )
    if not overlay_config.virtual_ebus_enabled and not overlay_config.simulated_ebus_enabled:
        raise ValueError("At least one of virtual_ebus or simulated_ebus must be enabled.")

    resolved_width = int(defaults.get("image_size", [512, 512])[0] if width is None else width)
    resolved_height = int(defaults.get("image_size", [512, 512])[1] if height is None else height)
    preset_roll_offset_deg = 0.0 if preset_overrides is None or preset_overrides.roll_offset_deg is None else float(preset_overrides.roll_offset_deg)
    configured_axis_sign_override = None if preset_overrides is None else preset_overrides.axis_sign_override
    configured_branch_hint = None if preset_overrides is None else preset_overrides.branch_hint
    configured_branch_shift_mm = None if preset_overrides is None else preset_overrides.branch_shift_mm
    resolved_gain = float(defaults.get("gain", 1.0))
    resolved_attenuation = float(defaults.get("attenuation", 0.15))
    resolved_slice_thickness_mm = _resolve_slice_thickness_mm(overlay_config.mode, slice_thickness_mm)
    fallback_probe_axis_world = np.asarray(
        pose.depth_axis if pose.depth_axis is not None else pose.default_depth_axis,
        dtype=np.float64,
    )
    branch_shift_seed = (
        None
        if configured_branch_hint is None or configured_branch_shift_mm is None
        else _resolve_branch_shift_seed(
            render_context,
            pose=pose,
            branch_hint=configured_branch_hint,
            branch_shift_mm=float(configured_branch_shift_mm),
            fallback_contact_world=np.asarray(pose.contact_world, dtype=np.float64),
            fallback_probe_axis_world=fallback_probe_axis_world,
        )
    )
    device_pose = build_device_pose(
        pose,
        device_name=device,
        ct_volume=render_context.ct_volume,
        airway_lumen=render_context.airway_lumen_volume,
        airway_solid=render_context.airway_solid_volume,
        raw_airway_mesh=render_context.airway_geometry_mesh,
        main_graph=render_context.main_graph,
        network_graph=render_context.network_graph,
        refine_contact=refine_contact,
        roll_offset_deg=preset_roll_offset_deg,
        axis_sign_override=configured_axis_sign_override,
        branch_hint=configured_branch_hint,
        contact_seed_world=(None if branch_shift_seed is None else branch_shift_seed[0]),
        shaft_axis_override=(None if branch_shift_seed is None else branch_shift_seed[1]),
        depth_axis_override=(None if branch_shift_seed is None else branch_shift_seed[2]),
    )
    baseline_device_pose = device_pose

    resolved_sector_angle_deg = float(device_pose.device_model.sector_angle_deg if sector_angle_deg is None else sector_angle_deg)
    resolved_max_depth_mm = float(device_pose.device_model.displayed_range_mm if max_depth_mm is None else max_depth_mm)
    resolved_source_oblique_size_mm = float(device_pose.device_model.source_oblique_size_mm if source_oblique_size_mm is None else source_oblique_size_mm)
    optimization_cache_key = None if configured_branch_hint is None or configured_branch_shift_mm is not None else (
        str(render_context.manifest.manifest_path),
        pose.preset_id,
        pose.contact_approach,
        device,
        configured_branch_hint,
        resolved_width,
        resolved_height,
        round(resolved_source_oblique_size_mm, 4),
        round(resolved_max_depth_mm, 4),
        round(resolved_sector_angle_deg, 4),
        round(resolved_slice_thickness_mm, 4),
        round(preset_roll_offset_deg, 4),
        configured_axis_sign_override,
    )
    optimization_result = None if optimization_cache_key is None else _LOCAL_POSE_OPTIMIZATION_CACHE.get(optimization_cache_key)
    if optimization_result is None and configured_branch_hint is not None and configured_branch_shift_mm is None:
        optimization_result = _optimize_flagged_pose_locally(
            render_context,
            pose=pose,
            preset_manifest=preset_manifest,
            device=device,
            branch_hint=configured_branch_hint,
            base_device_pose=device_pose,
            base_roll_offset_deg=preset_roll_offset_deg,
            base_axis_sign_override=configured_axis_sign_override,
            width=resolved_width,
            height=resolved_height,
            source_oblique_size_mm=resolved_source_oblique_size_mm,
            max_depth_mm=resolved_max_depth_mm,
            sector_angle_deg=resolved_sector_angle_deg,
            slice_thickness_mm=resolved_slice_thickness_mm,
        )
        if optimization_cache_key is not None and optimization_result is not None:
            _LOCAL_POSE_OPTIMIZATION_CACHE[optimization_cache_key] = optimization_result
    if optimization_result is not None:
        device_pose = optimization_result.device_pose
        preset_roll_offset_deg = float(optimization_result.roll_offset_deg)
        configured_axis_sign_override = optimization_result.axis_sign_override
    resolved_roll_deg = float(pose.roll_deg + preset_roll_offset_deg)
    resolved_reference_fov_mm = float(
        (
            device_pose.device_model.reference_fov_mm
            if preset_overrides is None or preset_overrides.reference_fov_mm is None
            else preset_overrides.reference_fov_mm
        )
        if reference_fov_mm is None
        else reference_fov_mm
    )

    contact_world = np.asarray(device_pose.contact_refinement.refined_contact_world, dtype=np.float64)
    original_contact_world = np.asarray(device_pose.contact_refinement.original_contact_world, dtype=np.float64)
    target_world = np.asarray(device_pose.target_world, dtype=np.float64)
    shaft_axis = np.asarray(device_pose.shaft_axis_world, dtype=np.float64)
    probe_axis = np.asarray(device_pose.probe_axis_world, dtype=np.float64)
    lateral_axis = np.asarray(device_pose.lateral_axis_world, dtype=np.float64)
    station_local = _extract_local_mask_points(
        _get_mask_volume(render_context, preset_manifest.station_mask),
        center_world=contact_world,
        lateral_axis=lateral_axis,
        probe_axis=probe_axis,
        shaft_axis=shaft_axis,
        radius_mm=max(resolved_max_depth_mm, 32.0),
        max_points=1200,
    )
    station_visibility_points_world = (
        None
        if station_local.size == 0
        else (
            contact_world[None, :]
            + station_local[:, [0]] * lateral_axis[None, :]
            + station_local[:, [1]] * probe_axis[None, :]
            + station_local[:, [2]] * shaft_axis[None, :]
        )
    )
    if render_context.airway_display_mesh is None:
        raise ValueError("Smoothed airway display mesh is unavailable at meshes/airway_endoluminal_surface_smoothed.vtp.")
    cutaway_display = build_display_cutaway(
        render_context.airway_display_mesh,
        mesh_source="smoothed",
        station=pose.station,
        approach=pose.contact_approach,
        mode=cutaway_config.mode,
        requested_side=cutaway_config.side,
        origin_mode=cutaway_config.origin_mode,
        depth_mm=cutaway_config.depth_mm,
        show_full_airway=cutaway_config.show_full_airway,
        contact_world=contact_world,
        target_world=target_world,
        lateral_axis_world=lateral_axis,
        probe_axis_world=probe_axis,
        shaft_axis_world=shaft_axis,
        probe_origin_world=np.asarray(device_pose.probe_origin_world, dtype=np.float64),
        custom_origin_world=cutaway_custom_origin_world,
        station_visibility_points_world=station_visibility_points_world,
    )

    depth_grid, lateral_grid, sector_mask, max_lateral_mm = _build_sector_grid(
        resolved_width,
        resolved_height,
        resolved_max_depth_mm,
        resolved_sector_angle_deg,
    )

    source_hu, _, _, _ = _sample_contact_plane(
        data=np.asarray(render_context.ct_volume.data, dtype=np.float32),
        inverse_affine_lps=render_context.ct_volume.inverse_affine_lps,
        origin_world=contact_world,
        depth_axis=probe_axis,
        lateral_axis=shaft_axis,
        thickness_axis=lateral_axis,
        depth_max_mm=resolved_source_oblique_size_mm,
        lateral_half_width_mm=(resolved_source_oblique_size_mm / 2.0),
        width=resolved_width,
        height=resolved_height,
        slice_thickness_mm=resolved_slice_thickness_mm,
        order=1,
        cval=-1000.0,
    )
    source_section = SourceSection(
        hu=source_hu,
        preprocessed_hu=_preprocess_source_section(source_hu),
        forward_max_mm=resolved_source_oblique_size_mm,
        shaft_half_width_mm=(resolved_source_oblique_size_mm / 2.0),
        width=resolved_width,
        height=resolved_height,
    )

    fan_hu = _map_plane_to_fan(
        source_section.hu,
        source_forward_max_mm=source_section.forward_max_mm,
        source_shaft_half_width_mm=source_section.shaft_half_width_mm,
        depth_grid_mm=depth_grid,
        display_lateral_grid_mm=lateral_grid,
        sector_mask=sector_mask,
        order=1,
        cval=-1000.0,
    )
    fan_preprocessed_hu = _map_plane_to_fan(
        source_section.preprocessed_hu,
        source_forward_max_mm=source_section.forward_max_mm,
        source_shaft_half_width_mm=source_section.shaft_half_width_mm,
        depth_grid_mm=depth_grid,
        display_lateral_grid_mm=lateral_grid,
        sector_mask=sector_mask,
        order=1,
        cval=-1000.0,
    )
    base_windowed = _window_ct(
        fan_hu[sector_mask],
        gain=resolved_gain,
        attenuation=resolved_attenuation,
        depths_mm=depth_grid[sector_mask],
        max_depth_mm=resolved_max_depth_mm,
    )
    virtual_rgb = np.zeros((resolved_height, resolved_width, 3), dtype=np.float32)
    virtual_rgb[sector_mask] = np.repeat(base_windowed[:, None], 3, axis=1)
    _apply_contour_overlay(virtual_rgb, sector_mask, FAN_BOUNDARY_COLOR)

    visible_layers: list[OverlayLayer] = []
    sampled_fan_masks: dict[str, np.ndarray] = {}

    def _sample_source_mask_to_fan(mask_path: Path) -> np.ndarray:
        cache_key = str(mask_path)
        cached = sampled_fan_masks.get(cache_key)
        if cached is not None:
            return cached
        mask_volume = _get_mask_volume(render_context, mask_path)
        source_mask, _, _, _ = _sample_contact_plane(
            data=np.asarray(mask_volume.data, dtype=np.float32),
            inverse_affine_lps=mask_volume.inverse_affine_lps,
            origin_world=contact_world,
            depth_axis=probe_axis,
            lateral_axis=shaft_axis,
            thickness_axis=lateral_axis,
            depth_max_mm=resolved_source_oblique_size_mm,
            lateral_half_width_mm=(resolved_source_oblique_size_mm / 2.0),
            width=resolved_width,
            height=resolved_height,
            slice_thickness_mm=resolved_slice_thickness_mm,
            order=0,
            cval=0.0,
        )
        fan_mask = _map_plane_to_fan(
            source_mask,
            source_forward_max_mm=source_section.forward_max_mm,
            source_shaft_half_width_mm=source_section.shaft_half_width_mm,
            depth_grid_mm=depth_grid,
            display_lateral_grid_mm=lateral_grid,
            sector_mask=sector_mask,
            order=0,
            cval=0.0,
        ) > 0.0
        sampled_fan_masks[cache_key] = fan_mask
        return fan_mask

    def _add_source_mask_layer(mask_path: Path, label: str, color_rgb: np.ndarray, *, label_enabled: bool) -> None:
        fan_mask = _sample_source_mask_to_fan(mask_path)
        filtered_mask = _filter_mask_components(
            np.logical_and(fan_mask, sector_mask),
            min_area_px=overlay_config.min_contour_area_px,
            min_length_px=overlay_config.min_contour_length_px,
        )
        if not np.any(filtered_mask):
            return
        _apply_contour_overlay(virtual_rgb, filtered_mask, color_rgb)
        visible_layers.append(
            OverlayLayer(
                key=label.lower().replace(" ", "_"),
                label=label,
                color_rgb=color_rgb,
                mask=filtered_mask,
                label_enabled=label_enabled,
            )
        )

    fan_lumen_mask = np.logical_and(_sample_source_mask_to_fan(render_context.manifest.airway_lumen_mask), sector_mask)
    fan_wall_mask = np.logical_and(_sample_source_mask_to_fan(render_context.manifest.airway_solid_mask), sector_mask)
    fan_vessel_mask = np.zeros_like(sector_mask, dtype=bool)
    for mask_path in render_context.manifest.overlay_masks.values():
        fan_vessel_mask |= np.logical_and(_sample_source_mask_to_fan(mask_path), sector_mask)

    if overlay_config.airway_lumen_enabled:
        _add_source_mask_layer(render_context.manifest.airway_lumen_mask, "Airway lumen", AIRWAY_LUMEN_COLOR, label_enabled=False)
    if overlay_config.airway_wall_enabled:
        _add_source_mask_layer(render_context.manifest.airway_solid_mask, "Airway wall", AIRWAY_WALL_COLOR, label_enabled=False)
    if overlay_config.station_enabled:
        _add_source_mask_layer(preset_manifest.station_mask, f"Station {preset_manifest.station.upper()} ROI", STATION_COLOR, label_enabled=True)
    for index, vessel_name in enumerate(overlay_config.vessel_names):
        _add_source_mask_layer(
            render_context.manifest.overlay_masks[vessel_name],
            vessel_name.replace("_", " ").title(),
            VESSEL_OVERLAY_PALETTE[index % len(VESSEL_OVERLAY_PALETTE)],
            label_enabled=True,
        )

    target_offset = target_world - contact_world
    target_depth_mm = float(np.dot(target_offset, probe_axis))
    target_lateral_offset_mm = float(np.dot(target_offset, shaft_axis))
    target_pixel = _fan_target_row_col(
        contact_world=contact_world,
        target_world=target_world,
        probe_axis=probe_axis,
        shaft_axis=shaft_axis,
        max_depth_mm=resolved_max_depth_mm,
        sector_angle_deg=resolved_sector_angle_deg,
        width=resolved_width,
        height=resolved_height,
        max_lateral_mm=max_lateral_mm,
    )
    if overlay_config.target_enabled and target_pixel is not None:
        _draw_cross_marker(virtual_rgb, row=target_pixel[0], column=target_pixel[1], color_rgb=TARGET_MARKER_COLOR, radius=4)
    if overlay_config.contact_enabled:
        _draw_cross_marker(virtual_rgb, row=0, column=resolved_width // 2, color_rgb=CONTACT_MARKER_COLOR, radius=4)

    legend_entries = [(layer.label, layer.color_rgb) for layer in visible_layers]
    visible_overlay_names = [layer.key for layer in visible_layers]
    if overlay_config.target_enabled and target_pixel is not None:
        legend_entries.append(("Target", TARGET_MARKER_COLOR))
        visible_overlay_names.append("target")
    if overlay_config.contact_enabled:
        legend_entries.append(("Refined contact", CONTACT_MARKER_COLOR))
        visible_overlay_names.append("contact")
    virtual_view = _annotate_legend_and_labels(
        virtual_rgb,
        visible_layers=visible_layers,
        show_legend=overlay_config.show_legend,
        label_overlays=overlay_config.label_overlays,
        legend_entries=legend_entries,
    )

    simulated_depth_gradient = np.abs(np.gradient(fan_preprocessed_hu, axis=0))
    simulated_lateral_gradient = np.abs(np.gradient(fan_preprocessed_hu, axis=1))
    interface_strength = np.clip((0.70 * simulated_depth_gradient + 0.30 * simulated_lateral_gradient) / 220.0, 0.0, 1.0)
    simulated_windowed = np.zeros((resolved_height, resolved_width), dtype=np.float32)
    simulated_windowed[sector_mask] = _window_ct(
        fan_preprocessed_hu[sector_mask],
        gain=resolved_gain,
        attenuation=resolved_attenuation,
        depths_mm=depth_grid[sector_mask],
        max_depth_mm=resolved_max_depth_mm,
    )
    attenuation_curve = np.exp(-resolved_attenuation * (depth_grid / max(resolved_max_depth_mm, 1e-6)))
    simulated_gray = np.clip((0.58 * simulated_windowed) + (0.42 * interface_strength * attenuation_curve), 0.0, 1.0)
    air_dominant_mask = np.logical_and(fan_preprocessed_hu < -800.0, sector_mask)
    suppression_mask = np.logical_or(fan_lumen_mask, air_dominant_mask)
    if np.any(suppression_mask):
        preserved_signal = np.where(~suppression_mask, simulated_gray, 0.0)
        ray_signal_envelope = np.maximum.accumulate(preserved_signal, axis=0)
        ray_interface_envelope = np.maximum.accumulate(interface_strength, axis=0)
        nearest_tissue_fill = np.zeros_like(simulated_gray)
        valid_signal_mask = np.logical_and(sector_mask, ~suppression_mask)
        if np.any(valid_signal_mask):
            _, nearest_indices = ndimage.distance_transform_edt(
                np.where(sector_mask, suppression_mask, True),
                return_indices=True,
            )
            nearest_tissue_fill = simulated_gray[nearest_indices[0], nearest_indices[1]]
            nearest_tissue_fill[~sector_mask] = 0.0
        suppressed_fill = np.clip(
            (0.18 * attenuation_curve)
            + (0.34 * ray_interface_envelope)
            + (0.36 * ray_signal_envelope),
            0.08,
            0.54,
        )
        simulated_gray[suppression_mask] = np.clip(
            np.maximum(suppressed_fill[suppression_mask], 0.72 * nearest_tissue_fill[suppression_mask]),
            0.10,
            0.58,
        )
    simulated_rgb = np.zeros((resolved_height, resolved_width, 3), dtype=np.float32)
    simulated_rgb[sector_mask] = np.repeat(simulated_gray[sector_mask, None], 3, axis=1)
    _apply_contour_overlay(simulated_rgb, sector_mask, FAN_BOUNDARY_COLOR * 0.9)
    simulated_view = np.asarray((simulated_rgb * 255.0).clip(0.0, 255.0), dtype=np.uint8)
    consistency_metrics = compute_render_consistency_metrics(
        image_gray=simulated_gray,
        sector_mask=sector_mask,
        depth_grid_mm=depth_grid,
        lateral_grid_mm=lateral_grid,
        max_depth_mm=resolved_max_depth_mm,
        sector_angle_deg=resolved_sector_angle_deg,
        target_depth_mm=target_depth_mm,
        target_lateral_offset_mm=target_lateral_offset_mm,
        airway_wall_mask=fan_wall_mask,
        vessel_mask=fan_vessel_mask,
        normalization_method="ct_window_interface_blend",
        normalization_lower_bound=0.0,
        normalization_upper_bound=1.0,
        compression_gain_factor=resolved_gain,
    )

    localizer_x_min_mm = -10.0
    localizer_x_max_mm = max(20.0, resolved_reference_fov_mm - 10.0)
    localizer_y_min_mm = -(resolved_reference_fov_mm / 2.0)
    localizer_y_max_mm = resolved_reference_fov_mm / 2.0
    localizer_ct, _, _, _ = _sample_plane(
        render_context,
        x_axis=probe_axis,
        y_axis=shaft_axis,
        thickness_axis=lateral_axis,
        center_world=contact_world,
        x_min_mm=localizer_x_min_mm,
        x_max_mm=localizer_x_max_mm,
        y_min_mm=localizer_y_min_mm,
        y_max_mm=localizer_y_max_mm,
        width=resolved_width,
        height=resolved_height,
        slice_thickness_mm=resolved_slice_thickness_mm,
        order=1,
        cval=-1000.0,
        data=np.asarray(render_context.ct_volume.data, dtype=np.float32),
        inverse_affine_lps=render_context.ct_volume.inverse_affine_lps,
    )
    localizer_rgb = np.repeat(
        _window_ct(
            localizer_ct.reshape(-1),
            gain=1.0,
            attenuation=0.0,
            depths_mm=np.zeros(resolved_width * resolved_height, dtype=np.float32),
            max_depth_mm=max(resolved_reference_fov_mm, 1.0),
        ).reshape((resolved_height, resolved_width))[:, :, None],
        3,
        axis=2,
    )
    localizer_layers: list[OverlayLayer] = []

    def _add_localizer_mask(mask_path: Path, label: str, color_rgb: np.ndarray, *, label_enabled: bool) -> None:
        mask_volume = _get_mask_volume(render_context, mask_path)
        samples, _, _, _ = _sample_plane(
            render_context,
            x_axis=probe_axis,
            y_axis=shaft_axis,
            thickness_axis=lateral_axis,
            center_world=contact_world,
            x_min_mm=localizer_x_min_mm,
            x_max_mm=localizer_x_max_mm,
            y_min_mm=localizer_y_min_mm,
            y_max_mm=localizer_y_max_mm,
            width=resolved_width,
            height=resolved_height,
            slice_thickness_mm=resolved_slice_thickness_mm,
            order=0,
            cval=0.0,
            data=np.asarray(mask_volume.data, dtype=np.float32),
            inverse_affine_lps=mask_volume.inverse_affine_lps,
        )
        filtered_mask = _filter_mask_components(
            samples > 0.0,
            min_area_px=overlay_config.min_contour_area_px,
            min_length_px=overlay_config.min_contour_length_px,
        )
        if not np.any(filtered_mask):
            return
        _apply_contour_overlay(localizer_rgb, filtered_mask, color_rgb)
        localizer_layers.append(
            OverlayLayer(
                key=label.lower().replace(" ", "_"),
                label=label,
                color_rgb=color_rgb,
                mask=filtered_mask,
                label_enabled=label_enabled,
            )
        )

    if overlay_config.airway_lumen_enabled:
        _add_localizer_mask(render_context.manifest.airway_lumen_mask, "Airway lumen", AIRWAY_LUMEN_COLOR, label_enabled=False)
    if overlay_config.airway_wall_enabled:
        _add_localizer_mask(render_context.manifest.airway_solid_mask, "Airway wall", AIRWAY_WALL_COLOR, label_enabled=False)
    if overlay_config.station_enabled:
        _add_localizer_mask(preset_manifest.station_mask, f"Station {preset_manifest.station.upper()} ROI", STATION_COLOR, label_enabled=True)
    for index, vessel_name in enumerate(overlay_config.vessel_names):
        _add_localizer_mask(
            render_context.manifest.overlay_masks[vessel_name],
            vessel_name.replace("_", " ").title(),
            VESSEL_OVERLAY_PALETTE[index % len(VESSEL_OVERLAY_PALETTE)],
            label_enabled=True,
        )

    if overlay_config.contact_enabled:
        row, column = _plane_point_to_pixel(
            0.0,
            0.0,
            x_min_mm=localizer_x_min_mm,
            x_max_mm=localizer_x_max_mm,
            y_min_mm=localizer_y_min_mm,
            y_max_mm=localizer_y_max_mm,
            width=resolved_width,
            height=resolved_height,
        )
        _draw_cross_marker(localizer_rgb, row=row, column=column, color_rgb=CONTACT_MARKER_COLOR, radius=4)
    if overlay_config.target_enabled:
        row, column = _plane_point_to_pixel(
            target_depth_mm,
            target_lateral_offset_mm,
            x_min_mm=localizer_x_min_mm,
            x_max_mm=localizer_x_max_mm,
            y_min_mm=localizer_y_min_mm,
            y_max_mm=localizer_y_max_mm,
            width=resolved_width,
            height=resolved_height,
        )
        _draw_cross_marker(localizer_rgb, row=row, column=column, color_rgb=TARGET_MARKER_COLOR, radius=4)
    localizer_view = _annotate_legend_and_labels(
        localizer_rgb,
        visible_layers=localizer_layers,
        show_legend=False,
        label_overlays=overlay_config.label_overlays,
        legend_entries=[],
    )

    context_snapshot = _build_cp_context_snapshot(
        render_context,
        pose=pose,
        device_pose=device_pose,
        overlay_config=overlay_config,
        cutaway_display=cutaway_display,
        station_local=station_local,
        width=resolved_width,
        height=resolved_height,
        max_depth_mm=resolved_max_depth_mm,
    )

    if overlay_config.diagnostic_panel_enabled:
        output_image_uint8 = _compose_cp_diagnostic_panel(virtual_view, simulated_view, localizer_view, context_snapshot)
        panel_layout = ["virtual_ebus", "simulated_ebus", "wide_ct_localizer", "context_3d"]
        view_kind = "diagnostic_panel"
    else:
        if overlay_config.virtual_ebus_enabled and (_overlay_summary(overlay_config) or overlay_config.mode == "debug"):
            output_image_uint8 = virtual_view
            view_kind = "localizer_virtual"
        else:
            output_image_uint8 = simulated_view
            view_kind = "localizer_simulated"
        panel_layout = []

    output_path = Path(output_path).expanduser().resolve()
    metadata_path = output_path.with_suffix(".json") if metadata_path is None else Path(metadata_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(output_image_uint8, mode="RGB").save(output_path)

    warnings = list(pose.warnings) + list(device_pose.contact_refinement.warnings) + list(cutaway_display.warnings)
    optimization_applied = (
        optimization_result is not None
        and (
            abs(float(optimization_result.branch_shift_mm)) > OPTIMIZATION_EPSILON
            or abs(float(optimization_result.roll_offset_deg) - float(0.0 if preset_overrides is None or preset_overrides.roll_offset_deg is None else preset_overrides.roll_offset_deg)) > OPTIMIZATION_EPSILON
            or optimization_result.axis_sign_override != (None if preset_overrides is None else preset_overrides.axis_sign_override)
            or float(
                np.linalg.norm(
                    np.asarray(optimization_result.device_pose.contact_refinement.refined_contact_world, dtype=np.float64)
                    - np.asarray(baseline_device_pose.contact_refinement.refined_contact_world, dtype=np.float64)
                )
            )
            > OPTIMIZATION_EPSILON
        )
    )
    if optimization_applied and optimization_result is not None:
        warnings.append(
            "Flagged local pose optimization selected "
            f"branch_shift_mm={optimization_result.branch_shift_mm:.1f}, "
            f"roll_offset_deg={optimization_result.roll_offset_deg:.1f}, "
            f"axis_sign_override={optimization_result.axis_sign_override!r}."
        )
    if overlay_config.single_vessel_name is not None and not any(layer.label == overlay_config.single_vessel_name.replace("_", " ").title() for layer in visible_layers):
        warnings.append(f"Single-vessel mode requested '{overlay_config.single_vessel_name}' but no contour intersected the displayed fan.")

    metadata = RenderMetadata(
        manifest_path=str(manifest.manifest_path),
        case_id=manifest.case_id,
        preset_id=pose.preset_id,
        approach=pose.contact_approach,
        mode=overlay_config.mode,
        output_path=str(output_path),
        metadata_path=str(metadata_path),
        engine=RenderEngine.LOCALIZER.value,
        engine_version=LOCALIZER_ENGINE_VERSION,
        seed=seed,
        view_kind=view_kind,
        image_size=[int(output_image_uint8.shape[1]), int(output_image_uint8.shape[0])],
        device_model=device_pose.device_model.name,
        device_label=device_pose.device_model.shaft_label,
        sector_angle_deg=resolved_sector_angle_deg,
        max_depth_mm=resolved_max_depth_mm,
        roll_deg=resolved_roll_deg,
        gain=resolved_gain,
        attenuation=resolved_attenuation,
        slice_thickness_mm=resolved_slice_thickness_mm,
        slice_samples=DEFAULT_SLAB_SAMPLES,
        source_oblique_size_mm=resolved_source_oblique_size_mm,
        reference_fov_mm=resolved_reference_fov_mm,
        source_plane="nUS_nB_with_lateral_thickness",
        display_plane="nUS_nB_fan",
        reference_plane="nUS_nB_with_lateral_thickness",
        refine_contact_enabled=bool(refine_contact),
        diagnostic_panel_enabled=overlay_config.diagnostic_panel_enabled,
        diagnostic_panel_layout=panel_layout,
        virtual_ebus_enabled=overlay_config.virtual_ebus_enabled,
        simulated_ebus_enabled=overlay_config.simulated_ebus_enabled,
        contact_world=list(device_pose.contact_refinement.refined_contact_world),
        original_contact_world=list(device_pose.contact_refinement.original_contact_world),
        voxel_refined_contact_world=list(device_pose.contact_refinement.voxel_refined_contact_world),
        refined_contact_world=list(device_pose.contact_refinement.refined_contact_world),
        tip_start_world=list(device_pose.tip_start_world),
        target_world=list(device_pose.target_world),
        nearest_centerline_point=pose.nearest_centerline_point,
        pose_axes={
            "shaft_axis": (None if pose.shaft_axis is None else [float(value) for value in pose.shaft_axis]),
            "depth_axis": (None if pose.depth_axis is None else [float(value) for value in pose.depth_axis]),
            "lateral_axis": (None if pose.lateral_axis is None else [float(value) for value in pose.lateral_axis]),
        },
        device_axes={
            "nB": list(device_pose.shaft_axis_world),
            "nC": list(device_pose.video_axis_world),
            "nUS": list(device_pose.probe_axis_world),
            "wall_normal": list(device_pose.wall_normal_world),
        },
        target_in_default_forward_hemisphere=pose.target_in_default_forward_hemisphere,
        contact_to_airway_distance_mm=float(device_pose.contact_refinement.refined_contact_to_airway_distance_mm),
        original_contact_to_airway_distance_mm=float(device_pose.contact_refinement.original_contact_to_airway_distance_mm),
        voxel_refined_contact_to_airway_distance_mm=float(device_pose.contact_refinement.voxel_refined_contact_to_airway_distance_mm),
        refined_contact_to_airway_distance_mm=float(device_pose.contact_refinement.refined_contact_to_airway_distance_mm),
        centerline_projection_distance_mm=pose.centerline_projection_distance_mm,
        contact_refinement_method=device_pose.contact_refinement.refinement_method,
        pose_comparison={
            "markup_contact_world": list(device_pose.contact_refinement.original_contact_world),
            "voxel_refined_contact_world": list(device_pose.contact_refinement.voxel_refined_contact_world),
            "mesh_refined_contact_world": list(device_pose.contact_refinement.refined_contact_world),
            "voxel_to_mesh_contact_distance_mm": float(device_pose.contact_refinement.voxel_to_mesh_contact_distance_mm),
            "voxel_contact_to_airway_distance_mm": float(device_pose.contact_refinement.voxel_refined_contact_to_airway_distance_mm),
            "mesh_contact_to_airway_distance_mm": float(device_pose.contact_refinement.refined_contact_to_airway_distance_mm),
            "voxel_refinement_method": device_pose.contact_refinement.voxel_refinement_method,
            "mesh_refinement_method": device_pose.contact_refinement.mesh_refinement_method,
            "candidate_branch_graph_name": device_pose.contact_refinement.candidate_branch_graph_name,
            "candidate_branch_line_index": device_pose.contact_refinement.candidate_branch_line_index,
            "branch_hint": device_pose.contact_refinement.branch_hint,
            "branch_hint_applied": device_pose.contact_refinement.branch_hint_applied,
            "branch_hint_match": device_pose.contact_refinement.branch_hint_match,
            "voxel_wall_normal_world": list(device_pose.voxel_wall_normal_world),
            "mesh_wall_normal_world": list(device_pose.wall_normal_world),
            "voxel_nUS_world": list(device_pose.voxel_probe_axis_world),
            "mesh_nUS_world": list(device_pose.probe_axis_world),
            "nUS_angular_difference_deg": float(
                np.degrees(
                    np.arccos(
                        np.clip(
                            float(np.dot(device_pose.voxel_probe_axis_world, device_pose.probe_axis_world)),
                            -1.0,
                            1.0,
                        )
                    )
                )
            ),
            "final_nB_world": list(device_pose.shaft_axis_world),
            "final_nUS_world": list(device_pose.probe_axis_world),
            "final_nC_world": list(device_pose.video_axis_world),
            "optimized_branch_shift_mm": (None if optimization_result is None else float(optimization_result.branch_shift_mm)),
            "optimized_roll_offset_deg": (None if optimization_result is None else float(optimization_result.roll_offset_deg)),
            "optimized_axis_sign_override": (None if optimization_result is None else optimization_result.axis_sign_override),
            "optimized_objective": (None if optimization_result is None else list(optimization_result.objective)),
            "warnings": list(device_pose.contact_refinement.warnings),
        },
        airway_overlay_enabled=(overlay_config.airway_lumen_enabled or overlay_config.airway_wall_enabled),
        airway_lumen_overlay_enabled=overlay_config.airway_lumen_enabled,
        airway_wall_overlay_enabled=overlay_config.airway_wall_enabled,
        target_overlay_enabled=overlay_config.target_enabled,
        contact_overlay_enabled=overlay_config.contact_enabled,
        station_overlay_enabled=overlay_config.station_enabled,
        vessel_overlay_names=list(overlay_config.vessel_names),
        single_vessel_name=overlay_config.single_vessel_name,
        show_legend=overlay_config.show_legend,
        label_overlays=overlay_config.label_overlays,
        show_frustum=overlay_config.show_frustum,
        cutaway_mode=cutaway_display.mode,
        cutaway_side=cutaway_display.side,
        cutaway_side_source=cutaway_display.side_source,
        cutaway_open_side=cutaway_display.open_side,
        cutaway_depth_mm=cutaway_display.depth_mm,
        cutaway_origin_mode=cutaway_display.origin_mode,
        cutaway_origin=[float(value) for value in cutaway_display.origin_world.tolist()],
        cutaway_normal=[float(value) for value in cutaway_display.normal_world.tolist()],
        cutaway_mesh_source=cutaway_display.mesh_source,
        show_full_airway=cutaway_display.show_full_airway,
        overlays_enabled=_overlay_summary(overlay_config),
        visible_overlay_names=list(dict.fromkeys(visible_overlay_names)),
        consistency_metrics=consistency_metrics,
        preset_override_applied=(preset_overrides is not None),
        preset_override_vessel_overlays=([] if preset_overrides is None or preset_overrides.vessel_overlays is None else list(preset_overrides.vessel_overlays)),
        preset_override_cutaway_side=(None if preset_overrides is None else preset_overrides.cutaway_side),
        preset_override_roll_offset_deg=preset_roll_offset_deg,
        preset_override_branch_hint=(None if preset_overrides is None else preset_overrides.branch_hint),
        preset_override_branch_shift_mm=(None if preset_overrides is None else preset_overrides.branch_shift_mm),
        preset_override_axis_sign_override=(None if preset_overrides is None else preset_overrides.axis_sign_override),
        preset_override_reference_fov_mm=(None if preset_overrides is None else preset_overrides.reference_fov_mm),
        preset_override_notes=(None if preset_overrides is None else preset_overrides.notes),
        warnings=warnings,
        engine_diagnostics={},
    )
    metadata_path.write_text(json.dumps(asdict(metadata), indent=2))

    rendered = RenderedPreset(
        image_rgb=output_image_uint8,
        sector_mask=sector_mask,
        metadata=metadata,
    )
    return RenderResult(
        engine=RenderEngine.LOCALIZER,
        engine_version=LOCALIZER_ENGINE_VERSION,
        rendered_preset=rendered,
    )
