from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

from ebus_simulator.acoustic_properties import AcousticField, map_acoustic_properties
from ebus_simulator.artifacts import PhysicsArtifactConfig, apply_physics_artifacts
from ebus_simulator.annotations import (
    _annotate_legend_and_labels,
    _apply_contour_overlay,
    _draw_cross_marker,
    _filter_mask_components,
)
from ebus_simulator.eval import summarize_bmode_regions
from ebus_simulator.render_engines import RenderEngine, RenderRequest, RenderResult
from ebus_simulator.render_profiles import PhysicsTuningProfile, load_physics_profile, resolve_physics_artifact_settings
from ebus_simulator.render_state import (
    OverlayLayer,
    RenderContext,
    RenderMetadata,
    RenderedPreset,
    _get_mask_volume,
    _overlay_summary,
    prepare_physics_render_state,
)
from ebus_simulator.transforms import (
    DEFAULT_SLAB_SAMPLES,
    OPTIMIZATION_EPSILON,
    _build_sector_grid,
    _fan_target_row_col,
    _points_to_voxel,
    _sample_slab,
)
from ebus_simulator.rendering import (
    AIRWAY_LUMEN_COLOR,
    AIRWAY_WALL_COLOR,
    CONTACT_MARKER_COLOR,
    STATION_COLOR,
    TARGET_MARKER_COLOR,
    VESSEL_OVERLAY_PALETTE,
    classify_render_consistency_bucket,
    compute_render_consistency_metrics,
    _optimize_flagged_pose_locally,
)


PHYSICS_ENGINE_VERSION = "physics-v1"


def _resolve_artifact_config(
    request: RenderRequest,
    *,
    profile: PhysicsTuningProfile,
) -> tuple[PhysicsArtifactConfig, dict[str, float], dict[str, float]]:
    effective_settings, explicit_overrides = resolve_physics_artifact_settings(
        profile,
        speckle_strength=request.speckle_strength,
        reverberation_strength=request.reverberation_strength,
        shadow_strength=request.shadow_strength,
    )
    return (
        PhysicsArtifactConfig(
            speckle_strength=float(effective_settings["speckle_strength"]),
            reverberation_strength=float(effective_settings["reverberation_strength"]),
            shadow_strength=float(effective_settings["shadow_strength"]),
        ),
        effective_settings,
        explicit_overrides,
    )


def _normalize_compressed_bmode(
    compressed: np.ndarray,
    *,
    profile: PhysicsTuningProfile,
) -> tuple[np.ndarray, dict[str, float | str | None]]:
    positive = np.asarray(compressed, dtype=np.float32)
    positive = positive[positive > 0.0]

    reference_percentile = float(profile.normalization_reference_percentile)
    aux_percentile = float(profile.normalization_aux_percentile)
    reference_value = float(np.percentile(positive, reference_percentile)) if positive.size > 0 else 0.0
    aux_value = float(np.percentile(positive, aux_percentile)) if positive.size > 0 else None

    if reference_value <= 0.0:
        scale = 1.0
        method = f"log_percentile_{reference_percentile:.1f}"
    else:
        scale = float(reference_value)
        method = f"log_percentile_{reference_percentile:.1f}"
        if aux_value is not None and aux_value > 0.0:
            spike_ratio = float(reference_value / aux_value)
            if spike_ratio > float(profile.normalization_spike_ratio):
                scale = max(
                    float(aux_value),
                    float(
                        ((1.0 - float(profile.normalization_aux_blend_weight)) * reference_value)
                        + (float(profile.normalization_aux_blend_weight) * aux_value)
                    ),
                )
                method = f"log_percentile_{reference_percentile:.1f}_blended_{aux_percentile:.1f}"

    return (
        np.clip(np.asarray(compressed, dtype=np.float32) / float(scale), 0.0, 1.0).astype(np.float32),
        {
            "method": method,
            "reference_percentile": reference_percentile,
            "reference_value": float(reference_value),
            "aux_percentile": aux_percentile,
            "aux_value": (None if aux_value is None else float(aux_value)),
            "lower_bound": 0.0,
            "upper_bound": float(scale),
            "compression_gain_factor": float(profile.log_compression_gain_factor),
        },
    )


def _normalize_support_map(values: np.ndarray, *, sector_mask: np.ndarray, percentile: float = 99.0) -> np.ndarray:
    image = np.asarray(values, dtype=np.float32)
    mask = np.asarray(sector_mask, dtype=bool)
    masked = image[mask]
    if masked.size == 0:
        return np.zeros_like(image, dtype=np.float32)
    upper = float(np.percentile(masked, percentile))
    if upper <= 1e-9:
        return np.zeros_like(image, dtype=np.float32)
    normalized = np.clip(image / upper, 0.0, 1.0)
    normalized[~mask] = 0.0
    return normalized.astype(np.float32)


def _apply_sparse_signal_support(
    display_bmode: np.ndarray,
    *,
    profile: PhysicsTuningProfile,
    sector_mask: np.ndarray,
    depth_grid_mm: np.ndarray,
    max_depth_mm: float,
    preliminary_metrics: dict[str, object],
    target_mask: np.ndarray,
    wall_mask: np.ndarray,
    vessel_mask: np.ndarray,
    diagnostics: dict[str, np.ndarray],
) -> tuple[np.ndarray, dict[str, object], dict[str, np.ndarray]]:
    bucket = str(preliminary_metrics.get("consistency_bucket", classify_render_consistency_bucket(preliminary_metrics)))
    empty_sector_fraction = float(preliminary_metrics.get("empty_sector_fraction", 0.0) or 0.0)
    target_coverage = float(preliminary_metrics.get("target_sector_coverage_fraction", 0.0) or 0.0)

    support_mode = "none"
    support_active = False
    floor_base = 0.0
    floor_scale = 0.0
    anatomy_weight = 0.0
    target_weight = 0.0
    wall_moderation = 1.0

    if bucket == "sparse_empty_dominant" and empty_sector_fraction >= float(profile.sparse_support_empty_fraction):
        support_active = True
        support_mode = "sparse_target_support" if target_coverage > 0.0 else "sparse_anatomy_support"
        floor_base = float(profile.sparse_support_floor_base)
        floor_scale = float(profile.sparse_support_floor_scale)
        anatomy_weight = float(profile.sparse_support_anatomy_weight)
        target_weight = float(profile.sparse_support_target_weight) if target_coverage > 0.0 else 0.0
    elif bucket == "wall_dominant" and empty_sector_fraction >= float(profile.wall_guardrail_empty_fraction):
        support_active = True
        support_mode = "sparse_wall_guardrail"
        floor_base = float(profile.wall_guardrail_floor_base)
        floor_scale = float(profile.wall_guardrail_floor_scale)
        anatomy_weight = float(profile.wall_guardrail_anatomy_weight)
        target_weight = float(profile.wall_guardrail_target_weight) if target_coverage > 0.0 else 0.0
        wall_moderation = float(profile.wall_guardrail_moderation)

    support_debug = {
        "support_map": np.zeros_like(display_bmode, dtype=np.float32),
        "support_floor_map": np.zeros_like(display_bmode, dtype=np.float32),
        "target_scanline_support_map": np.zeros_like(display_bmode, dtype=np.float32),
    }
    support_details = {
        "pre_support_consistency_bucket": bucket,
        "support_logic_active": support_active,
        "support_logic_mode": support_mode,
        "support_floor_base": float(floor_base),
        "support_floor_scale": float(floor_scale),
        "support_anatomy_weight": float(anatomy_weight),
        "support_target_scanline_weight": float(target_weight),
        "support_wall_moderation_factor": float(wall_moderation),
        "support_candidate_fraction": 0.0,
        "pre_support_empty_sector_fraction": empty_sector_fraction,
        "pre_support_non_background_occupancy_fraction": float(preliminary_metrics.get("non_background_occupancy_fraction", 0.0) or 0.0),
    }
    if not support_active:
        return np.asarray(display_bmode, dtype=np.float32), support_details, support_debug

    boundary_map = _normalize_support_map(diagnostics.get("boundary_map", np.zeros_like(display_bmode, dtype=np.float32)), sector_mask=sector_mask)
    transmission_map = _normalize_support_map(diagnostics.get("transmission_map", np.zeros_like(display_bmode, dtype=np.float32)), sector_mask=sector_mask)
    target_focus_map = _normalize_support_map(diagnostics.get("target_focus_map", np.zeros_like(display_bmode, dtype=np.float32)), sector_mask=sector_mask)
    compressed_map = np.asarray(diagnostics.get("compressed_map", np.zeros_like(display_bmode, dtype=np.float32)), dtype=np.float32)

    target_core = ndimage.gaussian_filter(np.asarray(target_mask, dtype=np.float32), sigma=(2.4, 1.2))
    target_columns = np.max(target_core, axis=0, keepdims=True)
    target_scanline_map = np.clip(
        (0.65 * target_focus_map) + (0.35 * np.broadcast_to(target_columns, target_core.shape)),
        0.0,
        1.0,
    ).astype(np.float32)
    anatomy_support_map = np.clip(
        (0.52 * boundary_map)
        + (0.28 * np.asarray(wall_mask, dtype=np.float32))
        + (0.18 * np.asarray(vessel_mask, dtype=np.float32))
        + (0.36 * target_focus_map),
        0.0,
        1.0,
    ).astype(np.float32)
    transmission_support = np.sqrt(np.clip(transmission_map, 0.0, 1.0)).astype(np.float32)
    sampled_anatomy_mask = np.asarray(
        sector_mask
        & (
            (compressed_map > (float(profile.sparse_signal_threshold) * 1.8))
            | np.asarray(wall_mask, dtype=bool)
            | np.asarray(vessel_mask, dtype=bool)
            | np.asarray(target_mask, dtype=bool)
            | (boundary_map > 0.18)
        ),
        dtype=bool,
    )
    sampled_anatomy_mask &= anatomy_support_map > 0.22
    support_details["support_candidate_fraction"] = float(
        np.count_nonzero(sampled_anatomy_mask) / max(1, np.count_nonzero(np.asarray(sector_mask, dtype=bool)))
    )

    support_floor_map = np.clip(
        float(floor_base) + (float(floor_scale) * anatomy_support_map * transmission_support),
        0.0,
        1.0,
    ).astype(np.float32)
    support_lift_map = (
        float(anatomy_weight) * anatomy_support_map * transmission_support
    ).astype(np.float32)
    if target_weight > 0.0:
        support_lift_map += (float(target_weight) * target_scanline_map).astype(np.float32)

    updated = np.asarray(display_bmode, dtype=np.float32).copy()
    low_signal_mask = sampled_anatomy_mask & (updated < 0.08)
    updated[low_signal_mask] = np.maximum(updated[low_signal_mask], support_floor_map[low_signal_mask])
    lift_gate = np.clip((0.14 - updated) / 0.14, 0.0, 1.0).astype(np.float32)
    gated_lift_map = support_lift_map * lift_gate
    updated[sampled_anatomy_mask] = np.clip(
        updated[sampled_anatomy_mask] + gated_lift_map[sampled_anatomy_mask],
        0.0,
        1.0,
    )

    if wall_moderation < 1.0:
        near_field_mask = np.asarray(depth_grid_mm, dtype=np.float32) <= (float(max_depth_mm) * 0.20)
        wall_guardrail_mask = np.asarray(sector_mask, dtype=bool) & near_field_mask & np.asarray(wall_mask, dtype=bool) & (updated > 0.14)
        updated[wall_guardrail_mask] *= float(wall_moderation)

    support_debug["support_map"] = gated_lift_map.astype(np.float32)
    support_debug["support_floor_map"] = np.where(sampled_anatomy_mask, support_floor_map, 0.0).astype(np.float32)
    support_debug["target_scanline_support_map"] = target_scanline_map.astype(np.float32)
    return np.clip(updated, 0.0, 1.0).astype(np.float32), support_details, support_debug


def _simulate_bmode_with_diagnostics(
    field: AcousticField,
    *,
    profile: PhysicsTuningProfile,
    depth_step_mm: float,
    gain: float,
    attenuation_scale: float,
    seed: int | None,
    artifact_config: PhysicsArtifactConfig,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, float | str | None]]:
    if depth_step_mm <= 0.0:
        raise ValueError(f"depth_step_mm must be positive, got {depth_step_mm!r}.")

    rng = np.random.default_rng(0 if seed is None else seed)

    impedance = np.asarray(field.impedance, dtype=np.float32)
    scatter = np.asarray(field.scatter, dtype=np.float32)
    attenuation = np.asarray(field.attenuation, dtype=np.float32)
    lumen = np.asarray(field.airway_lumen_mask, dtype=np.float32)
    vessel = np.asarray(field.vessel_mask, dtype=np.float32)
    target_focus = np.asarray(field.target_focus, dtype=np.float32)

    boundary = np.abs(np.diff(impedance, axis=0, prepend=impedance[[0], :]))
    boundary = np.maximum(boundary, np.asarray(field.airway_wall_mask, dtype=np.float32) * 0.18)
    air_interface_map = boundary * np.maximum(lumen, np.pad(lumen[1:, :], ((0, 1), (0, 0)), mode="constant"))

    boundary_texture = (0.95 + (0.35 * rng.random(boundary.shape))).astype(np.float32)
    scatter_component = scatter * 0.72
    boundary_component = boundary * boundary_texture

    depth_step_cm = float(depth_step_mm) / 10.0
    cumulative_attenuation = np.cumsum(attenuation * float(attenuation_scale), axis=0) * depth_step_cm
    transmission = np.exp(-cumulative_attenuation).astype(np.float32)
    vessel_suppression = (1.0 - (0.18 * vessel)).astype(np.float32)

    base_signal = ((0.65 * scatter_component) + (1.45 * boundary_component)) * transmission * vessel_suppression
    if np.any(target_focus > 0.0):
        base_signal *= (1.0 - (0.12 * target_focus))

    raw_with_artifacts, artifact_maps = apply_physics_artifacts(
        base_signal,
        air_interface_map=air_interface_map,
        airway_lumen_mask=field.airway_lumen_mask,
        vessel_mask=field.vessel_mask,
        depth_step_mm=depth_step_mm,
        config=artifact_config,
        rng=rng,
    )
    smoothed = ndimage.gaussian_filter(np.asarray(raw_with_artifacts, dtype=np.float32), sigma=(0.85, 0.45))
    smoothed *= np.linspace(0.92, 1.10, smoothed.shape[0], dtype=np.float32)[:, None]

    compressed = np.log1p(np.clip(smoothed, 0.0, None) * (float(profile.log_compression_gain_factor) * float(gain)))
    normalized, normalization_details = _normalize_compressed_bmode(compressed, profile=profile)

    diagnostics = {
        "boundary_map": boundary.astype(np.float32),
        "transmission_map": transmission.astype(np.float32),
        "shadow_map": artifact_maps.shadow_map.astype(np.float32),
        "reverberation_map": artifact_maps.reverberation_map.astype(np.float32),
        "speckle_map": artifact_maps.speckle_map.astype(np.float32),
        "precompression_map": smoothed.astype(np.float32),
        "compressed_map": compressed.astype(np.float32),
    }
    return normalized, diagnostics, normalization_details


def simulate_bmode_from_acoustic_field(
    field: AcousticField,
    *,
    depth_step_mm: float,
    gain: float,
    attenuation_scale: float,
    seed: int | None,
    physics_profile: str | None = None,
    speckle_strength: float | None = None,
    reverberation_strength: float | None = None,
    shadow_strength: float | None = None,
) -> np.ndarray:
    loaded_profile = load_physics_profile(physics_profile)
    effective_artifact_settings, _ = resolve_physics_artifact_settings(
        loaded_profile.settings,
        speckle_strength=speckle_strength,
        reverberation_strength=reverberation_strength,
        shadow_strength=shadow_strength,
    )
    artifact_config = PhysicsArtifactConfig(
        speckle_strength=float(effective_artifact_settings["speckle_strength"]),
        reverberation_strength=float(effective_artifact_settings["reverberation_strength"]),
        shadow_strength=float(effective_artifact_settings["shadow_strength"]),
    )
    image, _, _ = _simulate_bmode_with_diagnostics(
        field,
        profile=loaded_profile.settings,
        depth_step_mm=depth_step_mm,
        gain=gain,
        attenuation_scale=attenuation_scale,
        seed=seed,
        artifact_config=artifact_config,
    )
    return image


def _build_polar_grid(height: int, width: int, *, max_depth_mm: float, sector_angle_deg: float) -> tuple[np.ndarray, np.ndarray]:
    depths_mm = np.linspace(0.0, max_depth_mm, height, dtype=np.float64)
    half_angle_rad = np.deg2rad(sector_angle_deg / 2.0)
    angles_rad = np.linspace(-half_angle_rad, half_angle_rad, width, dtype=np.float64)
    depth_grid = np.broadcast_to(depths_mm[:, None], (height, width))
    angle_grid = np.broadcast_to(angles_rad[None, :], (height, width))
    return depth_grid, angle_grid


def _scan_convert_polar_to_sector(
    polar_values: np.ndarray,
    *,
    depth_grid_mm: np.ndarray,
    lateral_grid_mm: np.ndarray,
    sector_mask: np.ndarray,
    max_depth_mm: float,
    sector_angle_deg: float,
    order: int,
    cval: float,
) -> np.ndarray:
    mapped = np.full(depth_grid_mm.shape, cval, dtype=np.float32)
    if not np.any(sector_mask):
        return mapped

    half_angle_rad = np.deg2rad(sector_angle_deg / 2.0)
    phi = np.arctan2(lateral_grid_mm, np.maximum(depth_grid_mm, 1e-9))
    source_rows = (depth_grid_mm[sector_mask] / max(max_depth_mm, 1e-6)) * max(1, polar_values.shape[0] - 1)
    source_cols = ((phi[sector_mask] + half_angle_rad) / max(half_angle_rad * 2.0, 1e-6)) * max(1, polar_values.shape[1] - 1)
    sampled = ndimage.map_coordinates(
        np.asarray(polar_values, dtype=np.float32),
        [source_rows, source_cols],
        order=order,
        mode="constant",
        cval=cval,
    )
    mapped[sector_mask] = sampled
    return mapped


def _sample_mask_ray_domain(
    context: RenderContext,
    *,
    mask_path: Path,
    ray_points_world: np.ndarray,
    thickness_axis: np.ndarray,
    slice_thickness_mm: float,
    slab_reduce: str = "mean",
) -> np.ndarray:
    mask_volume = _get_mask_volume(context, mask_path)
    if slab_reduce == "mean":
        samples = _sample_slab(
            np.asarray(mask_volume.data, dtype=np.float32),
            base_points_lps=ray_points_world.reshape((-1, 3)),
            thickness_axis=thickness_axis,
            inverse_affine_lps=mask_volume.inverse_affine_lps,
            sample_count=DEFAULT_SLAB_SAMPLES,
            slab_thickness_mm=slice_thickness_mm,
            order=0,
            cval=0.0,
        )
        return samples.reshape(ray_points_world.shape[:2]) > 0.5
    if slab_reduce != "max":
        raise ValueError(f"Unsupported slab_reduce {slab_reduce!r}.")

    if DEFAULT_SLAB_SAMPLES <= 1 or slice_thickness_mm <= 0.0:
        offsets = np.asarray([0.0], dtype=np.float64)
    else:
        offsets = np.linspace(-slice_thickness_mm / 2.0, slice_thickness_mm / 2.0, DEFAULT_SLAB_SAMPLES, dtype=np.float64)

    base_points = ray_points_world.reshape((-1, 3))
    stacked_points = base_points[None, :, :] + offsets[:, None, None] * thickness_axis[None, None, :]
    voxel_points = _points_to_voxel(stacked_points.reshape((-1, 3)), mask_volume.inverse_affine_lps)
    sampled = ndimage.map_coordinates(
        np.asarray(mask_volume.data, dtype=np.float32),
        [voxel_points[:, 0], voxel_points[:, 1], voxel_points[:, 2]],
        order=0,
        mode="constant",
        cval=0.0,
    )
    occupancy = sampled.reshape((offsets.shape[0], base_points.shape[0])).max(axis=0)
    return occupancy.reshape(ray_points_world.shape[:2]) > 0.0


def _combine_vessel_masks(
    context: RenderContext,
    *,
    ray_points_world: np.ndarray,
    thickness_axis: np.ndarray,
    slice_thickness_mm: float,
) -> np.ndarray:
    combined = np.zeros(ray_points_world.shape[:2], dtype=bool)
    for mask_path in context.manifest.overlay_masks.values():
        combined |= _sample_mask_ray_domain(
            context,
            mask_path=mask_path,
            ray_points_world=ray_points_world,
            thickness_axis=thickness_axis,
            slice_thickness_mm=slice_thickness_mm,
        )
    return combined


def _scan_convert_display_mask(
    mask: np.ndarray,
    *,
    depth_grid_mm: np.ndarray,
    lateral_grid_mm: np.ndarray,
    sector_mask: np.ndarray,
    max_depth_mm: float,
    sector_angle_deg: float,
    min_area_px: float,
    min_length_px: float,
) -> np.ndarray:
    display_mask = _scan_convert_binary_mask(
        mask,
        depth_grid_mm=depth_grid_mm,
        lateral_grid_mm=lateral_grid_mm,
        sector_mask=sector_mask,
        max_depth_mm=max_depth_mm,
        sector_angle_deg=sector_angle_deg,
    )
    return _filter_mask_components(
        display_mask,
        min_area_px=min_area_px,
        min_length_px=min_length_px,
    )


def _scan_convert_binary_mask(
    mask: np.ndarray,
    *,
    depth_grid_mm: np.ndarray,
    lateral_grid_mm: np.ndarray,
    sector_mask: np.ndarray,
    max_depth_mm: float,
    sector_angle_deg: float,
) -> np.ndarray:
    display_mask = _scan_convert_polar_to_sector(
        mask.astype(np.float32),
        depth_grid_mm=depth_grid_mm,
        lateral_grid_mm=lateral_grid_mm,
        sector_mask=sector_mask,
        max_depth_mm=max_depth_mm,
        sector_angle_deg=sector_angle_deg,
        order=0,
        cval=0.0,
    ) > 0.5
    display_mask &= sector_mask
    return display_mask


def _resolve_eval_wall_mask(
    *,
    lumen_mask: np.ndarray,
    wall_mask: np.ndarray,
    sector_mask: np.ndarray,
) -> np.ndarray:
    if np.any(wall_mask):
        return wall_mask
    if not np.any(lumen_mask):
        return wall_mask

    shell = ndimage.binary_dilation(lumen_mask, structure=np.ones((3, 3), dtype=bool), iterations=1)
    return shell & ~lumen_mask & sector_mask


def _write_debug_map_png(values: np.ndarray, path: Path) -> None:
    image = np.asarray(values, dtype=np.float32)
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        normalized = np.zeros_like(image, dtype=np.float32)
    else:
        lower = float(np.min(finite))
        upper = float(np.percentile(finite, 99.5))
        if upper <= lower:
            normalized = np.clip(image, 0.0, 1.0)
        else:
            normalized = np.clip((image - lower) / (upper - lower), 0.0, 1.0)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((normalized * 255.0).astype(np.uint8), mode="L").save(path)


def render_physics_preset(
    request: RenderRequest,
    *,
    context: RenderContext | None = None,
) -> RenderResult:
    if request.engine is not RenderEngine.PHYSICS:
        raise ValueError(f"render_physics_preset expected engine=physics, got {request.engine.value!r}.")

    prepared_state = prepare_physics_render_state(
        request,
        context=context,
        optimize_pose_fn=_optimize_flagged_pose_locally,
    )

    render_context = prepared_state.context
    manifest = prepared_state.manifest
    pose = prepared_state.pose
    preset_manifest = prepared_state.preset_manifest
    preset_overrides = prepared_state.preset_overrides
    overlay_config = prepared_state.overlay_config
    cutaway_config = prepared_state.cutaway_config
    baseline_device_pose = prepared_state.baseline_device_pose
    device_pose = prepared_state.device_pose
    optimization_result = prepared_state.optimization_result
    resolved_width = prepared_state.resolved_width
    resolved_height = prepared_state.resolved_height
    resolved_sector_angle_deg = prepared_state.resolved_sector_angle_deg
    resolved_max_depth_mm = prepared_state.resolved_max_depth_mm
    resolved_source_oblique_size_mm = prepared_state.resolved_source_oblique_size_mm
    resolved_reference_fov_mm = prepared_state.resolved_reference_fov_mm
    resolved_roll_deg = prepared_state.resolved_roll_deg
    resolved_gain = prepared_state.resolved_gain
    resolved_attenuation = prepared_state.resolved_attenuation
    resolved_slice_thickness_mm = prepared_state.resolved_slice_thickness_mm
    preset_roll_offset_deg = prepared_state.preset_roll_offset_deg
    configured_axis_sign_override = prepared_state.configured_axis_sign_override
    configured_branch_hint = prepared_state.configured_branch_hint
    configured_branch_shift_mm = prepared_state.configured_branch_shift_mm
    contact_world = prepared_state.contact_world
    target_world = prepared_state.target_world
    probe_axis = prepared_state.probe_axis
    shaft_axis = prepared_state.shaft_axis
    lateral_axis = prepared_state.lateral_axis
    loaded_profile = load_physics_profile(request.physics_profile)
    artifact_config, effective_artifact_settings, explicit_profile_overrides = _resolve_artifact_config(
        request,
        profile=loaded_profile.settings,
    )
    effective_profile_settings = asdict(loaded_profile.settings)
    effective_profile_settings.update(
        {
            "speckle_strength_default": float(artifact_config.speckle_strength),
            "reverberation_strength_default": float(artifact_config.reverberation_strength),
            "shadow_strength_default": float(artifact_config.shadow_strength),
        }
    )

    ray_depth_grid_mm, ray_angle_grid_rad = _build_polar_grid(
        resolved_height,
        resolved_width,
        max_depth_mm=resolved_max_depth_mm,
        sector_angle_deg=resolved_sector_angle_deg,
    )
    ray_directions = (
        np.cos(ray_angle_grid_rad)[:, :, None] * probe_axis[None, None, :]
        + np.sin(ray_angle_grid_rad)[:, :, None] * shaft_axis[None, None, :]
    )
    ray_points_world = contact_world[None, None, :] + ray_depth_grid_mm[:, :, None] * ray_directions

    ct_ray = _sample_slab(
        np.asarray(render_context.ct_volume.data, dtype=np.float32),
        base_points_lps=ray_points_world.reshape((-1, 3)),
        thickness_axis=lateral_axis,
        inverse_affine_lps=render_context.ct_volume.inverse_affine_lps,
        sample_count=DEFAULT_SLAB_SAMPLES,
        slab_thickness_mm=resolved_slice_thickness_mm,
        order=1,
        cval=-1000.0,
    ).reshape((resolved_height, resolved_width))
    lumen_ray = _sample_mask_ray_domain(
        render_context,
        mask_path=manifest.airway_lumen_mask,
        ray_points_world=ray_points_world,
        thickness_axis=lateral_axis,
        slice_thickness_mm=resolved_slice_thickness_mm,
    )
    wall_ray = _sample_mask_ray_domain(
        render_context,
        mask_path=manifest.airway_solid_mask,
        ray_points_world=ray_points_world,
        thickness_axis=lateral_axis,
        slice_thickness_mm=resolved_slice_thickness_mm,
        slab_reduce="max",
    )
    wall_ray &= ~lumen_ray
    station_ray = _sample_mask_ray_domain(
        render_context,
        mask_path=preset_manifest.station_mask,
        ray_points_world=ray_points_world,
        thickness_axis=lateral_axis,
        slice_thickness_mm=resolved_slice_thickness_mm,
    )
    all_vessels_ray = _combine_vessel_masks(
        render_context,
        ray_points_world=ray_points_world,
        thickness_axis=lateral_axis,
        slice_thickness_mm=resolved_slice_thickness_mm,
    )

    target_distance_mm = np.linalg.norm(ray_points_world - target_world[None, None, :], axis=2)
    target_focus = np.exp(
        -0.5 * ((target_distance_mm / float(loaded_profile.settings.target_focus_sigma_mm)) ** 2)
    ).astype(np.float32)
    if np.any(station_ray):
        target_focus *= np.where(station_ray, 1.0, 0.35).astype(np.float32)

    acoustic_field = map_acoustic_properties(
        ct_hu=ct_ray,
        airway_lumen_mask=lumen_ray,
        airway_wall_mask=wall_ray,
        vessel_mask=all_vessels_ray,
        station_mask=station_ray,
        target_focus=target_focus,
    )
    depth_step_mm = float(resolved_max_depth_mm / max(1, resolved_height - 1))
    polar_bmode, polar_diagnostics, normalization_details = _simulate_bmode_with_diagnostics(
        acoustic_field,
        profile=loaded_profile.settings,
        depth_step_mm=depth_step_mm,
        gain=resolved_gain,
        attenuation_scale=(1.0 + (resolved_attenuation * 2.0)),
        seed=request.seed,
        artifact_config=artifact_config,
    )
    polar_diagnostics["target_focus_map"] = target_focus.astype(np.float32)

    display_depth_grid_mm, display_lateral_grid_mm, sector_mask, max_lateral_mm = _build_sector_grid(
        resolved_width,
        resolved_height,
        resolved_max_depth_mm,
        resolved_sector_angle_deg,
    )
    display_bmode = _scan_convert_polar_to_sector(
        polar_bmode,
        depth_grid_mm=display_depth_grid_mm,
        lateral_grid_mm=display_lateral_grid_mm,
        sector_mask=sector_mask,
        max_depth_mm=resolved_max_depth_mm,
        sector_angle_deg=resolved_sector_angle_deg,
        order=1,
        cval=0.0,
    )
    display_diagnostics = {
        name: _scan_convert_polar_to_sector(
            values,
            depth_grid_mm=display_depth_grid_mm,
            lateral_grid_mm=display_lateral_grid_mm,
            sector_mask=sector_mask,
            max_depth_mm=resolved_max_depth_mm,
            sector_angle_deg=resolved_sector_angle_deg,
            order=1,
            cval=0.0,
        )
        for name, values in polar_diagnostics.items()
    }
    eval_lumen_mask = _scan_convert_binary_mask(
        lumen_ray,
        depth_grid_mm=display_depth_grid_mm,
        lateral_grid_mm=display_lateral_grid_mm,
        sector_mask=sector_mask,
        max_depth_mm=resolved_max_depth_mm,
        sector_angle_deg=resolved_sector_angle_deg,
    )
    display_lumen_mask = _filter_mask_components(
        eval_lumen_mask,
        min_area_px=overlay_config.min_contour_area_px,
        min_length_px=overlay_config.min_contour_length_px,
    )
    eval_wall_mask = _scan_convert_binary_mask(
        wall_ray,
        depth_grid_mm=display_depth_grid_mm,
        lateral_grid_mm=display_lateral_grid_mm,
        sector_mask=sector_mask,
        max_depth_mm=resolved_max_depth_mm,
        sector_angle_deg=resolved_sector_angle_deg,
    )
    eval_wall_mask = _resolve_eval_wall_mask(
        lumen_mask=eval_lumen_mask,
        wall_mask=eval_wall_mask,
        sector_mask=sector_mask,
    )
    display_wall_mask = _filter_mask_components(
        eval_wall_mask,
        min_area_px=overlay_config.min_contour_area_px,
        min_length_px=overlay_config.min_contour_length_px,
    )
    display_station_mask = _scan_convert_display_mask(
        station_ray,
        depth_grid_mm=display_depth_grid_mm,
        lateral_grid_mm=display_lateral_grid_mm,
        sector_mask=sector_mask,
        max_depth_mm=resolved_max_depth_mm,
        sector_angle_deg=resolved_sector_angle_deg,
        min_area_px=overlay_config.min_contour_area_px,
        min_length_px=overlay_config.min_contour_length_px,
    )
    eval_vessel_mask = _scan_convert_binary_mask(
        all_vessels_ray,
        depth_grid_mm=display_depth_grid_mm,
        lateral_grid_mm=display_lateral_grid_mm,
        sector_mask=sector_mask,
        max_depth_mm=resolved_max_depth_mm,
        sector_angle_deg=resolved_sector_angle_deg,
    )
    display_vessel_mask = _filter_mask_components(
        eval_vessel_mask,
        min_area_px=overlay_config.min_contour_area_px,
        min_length_px=overlay_config.min_contour_length_px,
    )
    eval_target_region_mask = _scan_convert_binary_mask(
        target_focus > 0.35,
        depth_grid_mm=display_depth_grid_mm,
        lateral_grid_mm=display_lateral_grid_mm,
        sector_mask=sector_mask,
        max_depth_mm=resolved_max_depth_mm,
        sector_angle_deg=resolved_sector_angle_deg,
    )
    display_target_region_mask = _filter_mask_components(
        eval_target_region_mask,
        min_area_px=overlay_config.min_contour_area_px,
        min_length_px=overlay_config.min_contour_length_px,
    )
    target_offset = target_world - contact_world
    target_depth_mm = float(np.dot(target_offset, probe_axis))
    target_lateral_offset_mm = float(np.dot(target_offset, shaft_axis))
    preliminary_consistency_metrics = compute_render_consistency_metrics(
        image_gray=display_bmode,
        sector_mask=sector_mask,
        depth_grid_mm=display_depth_grid_mm,
        lateral_grid_mm=display_lateral_grid_mm,
        max_depth_mm=resolved_max_depth_mm,
        sector_angle_deg=resolved_sector_angle_deg,
        target_depth_mm=target_depth_mm,
        target_lateral_offset_mm=target_lateral_offset_mm,
        target_region_mask=eval_target_region_mask,
        airway_wall_mask=eval_wall_mask,
        vessel_mask=eval_vessel_mask,
        normalization_method=str(normalization_details["method"]),
        normalization_reference_percentile=(
            None
            if normalization_details["reference_percentile"] is None
            else float(normalization_details["reference_percentile"])
        ),
        normalization_reference_value=(
            None
            if normalization_details["reference_value"] is None
            else float(normalization_details["reference_value"])
        ),
        normalization_aux_percentile=(
            None
            if normalization_details["aux_percentile"] is None
            else float(normalization_details["aux_percentile"])
        ),
        normalization_aux_value=(
            None
            if normalization_details["aux_value"] is None
            else float(normalization_details["aux_value"])
        ),
        normalization_lower_bound=(
            None
            if normalization_details["lower_bound"] is None
            else float(normalization_details["lower_bound"])
        ),
        normalization_upper_bound=(
            None
            if normalization_details["upper_bound"] is None
            else float(normalization_details["upper_bound"])
        ),
        compression_gain_factor=(
            None
            if normalization_details["compression_gain_factor"] is None
            else float(normalization_details["compression_gain_factor"])
        ),
    )
    display_bmode, support_details, support_debug_maps = _apply_sparse_signal_support(
        display_bmode,
        profile=loaded_profile.settings,
        sector_mask=sector_mask,
        depth_grid_mm=display_depth_grid_mm,
        max_depth_mm=resolved_max_depth_mm,
        preliminary_metrics=preliminary_consistency_metrics,
        target_mask=eval_target_region_mask,
        wall_mask=eval_wall_mask,
        vessel_mask=eval_vessel_mask,
        diagnostics=display_diagnostics,
    )
    display_diagnostics.update(support_debug_maps)

    image_rgb = np.zeros((resolved_height, resolved_width, 3), dtype=np.float32)
    image_rgb[sector_mask] = np.repeat(display_bmode[sector_mask, None], 3, axis=1)

    visible_layers: list[OverlayLayer] = []
    legend_entries: list[tuple[str, np.ndarray]] = []

    def _overlay_display_mask(mask: np.ndarray, *, key: str, label: str, color_rgb: np.ndarray, enabled: bool) -> None:
        if not enabled:
            return
        if not np.any(mask):
            return
        _apply_contour_overlay(image_rgb, mask, color_rgb)
        visible_layers.append(OverlayLayer(key=key, label=label, color_rgb=color_rgb, mask=mask))
        legend_entries.append((label, color_rgb))

    _overlay_display_mask(
        display_lumen_mask,
        key="airway_lumen",
        label="Airway lumen",
        color_rgb=AIRWAY_LUMEN_COLOR,
        enabled=overlay_config.airway_lumen_enabled,
    )
    _overlay_display_mask(
        display_wall_mask,
        key="airway_wall",
        label="Airway wall",
        color_rgb=AIRWAY_WALL_COLOR,
        enabled=overlay_config.airway_wall_enabled,
    )
    _overlay_display_mask(
        display_station_mask,
        key="station",
        label=f"Station {pose.station.upper()}",
        color_rgb=STATION_COLOR,
        enabled=overlay_config.station_enabled,
    )
    for index, vessel_name in enumerate(overlay_config.vessel_names):
        vessel_mask = _sample_mask_ray_domain(
            render_context,
            mask_path=manifest.overlay_masks[vessel_name],
            ray_points_world=ray_points_world,
            thickness_axis=lateral_axis,
            slice_thickness_mm=resolved_slice_thickness_mm,
        )
        display_selected_vessel_mask = _scan_convert_display_mask(
            vessel_mask,
            depth_grid_mm=display_depth_grid_mm,
            lateral_grid_mm=display_lateral_grid_mm,
            sector_mask=sector_mask,
            max_depth_mm=resolved_max_depth_mm,
            sector_angle_deg=resolved_sector_angle_deg,
            min_area_px=overlay_config.min_contour_area_px,
            min_length_px=overlay_config.min_contour_length_px,
        )
        _overlay_display_mask(
            display_selected_vessel_mask,
            key=vessel_name,
            label=vessel_name.replace("_", " ").title(),
            color_rgb=VESSEL_OVERLAY_PALETTE[index % len(VESSEL_OVERLAY_PALETTE)],
            enabled=True,
        )

    if overlay_config.target_enabled:
        target_row_col = _fan_target_row_col(
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
        if target_row_col is not None:
            display_target_region_mask[target_row_col[0], target_row_col[1]] = True
            _draw_cross_marker(image_rgb, row=target_row_col[0], column=target_row_col[1], color_rgb=TARGET_MARKER_COLOR, radius=4)
            target_mask = np.zeros((resolved_height, resolved_width), dtype=bool)
            target_mask[target_row_col[0], target_row_col[1]] = True
            visible_layers.append(OverlayLayer(key="target", label="Target", color_rgb=TARGET_MARKER_COLOR, mask=target_mask, label_enabled=False))
            legend_entries.append(("Target", TARGET_MARKER_COLOR))

    if overlay_config.contact_enabled:
        _draw_cross_marker(image_rgb, row=0, column=resolved_width // 2, color_rgb=CONTACT_MARKER_COLOR, radius=4)
        contact_mask = np.zeros((resolved_height, resolved_width), dtype=bool)
        contact_mask[0, resolved_width // 2] = True
        visible_layers.append(OverlayLayer(key="contact", label="Contact", color_rgb=CONTACT_MARKER_COLOR, mask=contact_mask, label_enabled=False))
        legend_entries.append(("Contact", CONTACT_MARKER_COLOR))

    output_path = Path(request.output_path).expanduser().resolve()
    metadata_path = output_path.with_suffix(".json") if request.metadata_path is None else Path(request.metadata_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    output_image_uint8 = _annotate_legend_and_labels(
        image_rgb,
        visible_layers=visible_layers,
        show_legend=overlay_config.show_legend,
        label_overlays=overlay_config.label_overlays,
        legend_entries=legend_entries,
    )
    Image.fromarray(output_image_uint8, mode="RGB").save(output_path)

    debug_map_paths: dict[str, str] = {}
    if request.debug_map_dir is not None:
        debug_dir = Path(request.debug_map_dir).expanduser().resolve()
        for name, values in display_diagnostics.items():
            debug_path = debug_dir / f"{output_path.stem}_{name}.png"
            _write_debug_map_png(values, debug_path)
            debug_map_paths[name] = str(debug_path)

    eval_summary = summarize_bmode_regions(
        display_bmode,
        sector_mask=sector_mask,
        target_mask=eval_target_region_mask,
        wall_mask=eval_wall_mask,
        vessel_mask=eval_vessel_mask,
    )
    consistency_metrics = compute_render_consistency_metrics(
        image_gray=display_bmode,
        sector_mask=sector_mask,
        depth_grid_mm=display_depth_grid_mm,
        lateral_grid_mm=display_lateral_grid_mm,
        max_depth_mm=resolved_max_depth_mm,
        sector_angle_deg=resolved_sector_angle_deg,
        target_depth_mm=target_depth_mm,
        target_lateral_offset_mm=target_lateral_offset_mm,
        target_region_mask=eval_target_region_mask,
        airway_wall_mask=eval_wall_mask,
        vessel_mask=eval_vessel_mask,
        normalization_method=str(normalization_details["method"]),
        normalization_reference_percentile=(
            None
            if normalization_details["reference_percentile"] is None
            else float(normalization_details["reference_percentile"])
        ),
        normalization_reference_value=(
            None
            if normalization_details["reference_value"] is None
            else float(normalization_details["reference_value"])
        ),
        normalization_aux_percentile=(
            None
            if normalization_details["aux_percentile"] is None
            else float(normalization_details["aux_percentile"])
        ),
        normalization_aux_value=(
            None
            if normalization_details["aux_value"] is None
            else float(normalization_details["aux_value"])
        ),
        normalization_lower_bound=(
            None
            if normalization_details["lower_bound"] is None
            else float(normalization_details["lower_bound"])
        ),
        normalization_upper_bound=(
            None
            if normalization_details["upper_bound"] is None
            else float(normalization_details["upper_bound"])
        ),
        compression_gain_factor=(
            None
            if normalization_details["compression_gain_factor"] is None
            else float(normalization_details["compression_gain_factor"])
        ),
    )
    consistency_metrics.update(support_details)
    consistency_metrics["physics_profile_name"] = loaded_profile.name
    engine_diagnostics = {
        "profile": {
            "name": loaded_profile.name,
            "requested_name": request.physics_profile,
            "source_path": loaded_profile.source_path,
            "explicit_overrides": explicit_profile_overrides,
            "effective_settings": effective_profile_settings,
        },
        "artifact_settings": {
            "speckle_strength": float(artifact_config.speckle_strength),
            "reverberation_strength": float(artifact_config.reverberation_strength),
            "shadow_strength": float(artifact_config.shadow_strength),
        },
        "normalization": dict(normalization_details),
        "support_logic": dict(support_details),
        "eval_summary": eval_summary,
        "debug_map_paths": debug_map_paths,
    }

    warnings = list(pose.warnings) + list(device_pose.contact_refinement.warnings)
    if request.virtual_ebus:
        warnings.append("Physics engine currently renders only the simulated B-mode sector; virtual_ebus was ignored.")
    if request.diagnostic_panel:
        warnings.append("Physics engine does not yet provide a diagnostic panel; rendered the sector view only.")
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
    if overlay_config.single_vessel_name is not None and not any(layer.key == overlay_config.single_vessel_name for layer in visible_layers):
        warnings.append(f"Single-vessel mode requested '{overlay_config.single_vessel_name}' but no contour intersected the displayed fan.")

    cutaway_origin = (
        np.asarray(request.cutaway_custom_origin_world, dtype=np.float64)
        if cutaway_config.origin_mode == "custom" and request.cutaway_custom_origin_world is not None
        else contact_world
    )
    cutaway_side = cutaway_config.side
    cutaway_open_side = "left" if cutaway_side == "auto" else cutaway_side

    metadata = RenderMetadata(
        manifest_path=str(manifest.manifest_path),
        case_id=manifest.case_id,
        preset_id=pose.preset_id,
        approach=pose.contact_approach,
        mode=overlay_config.mode,
        output_path=str(output_path),
        metadata_path=str(metadata_path),
        engine=RenderEngine.PHYSICS.value,
        engine_version=PHYSICS_ENGINE_VERSION,
        seed=request.seed,
        view_kind="physics_bmode",
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
        source_plane="probe_ray_domain_nUS_nB",
        display_plane="nUS_nB_fan",
        reference_plane="probe_ray_domain_nUS_nB",
        refine_contact_enabled=bool(request.refine_contact),
        diagnostic_panel_enabled=False,
        diagnostic_panel_layout=[],
        virtual_ebus_enabled=False,
        simulated_ebus_enabled=True,
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
        show_frustum=False,
        cutaway_mode=cutaway_config.mode,
        cutaway_side=cutaway_side,
        cutaway_side_source="physics_placeholder",
        cutaway_open_side=cutaway_open_side,
        cutaway_depth_mm=cutaway_config.depth_mm,
        cutaway_origin_mode=cutaway_config.origin_mode,
        cutaway_origin=[float(value) for value in cutaway_origin.tolist()],
        cutaway_normal=[float(value) for value in lateral_axis.tolist()],
        cutaway_mesh_source="none",
        show_full_airway=cutaway_config.show_full_airway,
        overlays_enabled=_overlay_summary(overlay_config),
        visible_overlay_names=[layer.key for layer in visible_layers],
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
        engine_diagnostics=engine_diagnostics,
    )
    metadata_path.write_text(json.dumps(asdict(metadata), indent=2))

    rendered = RenderedPreset(
        image_rgb=output_image_uint8,
        sector_mask=sector_mask,
        metadata=metadata,
    )
    return RenderResult(
        engine=RenderEngine.PHYSICS,
        engine_version=PHYSICS_ENGINE_VERSION,
        rendered_preset=rendered,
    )
