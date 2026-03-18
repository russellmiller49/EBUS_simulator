from __future__ import annotations

from csv import DictWriter
from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path

import numpy as np

from ebus_simulator.manifest import resolve_preset_overrides
from ebus_simulator.review_rubric import render_review_rubric_template, render_review_sheet
from ebus_simulator.rendering import (
    _default_output_stem,
    _angle_deg,
    _resolve_preset_manifest,
    build_render_context,
    compute_pose_review_metrics,
    render_preset,
)


REQUIRED_COMPARISON_CASES = {
    ("station_11l_node_a", "default"),
    ("station_11ri_node_a", "default"),
    ("station_11rs_node_a", "default"),
    ("station_7_node_a", "lms"),
    ("station_7_node_a", "rms"),
    ("station_4r_node_b", "default"),
}


@dataclass(frozen=True)
class ReviewThresholds:
    nUS_delta_deg_from_voxel_baseline: float = 10.0
    contact_delta_mm_from_voxel_baseline: float = 1.5
    station_overlap_fraction_in_fan: float = 0.003
    target_contrast_vs_sector_min: float = 0.0
    vessel_contrast_vs_sector_max: float = -0.01
    wall_contrast_vs_sector_min: float | None = None


DEFAULT_REVIEW_THRESHOLDS = ReviewThresholds()


def _relative_to_output(path: str | Path, output_dir: Path) -> str:
    return str(Path(path).resolve().relative_to(output_dir))


def _iter_selected_presets(context, preset_ids: list[str] | None) -> list[tuple[str, str]]:
    selected = None if preset_ids is None else set(preset_ids)
    if selected is not None:
        available = {preset.id for preset in context.manifest.presets}
        unknown = sorted(selected - available)
        if unknown:
            raise ValueError(f"Unknown preset_id filter(s): {', '.join(unknown)}")
    pairs: list[tuple[str, str]] = []
    for preset in context.manifest.presets:
        if selected is not None and preset.id not in selected:
            continue
        for approach in preset.contacts:
            pairs.append((preset.id, approach))
    return sorted(pairs, key=lambda value: (value[0], value[1]))


def _compute_review_metrics(context, *, preset_manifest, clean_rendered) -> dict[str, object]:
    metadata = clean_rendered.metadata
    thickness_axis = np.cross(
        np.asarray(metadata.device_axes["nB"], dtype=np.float64),
        np.asarray(metadata.device_axes["nUS"], dtype=np.float64),
    )
    if float(np.linalg.norm(thickness_axis)) <= 1e-9:
        thickness_axis = (
            None
            if metadata.pose_axes["lateral_axis"] is None
            else np.asarray(metadata.pose_axes["lateral_axis"], dtype=np.float64)
        )
    metrics = compute_pose_review_metrics(
        context,
        preset_manifest=preset_manifest,
        target_world=np.asarray(metadata.target_world, dtype=np.float64),
        contact_world=np.asarray(metadata.refined_contact_world, dtype=np.float64),
        probe_axis=np.asarray(metadata.device_axes["nUS"], dtype=np.float64),
        shaft_axis=np.asarray(metadata.device_axes["nB"], dtype=np.float64),
        thickness_axis=thickness_axis,
        warnings=list(metadata.warnings),
        width=clean_rendered.sector_mask.shape[1],
        height=clean_rendered.sector_mask.shape[0],
        source_oblique_size_mm=metadata.source_oblique_size_mm,
        max_depth_mm=metadata.max_depth_mm,
        sector_angle_deg=metadata.sector_angle_deg,
        slice_thickness_mm=metadata.slice_thickness_mm,
        nUS_delta_deg_from_voxel_baseline=float(metadata.pose_comparison.get("nUS_angular_difference_deg", 0.0)),
        contact_delta_mm_from_voxel_baseline=float(metadata.pose_comparison.get("voxel_to_mesh_contact_distance_mm", 0.0)),
    )
    metrics.update(
        {
            "preset_id": metadata.preset_id,
            "approach": metadata.approach,
            "contact_to_mesh_mm": metadata.refined_contact_to_airway_distance_mm,
        }
    )
    return metrics


def _flag_geometry_metrics(
    metrics: dict[str, object],
    *,
    thresholds: ReviewThresholds = DEFAULT_REVIEW_THRESHOLDS,
) -> list[str]:
    reasons: list[str] = []
    if float(metrics["nUS_delta_deg_from_voxel_baseline"]) > thresholds.nUS_delta_deg_from_voxel_baseline:
        reasons.append(
            "nUS delta "
            f"{float(metrics['nUS_delta_deg_from_voxel_baseline']):.2f} deg"
            f" > {thresholds.nUS_delta_deg_from_voxel_baseline:.1f} deg"
        )
    if float(metrics["contact_delta_mm_from_voxel_baseline"]) > thresholds.contact_delta_mm_from_voxel_baseline:
        reasons.append(
            "contact delta "
            f"{float(metrics['contact_delta_mm_from_voxel_baseline']):.2f} mm"
            f" > {thresholds.contact_delta_mm_from_voxel_baseline:.1f} mm"
        )
    if not bool(metrics["target_in_sector"]):
        reasons.append("target not in displayed fan sector")
    if float(metrics["station_overlap_fraction_in_fan"]) < thresholds.station_overlap_fraction_in_fan:
        reasons.append(
            "station overlap "
            f"{float(metrics['station_overlap_fraction_in_fan']):.4f}"
            f" < {thresholds.station_overlap_fraction_in_fan:.4f}"
        )
    if bool(metrics["contact_refinement_ambiguity"]):
        reasons.append("contact refinement remained ambiguous")
    return reasons


def _region_pixel_count(summary: dict[str, object], key: str) -> int:
    region = summary.get(key)
    if not isinstance(region, dict):
        return 0
    return int(region.get("pixel_count", 0) or 0)


def _optional_float(summary: dict[str, object], key: str) -> float | None:
    value = summary.get(key)
    if value is None:
        return None
    return float(value)


def _flag_physics_eval_summary(
    physics_eval_summary: dict[str, object] | None,
    *,
    thresholds: ReviewThresholds = DEFAULT_REVIEW_THRESHOLDS,
) -> list[str]:
    if not physics_eval_summary:
        return []

    reasons: list[str] = []

    target_pixels = _region_pixel_count(physics_eval_summary, "target")
    target_contrast = _optional_float(physics_eval_summary, "target_contrast_vs_sector")
    if target_pixels == 0:
        reasons.append("target region missing from physics eval summary")
    elif target_contrast is not None and target_contrast < thresholds.target_contrast_vs_sector_min:
        reasons.append(
            "target contrast "
            f"{target_contrast:.3f}"
            f" < {thresholds.target_contrast_vs_sector_min:.3f}"
        )

    vessel_pixels = _region_pixel_count(physics_eval_summary, "vessel")
    vessel_contrast = _optional_float(physics_eval_summary, "vessel_contrast_vs_sector")
    if vessel_pixels > 0 and vessel_contrast is not None and vessel_contrast > thresholds.vessel_contrast_vs_sector_max:
        reasons.append(
            "vessel contrast "
            f"{vessel_contrast:.3f}"
            f" > {thresholds.vessel_contrast_vs_sector_max:.3f}"
        )

    if thresholds.wall_contrast_vs_sector_min is not None:
        wall_pixels = _region_pixel_count(physics_eval_summary, "wall")
        wall_contrast = _optional_float(physics_eval_summary, "wall_contrast_vs_sector")
        if wall_pixels == 0:
            reasons.append("wall region missing from physics eval summary")
        elif wall_contrast is not None and wall_contrast < thresholds.wall_contrast_vs_sector_min:
            reasons.append(
                "wall contrast "
                f"{wall_contrast:.3f}"
                f" < {thresholds.wall_contrast_vs_sector_min:.3f}"
            )

    return reasons


def _flag_review_metrics(
    metrics: dict[str, object],
    *,
    physics_eval_summary: dict[str, object] | None = None,
    thresholds: ReviewThresholds = DEFAULT_REVIEW_THRESHOLDS,
) -> list[str]:
    return _flag_geometry_metrics(metrics, thresholds=thresholds) + _flag_physics_eval_summary(
        physics_eval_summary,
        thresholds=thresholds,
    )


def _render_review_entry(
    context,
    *,
    preset_id: str,
    approach: str,
    output_dir: Path,
    width: int,
    height: int,
    device: str,
    roll_deg: float | None,
    sector_angle_deg: float | None,
    max_depth_mm: float | None,
    reference_fov_mm: float | None,
    slice_thickness_mm: float | None,
    cutaway_mode: str | None,
    cutaway_side: str | None,
    cutaway_depth_mm: float | None,
    cutaway_origin: str | None,
    show_full_airway: bool | None,
    vessel_overlay_names: list[str] | None,
    include_physics_debug_maps: bool,
    physics_speckle_strength: float | None,
    physics_reverberation_strength: float | None,
    physics_shadow_strength: float | None,
    review_thresholds: ReviewThresholds,
) -> dict[str, object]:
    entry_dir = output_dir / "presets" / preset_id / approach
    entry_dir.mkdir(parents=True, exist_ok=True)

    diagnostic_path = entry_dir / "localizer_panel.png"
    clean_path = entry_dir / "localizer_clean.png"
    physics_path = entry_dir / "physics.png"
    eval_summary_path = entry_dir / "eval_summary.json"
    review_json_path = entry_dir / "review_entry.json"
    review_sheet_path = entry_dir / "review_sheet.md"
    debug_map_dir = entry_dir / "debug_maps" if include_physics_debug_maps else None

    diagnostic = render_preset(
        context.manifest.manifest_path,
        preset_id,
        approach=approach,
        output_path=diagnostic_path,
        width=width,
        height=height,
        sector_angle_deg=sector_angle_deg,
        max_depth_mm=max_depth_mm,
        roll_deg=roll_deg,
        mode="debug",
        diagnostic_panel=True,
        device=device,
        refine_contact=True,
        virtual_ebus=True,
        simulated_ebus=True,
        reference_fov_mm=reference_fov_mm,
        airway_overlay=None,
        airway_lumen_overlay=None,
        airway_wall_overlay=None,
        target_overlay=None,
        contact_overlay=None,
        station_overlay=None,
        vessel_overlay_names=vessel_overlay_names,
        show_legend=None,
        label_overlays=None,
        show_contact=None,
        show_frustum=None,
        slice_thickness_mm=slice_thickness_mm,
        cutaway_mode=cutaway_mode,
        cutaway_side=cutaway_side,
        cutaway_depth_mm=cutaway_depth_mm,
        cutaway_origin=cutaway_origin,
        show_full_airway=show_full_airway,
        context=context,
    )
    clean = render_preset(
        context.manifest.manifest_path,
        preset_id,
        approach=approach,
        output_path=clean_path,
        width=width,
        height=height,
        sector_angle_deg=sector_angle_deg,
        max_depth_mm=max_depth_mm,
        roll_deg=roll_deg,
        mode="clean",
        diagnostic_panel=False,
        device=device,
        refine_contact=True,
        virtual_ebus=False,
        simulated_ebus=True,
        reference_fov_mm=reference_fov_mm,
        airway_overlay=None,
        airway_lumen_overlay=None,
        airway_wall_overlay=None,
        target_overlay=None,
        contact_overlay=False,
        station_overlay=None,
        vessel_overlay_names=vessel_overlay_names,
        show_legend=False,
        label_overlays=False,
        show_contact=False,
        show_frustum=False,
        slice_thickness_mm=slice_thickness_mm,
        cutaway_mode=cutaway_mode,
        cutaway_side=cutaway_side,
        cutaway_depth_mm=cutaway_depth_mm,
        cutaway_origin=cutaway_origin,
        show_full_airway=show_full_airway,
        context=context,
    )
    physics = render_preset(
        context.manifest.manifest_path,
        preset_id,
        approach=approach,
        output_path=physics_path,
        engine="physics",
        width=width,
        height=height,
        sector_angle_deg=sector_angle_deg,
        max_depth_mm=max_depth_mm,
        roll_deg=roll_deg,
        mode="clean",
        diagnostic_panel=False,
        device=device,
        refine_contact=True,
        virtual_ebus=False,
        simulated_ebus=True,
        reference_fov_mm=reference_fov_mm,
        airway_overlay=None,
        airway_lumen_overlay=None,
        airway_wall_overlay=None,
        target_overlay=None,
        contact_overlay=False,
        station_overlay=None,
        vessel_overlay_names=vessel_overlay_names,
        show_legend=False,
        label_overlays=False,
        show_contact=False,
        show_frustum=False,
        slice_thickness_mm=slice_thickness_mm,
        cutaway_mode=cutaway_mode,
        cutaway_side=cutaway_side,
        cutaway_depth_mm=cutaway_depth_mm,
        cutaway_origin=cutaway_origin,
        show_full_airway=show_full_airway,
        debug_map_dir=debug_map_dir,
        speckle_strength=physics_speckle_strength,
        reverberation_strength=physics_reverberation_strength,
        shadow_strength=physics_shadow_strength,
        context=context,
    )

    preset_manifest = _resolve_preset_manifest(context.manifest, preset_id)
    metrics = _compute_review_metrics(context, preset_manifest=preset_manifest, clean_rendered=clean)
    preset_overrides = resolve_preset_overrides(preset_manifest, approach=approach)
    physics_diagnostics = dict(physics.metadata.engine_diagnostics)
    physics_eval_summary = dict(physics_diagnostics.get("eval_summary", {}))
    physics_artifact_settings = dict(physics_diagnostics.get("artifact_settings", {}))
    physics_debug_maps = dict(physics_diagnostics.get("debug_map_paths", {}))
    geometry_flag_reasons = _flag_geometry_metrics(metrics, thresholds=review_thresholds)
    physics_flag_reasons = _flag_physics_eval_summary(physics_eval_summary, thresholds=review_thresholds)
    flag_reasons = geometry_flag_reasons + physics_flag_reasons

    eval_summary_payload = {
        "preset_id": preset_id,
        "approach": approach,
        "engine": physics.metadata.engine,
        "engine_version": physics.metadata.engine_version,
        "artifact_settings": physics_artifact_settings,
        "eval_summary": physics_eval_summary,
    }
    eval_summary_path.write_text(json.dumps(eval_summary_payload, indent=2))

    entry = {
        "preset_id": preset_id,
        "approach": approach,
        "localizer_panel_png": str(diagnostic_path),
        "localizer_panel_json": diagnostic.metadata.metadata_path,
        "localizer_clean_png": str(clean_path),
        "localizer_clean_json": clean.metadata.metadata_path,
        "physics_png": str(physics_path),
        "physics_json": physics.metadata.metadata_path,
        "eval_summary_json": str(eval_summary_path),
        "review_sheet_md": str(review_sheet_path),
        "physics_debug_maps": physics_debug_maps,
        "physics_debug_map_count": len(physics_debug_maps),
        "physics_artifact_settings": physics_artifact_settings,
        "physics_eval_summary": physics_eval_summary,
        "diagnostic_panel_png": str(diagnostic_path),
        "diagnostic_panel_json": diagnostic.metadata.metadata_path,
        "clean_simulated_png": str(clean_path),
        "clean_simulated_json": clean.metadata.metadata_path,
        "flagged": bool(flag_reasons),
        "flag_reasons": flag_reasons,
        "geometry_flag_reasons": geometry_flag_reasons,
        "physics_flag_reasons": physics_flag_reasons,
        "metrics": metrics,
        "cutaway_side": clean.metadata.cutaway_side,
        "cutaway_mode": clean.metadata.cutaway_mode,
        "vessel_overlay_names": list(diagnostic.metadata.vessel_overlay_names),
        "manifest_overrides": {
            "applied": clean.metadata.preset_override_applied,
            "vessel_overlays": list(clean.metadata.preset_override_vessel_overlays),
            "cutaway_side": clean.metadata.preset_override_cutaway_side,
            "roll_offset_deg": clean.metadata.preset_override_roll_offset_deg,
            "branch_hint": clean.metadata.preset_override_branch_hint,
            "branch_shift_mm": clean.metadata.preset_override_branch_shift_mm,
            "axis_sign_override": clean.metadata.preset_override_axis_sign_override,
            "reference_fov_mm": clean.metadata.preset_override_reference_fov_mm,
            "notes": clean.metadata.preset_override_notes,
            "raw_override_block": None
            if preset_overrides is None
            else {
                "vessel_overlays": preset_overrides.vessel_overlays,
                "cutaway_side": preset_overrides.cutaway_side,
                "roll_offset_deg": preset_overrides.roll_offset_deg,
                "branch_hint": preset_overrides.branch_hint,
                "branch_shift_mm": preset_overrides.branch_shift_mm,
                "axis_sign_override": preset_overrides.axis_sign_override,
                "reference_fov_mm": preset_overrides.reference_fov_mm,
                "notes": preset_overrides.notes,
            },
        },
        "warnings": list(dict.fromkeys(clean.metadata.warnings + diagnostic.metadata.warnings + physics.metadata.warnings)),
    }
    entry["review_json"] = str(review_json_path)
    entry["review_entry_json"] = str(review_json_path)
    review_json_path.write_text(json.dumps(entry, indent=2))
    review_sheet_path.write_text(
        render_review_sheet(
            preset_id=preset_id,
            approach=approach,
            localizer_panel_path=diagnostic_path,
            localizer_clean_path=clean_path,
            physics_path=physics_path,
            eval_summary_path=eval_summary_path,
            review_entry_path=review_json_path,
            warnings=entry["warnings"],
            geometry_flag_reasons=geometry_flag_reasons,
            physics_flag_reasons=physics_flag_reasons,
        )
    )
    return entry


def _write_summary_csv(summary_path: Path, entries: list[dict[str, object]]) -> None:
    with summary_path.open("w", newline="") as handle:
        writer = DictWriter(
            handle,
            fieldnames=[
                "preset_id",
                "approach",
                "flagged",
                "flag_reasons",
                "geometry_flag_reasons",
                "physics_flag_reasons",
                "contact_to_mesh_mm",
                "contact_to_centerline_mm",
                "target_depth_mm",
                "target_lateral_offset_mm",
                "target_in_sector",
                "target_in_forward_hemisphere",
                "station_overlap_fraction_in_fan",
                "nUS_delta_deg_from_voxel_baseline",
                "contact_delta_mm_from_voxel_baseline",
                "contact_refinement_ambiguity",
                "target_contrast_vs_sector",
                "wall_contrast_vs_sector",
                "vessel_contrast_vs_sector",
                "localizer_panel_png",
                "localizer_clean_png",
                "physics_png",
                "physics_json",
                "eval_summary_json",
                "review_sheet_md",
                "physics_debug_map_count",
                "cutaway_side",
                "cutaway_mode",
                "vessel_overlay_names",
                "review_json",
                "warnings",
            ],
        )
        writer.writeheader()
        for entry in entries:
            metrics = entry["metrics"]
            writer.writerow(
                {
                    "preset_id": entry["preset_id"],
                    "approach": entry["approach"],
                    "flagged": entry["flagged"],
                    "flag_reasons": " | ".join(entry["flag_reasons"]),
                    "geometry_flag_reasons": " | ".join(entry["geometry_flag_reasons"]),
                    "physics_flag_reasons": " | ".join(entry["physics_flag_reasons"]),
                    "contact_to_mesh_mm": metrics["contact_to_mesh_mm"],
                    "contact_to_centerline_mm": metrics["contact_to_centerline_mm"],
                    "target_depth_mm": metrics["target_depth_mm"],
                    "target_lateral_offset_mm": metrics["target_lateral_offset_mm"],
                    "target_in_sector": metrics["target_in_sector"],
                    "target_in_forward_hemisphere": metrics["target_in_forward_hemisphere"],
                    "station_overlap_fraction_in_fan": metrics["station_overlap_fraction_in_fan"],
                    "nUS_delta_deg_from_voxel_baseline": metrics["nUS_delta_deg_from_voxel_baseline"],
                    "contact_delta_mm_from_voxel_baseline": metrics["contact_delta_mm_from_voxel_baseline"],
                    "contact_refinement_ambiguity": metrics["contact_refinement_ambiguity"],
                    "target_contrast_vs_sector": entry["physics_eval_summary"].get("target_contrast_vs_sector"),
                    "wall_contrast_vs_sector": entry["physics_eval_summary"].get("wall_contrast_vs_sector"),
                    "vessel_contrast_vs_sector": entry["physics_eval_summary"].get("vessel_contrast_vs_sector"),
                    "localizer_panel_png": entry["localizer_panel_png"],
                    "localizer_clean_png": entry["localizer_clean_png"],
                    "physics_png": entry["physics_png"],
                    "physics_json": entry["physics_json"],
                    "eval_summary_json": entry["eval_summary_json"],
                    "review_sheet_md": entry["review_sheet_md"],
                    "physics_debug_map_count": entry["physics_debug_map_count"],
                    "cutaway_side": entry["cutaway_side"],
                    "cutaway_mode": entry["cutaway_mode"],
                    "vessel_overlay_names": ",".join(entry["vessel_overlay_names"]),
                    "review_json": entry["review_json"],
                    "warnings": " | ".join(entry["warnings"]),
                }
            )


def _write_summary_markdown(summary_path: Path, output_dir: Path, entries: list[dict[str, object]]) -> None:
    lines = [
        "# Review Index",
        "",
        "| Preset | Approach | Flagged | Localizer | Physics | Eval Summary | Review Sheet |",
        "|---|---|---|---|---|---|---|",
    ]
    for entry in entries:
        lines.append(
            "| {preset} | {approach} | {flagged} | [panel]({localizer}) | [physics]({physics}) | [eval]({eval}) | [sheet]({sheet}) |".format(
                preset=entry["preset_id"],
                approach=entry["approach"],
                flagged=("yes" if entry["flagged"] else "no"),
                localizer=_relative_to_output(entry["localizer_panel_png"], output_dir),
                physics=_relative_to_output(entry["physics_png"], output_dir),
                eval=_relative_to_output(entry["eval_summary_json"], output_dir),
                sheet=_relative_to_output(entry["review_sheet_md"], output_dir),
            )
        )
    lines.extend(
        [
            "",
            "## Auto-Flagged Entries",
            "",
        ]
    )
    flagged_entries = [entry for entry in entries if entry["flagged"]]
    if not flagged_entries:
        lines.append("- none")
    else:
        for entry in flagged_entries:
            reasons = "; ".join(entry["flag_reasons"])
            lines.append(f"- `{entry['preset_id']}` / `{entry['approach']}`: {reasons}")
    summary_path.write_text("\n".join(lines) + "\n")


def _render_comparison_bundle(
    context,
    voxel_context,
    *,
    preset_id: str,
    approach: str,
    output_dir: Path,
    width: int,
    height: int,
    device: str,
    roll_deg: float | None,
    sector_angle_deg: float | None,
    max_depth_mm: float | None,
    reference_fov_mm: float | None,
    slice_thickness_mm: float | None,
    cutaway_mode: str | None,
    cutaway_side: str | None,
    cutaway_depth_mm: float | None,
    cutaway_origin: str | None,
    show_full_airway: bool | None,
    vessel_overlay_names: list[str] | None,
    auto_flagged: bool,
    flag_reasons: list[str],
    metrics: dict[str, object],
) -> dict[str, object]:
    stem = _default_output_stem(preset_id, approach)
    comparison_dir = output_dir / "comparisons"
    mesh_path = comparison_dir / f"{stem}_mesh_panel.png"
    voxel_path = comparison_dir / f"{stem}_voxel_panel.png"
    summary_path = comparison_dir / f"{stem}_comparison.json"

    mesh_rendered = render_preset(
        context.manifest.manifest_path,
        preset_id,
        approach=approach,
        output_path=mesh_path,
        width=width,
        height=height,
        sector_angle_deg=sector_angle_deg,
        max_depth_mm=max_depth_mm,
        roll_deg=roll_deg,
        mode="debug",
        diagnostic_panel=True,
        device=device,
        refine_contact=True,
        virtual_ebus=True,
        simulated_ebus=True,
        reference_fov_mm=reference_fov_mm,
        vessel_overlay_names=vessel_overlay_names,
        slice_thickness_mm=slice_thickness_mm,
        cutaway_mode=cutaway_mode,
        cutaway_side=cutaway_side,
        cutaway_depth_mm=cutaway_depth_mm,
        cutaway_origin=cutaway_origin,
        show_full_airway=show_full_airway,
        context=context,
    )
    voxel_rendered = render_preset(
        voxel_context.manifest.manifest_path,
        preset_id,
        approach=approach,
        output_path=voxel_path,
        width=width,
        height=height,
        sector_angle_deg=sector_angle_deg,
        max_depth_mm=max_depth_mm,
        roll_deg=roll_deg,
        mode="debug",
        diagnostic_panel=True,
        device=device,
        refine_contact=True,
        virtual_ebus=True,
        simulated_ebus=True,
        reference_fov_mm=reference_fov_mm,
        vessel_overlay_names=vessel_overlay_names,
        slice_thickness_mm=slice_thickness_mm,
        cutaway_mode=cutaway_mode,
        cutaway_side=cutaway_side,
        cutaway_depth_mm=cutaway_depth_mm,
        cutaway_origin=cutaway_origin,
        show_full_airway=show_full_airway,
        context=voxel_context,
    )

    mesh_contact = np.asarray(mesh_rendered.metadata.refined_contact_world, dtype=np.float64)
    voxel_contact = np.asarray(voxel_rendered.metadata.refined_contact_world, dtype=np.float64)
    mesh_nUS = np.asarray(mesh_rendered.metadata.device_axes["nUS"], dtype=np.float64)
    voxel_nUS = np.asarray(voxel_rendered.metadata.device_axes["nUS"], dtype=np.float64)

    payload = {
        "preset_id": preset_id,
        "approach": approach,
        "required_comparison_case": (preset_id, approach) in REQUIRED_COMPARISON_CASES,
        "auto_flagged": auto_flagged,
        "flag_reasons": list(flag_reasons),
        "mesh_panel_png": str(mesh_path),
        "mesh_panel_json": mesh_rendered.metadata.metadata_path,
        "voxel_panel_png": str(voxel_path),
        "voxel_panel_json": voxel_rendered.metadata.metadata_path,
        "mesh_contact_world": mesh_rendered.metadata.refined_contact_world,
        "voxel_contact_world": voxel_rendered.metadata.refined_contact_world,
        "mesh_nUS_world": mesh_rendered.metadata.device_axes["nUS"],
        "voxel_nUS_world": voxel_rendered.metadata.device_axes["nUS"],
        "contact_delta_mm": float(np.linalg.norm(mesh_contact - voxel_contact)),
        "nUS_delta_deg": _angle_deg(mesh_nUS, voxel_nUS),
        "review_metrics": metrics,
        "mesh_warnings": list(mesh_rendered.metadata.warnings),
        "voxel_warnings": list(voxel_rendered.metadata.warnings),
    }
    summary_path.write_text(json.dumps(payload, indent=2))
    return payload


def review_presets(
    manifest_path: str | Path,
    *,
    output_dir: str | Path,
    width: int = 384,
    height: int = 384,
    device: str = "bf_uc180f",
    roll_deg: float | None = None,
    sector_angle_deg: float | None = None,
    max_depth_mm: float | None = None,
    reference_fov_mm: float | None = None,
    slice_thickness_mm: float | None = None,
    cutaway_mode: str | None = None,
    cutaway_side: str | None = None,
    cutaway_depth_mm: float | None = None,
    cutaway_origin: str | None = None,
    show_full_airway: bool | None = None,
    vessel_overlay_names: list[str] | None = None,
    preset_ids: list[str] | None = None,
    include_physics_debug_maps: bool = False,
    physics_speckle_strength: float | None = None,
    physics_reverberation_strength: float | None = None,
    physics_shadow_strength: float | None = None,
    review_thresholds: ReviewThresholds | None = None,
) -> dict[str, object]:
    resolved_thresholds = DEFAULT_REVIEW_THRESHOLDS if review_thresholds is None else review_thresholds
    context = build_render_context(manifest_path, roll_deg=roll_deg)
    voxel_context = replace(context, airway_geometry_mesh=None)
    output_dir = Path(output_dir).expanduser().resolve()
    (output_dir / "presets").mkdir(parents=True, exist_ok=True)
    (output_dir / "comparisons").mkdir(parents=True, exist_ok=True)

    entries: list[dict[str, object]] = []
    for preset_id, approach in _iter_selected_presets(context, preset_ids):
        entries.append(
            _render_review_entry(
                context,
                preset_id=preset_id,
                approach=approach,
                output_dir=output_dir,
                width=width,
                height=height,
                device=device,
                roll_deg=roll_deg,
                sector_angle_deg=sector_angle_deg,
                max_depth_mm=max_depth_mm,
                reference_fov_mm=reference_fov_mm,
                slice_thickness_mm=slice_thickness_mm,
                cutaway_mode=cutaway_mode,
                cutaway_side=cutaway_side,
                cutaway_depth_mm=cutaway_depth_mm,
                cutaway_origin=cutaway_origin,
                show_full_airway=show_full_airway,
                vessel_overlay_names=vessel_overlay_names,
                include_physics_debug_maps=include_physics_debug_maps,
                physics_speckle_strength=physics_speckle_strength,
                physics_reverberation_strength=physics_reverberation_strength,
                physics_shadow_strength=physics_shadow_strength,
                review_thresholds=resolved_thresholds,
            )
        )

    summary_json_path = output_dir / "review_summary.json"
    summary_csv_path = output_dir / "review_summary.csv"
    index_json_path = output_dir / "review_index.json"
    index_csv_path = output_dir / "review_index.csv"
    index_md_path = output_dir / "review_index.md"
    rubric_path = output_dir / "review_rubric_template.md"

    comparison_payloads: list[dict[str, object]] = []
    flagged_keys = {(entry["preset_id"], entry["approach"]) for entry in entries if entry["flagged"]}
    selected_keys = {(entry["preset_id"], entry["approach"]) for entry in entries}
    for preset_id, approach in sorted((flagged_keys | REQUIRED_COMPARISON_CASES) & selected_keys):
        matching_entry = next(entry for entry in entries if entry["preset_id"] == preset_id and entry["approach"] == approach)
        comparison_payloads.append(
            _render_comparison_bundle(
                context,
                voxel_context,
                preset_id=preset_id,
                approach=approach,
                output_dir=output_dir,
                width=width,
                height=height,
                device=device,
                roll_deg=roll_deg,
                sector_angle_deg=sector_angle_deg,
                max_depth_mm=max_depth_mm,
                reference_fov_mm=reference_fov_mm,
                slice_thickness_mm=slice_thickness_mm,
                cutaway_mode=cutaway_mode,
                cutaway_side=cutaway_side,
                cutaway_depth_mm=cutaway_depth_mm,
                cutaway_origin=cutaway_origin,
                show_full_airway=show_full_airway,
                vessel_overlay_names=vessel_overlay_names,
                auto_flagged=bool(matching_entry["flagged"]),
                flag_reasons=list(matching_entry["flag_reasons"]),
                metrics=matching_entry["metrics"],
            )
        )

    summary = {
        "manifest_path": str(context.manifest.manifest_path),
        "case_id": context.manifest.case_id,
        "output_dir": str(output_dir),
        "review_count": len(entries),
        "flagged_count": sum(1 for entry in entries if entry["flagged"]),
        "preset_filter": ([] if preset_ids is None else list(preset_ids)),
        "include_physics_debug_maps": bool(include_physics_debug_maps),
        "physics_settings": {
            "speckle_strength": physics_speckle_strength,
            "reverberation_strength": physics_reverberation_strength,
            "shadow_strength": physics_shadow_strength,
        },
        "rubric_template": str(rubric_path),
        "thresholds": {
            **asdict(resolved_thresholds),
        },
        "entries": entries,
        "comparison_bundles": comparison_payloads,
    }
    rubric_path.write_text(render_review_rubric_template())
    summary_json_path.write_text(json.dumps(summary, indent=2))
    index_json_path.write_text(json.dumps(summary, indent=2))
    _write_summary_csv(summary_csv_path, entries)
    _write_summary_csv(index_csv_path, entries)
    _write_summary_markdown(index_md_path, output_dir, entries)
    return summary
