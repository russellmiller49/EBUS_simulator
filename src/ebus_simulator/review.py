from __future__ import annotations

from csv import DictWriter
from dataclasses import replace
import json
from pathlib import Path

import numpy as np

from ebus_simulator.manifest import resolve_preset_overrides
from ebus_simulator.rendering import (
    _build_sector_grid,
    _default_output_stem,
    _fan_target_row_col,
    _get_mask_volume,
    _map_plane_to_fan,
    _resolve_preset_manifest,
    _sample_contact_plane,
    build_render_context,
    render_preset,
)


NUS_DELTA_WARN_DEG = 10.0
CONTACT_DELTA_WARN_MM = 1.5
STATION_OVERLAP_WARN_FRACTION = 0.003
REQUIRED_COMPARISON_CASES = {
    ("station_11l_node_a", "default"),
    ("station_7_node_a", "lms"),
    ("station_7_node_a", "rms"),
    ("station_4r_node_b", "default"),
}


def _normalize(vector: np.ndarray) -> np.ndarray | None:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-9:
        return None
    return np.asarray(vector, dtype=np.float64) / norm


def _angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    a_unit = _normalize(a)
    b_unit = _normalize(b)
    if a_unit is None or b_unit is None:
        return 0.0
    return float(np.degrees(np.arccos(np.clip(float(np.dot(a_unit, b_unit)), -1.0, 1.0))))


def _compute_contact_to_centerline_mm(context, contact_world: np.ndarray) -> tuple[float | None, str | None]:
    projection = context.main_graph.nearest_point(contact_world)
    if projection is not None:
        return float(projection.distance_mm), "main"
    projection = context.network_graph.nearest_point(contact_world)
    if projection is not None:
        return float(projection.distance_mm), "network"
    return None, None


def _compute_station_overlap_fraction_in_fan(
    context,
    *,
    preset_manifest,
    contact_world: np.ndarray,
    probe_axis: np.ndarray,
    shaft_axis: np.ndarray,
    thickness_axis: np.ndarray,
    width: int,
    height: int,
    source_oblique_size_mm: float,
    max_depth_mm: float,
    sector_angle_deg: float,
    slice_thickness_mm: float,
) -> float:
    station_mask = _get_mask_volume(context, preset_manifest.station_mask)
    source_mask, _, _, _ = _sample_contact_plane(
        data=np.asarray(station_mask.data, dtype=np.float32),
        inverse_affine_lps=station_mask.inverse_affine_lps,
        origin_world=contact_world,
        depth_axis=probe_axis,
        lateral_axis=shaft_axis,
        thickness_axis=thickness_axis,
        depth_max_mm=source_oblique_size_mm,
        lateral_half_width_mm=(source_oblique_size_mm / 2.0),
        width=width,
        height=height,
        slice_thickness_mm=slice_thickness_mm,
        order=0,
        cval=0.0,
    )
    depth_grid, lateral_grid, sector_mask, _ = _build_sector_grid(width, height, max_depth_mm, sector_angle_deg)
    fan_mask = _map_plane_to_fan(
        source_mask,
        source_forward_max_mm=source_oblique_size_mm,
        source_shaft_half_width_mm=(source_oblique_size_mm / 2.0),
        depth_grid_mm=depth_grid,
        display_lateral_grid_mm=lateral_grid,
        sector_mask=sector_mask,
        order=0,
        cval=0.0,
    ) > 0.0
    denominator = int(np.count_nonzero(sector_mask))
    if denominator <= 0:
        return 0.0
    return float(np.count_nonzero(np.logical_and(fan_mask, sector_mask)) / denominator)


def _compute_review_metrics(context, *, preset_manifest, clean_rendered) -> dict[str, object]:
    metadata = clean_rendered.metadata
    contact_world = np.asarray(metadata.refined_contact_world, dtype=np.float64)
    target_world = np.asarray(metadata.target_world, dtype=np.float64)
    probe_axis = np.asarray(metadata.device_axes["nUS"], dtype=np.float64)
    shaft_axis = np.asarray(metadata.device_axes["nB"], dtype=np.float64)
    thickness_axis = _normalize(np.cross(shaft_axis, probe_axis))
    if thickness_axis is None:
        thickness_axis = np.asarray(metadata.pose_axes["lateral_axis"], dtype=np.float64)

    target_offset = target_world - contact_world
    target_depth_mm = float(np.dot(target_offset, probe_axis))
    target_lateral_offset_mm = float(np.dot(target_offset, shaft_axis))
    target_in_forward_hemisphere = bool(target_depth_mm > 0.0)
    _, _, _, max_lateral_mm = _build_sector_grid(
        clean_rendered.sector_mask.shape[1],
        clean_rendered.sector_mask.shape[0],
        metadata.max_depth_mm,
        metadata.sector_angle_deg,
    )
    target_in_sector = (
        _fan_target_row_col(
            contact_world=contact_world,
            target_world=target_world,
            probe_axis=probe_axis,
            shaft_axis=shaft_axis,
            max_depth_mm=metadata.max_depth_mm,
            sector_angle_deg=metadata.sector_angle_deg,
            width=clean_rendered.sector_mask.shape[1],
            height=clean_rendered.sector_mask.shape[0],
            max_lateral_mm=max_lateral_mm,
        )
        is not None
    )

    contact_to_centerline_mm, contact_centerline_source = _compute_contact_to_centerline_mm(context, contact_world)
    station_overlap_fraction_in_fan = _compute_station_overlap_fraction_in_fan(
        context,
        preset_manifest=preset_manifest,
        contact_world=contact_world,
        probe_axis=probe_axis,
        shaft_axis=shaft_axis,
        thickness_axis=thickness_axis,
        width=clean_rendered.sector_mask.shape[1],
        height=clean_rendered.sector_mask.shape[0],
        source_oblique_size_mm=metadata.source_oblique_size_mm,
        max_depth_mm=metadata.max_depth_mm,
        sector_angle_deg=metadata.sector_angle_deg,
        slice_thickness_mm=metadata.slice_thickness_mm,
    )

    warnings = list(metadata.warnings)
    contact_refinement_ambiguity = any("ambiguous" in warning.lower() for warning in warnings)
    nUS_delta_deg = float(metadata.pose_comparison.get("nUS_angular_difference_deg", 0.0))
    contact_delta_mm = float(metadata.pose_comparison.get("voxel_to_mesh_contact_distance_mm", 0.0))

    return {
        "preset_id": metadata.preset_id,
        "approach": metadata.approach,
        "contact_to_mesh_mm": metadata.refined_contact_to_airway_distance_mm,
        "contact_to_centerline_mm": contact_to_centerline_mm,
        "contact_centerline_source": contact_centerline_source,
        "target_depth_mm": target_depth_mm,
        "target_lateral_offset_mm": target_lateral_offset_mm,
        "target_in_sector": target_in_sector,
        "target_in_forward_hemisphere": target_in_forward_hemisphere,
        "station_overlap_fraction_in_fan": station_overlap_fraction_in_fan,
        "nUS_delta_deg_from_voxel_baseline": nUS_delta_deg,
        "contact_delta_mm_from_voxel_baseline": contact_delta_mm,
        "contact_refinement_ambiguity": contact_refinement_ambiguity,
        "warnings": warnings,
    }


def _flag_review_metrics(metrics: dict[str, object]) -> list[str]:
    reasons: list[str] = []
    if float(metrics["nUS_delta_deg_from_voxel_baseline"]) > NUS_DELTA_WARN_DEG:
        reasons.append(f"nUS delta {float(metrics['nUS_delta_deg_from_voxel_baseline']):.2f} deg > {NUS_DELTA_WARN_DEG:.1f} deg")
    if float(metrics["contact_delta_mm_from_voxel_baseline"]) > CONTACT_DELTA_WARN_MM:
        reasons.append(f"contact delta {float(metrics['contact_delta_mm_from_voxel_baseline']):.2f} mm > {CONTACT_DELTA_WARN_MM:.1f} mm")
    if not bool(metrics["target_in_sector"]):
        reasons.append("target not in displayed fan sector")
    if float(metrics["station_overlap_fraction_in_fan"]) < STATION_OVERLAP_WARN_FRACTION:
        reasons.append(
            f"station overlap {float(metrics['station_overlap_fraction_in_fan']):.4f} < {STATION_OVERLAP_WARN_FRACTION:.4f}"
        )
    if bool(metrics["contact_refinement_ambiguity"]):
        reasons.append("contact refinement remained ambiguous")
    return reasons


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
) -> dict[str, object]:
    stem = _default_output_stem(preset_id, approach)
    review_dir = output_dir / "approaches"
    diagnostic_path = review_dir / f"{stem}_panel.png"
    clean_path = review_dir / f"{stem}_clean.png"
    review_json_path = review_dir / f"{stem}_review.json"

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

    preset_manifest = _resolve_preset_manifest(context.manifest, preset_id)
    metrics = _compute_review_metrics(context, preset_manifest=preset_manifest, clean_rendered=clean)
    flag_reasons = _flag_review_metrics(metrics)
    preset_overrides = resolve_preset_overrides(preset_manifest, approach=approach)

    entry = {
        "preset_id": preset_id,
        "approach": approach,
        "diagnostic_panel_png": str(diagnostic_path),
        "diagnostic_panel_json": diagnostic.metadata.metadata_path,
        "clean_simulated_png": str(clean_path),
        "clean_simulated_json": clean.metadata.metadata_path,
        "flagged": bool(flag_reasons),
        "flag_reasons": flag_reasons,
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
                "axis_sign_override": preset_overrides.axis_sign_override,
                "reference_fov_mm": preset_overrides.reference_fov_mm,
                "notes": preset_overrides.notes,
            },
        },
        "warnings": list(dict.fromkeys(clean.metadata.warnings + diagnostic.metadata.warnings)),
    }
    review_json_path.write_text(json.dumps(entry, indent=2))
    entry["review_json"] = str(review_json_path)
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
                "cutaway_side",
                "cutaway_mode",
                "vessel_overlay_names",
                "diagnostic_panel_png",
                "clean_simulated_png",
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
                    "cutaway_side": entry["cutaway_side"],
                    "cutaway_mode": entry["cutaway_mode"],
                    "vessel_overlay_names": ",".join(entry["vessel_overlay_names"]),
                    "diagnostic_panel_png": entry["diagnostic_panel_png"],
                    "clean_simulated_png": entry["clean_simulated_png"],
                    "review_json": entry["review_json"],
                    "warnings": " | ".join(entry["warnings"]),
                }
            )


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
) -> dict[str, object]:
    context = build_render_context(manifest_path, roll_deg=roll_deg)
    voxel_context = replace(context, airway_geometry_mesh=None)
    output_dir = Path(output_dir).expanduser().resolve()
    (output_dir / "approaches").mkdir(parents=True, exist_ok=True)
    (output_dir / "comparisons").mkdir(parents=True, exist_ok=True)

    entries: list[dict[str, object]] = []
    for preset in context.manifest.presets:
        for approach in preset.contacts:
            entries.append(
                _render_review_entry(
                    context,
                    preset_id=preset.id,
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
                )
            )

    summary_json_path = output_dir / "review_summary.json"
    summary_csv_path = output_dir / "review_summary.csv"

    comparison_payloads: list[dict[str, object]] = []
    flagged_keys = {(entry["preset_id"], entry["approach"]) for entry in entries if entry["flagged"]}
    for preset_id, approach in sorted(flagged_keys | REQUIRED_COMPARISON_CASES):
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
        "thresholds": {
            "nUS_delta_deg_from_voxel_baseline": NUS_DELTA_WARN_DEG,
            "contact_delta_mm_from_voxel_baseline": CONTACT_DELTA_WARN_MM,
            "station_overlap_fraction_in_fan": STATION_OVERLAP_WARN_FRACTION,
        },
        "entries": entries,
        "comparison_bundles": comparison_payloads,
    }
    summary_json_path.write_text(json.dumps(summary, indent=2))
    _write_summary_csv(summary_csv_path, entries)
    return summary
