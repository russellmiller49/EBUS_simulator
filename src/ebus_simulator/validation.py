from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np

from ebus_simulator.centerline import CenterlineGraph
from ebus_simulator.geometry import (
    build_centerline_segments,
    distance_to_mask_surface_mm,
    distance_to_mask_surface_points_mm,
    mask_contains_point,
    mask_contains_points,
    point_inside_volume,
    points_inside_volume,
    project_point_to_segments,
    sample_signed_distance_mm,
)
from ebus_simulator.io.mrkjson import load_mrk_json
from ebus_simulator.io.nifti import load_nifti
from ebus_simulator.io.vtp import RAS_TO_LPS_3X3, load_vtp_polydata
from ebus_simulator.manifest import load_case_manifest
from ebus_simulator.mesh_geometry import get_mesh_surface
from ebus_simulator.models import ContactValidation, PresetValidation, ValidationIssue, ValidationReport


CONTACT_AIRWAY_DISTANCE_WARN_MM = 4.0
TARGET_STATION_DISTANCE_WARN_MM = 4.0
CENTERLINE_PROJECTION_WARN_MM = 8.0
MESH_ALIGNMENT_WARN_MM = 3.0
TARGET_OUTSIDE_TOLERANCE_MM = -1.0
MESH_NORMAL_DELTA_MM = 0.75
MESH_QA_SAMPLE_POINTS = 2000
CENTERLINE_QA_SAMPLE_POINTS = 300


def _issue(severity: str, message: str, *, preset_id: str | None = None, approach: str | None = None, path: Path | None = None) -> ValidationIssue:
    return ValidationIssue(
        severity=severity,
        message=message,
        preset_id=preset_id,
        approach=approach,
        path=str(path) if path else None,
    )


def _first_defined_point(markup_path: Path) -> np.ndarray:
    markup = load_mrk_json(markup_path)
    for node in markup.markups:
        for point in node.control_points:
            if point.position_status == "defined":
                return np.asarray(point.position_lps, dtype=np.float64)
    raise ValueError(f"No defined control point found in {markup_path}")


def _asset_exists(path: Path | None) -> bool:
    return path is not None and path.exists() and path.is_file()


def _status_from_counts(error_count: int, warning_count: int) -> str:
    if error_count:
        return "failed"
    if warning_count:
        return "warning"
    return "passed"


def _normalize(vector: np.ndarray) -> np.ndarray | None:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-9:
        return None
    return np.asarray(vector, dtype=np.float64) / norm


def _project_perpendicular(vector: np.ndarray, axis: np.ndarray) -> np.ndarray:
    return np.asarray(vector, dtype=np.float64) - (float(np.dot(vector, axis)) * np.asarray(axis, dtype=np.float64))


def _candidate_branch_projection(
    point_lps: np.ndarray,
    *,
    main_graph: CenterlineGraph,
    network_graph: CenterlineGraph | None,
):
    projection = main_graph.nearest_point(point_lps)
    if projection is not None:
        return projection
    if network_graph is None:
        return None
    return network_graph.nearest_point(point_lps)


def _orient_mesh_normal(
    normal_world: np.ndarray,
    *,
    contact_world: np.ndarray,
    airway_lumen,
    main_graph: CenterlineGraph,
    network_graph: CenterlineGraph | None,
) -> np.ndarray:
    oriented = _normalize(normal_world)
    if oriented is None:
        return np.asarray(normal_world, dtype=np.float64)

    forward_sd = sample_signed_distance_mm(contact_world + (MESH_NORMAL_DELTA_MM * oriented), airway_lumen)
    backward_sd = sample_signed_distance_mm(contact_world - (MESH_NORMAL_DELTA_MM * oriented), airway_lumen)
    if forward_sd < backward_sd:
        oriented = -oriented

    projection = _candidate_branch_projection(contact_world, main_graph=main_graph, network_graph=network_graph)
    if projection is not None:
        tangent = projection.tangent_lps if projection.tangent_lps is not None else np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
        radial = _project_perpendicular(contact_world - projection.closest_point_lps, tangent)
        radial_unit = _normalize(radial)
        if radial_unit is not None and float(np.dot(oriented, radial_unit)) < 0.0:
            oriented = -oriented

    return oriented


def _stats(values: list[float]) -> dict[str, float | int] | None:
    if not values:
        return None
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "min": float(np.min(array)),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95.0)),
        "max": float(np.max(array)),
    }


def _sample_points(points_lps: np.ndarray, max_points: int) -> np.ndarray:
    if points_lps.shape[0] <= max_points:
        return points_lps
    sample_indices = np.linspace(0, points_lps.shape[0] - 1, max_points, dtype=int)
    return points_lps[sample_indices]


def _mesh_variant_summary(points_lps: np.ndarray, *, ct_volume, airway_lumen, airway_solid) -> dict[str, object]:
    sampled_points = _sample_points(points_lps, MESH_QA_SAMPLE_POINTS)
    inside_ct = points_inside_volume(sampled_points, ct_volume)
    lumen_distances = distance_to_mask_surface_points_mm(sampled_points, airway_lumen)
    solid_distances = distance_to_mask_surface_points_mm(sampled_points, airway_solid)
    lumen_inside = mask_contains_points(sampled_points, airway_lumen)
    solid_inside = mask_contains_points(sampled_points, airway_solid)
    return {
        "sampled_point_count": int(sampled_points.shape[0]),
        "inside_ct_fraction": float(np.mean(inside_ct)) if inside_ct.size else None,
        "inside_lumen_fraction": float(np.mean(lumen_inside)) if lumen_inside.size else None,
        "inside_solid_fraction": float(np.mean(solid_inside)) if solid_inside.size else None,
        "lumen_mask_distance_mm": _stats([float(value) for value in lumen_distances.tolist()]),
        "solid_mask_distance_mm": _stats([float(value) for value in solid_distances.tolist()]),
    }


def _mesh_summary(mesh, *, path: Path | None, ct_volume, airway_lumen, airway_solid) -> dict[str, object]:
    if mesh is None:
        return {
            "path": (None if path is None else str(path)),
            "present": False,
        }

    surface = get_mesh_surface(mesh)
    points = np.asarray(mesh.points_lps, dtype=np.float64)
    return {
        "path": str(mesh.path),
        "present": True,
        "source_space": mesh.source_space,
        "point_count": int(points.shape[0]),
        "polygon_count": int(len(mesh.polygons)),
        "triangle_count": int(surface.triangle_count),
        "bounds_lps": {
            "min": [float(value) for value in points.min(axis=0)],
            "max": [float(value) for value in points.max(axis=0)],
        },
        "alignment_identity": _mesh_variant_summary(points, ct_volume=ct_volume, airway_lumen=airway_lumen, airway_solid=airway_solid),
    }


def validate_case(manifest_path: str | Path) -> ValidationReport:
    manifest = load_case_manifest(manifest_path)
    issues: list[ValidationIssue] = []

    required_assets = [
        manifest.ct_image,
        manifest.centerline_main,
        manifest.centerline_network,
        manifest.airway_lumen_mask,
        manifest.airway_solid_mask,
    ]
    required_assets.extend(manifest.station_masks.values())
    for preset in manifest.presets:
        required_assets.append(preset.station_mask)
        required_assets.append(preset.target)
        required_assets.extend(preset.contacts.values())

    optional_assets = [manifest.airway_raw_mesh, manifest.airway_display_mesh, manifest.airway_cutaway_display_mesh]

    seen_paths: set[Path] = set()
    for asset_path in required_assets:
        if asset_path in seen_paths:
            continue
        seen_paths.add(asset_path)
        if not _asset_exists(asset_path):
            issues.append(_issue("error", "Referenced file is missing.", path=asset_path))
    for asset_path in optional_assets:
        if asset_path is None or asset_path in seen_paths:
            continue
        seen_paths.add(asset_path)
        if not _asset_exists(asset_path):
            issues.append(_issue("warning", "Optional mesh asset is referenced but missing.", path=asset_path))

    ct_volume = load_nifti(manifest.ct_image, kind="ct", load_data=False)
    airway_lumen = load_nifti(manifest.airway_lumen_mask, kind="mask", load_data=True)
    airway_solid = load_nifti(manifest.airway_solid_mask, kind="mask", load_data=True)
    station_mask_cache = {
        path: load_nifti(path, kind="mask", load_data=True)
        for path in {preset.station_mask for preset in manifest.presets}
    }

    centerline_main = load_vtp_polydata(manifest.centerline_main)
    centerline_network = load_vtp_polydata(manifest.centerline_network)
    main_graph = CenterlineGraph.from_vtp(str(manifest.centerline_main), name="main")
    network_graph = CenterlineGraph.from_vtp(str(manifest.centerline_network), name="network")
    main_starts, main_ends = build_centerline_segments(centerline_main)

    if centerline_main.source_space != "LPS":
        issues.append(_issue("warning", f"Main centerline SPACE is {centerline_main.source_space}, converted to internal LPS.", path=manifest.centerline_main))
    if centerline_network.source_space != "LPS":
        issues.append(_issue("warning", f"Network centerline SPACE is {centerline_network.source_space}, converted to internal LPS.", path=manifest.centerline_network))

    raw_mesh = load_vtp_polydata(manifest.airway_raw_mesh) if _asset_exists(manifest.airway_raw_mesh) else None
    display_mesh = load_vtp_polydata(manifest.airway_display_mesh) if _asset_exists(manifest.airway_display_mesh) else None
    cutaway_mesh = load_vtp_polydata(manifest.airway_cutaway_display_mesh) if _asset_exists(manifest.airway_cutaway_display_mesh) else None

    mesh_alignment_warnings: list[str] = []
    raw_mesh_contact_distances: list[float] = []
    target_signed_distances: list[float] = []
    target_outside_flags: list[bool] = []
    raw_surface = None if raw_mesh is None else get_mesh_surface(raw_mesh)

    mesh_summary = {
        "raw": _mesh_summary(raw_mesh, path=manifest.airway_raw_mesh, ct_volume=ct_volume, airway_lumen=airway_lumen, airway_solid=airway_solid),
        "display": _mesh_summary(display_mesh, path=manifest.airway_display_mesh, ct_volume=ct_volume, airway_lumen=airway_lumen, airway_solid=airway_solid),
        "cutaway_display": _mesh_summary(cutaway_mesh, path=manifest.airway_cutaway_display_mesh, ct_volume=ct_volume, airway_lumen=airway_lumen, airway_solid=airway_solid),
        "alignment": {
            "warnings": mesh_alignment_warnings,
        },
    }

    if raw_mesh is not None:
        if raw_mesh.source_space == "UNKNOWN":
            warning = "Raw airway mesh does not declare SPACE metadata; alignment is being validated against masks and centerline before using the current LPS assumption."
            mesh_alignment_warnings.append(warning)
            issues.append(_issue("warning", warning, path=raw_mesh.path))

        identity_alignment = mesh_summary["raw"]["alignment_identity"]
        flipped_points = (RAS_TO_LPS_3X3 @ raw_mesh.points_lps.T).T
        ras_flip_alignment = _mesh_variant_summary(flipped_points, ct_volume=ct_volume, airway_lumen=airway_lumen, airway_solid=airway_solid)
        mesh_summary["alignment"]["orientation_candidates"] = {
            "identity_lps_assumption": identity_alignment,
            "ras_xy_flip_candidate": ras_flip_alignment,
        }

        if (
            identity_alignment["inside_lumen_fraction"] is not None
            and ras_flip_alignment["inside_lumen_fraction"] is not None
            and float(ras_flip_alignment["inside_lumen_fraction"]) > float(identity_alignment["inside_lumen_fraction"]) + 0.15
        ):
            warning = "A RAS-to-LPS x/y flip scores substantially better than the current raw-mesh alignment; inspect the mesh coordinate frame."
            mesh_alignment_warnings.append(warning)
            issues.append(_issue("warning", warning, path=raw_mesh.path))

        if float(identity_alignment["inside_ct_fraction"]) < 0.95:
            warning = f"Only {identity_alignment['inside_ct_fraction']:.2%} of raw-mesh points fall inside CT bounds."
            mesh_alignment_warnings.append(warning)
            issues.append(_issue("warning", warning, path=raw_mesh.path))

        if float(identity_alignment["inside_lumen_fraction"]) < 0.75:
            warning = f"Only {identity_alignment['inside_lumen_fraction']:.2%} of raw-mesh points fall inside the airway lumen mask."
            mesh_alignment_warnings.append(warning)
            issues.append(_issue("warning", warning, path=raw_mesh.path))

        centerline_sample_points = _sample_points(centerline_main.points_lps, CENTERLINE_QA_SAMPLE_POINTS)
        centerline_to_mesh_distances = [float(raw_surface.nearest_point(point).distance_mm) for point in centerline_sample_points]
        mesh_summary["alignment"]["centerline_to_raw_mesh_distance_mm"] = _stats(centerline_to_mesh_distances)
        if mesh_summary["alignment"]["centerline_to_raw_mesh_distance_mm"] is not None and float(mesh_summary["alignment"]["centerline_to_raw_mesh_distance_mm"]["median"]) > 8.0:
            warning = "Main centerline points are unexpectedly far from the raw airway mesh; inspect mesh registration."
            mesh_alignment_warnings.append(warning)
            issues.append(_issue("warning", warning, path=raw_mesh.path))

    preset_results: list[PresetValidation] = []

    for preset in manifest.presets:
        preset_errors: list[str] = []
        preset_warnings: list[str] = []

        try:
            target_point = _first_defined_point(preset.target)
        except Exception as exc:  # pragma: no cover - exercised via CLI on real data
            message = f"Failed to load target markup: {exc}"
            preset_errors.append(message)
            issues.append(_issue("error", message, preset_id=preset.id, path=preset.target))
            preset_results.append(
                PresetValidation(
                    id=preset.id,
                    station=preset.station,
                    node=preset.node,
                    status="failed",
                    target_markup_path=str(preset.target),
                    station_mask_path=str(preset.station_mask),
                    target_point_lps=None,
                    target_inside_ct_bounds=None,
                    target_inside_station_mask=None,
                    target_station_distance_mm=None,
                    target_to_raw_mesh_signed_distance_mm=None,
                    target_raw_mesh_side=None,
                    target_raw_mesh_side_consistent=None,
                    contacts=[],
                    warnings=preset_warnings,
                    errors=preset_errors,
                )
            )
            continue

        station_mask = station_mask_cache[preset.station_mask]
        target_inside_ct = point_inside_volume(target_point, ct_volume)
        target_inside_station = mask_contains_point(target_point, station_mask)
        target_station_surface_distance = distance_to_mask_surface_mm(target_point, station_mask)
        target_station_distance = 0.0 if target_inside_station else target_station_surface_distance

        if not target_inside_ct:
            message = "Target point is outside CT bounds."
            preset_errors.append(message)
            issues.append(_issue("error", message, preset_id=preset.id, path=preset.target))

        if not target_inside_station and target_station_surface_distance > TARGET_STATION_DISTANCE_WARN_MM:
            message = f"Target is {target_station_surface_distance:.2f} mm from the station mask surface."
            preset_warnings.append(message)
            issues.append(_issue("warning", message, preset_id=preset.id, path=preset.station_mask))

        target_signed_distance = None
        target_side = None
        target_consistent = None
        if raw_surface is not None:
            query = raw_surface.nearest_point(target_point)
            normal = query.point_normal_lps if query.point_normal_lps is not None else query.face_normal_lps
            if normal is not None:
                oriented_normal = _orient_mesh_normal(
                    normal,
                    contact_world=query.closest_point_lps,
                    airway_lumen=airway_lumen,
                    main_graph=main_graph,
                    network_graph=network_graph,
                )
                target_signed_distance = float(np.dot(target_point - query.closest_point_lps, oriented_normal))
                target_side = "outside" if target_signed_distance > 0.5 else ("inside" if target_signed_distance < -0.5 else "surface")
                target_consistent = bool(target_signed_distance >= TARGET_OUTSIDE_TOLERANCE_MM)
                target_signed_distances.append(target_signed_distance)
                target_outside_flags.append(target_consistent)
                if not target_consistent:
                    message = f"Target lies on the luminal side of the raw airway mesh by {abs(target_signed_distance):.2f} mm."
                    preset_warnings.append(message)
                    issues.append(_issue("warning", message, preset_id=preset.id, path=manifest.airway_raw_mesh))

        contacts: list[ContactValidation] = []
        for approach, markup_path in preset.contacts.items():
            contact_errors: list[str] = []
            contact_warnings: list[str] = []

            try:
                contact_point = _first_defined_point(markup_path)
            except Exception as exc:  # pragma: no cover - exercised via CLI on real data
                message = f"Failed to load contact markup: {exc}"
                contact_errors.append(message)
                issues.append(_issue("error", message, preset_id=preset.id, approach=approach, path=markup_path))
                contacts.append(
                    ContactValidation(
                        approach=approach,
                        markup_path=str(markup_path),
                        point_lps=[],
                        inside_ct_bounds=False,
                        airway_surface_distance_mm=None,
                        raw_mesh_distance_mm=None,
                        centerline_projection_distance_mm=None,
                        tangent_defined=False,
                        closest_centerline_point_lps=None,
                        tangent_lps=None,
                        warnings=contact_warnings,
                        errors=contact_errors,
                    )
                )
                continue

            inside_ct = point_inside_volume(contact_point, ct_volume)
            airway_distance = min(distance_to_mask_surface_mm(contact_point, airway_lumen), distance_to_mask_surface_mm(contact_point, airway_solid))
            projection = project_point_to_segments(contact_point, main_starts, main_ends)
            raw_mesh_distance = None if raw_surface is None else float(raw_surface.nearest_point(contact_point).distance_mm)
            if raw_mesh_distance is not None:
                raw_mesh_contact_distances.append(raw_mesh_distance)

            if not inside_ct:
                message = "Contact point is outside CT bounds."
                contact_errors.append(message)
                issues.append(_issue("error", message, preset_id=preset.id, approach=approach, path=markup_path))

            if airway_distance > CONTACT_AIRWAY_DISTANCE_WARN_MM:
                message = f"Contact is {airway_distance:.2f} mm from the airway surface."
                contact_warnings.append(message)
                issues.append(_issue("warning", message, preset_id=preset.id, approach=approach, path=markup_path))

            if raw_mesh_distance is not None and raw_mesh_distance > MESH_ALIGNMENT_WARN_MM:
                message = f"Contact is {raw_mesh_distance:.2f} mm from the raw airway mesh."
                contact_warnings.append(message)
                issues.append(_issue("warning", message, preset_id=preset.id, approach=approach, path=manifest.airway_raw_mesh))

            if projection is None:
                message = "Contact could not be projected to the centerline."
                contact_errors.append(message)
                issues.append(_issue("error", message, preset_id=preset.id, approach=approach, path=markup_path))
            else:
                if projection["distance_mm"] > CENTERLINE_PROJECTION_WARN_MM:
                    message = f"Contact projects {projection['distance_mm']:.2f} mm away from the centerline."
                    contact_warnings.append(message)
                    issues.append(_issue("warning", message, preset_id=preset.id, approach=approach, path=markup_path))
                if not projection["tangent_defined"]:
                    message = "Centerline tangent is undefined at the projected contact."
                    contact_errors.append(message)
                    issues.append(_issue("error", message, preset_id=preset.id, approach=approach, path=markup_path))

            contacts.append(
                ContactValidation(
                    approach=approach,
                    markup_path=str(markup_path),
                    point_lps=[float(value) for value in contact_point.tolist()],
                    inside_ct_bounds=inside_ct,
                    airway_surface_distance_mm=float(airway_distance),
                    raw_mesh_distance_mm=raw_mesh_distance,
                    centerline_projection_distance_mm=(None if projection is None else float(projection["distance_mm"])),
                    tangent_defined=(False if projection is None else bool(projection["tangent_defined"])),
                    closest_centerline_point_lps=(None if projection is None else [float(value) for value in projection["closest_point_lps"]]),
                    tangent_lps=(None if projection is None else [float(value) for value in projection["tangent_lps"]]),
                    warnings=contact_warnings,
                    errors=contact_errors,
                )
            )

        preset_status = _status_from_counts(
            error_count=len(preset_errors) + sum(len(contact.errors) for contact in contacts),
            warning_count=len(preset_warnings) + sum(len(contact.warnings) for contact in contacts),
        )
        preset_results.append(
            PresetValidation(
                id=preset.id,
                station=preset.station,
                node=preset.node,
                status=preset_status,
                target_markup_path=str(preset.target),
                station_mask_path=str(preset.station_mask),
                target_point_lps=[float(value) for value in target_point.tolist()],
                target_inside_ct_bounds=target_inside_ct,
                target_inside_station_mask=target_inside_station,
                target_station_distance_mm=float(target_station_distance),
                target_to_raw_mesh_signed_distance_mm=target_signed_distance,
                target_raw_mesh_side=target_side,
                target_raw_mesh_side_consistent=target_consistent,
                contacts=contacts,
                warnings=preset_warnings,
                errors=preset_errors,
            )
        )

    if raw_surface is not None:
        mesh_summary["alignment"]["contact_to_raw_mesh_distance_mm"] = _stats(raw_mesh_contact_distances)
        mesh_summary["alignment"]["target_to_raw_mesh_signed_distance_mm"] = _stats(target_signed_distances)
        mesh_summary["alignment"]["target_outside_fraction"] = float(np.mean(target_outside_flags)) if target_outside_flags else None
    else:
        mesh_summary["alignment"]["contact_to_raw_mesh_distance_mm"] = None
        mesh_summary["alignment"]["target_to_raw_mesh_signed_distance_mm"] = None
        mesh_summary["alignment"]["target_outside_fraction"] = None

    error_count = sum(1 for issue in issues if issue.severity == "error")
    warning_count = sum(1 for issue in issues if issue.severity == "warning")

    return ValidationReport(
        manifest_path=str(manifest.manifest_path),
        case_id=manifest.case_id,
        dataset_root=str(manifest.root),
        internal_world_frame="LPS",
        preset_count=len(manifest.presets),
        status=_status_from_counts(error_count=error_count, warning_count=warning_count),
        ct={
            "path": str(manifest.ct_image),
            "shape": list(ct_volume.shape),
            "dtype": ct_volume.dtype,
            "voxel_sizes_mm": [float(value) for value in ct_volume.voxel_sizes_mm.tolist()],
            "axis_codes_ras": list(ct_volume.axis_codes_ras),
        },
        centerlines={
            "main_path": str(manifest.centerline_main),
            "network_path": str(manifest.centerline_network),
            "main_point_count": int(centerline_main.points_lps.shape[0]),
            "main_line_count": int(len(centerline_main.lines)),
            "main_segment_count": int(main_starts.shape[0]),
            "network_point_count": int(centerline_network.points_lps.shape[0]),
            "network_line_count": int(len(centerline_network.lines)),
            "main_space": centerline_main.source_space,
            "network_space": centerline_network.source_space,
        },
        meshes=mesh_summary,
        issues=issues,
        presets=preset_results,
    )


def report_to_dict(report: ValidationReport) -> dict:
    return asdict(report)
