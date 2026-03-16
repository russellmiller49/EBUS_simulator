from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from ebus_simulator.geometry import (
    build_centerline_segments,
    distance_to_mask_surface_mm,
    mask_contains_point,
    point_inside_volume,
    project_point_to_segments,
)
from ebus_simulator.io.mrkjson import load_mrk_json
from ebus_simulator.io.nifti import load_nifti
from ebus_simulator.io.vtp import load_vtp_polydata
from ebus_simulator.manifest import load_case_manifest
from ebus_simulator.models import (
    ContactValidation,
    PresetValidation,
    ValidationIssue,
    ValidationReport,
)


CONTACT_AIRWAY_DISTANCE_WARN_MM = 4.0
TARGET_STATION_DISTANCE_WARN_MM = 4.0
CENTERLINE_PROJECTION_WARN_MM = 8.0


def _issue(severity: str, message: str, *, preset_id: str | None = None, approach: str | None = None, path: Path | None = None) -> ValidationIssue:
    return ValidationIssue(
        severity=severity,
        message=message,
        preset_id=preset_id,
        approach=approach,
        path=str(path) if path else None,
    )


def _first_defined_point(markup_path: Path):
    markup = load_mrk_json(markup_path)
    for node in markup.markups:
        for point in node.control_points:
            if point.position_status == "defined":
                return point.position_lps
    raise ValueError(f"No defined control point found in {markup_path}")


def _asset_exists(path: Path) -> bool:
    return path.exists() and path.is_file()


def _status_from_counts(error_count: int, warning_count: int) -> str:
    if error_count:
        return "failed"
    if warning_count:
        return "warning"
    return "passed"


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

    seen_paths: set[Path] = set()
    for asset_path in required_assets:
        if asset_path in seen_paths:
            continue
        seen_paths.add(asset_path)
        if not _asset_exists(asset_path):
            issues.append(_issue("error", "Referenced file is missing.", path=asset_path))

    ct_volume = load_nifti(manifest.ct_image, kind="ct", load_data=False)
    airway_lumen = load_nifti(manifest.airway_lumen_mask, kind="mask", load_data=True)
    airway_solid = load_nifti(manifest.airway_solid_mask, kind="mask", load_data=True)
    station_mask_cache = {
        path: load_nifti(path, kind="mask", load_data=True)
        for path in {preset.station_mask for preset in manifest.presets}
    }

    centerline_main = load_vtp_polydata(manifest.centerline_main)
    centerline_network = load_vtp_polydata(manifest.centerline_network)
    main_starts, main_ends = build_centerline_segments(centerline_main)

    if centerline_main.source_space != "LPS":
        issues.append(
            _issue(
                "warning",
                f"Main centerline SPACE is {centerline_main.source_space}, converted to internal LPS.",
                path=manifest.centerline_main,
            )
        )
    if centerline_network.source_space != "LPS":
        issues.append(
            _issue(
                "warning",
                f"Network centerline SPACE is {centerline_network.source_space}, converted to internal LPS.",
                path=manifest.centerline_network,
            )
        )

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
            airway_distance = min(
                distance_to_mask_surface_mm(contact_point, airway_lumen),
                distance_to_mask_surface_mm(contact_point, airway_solid),
            )
            projection = project_point_to_segments(contact_point, main_starts, main_ends)

            if not inside_ct:
                message = "Contact point is outside CT bounds."
                contact_errors.append(message)
                issues.append(_issue("error", message, preset_id=preset.id, approach=approach, path=markup_path))

            if airway_distance > CONTACT_AIRWAY_DISTANCE_WARN_MM:
                message = f"Contact is {airway_distance:.2f} mm from the airway surface."
                contact_warnings.append(message)
                issues.append(_issue("warning", message, preset_id=preset.id, approach=approach, path=markup_path))

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
                    airway_surface_distance_mm=airway_distance,
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
                target_station_distance_mm=target_station_distance,
                contacts=contacts,
                warnings=preset_warnings,
                errors=preset_errors,
            )
        )

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
        issues=issues,
        presets=preset_results,
    )


def report_to_dict(report: ValidationReport) -> dict:
    return asdict(report)
