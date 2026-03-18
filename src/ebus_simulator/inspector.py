from __future__ import annotations

from dataclasses import dataclass
from html import escape
from typing import Mapping, Sequence

import numpy as np


_RESERVED_OVERLAY_KEYS = {
    "airway_lumen",
    "airway_wall",
    "station",
    "target",
    "contact",
}


@dataclass(slots=True, frozen=True)
class InspectorField:
    label: str
    value: str


@dataclass(slots=True, frozen=True)
class InspectorSection:
    title: str
    fields: tuple[InspectorField, ...]


def _coerce_vector3(value: object) -> np.ndarray | None:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return None
    try:
        return np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return None


def _coerce_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_bool(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    return None


def compute_target_offsets_mm(metadata: Mapping[str, object]) -> tuple[float | None, float | None]:
    contact_world = _coerce_vector3(metadata.get("contact_world"))
    target_world = _coerce_vector3(metadata.get("target_world"))
    pose_axes = metadata.get("pose_axes")
    if (
        contact_world is None
        or target_world is None
        or not isinstance(pose_axes, Mapping)
    ):
        return None, None

    depth_axis = _coerce_vector3(pose_axes.get("depth_axis"))
    lateral_axis = _coerce_vector3(pose_axes.get("lateral_axis"))
    if depth_axis is None:
        return None, None

    offset = target_world - contact_world
    target_depth_mm = float(np.dot(offset, depth_axis))
    target_lateral_mm = None if lateral_axis is None else float(np.dot(offset, lateral_axis))
    return target_depth_mm, target_lateral_mm


def compute_target_in_sector(
    metadata: Mapping[str, object],
    *,
    target_depth_mm: float | None,
    target_lateral_mm: float | None,
) -> bool | None:
    if target_depth_mm is None or target_lateral_mm is None:
        return None
    max_depth_mm = _coerce_float(metadata.get("max_depth_mm"))
    sector_angle_deg = _coerce_float(metadata.get("sector_angle_deg"))
    if max_depth_mm is None or sector_angle_deg is None:
        return None
    fan_half_tan = float(np.tan(np.deg2rad(sector_angle_deg / 2.0)))
    return bool(
        0.0 <= target_depth_mm <= max_depth_mm
        and abs(target_lateral_mm) <= ((target_depth_mm * fan_half_tan) + 1e-9)
    )


def _format_float(value: float | None, *, precision: int = 2, suffix: str = "") -> str:
    if value is None:
        return "n/a"
    return f"{value:.{precision}f}{suffix}"


def _format_optional_float(value: float | None, *, precision: int = 2, suffix: str = "") -> str | None:
    if value is None:
        return None
    return f"{value:.{precision}f}{suffix}"


def _format_seed(value: object) -> str:
    return "not set" if value is None else str(value)


def _titleize(value: str | None) -> str:
    if value is None:
        return "n/a"
    if len(value) <= 4 and value.isalpha():
        return value.upper()
    return value.replace("_", " ").strip().title()


def _format_station_label(value: str | None) -> str:
    if value is None or not value:
        return "n/a"
    return f"Station {value.upper()}"


def _format_node_label(value: str | None) -> str:
    if value is None or not value:
        return "n/a"
    return f"Node {value.upper()}"


def _format_bool(value: bool | None) -> str:
    if value is None:
        return "n/a"
    return "Yes" if value else "No"


def _format_presence(value: bool | None, *, source: str | None = None) -> str:
    base = _format_bool(value)
    if source is None or base == "n/a":
        return base
    return f"{base} ({source})"


def _format_list(value: Sequence[object] | None, *, empty: str = "none") -> str:
    if not value:
        return empty
    parts = [str(item) for item in value if str(item)]
    return ", ".join(parts) if parts else empty


def _format_multiline_list(value: Sequence[str] | None, *, empty: str = "None") -> str:
    if not value:
        return empty
    return "\n".join(str(item) for item in value if str(item))


def _extract_mapping(parent: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = parent.get(key)
    return value if isinstance(value, Mapping) else {}


def _overlay_names(metadata: Mapping[str, object], key: str) -> list[str]:
    value = metadata.get(key)
    if not isinstance(value, list):
        return []
    return [str(name) for name in value if str(name)]


def _eval_summary(sector_metadata: Mapping[str, object]) -> Mapping[str, object]:
    engine_diagnostics = sector_metadata.get("engine_diagnostics")
    if not isinstance(engine_diagnostics, Mapping):
        return {}
    value = engine_diagnostics.get("eval_summary")
    return value if isinstance(value, Mapping) else {}


def _artifact_settings(sector_metadata: Mapping[str, object]) -> Mapping[str, object]:
    engine_diagnostics = sector_metadata.get("engine_diagnostics")
    if not isinstance(engine_diagnostics, Mapping):
        return {}
    value = engine_diagnostics.get("artifact_settings")
    return value if isinstance(value, Mapping) else {}


def _consistency_metrics(sector_metadata: Mapping[str, object]) -> Mapping[str, object]:
    value = sector_metadata.get("consistency_metrics")
    return value if isinstance(value, Mapping) else {}


def _region_pixel_count(summary: Mapping[str, object], key: str) -> int | None:
    region = summary.get(key)
    if not isinstance(region, Mapping):
        return None
    value = region.get("pixel_count")
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _region_mean(summary: Mapping[str, object], key: str) -> float | None:
    region = summary.get(key)
    if not isinstance(region, Mapping):
        return None
    return _coerce_float(region.get("mean"))


def _render_eval_region_summary(summary: Mapping[str, object]) -> str | None:
    parts: list[str] = []
    sector_mean = _region_mean(summary, "sector")
    if sector_mean is not None:
        parts.append(f"sector mean {sector_mean:.4f}")
    for name in ("target", "wall", "vessel"):
        pixel_count = _region_pixel_count(summary, name)
        mean = _region_mean(summary, name)
        if pixel_count is None and mean is None:
            continue
        label = name.title()
        if pixel_count is None:
            parts.append(f"{label} mean {mean:.4f}")
        elif mean is None:
            parts.append(f"{label} {pixel_count} px")
        else:
            parts.append(f"{label} {pixel_count} px / {mean:.4f}")
    return "; ".join(parts) if parts else None


def _dedupe(values: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


def _visible_overlay_names(*metadata_values: Mapping[str, object]) -> list[str]:
    names: list[str] = []
    for metadata in metadata_values:
        names.extend(_overlay_names(metadata, "visible_overlay_names"))
    return _dedupe(names)


def _enabled_overlay_names(*metadata_values: Mapping[str, object]) -> list[str]:
    names: list[str] = []
    for metadata in metadata_values:
        names.extend(_overlay_names(metadata, "overlays_enabled"))
    return _dedupe(names)


def _visible_vessel_names(*metadata_values: Mapping[str, object]) -> list[str]:
    return [name for name in _visible_overlay_names(*metadata_values) if name not in _RESERVED_OVERLAY_KEYS]


def _enabled_vessel_names(*metadata_values: Mapping[str, object]) -> list[str]:
    return [name for name in _enabled_overlay_names(*metadata_values) if name not in _RESERVED_OVERLAY_KEYS]


def _derive_target_present(
    eval_summary: Mapping[str, object],
    *,
    visible_overlay_names: Sequence[str],
    target_in_sector: bool | None,
) -> tuple[bool | None, str | None]:
    target_pixels = _region_pixel_count(eval_summary, "target")
    if target_pixels is not None:
        return target_pixels > 0, "eval"
    if "target" in visible_overlay_names:
        return True, "overlay"
    if target_in_sector is not None:
        return target_in_sector, "geometry"
    return None, None


def _derive_wall_present(
    eval_summary: Mapping[str, object],
    *,
    visible_overlay_names: Sequence[str],
    enabled_overlay_names: Sequence[str],
) -> tuple[bool | None, str | None]:
    wall_pixels = _region_pixel_count(eval_summary, "wall")
    if wall_pixels is not None:
        return wall_pixels > 0, "eval"
    if "airway_wall" in visible_overlay_names:
        return True, "overlay"
    if "airway_wall" in enabled_overlay_names:
        return False, "overlay"
    return None, None


def _derive_vessel_present(
    eval_summary: Mapping[str, object],
    *,
    visible_vessel_names: Sequence[str],
    enabled_vessel_names: Sequence[str],
) -> tuple[bool | None, str | None]:
    vessel_pixels = _region_pixel_count(eval_summary, "vessel")
    if vessel_pixels is not None:
        return vessel_pixels > 0, "eval"
    if visible_vessel_names:
        return True, "overlay"
    if enabled_vessel_names:
        return False, "overlay"
    return None, None


def _view_description(metadata: Mapping[str, object]) -> str:
    engine = _titleize(str(metadata.get("engine", "n/a")))
    mode = _titleize(str(metadata.get("mode", "n/a")))
    view_kind = metadata.get("view_kind")
    if view_kind is None:
        return f"{engine} / {mode}"
    return f"{engine} / {mode} ({view_kind})"


def _cutaway_description(metadata: Mapping[str, object]) -> str | None:
    mode = metadata.get("cutaway_mode")
    side = metadata.get("cutaway_side")
    if mode is None and side is None:
        return None
    return f"{_titleize(None if side is None else str(side))} / {_titleize(None if mode is None else str(mode))}"


def _format_artifact_settings(settings: Mapping[str, object]) -> str | None:
    values = [
        ("Speckle", _coerce_float(settings.get("speckle_strength"))),
        ("Reverberation", _coerce_float(settings.get("reverberation_strength"))),
        ("Shadow", _coerce_float(settings.get("shadow_strength"))),
    ]
    parts = [f"{label} {value:.2f}" for label, value in values if value is not None]
    return ", ".join(parts) if parts else None


def _field_if_value(fields: list[InspectorField], label: str, value: str | None) -> None:
    if value is None:
        return
    fields.append(InspectorField(label=label, value=value))


def build_render_inspector_sections(
    sector_metadata: Mapping[str, object],
    context_metadata: Mapping[str, object],
    *,
    station_label: str | None = None,
    node_label: str | None = None,
    review_metrics: Mapping[str, object] | None = None,
    flag_reasons: Sequence[str] | None = None,
    warnings: Sequence[str] | None = None,
    screenshot_name_hint: str | None = None,
) -> list[InspectorSection]:
    metrics = review_metrics if isinstance(review_metrics, Mapping) else {}
    consistency_metrics = _consistency_metrics(sector_metadata)
    combined_metrics = dict(consistency_metrics)
    combined_metrics.update(metrics)
    eval_summary = _eval_summary(sector_metadata)
    artifact_settings = _artifact_settings(sector_metadata)
    pose_comparison = _extract_mapping(sector_metadata, "pose_comparison")
    target_depth_mm, target_lateral_mm = compute_target_offsets_mm(sector_metadata)
    target_in_sector = _coerce_bool(combined_metrics.get("target_in_sector"))
    if target_in_sector is None:
        target_in_sector = compute_target_in_sector(
            sector_metadata,
            target_depth_mm=target_depth_mm,
            target_lateral_mm=target_lateral_mm,
        )

    visible_overlay_names = _visible_overlay_names(sector_metadata, context_metadata)
    enabled_overlay_names = _enabled_overlay_names(sector_metadata, context_metadata)
    visible_vessel_names = _visible_vessel_names(sector_metadata, context_metadata)
    enabled_vessel_names = _enabled_vessel_names(sector_metadata, context_metadata)
    target_present, target_present_source = _derive_target_present(
        eval_summary,
        visible_overlay_names=visible_overlay_names,
        target_in_sector=target_in_sector,
    )
    wall_present, wall_present_source = _derive_wall_present(
        eval_summary,
        visible_overlay_names=visible_overlay_names,
        enabled_overlay_names=enabled_overlay_names,
    )
    vessel_present, vessel_present_source = _derive_vessel_present(
        eval_summary,
        visible_vessel_names=visible_vessel_names,
        enabled_vessel_names=enabled_vessel_names,
    )

    preset_fields = [
        InspectorField("Preset ID", str(sector_metadata.get("preset_id", "n/a"))),
        InspectorField("Station", _format_station_label(station_label)),
        InspectorField("Target / Node", _format_node_label(node_label)),
        InspectorField("Approach", _titleize(None if sector_metadata.get("approach") is None else str(sector_metadata.get("approach")))),
        InspectorField("2D Engine", _titleize(None if sector_metadata.get("engine") is None else str(sector_metadata.get("engine")))),
    ]
    _field_if_value(
        preset_fields,
        "Preset Notes",
        None if sector_metadata.get("preset_override_notes") is None else str(sector_metadata.get("preset_override_notes")),
    )

    pose_fields = [
        InspectorField("Depth", _format_float(_coerce_float(sector_metadata.get("max_depth_mm")), precision=1, suffix=" mm")),
        InspectorField("Sector Angle", _format_float(_coerce_float(sector_metadata.get("sector_angle_deg")), precision=1, suffix=" deg")),
        InspectorField("Fine Roll", _format_float(_coerce_float(sector_metadata.get("roll_deg")), precision=1, suffix=" deg")),
        InspectorField("Gain", _format_float(_coerce_float(sector_metadata.get("gain")), precision=2)),
        InspectorField("Attenuation", _format_float(_coerce_float(sector_metadata.get("attenuation")), precision=2)),
        InspectorField("Seed", _format_seed(sector_metadata.get("seed"))),
        InspectorField(
            "Contact Refinement",
            _titleize(None if sector_metadata.get("contact_refinement_method") is None else str(sector_metadata.get("contact_refinement_method"))),
        ),
        InspectorField("Target Depth", _format_float(target_depth_mm, precision=2, suffix=" mm")),
        InspectorField("Target Lateral", _format_float(target_lateral_mm, precision=2, suffix=" mm")),
        InspectorField(
            "Contact-Airway Distance",
            _format_float(_coerce_float(sector_metadata.get("refined_contact_to_airway_distance_mm")), precision=2, suffix=" mm"),
        ),
        InspectorField(
            "Centerline Distance",
            _format_float(_coerce_float(sector_metadata.get("centerline_projection_distance_mm")), precision=2, suffix=" mm"),
        ),
    ]
    branch_hint = pose_comparison.get("branch_hint")
    if branch_hint is not None:
        pose_fields.append(InspectorField("Branch Hint", str(branch_hint)))
    branch_hint_applied = _coerce_bool(pose_comparison.get("branch_hint_applied"))
    if branch_hint_applied is not None:
        pose_fields.append(InspectorField("Branch Hint Applied", _format_bool(branch_hint_applied)))

    anatomy_fields = [
        InspectorField("Target Present", _format_presence(target_present, source=target_present_source)),
        InspectorField("Target In Sector", _format_bool(target_in_sector)),
        InspectorField("Airway Wall Present", _format_presence(wall_present, source=wall_present_source)),
        InspectorField("Vessel Present", _format_presence(vessel_present, source=vessel_present_source)),
        InspectorField("Contact Visible", _format_bool("contact" in visible_overlay_names)),
        InspectorField("2D Visible Overlays", _format_list(_overlay_names(sector_metadata, "visible_overlay_names"))),
        InspectorField("3D Visible Overlays", _format_list(_overlay_names(context_metadata, "visible_overlay_names"))),
    ]
    _field_if_value(
        anatomy_fields,
        "Target Coverage",
        _format_optional_float(_coerce_float(combined_metrics.get("target_sector_coverage_fraction")), precision=4),
    )
    _field_if_value(
        anatomy_fields,
        "Target Centerline Offset",
        _format_optional_float(_coerce_float(combined_metrics.get("target_centerline_offset_fraction")), precision=3),
    )
    _field_if_value(
        anatomy_fields,
        "Near-Field Wall Occupancy",
        _format_optional_float(_coerce_float(combined_metrics.get("near_field_wall_occupancy_fraction")), precision=4),
    )
    _field_if_value(
        anatomy_fields,
        "Non-Background Occupancy",
        _format_optional_float(_coerce_float(combined_metrics.get("non_background_occupancy_fraction")), precision=4),
    )
    _field_if_value(
        anatomy_fields,
        "Empty Sector",
        _format_optional_float(_coerce_float(combined_metrics.get("empty_sector_fraction")), precision=4),
    )

    review_fields: list[InspectorField] = []
    for label, key in (
        ("Target Contrast", "target_contrast_vs_sector"),
        ("Vessel Contrast", "vessel_contrast_vs_sector"),
        ("Wall Contrast", "wall_contrast_vs_sector"),
    ):
        review_fields.append(
            InspectorField(
                label,
                _format_float(_coerce_float(eval_summary.get(key)), precision=4),
            )
        )
    eval_regions = _render_eval_region_summary(eval_summary)
    _field_if_value(review_fields, "Eval Regions", eval_regions)
    _field_if_value(
        review_fields,
        "Target Region Contrast",
        _format_optional_float(_coerce_float(combined_metrics.get("target_region_contrast_vs_sector")), precision=4),
    )
    _field_if_value(
        review_fields,
        "Wall Region Contrast",
        _format_optional_float(_coerce_float(combined_metrics.get("wall_region_contrast_vs_sector")), precision=4),
    )
    _field_if_value(
        review_fields,
        "Sector Brightness Mean",
        _format_optional_float(_coerce_float(combined_metrics.get("sector_brightness_mean")), precision=4),
    )
    _field_if_value(
        review_fields,
        "Near-Field Brightness Mean",
        _format_optional_float(_coerce_float(combined_metrics.get("near_field_brightness_mean")), precision=4),
    )
    _field_if_value(
        review_fields,
        "nUS Delta vs Voxel Baseline",
        _format_optional_float(_coerce_float(combined_metrics.get("nUS_delta_deg_from_voxel_baseline")), precision=2, suffix=" deg"),
    )
    _field_if_value(
        review_fields,
        "Contact Delta vs Voxel Baseline",
        _format_optional_float(_coerce_float(combined_metrics.get("contact_delta_mm_from_voxel_baseline")), precision=2, suffix=" mm"),
    )
    _field_if_value(
        review_fields,
        "Station Overlap in Fan",
        _format_optional_float(_coerce_float(combined_metrics.get("station_overlap_fraction_in_fan")), precision=4),
    )
    review_fields.append(InspectorField("Auto-Flags", _format_multiline_list([str(reason) for reason in flag_reasons or []])))
    review_fields.append(InspectorField("Warnings", _format_multiline_list([str(reason) for reason in warnings or []])))

    render_fields = [
        InspectorField("2D Render", _view_description(sector_metadata)),
        InspectorField("3D Context", _view_description(context_metadata)),
        InspectorField("2D Enabled Overlays", _format_list(_overlay_names(sector_metadata, "overlays_enabled"))),
        InspectorField("3D Enabled Overlays", _format_list(_overlay_names(context_metadata, "overlays_enabled"))),
    ]
    _field_if_value(render_fields, "Cutaway", _cutaway_description(context_metadata))
    _field_if_value(render_fields, "Artifact Settings", _format_artifact_settings(artifact_settings))
    _field_if_value(
        render_fields,
        "Normalization",
        None if consistency_metrics.get("normalization_method") is None else str(consistency_metrics.get("normalization_method")),
    )
    normalization_reference_value = _coerce_float(consistency_metrics.get("normalization_reference_value"))
    normalization_reference_percentile = _coerce_float(consistency_metrics.get("normalization_reference_percentile"))
    if normalization_reference_value is not None:
        if normalization_reference_percentile is None:
            render_fields.append(InspectorField("Normalization Reference", f"{normalization_reference_value:.4f}"))
        else:
            render_fields.append(
                InspectorField(
                    "Normalization Reference",
                    f"p{normalization_reference_percentile:.1f} = {normalization_reference_value:.4f}",
                )
            )
    _field_if_value(render_fields, "Export Filename Hint", screenshot_name_hint)

    return [
        InspectorSection("Preset", tuple(preset_fields)),
        InspectorSection("Pose", tuple(pose_fields)),
        InspectorSection("Anatomy In Fan", tuple(anatomy_fields)),
        InspectorSection("Review / Eval", tuple(review_fields)),
        InspectorSection("Render Settings", tuple(render_fields)),
    ]


def format_inspector_sections_text(sections: Sequence[InspectorSection]) -> str:
    blocks: list[str] = []
    for section in sections:
        lines = [section.title]
        for field in section.fields:
            field_lines = field.value.splitlines() or [field.value]
            if len(field_lines) == 1:
                lines.append(f"- {field.label}: {field_lines[0]}")
                continue
            lines.append(f"- {field.label}:")
            lines.extend(f"  {line}" for line in field_lines)
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def render_inspector_sections_html(sections: Sequence[InspectorSection]) -> str:
    parts = [
        "<html><body style=\"font-family: 'Segoe UI', sans-serif; font-size: 12px; color: #e8e8e8; "
        "background-color: transparent;\">"
    ]
    for section in sections:
        parts.append(
            "<div style=\"margin-bottom: 14px; padding: 10px; border: 1px solid #303030; "
            "border-radius: 8px; background-color: #181818;\">"
        )
        parts.append(
            f"<div style=\"font-size: 13px; font-weight: 600; margin-bottom: 8px; color: #f2f2f2;\">{escape(section.title)}</div>"
        )
        parts.append("<table cellspacing=\"0\" cellpadding=\"0\" style=\"width: 100%;\">")
        for field in section.fields:
            value_lines = field.value.splitlines() or [field.value]
            value_html = "<br/>".join(escape(line) for line in value_lines)
            parts.append(
                "<tr>"
                f"<td style=\"vertical-align: top; width: 42%; padding: 2px 10px 4px 0; color: #9ea3ad; font-weight: 600;\">{escape(field.label)}</td>"
                f"<td style=\"vertical-align: top; padding: 2px 0 4px 0; color: #f6f7f9;\">{value_html}</td>"
                "</tr>"
            )
        parts.append("</table></div>")
    parts.append("</body></html>")
    return "".join(parts)


def build_render_summary_text(
    sector_metadata: Mapping[str, object],
    context_metadata: Mapping[str, object],
    *,
    station_label: str | None = None,
    node_label: str | None = None,
    review_metrics: Mapping[str, object] | None = None,
    flag_reasons: Sequence[str] | None = None,
    warnings: Sequence[str] | None = None,
    screenshot_name_hint: str | None = None,
) -> str:
    sections = build_render_inspector_sections(
        sector_metadata,
        context_metadata,
        station_label=station_label,
        node_label=node_label,
        review_metrics=review_metrics,
        flag_reasons=flag_reasons,
        warnings=warnings,
        screenshot_name_hint=screenshot_name_hint,
    )
    return format_inspector_sections_text(sections)
