from __future__ import annotations

from csv import DictWriter
import json
from pathlib import Path
from typing import Mapping

from ebus_simulator.rendering import build_render_context, render_preset


TARGET_PROMINENCE_DELTA_WARN = 0.12
OCCUPANCY_DELTA_WARN = 0.18
BRIGHTNESS_DELTA_WARN = 0.14
TARGET_EDGE_OFFSET_WARN = 0.85
NEAR_FIELD_WALL_WARN = 0.15
EMPTY_SECTOR_WARN = 0.78
NORMALIZATION_TAIL_RATIO_WARN = 1.22


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


def _metric(metrics: Mapping[str, object], key: str) -> float | None:
    value = metrics.get(key)
    if value is None:
        return None
    return float(value)


def _delta(left: Mapping[str, object], right: Mapping[str, object], key: str) -> float | None:
    left_value = _metric(left, key)
    right_value = _metric(right, key)
    if left_value is None or right_value is None:
        return None
    return abs(left_value - right_value)


def _entry_key(entry: Mapping[str, object]) -> tuple[str, str]:
    return str(entry["preset_id"]), str(entry["approach"])


def _occupancy_gap(localizer_metrics: Mapping[str, object], physics_metrics: Mapping[str, object]) -> float | None:
    return _delta(localizer_metrics, physics_metrics, "non_background_occupancy_fraction")


def _entry_metrics(entry: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = entry.get(key)
    return value if isinstance(value, Mapping) else {}


def _tail_ratio(metrics: Mapping[str, object]) -> float | None:
    reference_value = _metric(metrics, "normalization_reference_value")
    aux_value = _metric(metrics, "normalization_aux_value")
    if reference_value is None or aux_value is None or aux_value <= 0.0:
        return None
    return float(reference_value / aux_value)


def _entry_reasons(
    *,
    localizer_metrics: Mapping[str, object],
    physics_metrics: Mapping[str, object],
) -> list[str]:
    reasons: list[str] = []

    target_prominence_delta = _delta(localizer_metrics, physics_metrics, "target_region_contrast_vs_sector")
    if target_prominence_delta is not None and target_prominence_delta >= TARGET_PROMINENCE_DELTA_WARN:
        reasons.append(f"localizer/physics target prominence delta {target_prominence_delta:.3f}")

    occupancy_delta = _delta(localizer_metrics, physics_metrics, "non_background_occupancy_fraction")
    if occupancy_delta is not None and occupancy_delta >= OCCUPANCY_DELTA_WARN:
        reasons.append(f"localizer/physics occupancy delta {occupancy_delta:.3f}")

    brightness_delta = _delta(localizer_metrics, physics_metrics, "sector_brightness_mean")
    if brightness_delta is not None and brightness_delta >= BRIGHTNESS_DELTA_WARN:
        reasons.append(f"localizer/physics brightness delta {brightness_delta:.3f}")

    target_offset_fraction = _metric(physics_metrics, "target_centerline_offset_fraction")
    if target_offset_fraction is not None and target_offset_fraction >= TARGET_EDGE_OFFSET_WARN:
        reasons.append(f"target near sector edge ({target_offset_fraction:.3f})")

    near_field_wall = _metric(physics_metrics, "near_field_wall_occupancy_fraction")
    target_contrast = _metric(physics_metrics, "target_region_contrast_vs_sector")
    if near_field_wall is not None and near_field_wall >= NEAR_FIELD_WALL_WARN and (target_contrast is None or target_contrast <= 0.02):
        reasons.append(f"near-field wall occupancy high ({near_field_wall:.3f})")

    empty_sector = _metric(physics_metrics, "empty_sector_fraction")
    if empty_sector is not None and empty_sector >= EMPTY_SECTOR_WARN:
        reasons.append(f"sector mostly empty ({empty_sector:.3f})")

    tail_ratio = _tail_ratio(physics_metrics)
    if tail_ratio is not None and tail_ratio >= NORMALIZATION_TAIL_RATIO_WARN:
        reasons.append(f"physics normalization upper tail ratio {tail_ratio:.3f}")

    return reasons


def _divergence_score(localizer_metrics: Mapping[str, object], physics_metrics: Mapping[str, object]) -> float:
    target_prominence_delta = _delta(localizer_metrics, physics_metrics, "target_region_contrast_vs_sector") or 0.0
    occupancy_delta = _delta(localizer_metrics, physics_metrics, "non_background_occupancy_fraction") or 0.0
    brightness_delta = _delta(localizer_metrics, physics_metrics, "sector_brightness_mean") or 0.0
    return float(target_prominence_delta + (0.75 * occupancy_delta) + (0.5 * brightness_delta))


def _reduced_entry(entry: Mapping[str, object]) -> dict[str, object]:
    return {
        "preset_id": entry["preset_id"],
        "approach": entry["approach"],
        "divergence_score": entry["divergence_score"],
        "divergence_reasons": list(entry["divergence_reasons"]),
        "physics_png": entry["physics_png"],
        "localizer_png": entry["localizer_png"],
    }


def _reduced_or_none(entry: Mapping[str, object] | None) -> dict[str, object] | None:
    return None if entry is None else _reduced_entry(entry)


def _select_extreme_entry(entries: list[dict[str, object]], key: str, *, reverse: bool = True) -> dict[str, object] | None:
    ranked = [entry for entry in entries if entry.get(key) is not None]
    if not ranked:
        return None
    return max(ranked, key=lambda entry: float(entry[key])) if reverse else min(ranked, key=lambda entry: float(entry[key]))


def _write_summary_csv(summary_path: Path, entries: list[dict[str, object]]) -> None:
    with summary_path.open("w", newline="") as handle:
        writer = DictWriter(
            handle,
            fieldnames=[
                "preset_id",
                "approach",
                "divergence_score",
                "divergence_reasons",
                "localizer_target_region_contrast_vs_sector",
                "physics_target_region_contrast_vs_sector",
                "localizer_non_background_occupancy_fraction",
                "physics_non_background_occupancy_fraction",
                "localizer_sector_brightness_mean",
                "physics_sector_brightness_mean",
                "physics_near_field_wall_occupancy_fraction",
                "physics_empty_sector_fraction",
                "physics_target_centerline_offset_fraction",
                "physics_consistency_bucket",
                "physics_support_logic_active",
                "physics_support_logic_mode",
                "physics_normalization_method",
                "physics_normalization_reference_value",
                "physics_normalization_aux_value",
                "occupancy_gap",
                "localizer_png",
                "physics_png",
            ],
        )
        writer.writeheader()
        for entry in entries:
            writer.writerow(
                {
                    "preset_id": entry["preset_id"],
                    "approach": entry["approach"],
                    "divergence_score": entry["divergence_score"],
                    "divergence_reasons": " | ".join(entry["divergence_reasons"]),
                    "localizer_target_region_contrast_vs_sector": entry["localizer_consistency_metrics"].get("target_region_contrast_vs_sector"),
                    "physics_target_region_contrast_vs_sector": entry["physics_consistency_metrics"].get("target_region_contrast_vs_sector"),
                    "localizer_non_background_occupancy_fraction": entry["localizer_consistency_metrics"].get("non_background_occupancy_fraction"),
                    "physics_non_background_occupancy_fraction": entry["physics_consistency_metrics"].get("non_background_occupancy_fraction"),
                    "localizer_sector_brightness_mean": entry["localizer_consistency_metrics"].get("sector_brightness_mean"),
                    "physics_sector_brightness_mean": entry["physics_consistency_metrics"].get("sector_brightness_mean"),
                    "physics_near_field_wall_occupancy_fraction": entry["physics_consistency_metrics"].get("near_field_wall_occupancy_fraction"),
                    "physics_empty_sector_fraction": entry["physics_consistency_metrics"].get("empty_sector_fraction"),
                    "physics_target_centerline_offset_fraction": entry["physics_consistency_metrics"].get("target_centerline_offset_fraction"),
                    "physics_consistency_bucket": entry["physics_consistency_metrics"].get("consistency_bucket"),
                    "physics_support_logic_active": entry["physics_consistency_metrics"].get("support_logic_active"),
                    "physics_support_logic_mode": entry["physics_consistency_metrics"].get("support_logic_mode"),
                    "physics_normalization_method": entry["physics_consistency_metrics"].get("normalization_method"),
                    "physics_normalization_reference_value": entry["physics_consistency_metrics"].get("normalization_reference_value"),
                    "physics_normalization_aux_value": entry["physics_consistency_metrics"].get("normalization_aux_value"),
                    "occupancy_gap": entry.get("occupancy_gap"),
                    "localizer_png": entry["localizer_png"],
                    "physics_png": entry["physics_png"],
                }
            )


def _write_summary_markdown(summary_path: Path, summary: dict[str, object]) -> None:
    lines = [
        "# Render Consistency Summary",
        "",
        f"- case_id: {summary['case_id']}",
        f"- analysis_count: {summary['analysis_count']}",
        f"- output_dir: {summary['output_dir']}",
        "",
        "## Most Divergent Presets",
        "",
        "| Preset | Approach | Score | Reasons |",
        "|---|---|---|---|",
    ]
    for entry in summary["most_divergent_presets"]:
        lines.append(
            "| {preset} | {approach} | {score:.3f} | {reasons} |".format(
                preset=entry["preset_id"],
                approach=entry["approach"],
                score=float(entry["divergence_score"]),
                reasons="; ".join(entry["divergence_reasons"]) or "none",
            )
        )

    lines.extend(["", "## Representative Cases", ""])
    representative_cases = summary["representative_cases"]
    for label, entry in representative_cases.items():
        lines.append(f"### {label.replace('_', ' ').title()}")
        lines.append("")
        if entry is None:
            lines.append("- none")
        else:
            lines.append(
                f"- `{entry['preset_id']}` / `{entry['approach']}`: "
                f"{'; '.join(entry['divergence_reasons']) or 'no dominant divergence flags'}"
            )
        lines.append("")

        lines.extend(
        [
            "## Heuristic Breakdown",
            "",
            f"- target_prominence_disagreements: {summary['heuristic_breakdown']['target_prominence_disagreements']}",
            f"- occupancy_disagreements: {summary['heuristic_breakdown']['occupancy_disagreements']}",
            f"- brightness_disagreements: {summary['heuristic_breakdown']['brightness_disagreements']}",
            f"- support_logic_activations: {summary['heuristic_breakdown']['support_logic_activations']}",
            f"- edge_target_cases: {summary['heuristic_breakdown']['edge_target_cases']}",
            f"- wall_dominant_cases: {summary['heuristic_breakdown']['wall_dominant_cases']}",
            f"- sparse_sector_cases: {summary['heuristic_breakdown']['sparse_sector_cases']}",
            f"- normalization_tail_cases: {summary['heuristic_breakdown']['normalization_tail_cases']}",
            "",
        ]
    )
    summary_path.write_text("\n".join(lines))


def analyze_render_consistency(
    manifest_path: str | Path,
    *,
    output_dir: str | Path,
    width: int = 128,
    height: int = 128,
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
    physics_speckle_strength: float | None = None,
    physics_reverberation_strength: float | None = None,
    physics_shadow_strength: float | None = None,
) -> dict[str, object]:
    context = build_render_context(manifest_path, roll_deg=roll_deg)
    output_dir = Path(output_dir).expanduser().resolve()
    render_dir = output_dir / "presets"
    render_dir.mkdir(parents=True, exist_ok=True)

    entries: list[dict[str, object]] = []
    for preset_id, approach in _iter_selected_presets(context, preset_ids):
        entry_dir = render_dir / preset_id / approach
        entry_dir.mkdir(parents=True, exist_ok=True)
        localizer_path = entry_dir / "localizer_clean.png"
        physics_path = entry_dir / "physics_clean.png"

        localizer = render_preset(
            context.manifest.manifest_path,
            preset_id,
            approach=approach,
            output_path=localizer_path,
            width=width,
            height=height,
            engine="localizer",
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
            width=width,
            height=height,
            engine="physics",
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
            speckle_strength=physics_speckle_strength,
            reverberation_strength=physics_reverberation_strength,
            shadow_strength=physics_shadow_strength,
            context=context,
        )

        localizer_metrics = dict(localizer.metadata.consistency_metrics)
        physics_metrics = dict(physics.metadata.consistency_metrics)
        reasons = _entry_reasons(localizer_metrics=localizer_metrics, physics_metrics=physics_metrics)

        entries.append(
            {
                "preset_id": preset_id,
                "approach": approach,
                "localizer_png": str(localizer_path),
                "localizer_json": localizer.metadata.metadata_path,
                "physics_png": str(physics_path),
                "physics_json": physics.metadata.metadata_path,
                "localizer_consistency_metrics": localizer_metrics,
                "physics_consistency_metrics": physics_metrics,
                "physics_eval_summary": dict(physics.metadata.engine_diagnostics.get("eval_summary", {})),
                "divergence_score": _divergence_score(localizer_metrics, physics_metrics),
                "divergence_reasons": reasons,
                "target_prominence_delta": _delta(localizer_metrics, physics_metrics, "target_region_contrast_vs_sector"),
                "occupancy_delta": _delta(localizer_metrics, physics_metrics, "non_background_occupancy_fraction"),
                "occupancy_gap": _occupancy_gap(localizer_metrics, physics_metrics),
                "brightness_delta": _delta(localizer_metrics, physics_metrics, "sector_brightness_mean"),
                "physics_tail_ratio": _tail_ratio(physics_metrics),
                "physics_near_field_wall_occupancy_fraction": physics_metrics.get("near_field_wall_occupancy_fraction"),
                "physics_empty_sector_fraction": physics_metrics.get("empty_sector_fraction"),
                "physics_target_centerline_offset_fraction": physics_metrics.get("target_centerline_offset_fraction"),
                "physics_target_region_contrast_vs_sector": physics_metrics.get("target_region_contrast_vs_sector"),
                "physics_sector_brightness_mean": physics_metrics.get("sector_brightness_mean"),
            }
        )

    entries.sort(key=lambda entry: (float(entry["divergence_score"]), entry["preset_id"], entry["approach"]), reverse=True)
    most_divergent_presets = [_reduced_entry(entry) for entry in entries[:5]]

    summary = {
        "manifest_path": str(context.manifest.manifest_path),
        "case_id": context.manifest.case_id,
        "output_dir": str(output_dir),
        "analysis_count": len(entries),
        "preset_filter": ([] if preset_ids is None else list(preset_ids)),
        "physics_settings": {
            "speckle_strength": physics_speckle_strength,
            "reverberation_strength": physics_reverberation_strength,
            "shadow_strength": physics_shadow_strength,
        },
        "most_divergent_presets": most_divergent_presets,
        "representative_cases": {
            "wall_dominant": _reduced_or_none(_select_extreme_entry(entries, "physics_near_field_wall_occupancy_fraction", reverse=True)),
            "target_prominent": _reduced_or_none(_select_extreme_entry(entries, "physics_target_region_contrast_vs_sector", reverse=True)),
            "sparse_dark": _reduced_or_none(_select_extreme_entry(entries, "physics_empty_sector_fraction", reverse=True)),
        },
        "heuristic_breakdown": {
            "target_prominence_disagreements": sum(
                1
                for entry in entries
                if entry["target_prominence_delta"] is not None and float(entry["target_prominence_delta"]) >= TARGET_PROMINENCE_DELTA_WARN
            ),
            "occupancy_disagreements": sum(
                1
                for entry in entries
                if entry["occupancy_delta"] is not None and float(entry["occupancy_delta"]) >= OCCUPANCY_DELTA_WARN
            ),
            "brightness_disagreements": sum(
                1
                for entry in entries
                if entry["brightness_delta"] is not None and float(entry["brightness_delta"]) >= BRIGHTNESS_DELTA_WARN
            ),
            "support_logic_activations": sum(
                1
                for entry in entries
                if bool(entry["physics_consistency_metrics"].get("support_logic_active"))
            ),
            "edge_target_cases": sum(
                1
                for entry in entries
                if entry["physics_target_centerline_offset_fraction"] is not None
                and float(entry["physics_target_centerline_offset_fraction"]) >= TARGET_EDGE_OFFSET_WARN
            ),
            "wall_dominant_cases": sum(
                1
                for entry in entries
                if entry["physics_near_field_wall_occupancy_fraction"] is not None
                and float(entry["physics_near_field_wall_occupancy_fraction"]) >= NEAR_FIELD_WALL_WARN
            ),
            "sparse_sector_cases": sum(
                1
                for entry in entries
                if entry["physics_empty_sector_fraction"] is not None
                and float(entry["physics_empty_sector_fraction"]) >= EMPTY_SECTOR_WARN
            ),
            "normalization_tail_cases": sum(
                1
                for entry in entries
                if entry["physics_tail_ratio"] is not None
                and float(entry["physics_tail_ratio"]) >= NORMALIZATION_TAIL_RATIO_WARN
            ),
        },
        "entries": entries,
    }

    summary_json_path = output_dir / "consistency_summary.json"
    summary_csv_path = output_dir / "consistency_entries.csv"
    summary_md_path = output_dir / "consistency_summary.md"
    summary_json_path.write_text(json.dumps(summary, indent=2))
    _write_summary_csv(summary_csv_path, entries)
    _write_summary_markdown(summary_md_path, summary)
    return summary


def compare_consistency_summaries(
    before_summary: Mapping[str, object],
    after_summary: Mapping[str, object],
) -> dict[str, object]:
    before_entries = {
        _entry_key(entry): entry
        for entry in before_summary.get("entries", [])
        if isinstance(entry, Mapping)
    }
    after_entries = {
        _entry_key(entry): entry
        for entry in after_summary.get("entries", [])
        if isinstance(entry, Mapping)
    }
    matched_keys = sorted(set(before_entries) & set(after_entries))

    rows: list[dict[str, object]] = []
    for key in matched_keys:
        before_entry = before_entries[key]
        after_entry = after_entries[key]
        before_localizer_metrics = _entry_metrics(before_entry, "localizer_consistency_metrics")
        before_physics_metrics = _entry_metrics(before_entry, "physics_consistency_metrics")
        after_localizer_metrics = _entry_metrics(after_entry, "localizer_consistency_metrics")
        after_physics_metrics = _entry_metrics(after_entry, "physics_consistency_metrics")
        rows.append(
            {
                "preset_id": key[0],
                "approach": key[1],
                "before_bucket": before_physics_metrics.get("consistency_bucket"),
                "after_bucket": after_physics_metrics.get("consistency_bucket"),
                "before_support_logic_active": before_physics_metrics.get("support_logic_active", False),
                "after_support_logic_active": after_physics_metrics.get("support_logic_active", False),
                "before_empty_sector_fraction": before_physics_metrics.get("empty_sector_fraction"),
                "after_empty_sector_fraction": after_physics_metrics.get("empty_sector_fraction"),
                "before_non_background_occupancy_fraction": before_physics_metrics.get("non_background_occupancy_fraction"),
                "after_non_background_occupancy_fraction": after_physics_metrics.get("non_background_occupancy_fraction"),
                "before_target_region_contrast_vs_sector": before_physics_metrics.get("target_region_contrast_vs_sector"),
                "after_target_region_contrast_vs_sector": after_physics_metrics.get("target_region_contrast_vs_sector"),
                "before_occupancy_gap": (
                    before_entry.get("occupancy_gap")
                    if before_entry.get("occupancy_gap") is not None
                    else _occupancy_gap(before_localizer_metrics, before_physics_metrics)
                ),
                "after_occupancy_gap": (
                    after_entry.get("occupancy_gap")
                    if after_entry.get("occupancy_gap") is not None
                    else _occupancy_gap(after_localizer_metrics, after_physics_metrics)
                ),
            }
        )

    return {
        "before_case_id": before_summary.get("case_id"),
        "after_case_id": after_summary.get("case_id"),
        "matched_entry_count": len(rows),
        "improved_empty_sector_count": sum(
            1
            for row in rows
            if row["before_empty_sector_fraction"] is not None
            and row["after_empty_sector_fraction"] is not None
            and float(row["after_empty_sector_fraction"]) < float(row["before_empty_sector_fraction"])
        ),
        "improved_non_background_occupancy_count": sum(
            1
            for row in rows
            if row["before_non_background_occupancy_fraction"] is not None
            and row["after_non_background_occupancy_fraction"] is not None
            and float(row["after_non_background_occupancy_fraction"]) > float(row["before_non_background_occupancy_fraction"])
        ),
        "improved_target_contrast_count": sum(
            1
            for row in rows
            if row["before_target_region_contrast_vs_sector"] is not None
            and row["after_target_region_contrast_vs_sector"] is not None
            and float(row["after_target_region_contrast_vs_sector"]) > float(row["before_target_region_contrast_vs_sector"])
        ),
        "improved_occupancy_gap_count": sum(
            1
            for row in rows
            if row["before_occupancy_gap"] is not None
            and row["after_occupancy_gap"] is not None
            and float(row["after_occupancy_gap"]) < float(row["before_occupancy_gap"])
        ),
        "rows": rows,
    }
