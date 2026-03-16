from __future__ import annotations

import argparse
import json
from pathlib import Path

from ebus_simulator.validation import report_to_dict, validate_case


def _format_summary(report_dict: dict) -> str:
    lines = [
        f"case_id: {report_dict['case_id']}",
        f"manifest: {report_dict['manifest_path']}",
        f"dataset_root: {report_dict['dataset_root']}",
        f"internal_world_frame: {report_dict['internal_world_frame']}",
        f"status: {report_dict['status']}",
        f"preset_count: {report_dict['preset_count']}",
        "ct:",
        f"  path: {report_dict['ct']['path']}",
        f"  shape: {tuple(report_dict['ct']['shape'])}",
        f"  voxel_sizes_mm: {tuple(report_dict['ct']['voxel_sizes_mm'])}",
        f"  axis_codes_ras: {tuple(report_dict['ct']['axis_codes_ras'])}",
        "centerlines:",
        f"  main_path: {report_dict['centerlines']['main_path']}",
        f"  network_path: {report_dict['centerlines']['network_path']}",
        f"  main_point_count: {report_dict['centerlines']['main_point_count']}",
        f"  main_line_count: {report_dict['centerlines']['main_line_count']}",
        f"  main_segment_count: {report_dict['centerlines']['main_segment_count']}",
        f"  network_point_count: {report_dict['centerlines']['network_point_count']}",
        f"  network_line_count: {report_dict['centerlines']['network_line_count']}",
        "meshes:",
        f"  raw_path: {report_dict['meshes']['raw'].get('path')}",
        f"  raw_present: {report_dict['meshes']['raw'].get('present')}",
        f"  raw_space: {report_dict['meshes']['raw'].get('source_space')}",
        f"  raw_point_count: {report_dict['meshes']['raw'].get('point_count')}",
        f"  raw_triangle_count: {report_dict['meshes']['raw'].get('triangle_count')}",
        f"  raw_inside_ct_fraction: {report_dict['meshes']['raw'].get('alignment_identity', {}).get('inside_ct_fraction')}",
        f"  raw_inside_lumen_fraction: {report_dict['meshes']['raw'].get('alignment_identity', {}).get('inside_lumen_fraction')}",
        f"  centerline_to_raw_mesh_distance_mm: {report_dict['meshes']['alignment'].get('centerline_to_raw_mesh_distance_mm')}",
        f"  contact_to_raw_mesh_distance_mm: {report_dict['meshes']['alignment'].get('contact_to_raw_mesh_distance_mm')}",
        f"  target_to_raw_mesh_signed_distance_mm: {report_dict['meshes']['alignment'].get('target_to_raw_mesh_signed_distance_mm')}",
        "presets:",
    ]

    for preset in report_dict["presets"]:
        lines.append(
            f"  - {preset['id']}: status={preset['status']} target_inside_ct={preset['target_inside_ct_bounds']} "
            f"target_inside_station={preset['target_inside_station_mask']} target_station_distance_mm={preset['target_station_distance_mm']:.3f}"
        )
        for contact in preset["contacts"]:
            airway_distance = "n/a" if contact["airway_surface_distance_mm"] is None else f"{contact['airway_surface_distance_mm']:.3f}"
            raw_mesh_distance = "n/a" if contact["raw_mesh_distance_mm"] is None else f"{contact['raw_mesh_distance_mm']:.3f}"
            centerline_distance = (
                "n/a"
                if contact["centerline_projection_distance_mm"] is None
                else f"{contact['centerline_projection_distance_mm']:.3f}"
            )
            lines.append(
                f"    contact[{contact['approach']}]: inside_ct={contact['inside_ct_bounds']} "
                f"airway_surface_distance_mm={airway_distance} "
                f"raw_mesh_distance_mm={raw_mesh_distance} "
                f"centerline_projection_distance_mm={centerline_distance} "
                f"tangent_defined={contact['tangent_defined']}"
            )

    warnings = [issue for issue in report_dict["issues"] if issue["severity"] == "warning"]
    errors = [issue for issue in report_dict["issues"] if issue["severity"] == "error"]

    lines.append(f"errors: {len(errors)}")
    for issue in errors:
        lines.append(
            f"  - {issue['message']}"
            + (f" preset={issue['preset_id']}" if issue["preset_id"] else "")
            + (f" approach={issue['approach']}" if issue["approach"] else "")
            + (f" path={issue['path']}" if issue["path"] else "")
        )

    lines.append(f"warnings: {len(warnings)}")
    for issue in warnings:
        lines.append(
            f"  - {issue['message']}"
            + (f" preset={issue['preset_id']}" if issue["preset_id"] else "")
            + (f" approach={issue['approach']}" if issue["approach"] else "")
            + (f" path={issue['path']}" if issue["path"] else "")
        )

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a linear EBUS case manifest.")
    parser.add_argument("manifest", help="Path to the case manifest YAML file.")
    parser.add_argument("--report-json", help="Optional path to write a machine-readable JSON report.")
    args = parser.parse_args()

    report = validate_case(args.manifest)
    report_dict = report_to_dict(report)

    print(_format_summary(report_dict))

    if args.report_json:
        output_path = Path(args.report_json).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report_dict, indent=2))
        print(f"json_report: {output_path}")

    return 0 if report.status != "failed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
