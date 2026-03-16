from __future__ import annotations

import argparse
import json
from pathlib import Path

from ebus_simulator.poses import generate_pose_report, pose_report_to_dict


def _format_summary(report_dict: dict) -> str:
    lines = [
        f"case_id: {report_dict['case_id']}",
        f"manifest: {report_dict['manifest_path']}",
        f"dataset_root: {report_dict['dataset_root']}",
        f"internal_world_frame: {report_dict['internal_world_frame']}",
        f"status: {report_dict['status']}",
        f"roll_deg: {report_dict['roll_deg']}",
        f"preset_count: {report_dict['preset_count']}",
        f"approach_count: {report_dict['approach_count']}",
        "centerlines:",
        f"  main_line_count: {report_dict['centerlines']['main_line_count']}",
        f"  main_segment_count: {report_dict['centerlines']['main_segment_count']}",
        f"  network_line_count: {report_dict['centerlines']['network_line_count']}",
        f"  network_segment_count: {report_dict['centerlines']['network_segment_count']}",
        "poses:",
    ]

    warning_count = 0
    error_count = 0
    for pose in report_dict["poses"]:
        warning_count += len(pose["warnings"])
        error_count += len(pose["errors"])
        orthogonality = pose["orthogonality"]
        max_abs_dot = "n/a" if orthogonality is None else f"{orthogonality['max_abs_dot']:.6f}"
        airway_mm = "n/a" if pose["contact_to_airway_distance_mm"] is None else f"{pose['contact_to_airway_distance_mm']:.3f}"
        centerline_mm = (
            "n/a"
            if pose["centerline_projection_distance_mm"] is None
            else f"{pose['centerline_projection_distance_mm']:.3f}"
        )
        lines.append(
            f"  - {pose['preset_id']}[{pose['contact_approach']}]: status={pose['status']} "
            f"forward={pose['target_in_default_forward_hemisphere']} "
            f"airway_mm={airway_mm} "
            f"centerline_mm={centerline_mm} "
            f"max_abs_dot={max_abs_dot}"
        )

    lines.append(f"errors: {error_count}")
    for pose in report_dict["poses"]:
        for error in pose["errors"]:
            lines.append(f"  - {error} preset={pose['preset_id']} approach={pose['contact_approach']}")

    lines.append(f"warnings: {warning_count}")
    for pose in report_dict["poses"]:
        for warning in pose["warnings"]:
            lines.append(f"  - {warning} preset={pose['preset_id']} approach={pose['contact_approach']}")

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate Phase 2 pose report for a linear EBUS case manifest.")
    parser.add_argument("manifest", help="Path to the case manifest YAML file.")
    parser.add_argument("--roll-deg", type=float, default=None, help="Optional fine roll to apply around the shaft axis.")
    parser.add_argument("--report-json", help="Optional path to write a machine-readable JSON report.")
    args = parser.parse_args()

    report = generate_pose_report(args.manifest, roll_deg=args.roll_deg)
    report_dict = pose_report_to_dict(report)

    print(_format_summary(report_dict))

    if args.report_json:
        output_path = Path(args.report_json).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report_dict, indent=2))
        print(f"json_report: {output_path}")

    return 0 if report.status != "failed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
