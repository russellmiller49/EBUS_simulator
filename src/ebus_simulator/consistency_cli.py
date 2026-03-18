from __future__ import annotations

import argparse

from ebus_simulator.consistency import analyze_render_consistency
from ebus_simulator.render_profiles import DEFAULT_PHYSICS_PROFILE_NAME, list_physics_profile_names
from ebus_simulator.render_cli import _parse_bool, _parse_vessel_overlays


def main() -> int:
    available_profiles = ", ".join(list_physics_profile_names())
    parser = argparse.ArgumentParser(description="Analyze cross-preset render consistency for localizer and physics outputs.")
    parser.add_argument("manifest", help="Path to the case manifest YAML file.")
    parser.add_argument("--output-dir", required=True, help="Directory for rendered presets and consistency summaries.")
    parser.add_argument("--width", type=int, default=128, help="Per-render output width in pixels.")
    parser.add_argument("--height", type=int, default=128, help="Per-render output height in pixels.")
    parser.add_argument("--device", default="bf_uc180f", help="CP-EBUS device model key. Default: bf_uc180f.")
    parser.add_argument("--roll-deg", type=float, help="Optional global fine roll in degrees before per-preset overrides.")
    parser.add_argument("--sector-angle-deg", type=float, help="Optional sector angle override in degrees.")
    parser.add_argument("--max-depth-mm", type=float, help="Optional displayed fan depth override in millimeters.")
    parser.add_argument("--reference-fov-mm", type=float, help="Optional reference/localizer field-of-view override in millimeters.")
    parser.add_argument("--slice-thickness-mm", type=float, help="Optional slab thickness override in millimeters.")
    parser.add_argument("--cutaway-mode", choices=["lateral", "probe_axis", "shaft_axis"], default=None, help="3D context airway cutaway plane mode.")
    parser.add_argument("--cutaway-side", choices=["auto", "left", "right"], default=None, help="Requested cutaway opening side.")
    parser.add_argument("--cutaway-depth-mm", type=float, help="Additional cutaway depth in millimeters.")
    parser.add_argument("--cutaway-origin", choices=["contact", "probe_origin", "custom"], default=None, help="Reference origin used to derive the cutaway plane.")
    parser.add_argument("--show-full-airway", type=_parse_bool, default=None, help="Show the full smoothed airway mesh instead of clipping it in the context panel.")
    parser.add_argument("--overlay-vessels", type=_parse_vessel_overlays, help="Optional comma-separated vessel overlay override for all analysis renders.")
    parser.add_argument("--preset-id", action="append", dest="preset_ids", help="Optional preset identifier filter. Repeat to analyze multiple presets.")
    parser.add_argument(
        "--physics-profile",
        help=(
            "Optional physics tuning profile name. "
            f"Defaults to {DEFAULT_PHYSICS_PROFILE_NAME}. Available: {available_profiles}."
        ),
    )
    parser.add_argument("--physics-speckle-strength", type=float, help="Optional physics speckle strength override.")
    parser.add_argument("--physics-reverberation-strength", type=float, help="Optional physics reverberation strength override.")
    parser.add_argument("--physics-shadow-strength", type=float, help="Optional physics distal shadow strength override.")
    args = parser.parse_args()

    summary = analyze_render_consistency(
        args.manifest,
        output_dir=args.output_dir,
        width=args.width,
        height=args.height,
        device=args.device,
        roll_deg=args.roll_deg,
        sector_angle_deg=args.sector_angle_deg,
        max_depth_mm=args.max_depth_mm,
        reference_fov_mm=args.reference_fov_mm,
        slice_thickness_mm=args.slice_thickness_mm,
        cutaway_mode=args.cutaway_mode,
        cutaway_side=args.cutaway_side,
        cutaway_depth_mm=args.cutaway_depth_mm,
        cutaway_origin=args.cutaway_origin,
        show_full_airway=args.show_full_airway,
        vessel_overlay_names=args.overlay_vessels,
        preset_ids=args.preset_ids,
        physics_profile=args.physics_profile,
        physics_speckle_strength=args.physics_speckle_strength,
        physics_reverberation_strength=args.physics_reverberation_strength,
        physics_shadow_strength=args.physics_shadow_strength,
    )

    print(f"case_id: {summary['case_id']}")
    print(f"output_dir: {summary['output_dir']}")
    print(f"analysis_count: {summary['analysis_count']}")
    print(f"summary_json: {summary['output_dir']}/consistency_summary.json")
    print(f"summary_csv: {summary['output_dir']}/consistency_entries.csv")
    print(f"summary_md: {summary['output_dir']}/consistency_summary.md")
    print(f"most_divergent_presets: {summary['most_divergent_presets']}")
    print(f"representative_cases: {summary['representative_cases']}")
    print(f"heuristic_breakdown: {summary['heuristic_breakdown']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
