from __future__ import annotations

import argparse

from ebus_simulator.render_profiles import DEFAULT_PHYSICS_PROFILE_NAME, list_physics_profile_names
from ebus_simulator.render_cli import _parse_bool, _parse_vessel_overlays
from ebus_simulator.review import DEFAULT_REVIEW_THRESHOLDS, ReviewThresholds, review_presets


def _parse_optional_threshold(value: str) -> float | None:
    normalized = value.strip().lower()
    if normalized in {"off", "none", "disable", "disabled"}:
        return None
    return float(value)


def _build_review_thresholds(args: argparse.Namespace) -> ReviewThresholds:
    return ReviewThresholds(
        nUS_delta_deg_from_voxel_baseline=(
            DEFAULT_REVIEW_THRESHOLDS.nUS_delta_deg_from_voxel_baseline
            if args.warn_nus_delta_deg is None
            else args.warn_nus_delta_deg
        ),
        contact_delta_mm_from_voxel_baseline=(
            DEFAULT_REVIEW_THRESHOLDS.contact_delta_mm_from_voxel_baseline
            if args.warn_contact_delta_mm is None
            else args.warn_contact_delta_mm
        ),
        station_overlap_fraction_in_fan=(
            DEFAULT_REVIEW_THRESHOLDS.station_overlap_fraction_in_fan
            if args.warn_station_overlap_fraction is None
            else args.warn_station_overlap_fraction
        ),
        target_contrast_vs_sector_min=(
            DEFAULT_REVIEW_THRESHOLDS.target_contrast_vs_sector_min
            if args.warn_min_target_contrast is None
            else args.warn_min_target_contrast
        ),
        vessel_contrast_vs_sector_max=(
            DEFAULT_REVIEW_THRESHOLDS.vessel_contrast_vs_sector_max
            if args.warn_max_vessel_contrast is None
            else args.warn_max_vessel_contrast
        ),
        wall_contrast_vs_sector_min=(
            DEFAULT_REVIEW_THRESHOLDS.wall_contrast_vs_sector_min
            if not hasattr(args, "warn_min_wall_contrast")
            else args.warn_min_wall_contrast
        ),
    )


def main() -> int:
    available_profiles = ", ".join(list_physics_profile_names())
    parser = argparse.ArgumentParser(description="Render and review all preset/contact approaches with mesh-backed pose metrics.")
    parser.add_argument("manifest", help="Path to the case manifest YAML file.")
    parser.add_argument("--output-dir", required=True, help="Directory for review PNGs, sidecars, summaries, and comparison bundles.")
    parser.add_argument("--width", type=int, default=384, help="Per-tile output width in pixels.")
    parser.add_argument("--height", type=int, default=384, help="Per-tile output height in pixels.")
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
    parser.add_argument("--overlay-vessels", type=_parse_vessel_overlays, help="Optional comma-separated vessel overlay override for all review renders.")
    parser.add_argument("--preset-id", action="append", dest="preset_ids", help="Optional preset identifier filter. Repeat to review multiple presets.")
    parser.add_argument("--physics-debug-maps", action="store_true", help="Bundle physics debug maps into each review entry.")
    parser.add_argument(
        "--reference-manifest",
        help="Optional reference image manifest YAML. When provided, matching reference images are copied into each review entry.",
    )
    parser.add_argument(
        "--physics-profile",
        help=(
            "Optional physics tuning profile name. "
            f"Defaults to {DEFAULT_PHYSICS_PROFILE_NAME}. Available: {available_profiles}."
        ),
    )
    parser.add_argument("--physics-speckle-strength", type=float, help="Optional physics speckle strength override for review renders.")
    parser.add_argument("--physics-reverberation-strength", type=float, help="Optional physics reverberation strength override for review renders.")
    parser.add_argument("--physics-shadow-strength", type=float, help="Optional physics distal shadow strength override for review renders.")
    parser.add_argument("--warn-nus-delta-deg", type=float, help="Override the nUS delta auto-flag threshold in degrees.")
    parser.add_argument("--warn-contact-delta-mm", type=float, help="Override the contact delta auto-flag threshold in millimeters.")
    parser.add_argument("--warn-station-overlap-fraction", type=float, help="Override the minimum station overlap auto-flag threshold.")
    parser.add_argument("--warn-min-target-contrast", type=float, help="Override the minimum target-vs-sector contrast auto-flag threshold.")
    parser.add_argument("--warn-max-vessel-contrast", type=float, help="Override the maximum vessel-vs-sector contrast auto-flag threshold.")
    parser.add_argument(
        "--warn-min-wall-contrast",
        default=argparse.SUPPRESS,
        type=_parse_optional_threshold,
        help=(
            "Minimum wall-vs-sector contrast threshold. "
            "Defaults to the built-in wall threshold; pass 'off' to disable wall auto-flagging."
        ),
    )
    args = parser.parse_args()

    review_thresholds = _build_review_thresholds(args)

    summary = review_presets(
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
        include_physics_debug_maps=args.physics_debug_maps,
        reference_manifest_path=args.reference_manifest,
        physics_profile=args.physics_profile,
        physics_speckle_strength=args.physics_speckle_strength,
        physics_reverberation_strength=args.physics_reverberation_strength,
        physics_shadow_strength=args.physics_shadow_strength,
        review_thresholds=review_thresholds,
    )

    print(f"case_id: {summary['case_id']}")
    print(f"output_dir: {summary['output_dir']}")
    print(f"review_count: {summary['review_count']}")
    print(f"flagged_count: {summary['flagged_count']}")
    print(f"physics_debug_maps: {summary['include_physics_debug_maps']}")
    print(f"reference_manifest: {None if summary['reference_manifest'] is None else summary['reference_manifest']['manifest_path']}")
    print(f"summary_json: {summary['output_dir']}/review_summary.json")
    print(f"summary_csv: {summary['output_dir']}/review_summary.csv")
    print(f"index_json: {summary['output_dir']}/review_index.json")
    print(f"index_csv: {summary['output_dir']}/review_index.csv")
    print(f"index_md: {summary['output_dir']}/review_index.md")
    print(f"rubric_template: {summary['rubric_template']}")
    print(f"thresholds: {summary['thresholds']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
