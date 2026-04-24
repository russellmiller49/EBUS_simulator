from __future__ import annotations

import argparse

from ebus_simulator.render_engines import RenderEngine
from ebus_simulator.rendering import render_preset


def _parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes", "on"}:
        return True
    if normalized in {"false", "0", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected true or false, got {value!r}.")


def _parse_vessel_overlays(value: str) -> list[str]:
    normalized = value.strip()
    if not normalized or normalized.lower() == "none":
        return []
    return [item.strip() for item in normalized.split(",") if item.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description="Render a single linear EBUS preset to PNG.")
    parser.add_argument("manifest", help="Path to the case manifest YAML file.")
    parser.add_argument("preset_id", help="Preset identifier to render.")
    parser.add_argument("--approach", help="Contact approach key for presets with multiple contacts.")
    parser.add_argument("--output", required=True, help="PNG output path.")
    parser.add_argument("--metadata-json", help="Optional JSON sidecar path. Defaults to <output>.json.")
    parser.add_argument("--engine", choices=[engine.value for engine in RenderEngine], default=RenderEngine.LOCALIZER.value, help="Render engine. Default: localizer.")
    parser.add_argument("--seed", type=int, help="Optional deterministic seed recorded in render metadata.")
    parser.add_argument("--width", type=int, help="Output image width in pixels.")
    parser.add_argument("--height", type=int, help="Output image height in pixels.")
    parser.add_argument("--sector-angle-deg", type=float, help="Sector angle in degrees.")
    parser.add_argument("--max-depth-mm", type=float, help="Maximum render depth in millimeters.")
    parser.add_argument("--roll-deg", type=float, help="Optional fine roll in degrees.")
    parser.add_argument("--mode", choices=["clean", "debug"], default=None, help="Render mode preset for overlay defaults.")
    parser.add_argument("--diagnostic-panel", action="store_true", help="Export a 2x2 diagnostic panel instead of a single sector image.")
    parser.add_argument("--device", default="bf_uc180f", help="CP-EBUS device model key. Default: bf_uc180f.")
    parser.add_argument("--refine-contact", type=_parse_bool, default=True, help="Enable or disable airway-wall contact refinement.")
    parser.add_argument("--virtual-ebus", type=_parse_bool, default=True, help="Enable or disable the virtual EBUS structural/debug view.")
    parser.add_argument("--simulated-ebus", type=_parse_bool, default=True, help="Enable or disable the simulated EBUS grayscale view.")
    parser.add_argument("--reference-fov-mm", type=float, help="Wide CT localizer field of view in millimeters.")
    parser.add_argument("--source-oblique-size-mm", type=float, help="Source oblique CP-EBUS section size in millimeters.")
    parser.add_argument("--overlay-airway", type=_parse_bool, help="Enable or disable both airway lumen and wall overlays.")
    parser.add_argument("--overlay-airway-lumen", type=_parse_bool, help="Enable or disable the airway lumen contour overlay.")
    parser.add_argument("--overlay-airway-wall", type=_parse_bool, help="Enable or disable the airway wall contour overlay.")
    parser.add_argument("--overlay-target", type=_parse_bool, help="Enable or disable the target marker overlay.")
    parser.add_argument("--overlay-contact", type=_parse_bool, help="Enable or disable the contact/apex marker overlay.")
    parser.add_argument("--overlay-station", type=_parse_bool, help="Enable or disable the station mask overlay.")
    parser.add_argument("--overlay-vessels", type=_parse_vessel_overlays, help="Comma-separated vessel overlay keys, or 'none'.")
    parser.add_argument("--single-vessel", help="Enable single-vessel mode for exactly one named vessel overlay.")
    parser.add_argument("--show-legend", action=argparse.BooleanOptionalAction, default=None, help="Show or hide the virtual-view legend.")
    parser.add_argument("--label-overlays", action=argparse.BooleanOptionalAction, default=None, help="Show or hide direct contour labels.")
    parser.add_argument("--show-contact", action=argparse.BooleanOptionalAction, default=None, help="Show or hide the refined contact/apex marker.")
    parser.add_argument("--show-frustum", action=argparse.BooleanOptionalAction, default=None, help="Show or hide the sector frustum in the 3D context view.")
    parser.add_argument("--min-contour-area-px", type=float, default=20.0, help="Suppress contour fragments smaller than this area.")
    parser.add_argument("--min-contour-length-px", type=float, default=15.0, help="Suppress contour fragments shorter than this contour length.")
    parser.add_argument("--slice-thickness-mm", type=float, help="Debug/reference slab thickness in millimeters.")
    parser.add_argument("--cutaway-mode", choices=["lateral", "probe_axis", "shaft_axis"], default=None, help="3D context airway cutaway plane mode.")
    parser.add_argument("--cutaway-side", choices=["auto", "left", "right"], default=None, help="Requested cutaway opening side for the 3D context airway cutaway.")
    parser.add_argument("--cutaway-depth-mm", type=float, help="Additional cutaway depth in millimeters along the selected cutaway normal.")
    parser.add_argument("--cutaway-origin", choices=["contact", "probe_origin", "custom"], default=None, help="Reference origin used to derive the cutaway plane.")
    parser.add_argument("--show-full-airway", type=_parse_bool, default=None, help="Show the full smoothed airway mesh instead of clipping it for the 3D context panel.")
    parser.add_argument("--debug-map-dir", help="Optional directory for engine-specific debug map exports.")
    parser.add_argument("--speckle-strength", type=float, help="Optional physics speckle strength override.")
    parser.add_argument("--reverberation-strength", type=float, help="Optional physics reverberation strength override.")
    parser.add_argument("--shadow-strength", type=float, help="Optional physics distal shadow strength override.")
    parser.add_argument("--physics-profile", help="Physics appearance profile name or YAML/JSON path. Default: review_realistic_v1.")
    args = parser.parse_args()

    rendered = render_preset(
        args.manifest,
        args.preset_id,
        approach=args.approach,
        output_path=args.output,
        metadata_path=args.metadata_json,
        engine=args.engine,
        seed=args.seed,
        width=args.width,
        height=args.height,
        sector_angle_deg=args.sector_angle_deg,
        max_depth_mm=args.max_depth_mm,
        roll_deg=args.roll_deg,
        mode=args.mode,
        diagnostic_panel=args.diagnostic_panel,
        device=args.device,
        refine_contact=args.refine_contact,
        virtual_ebus=args.virtual_ebus,
        simulated_ebus=args.simulated_ebus,
        reference_fov_mm=args.reference_fov_mm,
        source_oblique_size_mm=args.source_oblique_size_mm,
        airway_overlay=args.overlay_airway,
        airway_lumen_overlay=args.overlay_airway_lumen,
        airway_wall_overlay=args.overlay_airway_wall,
        target_overlay=args.overlay_target,
        contact_overlay=args.overlay_contact,
        station_overlay=args.overlay_station,
        vessel_overlay_names=args.overlay_vessels,
        single_vessel=args.single_vessel,
        show_legend=args.show_legend,
        label_overlays=args.label_overlays,
        min_contour_area_px=args.min_contour_area_px,
        min_contour_length_px=args.min_contour_length_px,
        show_contact=args.show_contact,
        show_frustum=args.show_frustum,
        slice_thickness_mm=args.slice_thickness_mm,
        cutaway_mode=args.cutaway_mode,
        cutaway_side=args.cutaway_side,
        cutaway_depth_mm=args.cutaway_depth_mm,
        cutaway_origin=args.cutaway_origin,
        show_full_airway=args.show_full_airway,
        debug_map_dir=args.debug_map_dir,
        speckle_strength=args.speckle_strength,
        reverberation_strength=args.reverberation_strength,
        shadow_strength=args.shadow_strength,
        physics_profile=args.physics_profile,
    )

    print(f"preset_id: {rendered.metadata.preset_id}")
    print(f"approach: {rendered.metadata.approach}")
    print(f"engine: {rendered.metadata.engine}")
    print(f"engine_version: {rendered.metadata.engine_version}")
    print(f"view_kind: {rendered.metadata.view_kind}")
    print(f"mode: {rendered.metadata.mode}")
    print(f"diagnostic_panel: {rendered.metadata.diagnostic_panel_enabled}")
    print(f"device_model: {rendered.metadata.device_model}")
    print(f"output: {rendered.metadata.output_path}")
    print(f"metadata_json: {rendered.metadata.metadata_path}")
    print(f"image_size: {tuple(rendered.metadata.image_size)}")
    print(f"sector_angle_deg: {rendered.metadata.sector_angle_deg}")
    print(f"max_depth_mm: {rendered.metadata.max_depth_mm}")
    print(f"roll_deg: {rendered.metadata.roll_deg}")
    print(f"slice_thickness_mm: {rendered.metadata.slice_thickness_mm}")
    print(f"source_oblique_size_mm: {rendered.metadata.source_oblique_size_mm}")
    print(f"reference_fov_mm: {rendered.metadata.reference_fov_mm}")
    print(f"display_plane: {rendered.metadata.display_plane}")
    print(f"reference_plane: {rendered.metadata.reference_plane}")
    print(f"original_contact_world: {rendered.metadata.original_contact_world}")
    print(f"refined_contact_world: {rendered.metadata.refined_contact_world}")
    print(f"cutaway_mode: {rendered.metadata.cutaway_mode}")
    print(f"cutaway_side: {rendered.metadata.cutaway_side}")
    print(f"cutaway_open_side: {rendered.metadata.cutaway_open_side}")
    print(f"cutaway_origin_mode: {rendered.metadata.cutaway_origin_mode}")
    print(f"cutaway_origin: {rendered.metadata.cutaway_origin}")
    print(f"cutaway_normal: {rendered.metadata.cutaway_normal}")
    print(f"cutaway_mesh_source: {rendered.metadata.cutaway_mesh_source}")
    print(f"show_full_airway: {rendered.metadata.show_full_airway}")
    print(f"overlays_enabled: {rendered.metadata.overlays_enabled}")
    if rendered.metadata.engine_diagnostics:
        print(f"engine_diagnostics: {sorted(rendered.metadata.engine_diagnostics.keys())}")
    print(f"warnings: {len(rendered.metadata.warnings)}")
    for warning in rendered.metadata.warnings:
        print(f"  - {warning}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
