from __future__ import annotations

import argparse

from ebus_simulator.render_cli import _parse_bool, _parse_vessel_overlays
from ebus_simulator.rendering import render_all_presets


def main() -> int:
    parser = argparse.ArgumentParser(description="Render all configured presets/contact approaches.")
    parser.add_argument("manifest", help="Path to the case manifest YAML file.")
    parser.add_argument("--output-dir", required=True, help="Directory for PNGs, sidecars, and index files.")
    parser.add_argument("--width", type=int, help="Output image width in pixels.")
    parser.add_argument("--height", type=int, help="Output image height in pixels.")
    parser.add_argument("--sector-angle-deg", type=float, help="Sector angle in degrees.")
    parser.add_argument("--max-depth-mm", type=float, help="Maximum render depth in millimeters.")
    parser.add_argument("--roll-deg", type=float, help="Optional fine roll in degrees.")
    parser.add_argument("--mode", choices=["clean", "debug"], default=None, help="Render mode preset for overlay defaults.")
    parser.add_argument("--overlay-airway", type=_parse_bool, help="Enable or disable the airway overlay.")
    parser.add_argument("--overlay-target", type=_parse_bool, help="Enable or disable the target marker overlay.")
    parser.add_argument("--overlay-station", type=_parse_bool, help="Enable or disable the station mask overlay.")
    parser.add_argument("--overlay-vessels", type=_parse_vessel_overlays, help="Comma-separated vessel overlay keys, or 'none'.")
    args = parser.parse_args()

    index = render_all_presets(
        args.manifest,
        output_dir=args.output_dir,
        width=args.width,
        height=args.height,
        sector_angle_deg=args.sector_angle_deg,
        max_depth_mm=args.max_depth_mm,
        roll_deg=args.roll_deg,
        mode=args.mode,
        airway_overlay=args.overlay_airway,
        target_overlay=args.overlay_target,
        station_overlay=args.overlay_station,
        vessel_overlay_names=args.overlay_vessels,
    )

    print(f"case_id: {index.case_id}")
    print(f"mode: {index.mode}")
    print(f"output_dir: {index.output_dir}")
    print(f"render_count: {index.render_count}")
    print(f"index_json: {index.output_dir}/index.json")
    print(f"index_csv: {index.output_dir}/index.csv")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
