from __future__ import annotations

import argparse

from ebus_simulator.video_reference import build_reference_library


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a de-identified station reference keyframe library.")
    parser.add_argument("reference_config", help="Path to video reference YAML config.")
    parser.add_argument("--output-dir", required=True, help="Directory for reference keyframes and reference_library.json.")
    parser.add_argument("--frame-size-px", type=int, help="Maximum output keyframe dimension in pixels.")
    parser.add_argument("--overwrite", action="store_true", help="Re-extract keyframes that already exist.")
    args = parser.parse_args()

    library = build_reference_library(
        args.reference_config,
        output_dir=args.output_dir,
        frame_size_px=args.frame_size_px,
        overwrite=args.overwrite,
    )
    print(f"output_dir: {library.output_dir}")
    print(f"video_count: {len(library.videos)}")
    print(f"keyframe_count: {len(library.keyframes)}")
    print(f"warnings: {len(library.warnings)}")
    for warning in library.warnings:
        print(f"  - {warning}")
    print(f"library_json: {library.output_dir}/reference_library.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
