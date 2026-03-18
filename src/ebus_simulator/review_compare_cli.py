from __future__ import annotations

import argparse

from ebus_simulator.review import compare_review_bundle_files


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare two review-presets bundle summaries and write before/after calibration artifacts."
    )
    parser.add_argument("before_summary", help="Path to the baseline review_summary.json file.")
    parser.add_argument("after_summary", help="Path to the updated review_summary.json file.")
    parser.add_argument(
        "--output-dir",
        help="Directory for before_after_summary.{json,csv,md}. Defaults to the updated bundle directory.",
    )
    args = parser.parse_args()

    comparison = compare_review_bundle_files(
        args.before_summary,
        args.after_summary,
        output_dir=args.output_dir,
    )

    print(f"before_summary_json: {comparison['before_summary_json']}")
    print(f"after_summary_json: {comparison['after_summary_json']}")
    print(f"matched_entry_count: {comparison['matched_entry_count']}")
    print(f"before_flagged_count: {comparison['before_flagged_count']}")
    print(f"after_flagged_count: {comparison['after_flagged_count']}")
    print(f"resolved_flagged_count: {comparison['resolved_flagged_count']}")
    print(f"regressed_flagged_count: {comparison['regressed_flagged_count']}")
    print(f"comparison_json: {comparison['comparison_json']}")
    print(f"comparison_csv: {comparison['comparison_csv']}")
    print(f"comparison_md: {comparison['comparison_md']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
