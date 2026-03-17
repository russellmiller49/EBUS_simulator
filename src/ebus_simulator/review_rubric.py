from __future__ import annotations

from pathlib import Path


RUBRIC_ITEMS = (
    "Airway wall plausibility",
    "Vessel wall / lumen plausibility",
    "Node conspicuity",
    "Overall resemblance to CP-EBUS",
)


def render_review_rubric_template() -> str:
    lines = [
        "# CP-EBUS Review Rubric",
        "",
        "Use this sheet for descriptive human review only.",
        "Do not treat it as an automated scoring engine.",
        "",
        "| Criterion | Rating (1-5) | Notes |",
        "|---|---|---|",
    ]
    for item in RUBRIC_ITEMS:
        lines.append(f"| {item} |  |  |")
    lines.extend(
        [
            "",
            "## Freeform Comments",
            "",
            "- strengths:",
            "- weaknesses:",
            "- follow-up tuning ideas:",
            "",
        ]
    )
    return "\n".join(lines)


def render_review_sheet(
    *,
    preset_id: str,
    approach: str,
    localizer_panel_path: Path,
    localizer_clean_path: Path,
    physics_path: Path,
    eval_summary_path: Path,
    review_entry_path: Path,
    warnings: list[str],
    flag_reasons: list[str],
) -> str:
    lines = [
        f"# Review Sheet: {preset_id} / {approach}",
        "",
        "## Assets",
        "",
        f"- localizer diagnostic panel: [{localizer_panel_path.name}]({localizer_panel_path.name})",
        f"- localizer clean render: [{localizer_clean_path.name}]({localizer_clean_path.name})",
        f"- physics render: [{physics_path.name}]({physics_path.name})",
        f"- eval summary: [{eval_summary_path.name}]({eval_summary_path.name})",
        f"- review entry: [{review_entry_path.name}]({review_entry_path.name})",
        "",
        "## Warnings",
        "",
    ]
    if warnings:
        lines.extend(f"- {warning}" for warning in warnings)
    else:
        lines.append("- none")

    lines.extend(
        [
            "",
            "## Auto Flags",
            "",
        ]
    )
    if flag_reasons:
        lines.extend(f"- {reason}" for reason in flag_reasons)
    else:
        lines.append("- none")

    lines.extend(
        [
            "",
            "## Rubric",
            "",
            "| Criterion | Rating (1-5) | Notes |",
            "|---|---|---|",
        ]
    )
    for item in RUBRIC_ITEMS:
        lines.append(f"| {item} |  |  |")
    lines.extend(
        [
            "",
            "## Comments",
            "",
            "- ",
            "",
        ]
    )
    return "\n".join(lines)
