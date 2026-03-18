#!/usr/bin/env python3
"""
Generate a single-file Markdown ingest document for NLP workflows.

Includes:
- Repo metadata (branch, commit, remotes, status)
- Tracked file inventory
- Text content for tracked files, excluding binary/large assets

Skips:
- Binary files (detected by extension and null-byte heuristic)
- Files larger than a configurable threshold
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import subprocess
from pathlib import Path


DEFAULT_OUTPUT = "repo_ingest.md"
DEFAULT_MAX_BYTES = 512 * 1024  # 512 KiB per file

# Conservative list of binary-heavy extensions.
BINARY_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".bmp",
    ".tif",
    ".tiff",
    ".ico",
    ".pdf",
    ".zip",
    ".gz",
    ".tar",
    ".7z",
    ".rar",
    ".bin",
    ".exe",
    ".dll",
    ".so",
    ".dylib",
    ".pyd",
    ".mp3",
    ".wav",
    ".ogg",
    ".mp4",
    ".mov",
    ".avi",
    ".mkv",
    ".obj",
    ".stl",
    ".ply",
    ".vtk",
    ".vtp",
    ".vtu",
    ".npy",
    ".npz",
    ".h5",
    ".hdf5",
}


def run_git(args: list[str], cwd: Path) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return result.stdout.rstrip("\n")


def looks_binary(path: Path) -> bool:
    suffix = path.suffix.lower()
    if suffix in BINARY_EXTENSIONS:
        return True

    try:
        with path.open("rb") as f:
            sample = f.read(8192)
    except OSError:
        return True

    if b"\x00" in sample:
        return True

    return False


def safe_read_text(path: Path) -> str:
    # Use replacement to avoid failing on odd encodings.
    return path.read_text(encoding="utf-8", errors="replace")


def build_markdown(repo_root: Path, output_path: Path, max_bytes: int) -> str:
    now = dt.datetime.now(dt.timezone.utc).isoformat()
    branch = run_git(["rev-parse", "--abbrev-ref", "HEAD"], repo_root)
    commit = run_git(["rev-parse", "HEAD"], repo_root)
    remotes = run_git(["remote", "-v"], repo_root)
    status_short = run_git(["status", "--short"], repo_root)
    tracked = run_git(["ls-files"], repo_root).splitlines()

    included: list[str] = []
    skipped: list[tuple[str, str]] = []
    sections: list[str] = []

    for rel in tracked:
        file_path = repo_root / rel
        if not file_path.exists():
            skipped.append((rel, "missing-in-working-tree"))
            continue

        if file_path.is_dir():
            skipped.append((rel, "is-directory"))
            continue

        size = file_path.stat().st_size
        if size > max_bytes:
            skipped.append((rel, f"too-large ({size} bytes)"))
            continue

        if looks_binary(file_path):
            skipped.append((rel, "binary-or-nontext"))
            continue

        content = safe_read_text(file_path)
        included.append(rel)
        sections.append(
            "\n".join(
                [
                    f"## File: `{rel}`",
                    "",
                    "```text",
                    content.rstrip("\n"),
                    "```",
                    "",
                ]
            )
        )

    lines: list[str] = [
        "# Repository Ingest",
        "",
        f"- Generated (UTC): `{now}`",
        f"- Repository root: `{repo_root}`",
        f"- Branch: `{branch}`",
        f"- Commit: `{commit}`",
        f"- Max included file size: `{max_bytes}` bytes",
        f"- Output path: `{output_path}`",
        "",
        "## Remotes",
        "",
        "```text",
        remotes or "(none)",
        "```",
        "",
        "## Working Tree Status (short)",
        "",
        "```text",
        status_short or "(clean)",
        "```",
        "",
        "## Tracked File Inventory",
        "",
        f"- Total tracked files: `{len(tracked)}`",
        f"- Included text files: `{len(included)}`",
        f"- Skipped files: `{len(skipped)}`",
        "",
        "### Included Files",
        "",
    ]

    lines.extend(f"- `{path}`" for path in included)
    lines.extend(["", "### Skipped Files", ""])
    lines.extend(f"- `{path}` — {reason}" for path, reason in skipped)
    lines.extend(["", "## File Contents", ""])
    lines.extend(sections)

    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate a Markdown repo-ingest document from tracked files."
    )
    parser.add_argument(
        "--repo-root",
        default=".",
        help="Path to the git repository root (default: current directory).",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"Output markdown path (default: {DEFAULT_OUTPUT}).",
    )
    parser.add_argument(
        "--max-bytes",
        type=int,
        default=DEFAULT_MAX_BYTES,
        help=f"Max file size to include in bytes (default: {DEFAULT_MAX_BYTES}).",
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    output_path = Path(args.output).resolve()

    # Ensure we are in a git repo.
    run_git(["rev-parse", "--is-inside-work-tree"], repo_root)

    markdown = build_markdown(
        repo_root=repo_root,
        output_path=output_path,
        max_bytes=args.max_bytes,
    )
    output_path.write_text(markdown, encoding="utf-8")

    print(f"Wrote ingest file: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
