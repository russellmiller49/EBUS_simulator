from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
from typing import Any

from PIL import Image, ImageDraw
import yaml


DEFAULT_EBUS_VIDEO_ROOT = Path("/Users/russellmiller/Documents/EBUS_video")
REFERENCE_LIBRARY_VERSION = 1
REFERENCE_KINDS = {"ebus", "white_light"}


@dataclass(slots=True)
class DeidentifyRegion:
    x: float
    y: float
    width: float
    height: float


@dataclass(slots=True)
class ReferenceVideo:
    id: str
    stations: list[str]
    kind: str
    path: Path
    preset_ids: list[str] = field(default_factory=list)
    keyframe_count: int | None = None
    sample_times_seconds: list[float] | None = None
    deidentify_regions: list[DeidentifyRegion] = field(default_factory=list)
    notes: str | None = None


@dataclass(slots=True)
class ReferenceKeyframe:
    id: str
    video_id: str
    stations: list[str]
    kind: str
    source_video_path: str
    image_path: str
    time_seconds: float | None
    frame_index: int
    width: int
    height: int
    extraction_method: str
    warnings: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ReferenceVideoConfig:
    config_path: Path
    root: Path
    videos: list[ReferenceVideo]
    default_ebus_keyframes: int = 5
    default_white_light_keyframes: int = 3
    frame_size_px: int = 720


@dataclass(slots=True)
class ReferenceLibrary:
    version: int
    config_path: str
    output_dir: str
    root: str
    videos: list[dict[str, Any]]
    keyframes: list[dict[str, Any]]
    warnings: list[str] = field(default_factory=list)


_TOKEN_PATTERN = re.compile(r"\$\{([A-Z0-9_]+)\}")


def _expand_tokens(value: str, *, config_path: Path) -> str:
    environment = dict(os.environ)
    environment.setdefault("EBUS_VIDEO_ROOT", str(DEFAULT_EBUS_VIDEO_ROOT))
    environment.setdefault("REPO_ROOT", str(_discover_repo_root(config_path.parent)))

    def _replace(match: re.Match[str]) -> str:
        key = match.group(1)
        if key not in environment:
            raise ValueError(f"Reference video path token ${{{key}}} is not defined.")
        return environment[key]

    return _TOKEN_PATTERN.sub(_replace, value)


def _discover_repo_root(start: Path) -> Path:
    current = start.resolve()
    for candidate in (current, *current.parents):
        if (candidate / ".git").exists():
            return candidate
    return current.parent


def _resolve_reference_path(root: Path, value: str, *, config_path: Path) -> Path:
    expanded = Path(_expand_tokens(value, config_path=config_path)).expanduser()
    return expanded if expanded.is_absolute() else root / expanded


def _parse_deidentify_regions(payload: list[dict[str, object]] | None) -> list[DeidentifyRegion]:
    regions: list[DeidentifyRegion] = []
    for raw in payload or []:
        regions.append(
            DeidentifyRegion(
                x=float(raw.get("x", 0.0)),
                y=float(raw.get("y", 0.0)),
                width=float(raw.get("width", 0.0)),
                height=float(raw.get("height", 0.0)),
            )
        )
    return regions


def _normalize_stations(payload: dict[str, object]) -> list[str]:
    if payload.get("stations") is not None:
        raw_stations = payload["stations"]
        if not isinstance(raw_stations, list):
            raise ValueError("'stations' must be a list when provided.")
        return [str(item).lower() for item in raw_stations]
    if payload.get("station") is None:
        raise ValueError("Reference video entries require 'station' or 'stations'.")
    return [str(payload["station"]).lower()]


def load_video_reference_config(path: str | Path) -> ReferenceVideoConfig:
    config_path = Path(path).expanduser().resolve()
    payload = yaml.safe_load(config_path.read_text())
    root = Path(_expand_tokens(str(payload.get("root", "${EBUS_VIDEO_ROOT}")), config_path=config_path)).expanduser()
    if not root.is_absolute():
        root = (config_path.parent / root).resolve()

    defaults = payload.get("defaults", {})
    videos: list[ReferenceVideo] = []
    seen: set[str] = set()
    for raw_video in payload.get("videos", []):
        video_id = str(raw_video["id"])
        if video_id in seen:
            raise ValueError(f"Duplicate reference video id {video_id!r}.")
        seen.add(video_id)

        kind = str(raw_video.get("kind", "")).lower()
        if kind not in REFERENCE_KINDS:
            raise ValueError(f"Reference video {video_id!r} has unsupported kind {kind!r}.")

        sample_times = raw_video.get("sample_times_seconds")
        if sample_times is not None and not isinstance(sample_times, list):
            raise ValueError(f"Reference video {video_id!r} sample_times_seconds must be a list.")

        videos.append(
            ReferenceVideo(
                id=video_id,
                stations=_normalize_stations(raw_video),
                kind=kind,
                path=_resolve_reference_path(root, str(raw_video["path"]), config_path=config_path),
                preset_ids=[str(item) for item in raw_video.get("preset_ids", [])],
                keyframe_count=(
                    None
                    if raw_video.get("keyframe_count") is None
                    else int(raw_video["keyframe_count"])
                ),
                sample_times_seconds=(
                    None if sample_times is None else [float(value) for value in sample_times]
                ),
                deidentify_regions=_parse_deidentify_regions(raw_video.get("deidentify_regions")),
                notes=(None if raw_video.get("notes") is None else str(raw_video["notes"])),
            )
        )

    return ReferenceVideoConfig(
        config_path=config_path,
        root=root,
        videos=videos,
        default_ebus_keyframes=int(defaults.get("ebus_keyframes", 5)),
        default_white_light_keyframes=int(defaults.get("white_light_keyframes", 3)),
        frame_size_px=int(defaults.get("frame_size_px", 720)),
    )


def _jsonable_video(video: ReferenceVideo) -> dict[str, Any]:
    return {
        **asdict(video),
        "path": str(video.path),
        "deidentify_regions": [asdict(region) for region in video.deidentify_regions],
    }


def _duration_seconds(path: Path) -> float | None:
    ffprobe = shutil.which("ffprobe")
    if ffprobe is not None:
        command = [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ]
        completed = subprocess.run(command, check=False, capture_output=True, text=True)
        if completed.returncode == 0:
            try:
                return float(completed.stdout.strip())
            except ValueError:
                pass

    mdls = shutil.which("mdls")
    if mdls is not None:
        completed = subprocess.run(
            [mdls, "-raw", "-name", "kMDItemDurationSeconds", str(path)],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode == 0:
            raw_value = completed.stdout.strip()
            if raw_value and raw_value != "(null)":
                try:
                    return float(raw_value)
                except ValueError:
                    return None
    return None


def _sample_times(video: ReferenceVideo, *, config: ReferenceVideoConfig) -> list[float | None]:
    if video.sample_times_seconds is not None:
        return list(video.sample_times_seconds)

    default_count = (
        config.default_ebus_keyframes
        if video.kind == "ebus"
        else config.default_white_light_keyframes
    )
    count = max(1, int(video.keyframe_count or default_count))
    duration = _duration_seconds(video.path)
    if duration is None or duration <= 0.0:
        return [None]
    if count == 1:
        return [duration / 2.0]
    start = duration * 0.12
    stop = duration * 0.88
    step = (stop - start) / float(count - 1)
    return [start + (step * index) for index in range(count)]


def _default_deidentify_regions(video: ReferenceVideo) -> list[DeidentifyRegion]:
    if video.deidentify_regions:
        return video.deidentify_regions
    if video.kind == "ebus":
        return [DeidentifyRegion(x=0.76, y=0.0, width=0.24, height=0.14)]
    return []


def deidentify_image(image: Image.Image, regions: list[DeidentifyRegion]) -> Image.Image:
    output = image.convert("RGB")
    if not regions:
        return output
    draw = ImageDraw.Draw(output)
    width, height = output.size
    for region in regions:
        x0 = int(max(0.0, min(1.0, region.x)) * width)
        y0 = int(max(0.0, min(1.0, region.y)) * height)
        x1 = int(max(0.0, min(1.0, region.x + region.width)) * width)
        y1 = int(max(0.0, min(1.0, region.y + region.height)) * height)
        if x1 > x0 and y1 > y0:
            draw.rectangle((x0, y0, x1, y1), fill=(0, 0, 0))
    return output


def _resize_image(image: Image.Image, *, frame_size_px: int) -> Image.Image:
    output = image.convert("RGB")
    output.thumbnail((frame_size_px, frame_size_px), Image.Resampling.LANCZOS)
    return output


def _write_image_keyframe(
    source_path: Path,
    output_path: Path,
    *,
    frame_size_px: int,
    deidentify_regions: list[DeidentifyRegion],
) -> tuple[int, int]:
    image = Image.open(source_path)
    image = _resize_image(image, frame_size_px=frame_size_px)
    image = deidentify_image(image, deidentify_regions)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    return image.size


def _extract_with_ffmpeg(
    source_path: Path,
    output_path: Path,
    *,
    time_seconds: float,
    frame_size_px: int,
    deidentify_regions: list[DeidentifyRegion],
) -> tuple[int, int] | None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return None

    temp_path = output_path.with_suffix(".ffmpeg.png")
    command = [
        ffmpeg,
        "-y",
        "-ss",
        f"{time_seconds:.3f}",
        "-i",
        str(source_path),
        "-frames:v",
        "1",
        "-vf",
        f"scale={frame_size_px}:{frame_size_px}:force_original_aspect_ratio=decrease",
        str(temp_path),
    ]
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode != 0 or not temp_path.exists():
        return None
    try:
        return _write_image_keyframe(
            temp_path,
            output_path,
            frame_size_px=frame_size_px,
            deidentify_regions=deidentify_regions,
        )
    finally:
        temp_path.unlink(missing_ok=True)


def _extract_with_quicklook(
    source_path: Path,
    output_path: Path,
    *,
    frame_size_px: int,
    deidentify_regions: list[DeidentifyRegion],
) -> tuple[int, int] | None:
    qlmanage = shutil.which("qlmanage")
    if qlmanage is None:
        return None
    temp_dir = output_path.parent / "_quicklook_tmp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        [qlmanage, "-t", "-s", str(frame_size_px), "-o", str(temp_dir), str(source_path)],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        shutil.rmtree(temp_dir, ignore_errors=True)
        return None
    candidates = sorted(temp_dir.glob("*.png"))
    if not candidates:
        shutil.rmtree(temp_dir, ignore_errors=True)
        return None
    try:
        return _write_image_keyframe(
            candidates[0],
            output_path,
            frame_size_px=frame_size_px,
            deidentify_regions=deidentify_regions,
        )
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def _extract_keyframe(
    video: ReferenceVideo,
    output_path: Path,
    *,
    time_seconds: float | None,
    frame_size_px: int,
) -> tuple[str, tuple[int, int], list[str]]:
    warnings: list[str] = []
    deidentify_regions = _default_deidentify_regions(video)
    suffix = video.path.suffix.lower()

    if suffix in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
        size = _write_image_keyframe(
            video.path,
            output_path,
            frame_size_px=frame_size_px,
            deidentify_regions=deidentify_regions,
        )
        return "image", size, warnings

    if time_seconds is not None:
        ffmpeg_size = _extract_with_ffmpeg(
            video.path,
            output_path,
            time_seconds=time_seconds,
            frame_size_px=frame_size_px,
            deidentify_regions=deidentify_regions,
        )
        if ffmpeg_size is not None:
            return "ffmpeg", ffmpeg_size, warnings

    quicklook_size = _extract_with_quicklook(
        video.path,
        output_path,
        frame_size_px=frame_size_px,
        deidentify_regions=deidentify_regions,
    )
    if quicklook_size is not None:
        if time_seconds is not None:
            warnings.append("ffmpeg unavailable; Quick Look thumbnail used instead of requested timestamp.")
        return "quicklook", quicklook_size, warnings

    raise RuntimeError(
        f"Could not extract a keyframe from {video.path}. Install ffmpeg, or run on macOS with Quick Look support."
    )


def build_reference_library(
    config_path: str | Path,
    *,
    output_dir: str | Path,
    frame_size_px: int | None = None,
    overwrite: bool = False,
) -> ReferenceLibrary:
    config = load_video_reference_config(config_path)
    output_dir = Path(output_dir).expanduser().resolve()
    keyframe_dir = output_dir / "keyframes"
    output_dir.mkdir(parents=True, exist_ok=True)
    keyframe_dir.mkdir(parents=True, exist_ok=True)

    resolved_frame_size = config.frame_size_px if frame_size_px is None else int(frame_size_px)
    warnings: list[str] = []
    keyframes: list[dict[str, Any]] = []

    for video in config.videos:
        if not video.path.exists():
            warnings.append(f"Reference video {video.id!r} is missing: {video.path}")
            continue
        times = _sample_times(video, config=config)
        if shutil.which("ffmpeg") is None and video.path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
            times = times[:1]
        for index, time_seconds in enumerate(times):
            keyframe_id = f"{video.id}_kf{index:02d}"
            output_path = keyframe_dir / f"{keyframe_id}.png"
            if output_path.exists() and not overwrite:
                with Image.open(output_path) as existing:
                    width, height = existing.size
                method = "existing"
                frame_warnings: list[str] = []
            else:
                method, (width, height), frame_warnings = _extract_keyframe(
                    video,
                    output_path,
                    time_seconds=time_seconds,
                    frame_size_px=resolved_frame_size,
                )
            keyframe = ReferenceKeyframe(
                id=keyframe_id,
                video_id=video.id,
                stations=list(video.stations),
                kind=video.kind,
                source_video_path=str(video.path),
                image_path=str(output_path),
                time_seconds=None if time_seconds is None else float(time_seconds),
                frame_index=index,
                width=int(width),
                height=int(height),
                extraction_method=method,
                warnings=frame_warnings,
            )
            keyframes.append(asdict(keyframe))

    library = ReferenceLibrary(
        version=REFERENCE_LIBRARY_VERSION,
        config_path=str(config.config_path),
        output_dir=str(output_dir),
        root=str(config.root),
        videos=[_jsonable_video(video) for video in config.videos],
        keyframes=keyframes,
        warnings=warnings,
    )
    (output_dir / "reference_library.json").write_text(json.dumps(asdict(library), indent=2))
    return library


def load_reference_library(path: str | Path) -> ReferenceLibrary:
    payload = json.loads(Path(path).expanduser().resolve().read_text())
    return ReferenceLibrary(
        version=int(payload["version"]),
        config_path=str(payload["config_path"]),
        output_dir=str(payload["output_dir"]),
        root=str(payload["root"]),
        videos=list(payload.get("videos", [])),
        keyframes=list(payload.get("keyframes", [])),
        warnings=list(payload.get("warnings", [])),
    )


def station_reference_keyframes(
    library: ReferenceLibrary,
    station: str,
    *,
    max_items: int = 4,
) -> list[dict[str, Any]]:
    normalized_station = station.lower()
    matches = [
        dict(keyframe)
        for keyframe in library.keyframes
        if normalized_station in [str(item).lower() for item in keyframe.get("stations", [])]
    ]
    matches.sort(key=lambda item: (0 if item.get("kind") == "ebus" else 1, item.get("video_id", ""), item.get("frame_index", 0)))
    return matches[:max_items]


def station_reference_status(library: ReferenceLibrary, station: str) -> str:
    matches = station_reference_keyframes(library, station, max_items=1000)
    if any(match.get("kind") == "ebus" for match in matches):
        return "ebus_reference"
    if any(match.get("kind") == "white_light" for match in matches):
        return "white_light_only"
    return "missing_reference"
