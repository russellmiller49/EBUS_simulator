from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any


class RenderEngine(str, Enum):
    LOCALIZER = "localizer"
    PHYSICS = "physics"


def parse_render_engine(value: str | RenderEngine | None) -> RenderEngine:
    if value is None:
        return RenderEngine.LOCALIZER
    if isinstance(value, RenderEngine):
        return value
    normalized = value.strip().lower()
    for engine in RenderEngine:
        if engine.value == normalized:
            return engine
    supported = ", ".join(engine.value for engine in RenderEngine)
    raise ValueError(f"Unsupported render engine {value!r}. Expected one of: {supported}.")


@dataclass(slots=True)
class RenderRequest:
    manifest_path: str | Path
    preset_id: str
    output_path: str | Path
    approach: str | None = None
    metadata_path: str | Path | None = None
    engine: RenderEngine = RenderEngine.LOCALIZER
    seed: int | None = None
    width: int | None = None
    height: int | None = None
    sector_angle_deg: float | None = None
    max_depth_mm: float | None = None
    roll_deg: float | None = None
    mode: str | None = None
    airway_overlay: bool | None = None
    airway_lumen_overlay: bool | None = None
    airway_wall_overlay: bool | None = None
    target_overlay: bool | None = None
    contact_overlay: bool | None = None
    station_overlay: bool | None = None
    vessel_overlay_names: list[str] | None = None
    slice_thickness_mm: float | None = None
    diagnostic_panel: bool = False
    device: str = "bf_uc180f"
    refine_contact: bool = True
    virtual_ebus: bool = True
    simulated_ebus: bool = True
    reference_fov_mm: float | None = None
    source_oblique_size_mm: float | None = None
    single_vessel: str | None = None
    show_legend: bool | None = None
    label_overlays: bool | None = None
    min_contour_area_px: float = 20.0
    min_contour_length_px: float = 15.0
    show_contact: bool | None = None
    show_frustum: bool | None = None
    cutaway_mode: str | None = None
    cutaway_side: str | None = None
    cutaway_depth_mm: float | None = None
    cutaway_origin: str | None = None
    show_full_airway: bool | None = None
    cutaway_custom_origin_world: Any | None = None
    debug_map_dir: str | Path | None = None
    speckle_strength: float | None = None
    reverberation_strength: float | None = None
    shadow_strength: float | None = None


@dataclass(slots=True)
class RenderResult:
    engine: RenderEngine
    engine_version: str
    rendered_preset: Any
