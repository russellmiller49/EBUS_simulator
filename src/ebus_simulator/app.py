from __future__ import annotations

import atexit
import argparse
from dataclasses import dataclass
from pathlib import Path
from queue import Empty, SimpleQueue
import re
import tempfile
import threading
from typing import TYPE_CHECKING, Mapping

import numpy as np
from PIL import Image

from ebus_simulator.rendering import build_render_context, render_metadata_to_dict, render_preset


if TYPE_CHECKING:
    from ebus_simulator.rendering import RenderContext


DEFAULT_APP_TILE_SIZE = 384


@dataclass(slots=True)
class PresetBrowserOption:
    preset_id: str
    approaches: list[str]


@dataclass(slots=True)
class PresetBrowserState:
    preset_id: str
    approach: str
    engine: str = "physics"
    max_depth_mm: float | None = None
    sector_angle_deg: float | None = None
    roll_deg: float | None = None
    gain: float = 1.0
    attenuation: float = 0.15
    overlay_airway: bool = True
    overlay_target: bool = True
    overlay_station: bool = False
    overlay_vessels: bool = False


@dataclass(slots=True)
class PresetBrowserRender:
    preset_id: str
    approach: str
    engine: str
    state: PresetBrowserState
    sector_rgb: np.ndarray
    context_rgb: np.ndarray
    sector_metadata: dict[str, object]
    context_metadata: dict[str, object]
    sector_metadata_path: str
    context_metadata_path: str
    summary_text: str
    warnings: list[str]


def collect_preset_browser_options(context: RenderContext) -> list[PresetBrowserOption]:
    return [
        PresetBrowserOption(
            preset_id=preset.id,
            approaches=sorted(preset.contacts),
        )
        for preset in sorted(context.manifest.presets, key=lambda item: item.id)
    ]


def extract_context_tile(panel_rgb: np.ndarray) -> np.ndarray:
    image = np.asarray(panel_rgb, dtype=np.uint8)
    height, width = image.shape[:2]
    return np.array(image[height // 2 :, width // 2 :, :], copy=True)


def build_screenshot_strip(sector_rgb: np.ndarray, context_rgb: np.ndarray) -> np.ndarray:
    sector = np.asarray(sector_rgb, dtype=np.uint8)
    context = np.asarray(context_rgb, dtype=np.uint8)
    if sector.shape[0] != context.shape[0]:
        raise ValueError("sector_rgb and context_rgb must have the same height.")
    return np.concatenate((sector, context), axis=1)


@dataclass(slots=True)
class _RenderOutcome:
    state: PresetBrowserState
    rendered: PresetBrowserRender | None = None
    error_message: str | None = None


def _coerce_vector3(value: object) -> np.ndarray | None:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return None
    try:
        return np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return None


def _coerce_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def compute_target_offsets_mm(metadata: Mapping[str, object]) -> tuple[float | None, float | None]:
    contact_world = _coerce_vector3(metadata.get("contact_world"))
    target_world = _coerce_vector3(metadata.get("target_world"))
    pose_axes = metadata.get("pose_axes")
    if (
        contact_world is None
        or target_world is None
        or not isinstance(pose_axes, Mapping)
    ):
        return None, None

    depth_axis = _coerce_vector3(pose_axes.get("depth_axis"))
    lateral_axis = _coerce_vector3(pose_axes.get("lateral_axis"))
    if depth_axis is None:
        return None, None

    offset = target_world - contact_world
    target_depth_mm = float(np.dot(offset, depth_axis))
    target_lateral_mm = None if lateral_axis is None else float(np.dot(offset, lateral_axis))
    return target_depth_mm, target_lateral_mm


def _target_in_sector(metadata: Mapping[str, object], *, target_depth_mm: float | None, target_lateral_mm: float | None) -> bool | None:
    if target_depth_mm is None or target_lateral_mm is None:
        return None
    max_depth_mm = _coerce_float(metadata.get("max_depth_mm"))
    sector_angle_deg = _coerce_float(metadata.get("sector_angle_deg"))
    if max_depth_mm is None or sector_angle_deg is None:
        return None
    fan_half_tan = float(np.tan(np.deg2rad(sector_angle_deg / 2.0)))
    return bool(
        0.0 <= target_depth_mm <= max_depth_mm
        and abs(target_lateral_mm) <= ((target_depth_mm * fan_half_tan) + 1e-9)
    )


def _format_float(value: float | None, *, precision: int = 2, suffix: str = "") -> str:
    return "n/a" if value is None else f"{value:.{precision}f}{suffix}"


def _format_overlay_names(value: object) -> str:
    if not isinstance(value, list):
        return "none"
    names = [str(name) for name in value if str(name)]
    return ", ".join(names) if names else "none"


def _format_image_size(value: object) -> str:
    if not isinstance(value, list) or len(value) != 2:
        return "n/a"
    return f"{value[0]}x{value[1]}"


def build_render_summary_text(
    sector_metadata: Mapping[str, object],
    context_metadata: Mapping[str, object],
) -> str:
    target_depth_mm, target_lateral_mm = compute_target_offsets_mm(sector_metadata)
    target_in_sector = _target_in_sector(
        sector_metadata,
        target_depth_mm=target_depth_mm,
        target_lateral_mm=target_lateral_mm,
    )
    lines = [
        f"Preset: {sector_metadata.get('preset_id', 'n/a')} / {sector_metadata.get('approach', 'n/a')}",
        f"2D engine: {sector_metadata.get('engine', 'n/a')}",
        (
            "Display: "
            f"{_format_image_size(sector_metadata.get('image_size'))} px, "
            f"depth {_format_float(_coerce_float(sector_metadata.get('max_depth_mm')), precision=1, suffix=' mm')}, "
            f"sector {_format_float(_coerce_float(sector_metadata.get('sector_angle_deg')), precision=1, suffix=' deg')}, "
            f"roll {_format_float(_coerce_float(sector_metadata.get('roll_deg')), precision=1, suffix=' deg')}"
        ),
        (
            "Gain / attenuation: "
            f"{_format_float(_coerce_float(sector_metadata.get('gain')))} / "
            f"{_format_float(_coerce_float(sector_metadata.get('attenuation')))}"
        ),
        (
            "Pose: "
            f"target depth {_format_float(target_depth_mm, suffix=' mm')}, "
            f"lateral {_format_float(target_lateral_mm, suffix=' mm')}, "
            f"in sector {target_in_sector if target_in_sector is not None else 'n/a'}"
        ),
        (
            "Contact: "
            f"airway {_format_float(_coerce_float(sector_metadata.get('contact_to_airway_distance_mm')), suffix=' mm')}, "
            f"centerline {_format_float(_coerce_float(sector_metadata.get('centerline_projection_distance_mm')), suffix=' mm')}"
        ),
        f"2D overlays: {_format_overlay_names(sector_metadata.get('overlays_enabled'))}",
        f"Context cutaway: {context_metadata.get('cutaway_side', 'n/a')}",
        f"Context overlays: {_format_overlay_names(context_metadata.get('overlays_enabled'))}",
    ]

    engine_diagnostics = sector_metadata.get("engine_diagnostics")
    if isinstance(engine_diagnostics, Mapping):
        eval_summary = engine_diagnostics.get("eval_summary")
        if isinstance(eval_summary, Mapping):
            lines.append(
                "Physics eval: "
                f"target {_format_float(_coerce_float(eval_summary.get('target_contrast_vs_sector')), precision=4)}, "
                f"wall {_format_float(_coerce_float(eval_summary.get('wall_contrast_vs_sector')), precision=4)}, "
                f"vessel {_format_float(_coerce_float(eval_summary.get('vessel_contrast_vs_sector')), precision=4)}"
            )
        artifact_settings = engine_diagnostics.get("artifact_settings")
        if isinstance(artifact_settings, Mapping):
            lines.append(
                "Artifacts: "
                f"speckle {_format_float(_coerce_float(artifact_settings.get('speckle_strength')))}, "
                f"reverberation {_format_float(_coerce_float(artifact_settings.get('reverberation_strength')))}, "
                f"shadow {_format_float(_coerce_float(artifact_settings.get('shadow_strength')))}"
            )

    lines.append(f"Sector sidecar: {sector_metadata.get('metadata_path', 'n/a')}")
    lines.append(f"Context sidecar: {context_metadata.get('metadata_path', 'n/a')}")
    return "\n".join(lines)


def build_browser_screenshot_name(rendered: PresetBrowserRender) -> str:
    raw_name = (
        f"{rendered.preset_id}_{rendered.approach}_{rendered.engine}_"
        f"depth{rendered.state.max_depth_mm or 0.0:.1f}_"
        f"angle{rendered.state.sector_angle_deg or 0.0:.1f}_"
        f"roll{rendered.state.roll_deg or 0.0:.1f}_browser.png"
    )
    return re.sub(r"[^A-Za-z0-9._-]+", "_", raw_name)


class PresetBrowserSession:
    def __init__(
        self,
        manifest_path: str | Path,
        *,
        width: int = DEFAULT_APP_TILE_SIZE,
        height: int = DEFAULT_APP_TILE_SIZE,
    ) -> None:
        self.context = build_render_context(manifest_path)
        self.width = int(width)
        self.height = int(height)
        self._temp_dir = tempfile.TemporaryDirectory(prefix="ebus_preset_browser_")
        self._temp_root = Path(self._temp_dir.name)
        self.preset_options = collect_preset_browser_options(self.context)
        if not self.preset_options:
            raise ValueError("The manifest does not define any presets.")

    def close(self) -> None:
        self._temp_dir.cleanup()

    def available_approaches(self, preset_id: str) -> list[str]:
        for option in self.preset_options:
            if option.preset_id == preset_id:
                return list(option.approaches)
        raise ValueError(f"Unknown preset_id {preset_id!r}.")

    def default_state(self) -> PresetBrowserState:
        defaults = self.context.manifest.render_defaults
        overlay_defaults = defaults.get("default_overlays", {})
        default_preset = self.preset_options[0]
        return PresetBrowserState(
            preset_id=default_preset.preset_id,
            approach=default_preset.approaches[0],
            engine="physics",
            max_depth_mm=float(defaults.get("max_depth_mm", 40.0)),
            sector_angle_deg=float(defaults.get("sector_angle_deg", 60.0)),
            roll_deg=float(defaults.get("roll_deg", 0.0)),
            gain=float(defaults.get("gain", 1.0)),
            attenuation=float(defaults.get("attenuation", 0.15)),
            overlay_airway=bool(overlay_defaults.get("airway", True)),
            overlay_target=bool(overlay_defaults.get("target", True)),
            overlay_station=bool(overlay_defaults.get("station_mask", False)),
            overlay_vessels=bool(overlay_defaults.get("vessels", False)),
        )

    def render(self, state: PresetBrowserState) -> PresetBrowserRender:
        if state.approach not in self.available_approaches(state.preset_id):
            raise ValueError(f"Preset {state.preset_id!r} does not support approach {state.approach!r}.")

        defaults = self.context.manifest.render_defaults
        original_gain = defaults.get("gain")
        original_attenuation = defaults.get("attenuation")
        defaults["gain"] = float(state.gain)
        defaults["attenuation"] = float(state.attenuation)

        stem = f"{state.preset_id}_{state.approach}_{state.engine}"
        sector_path = self._temp_root / f"{stem}_sector.png"
        sector_metadata_path = self._temp_root / f"{stem}_sector.json"
        context_panel_path = self._temp_root / f"{stem}_context_panel.png"
        context_panel_metadata_path = self._temp_root / f"{stem}_context_panel.json"
        vessel_overlay_names = None if state.overlay_vessels else []

        try:
            sector = render_preset(
                self.context.manifest.manifest_path,
                state.preset_id,
                approach=state.approach,
                output_path=sector_path,
                metadata_path=sector_metadata_path,
                engine=state.engine,
                width=self.width,
                height=self.height,
                sector_angle_deg=state.sector_angle_deg,
                max_depth_mm=state.max_depth_mm,
                roll_deg=state.roll_deg,
                mode="clean",
                diagnostic_panel=False,
                virtual_ebus=False,
                simulated_ebus=True,
                airway_overlay=state.overlay_airway,
                target_overlay=state.overlay_target,
                contact_overlay=False,
                station_overlay=state.overlay_station,
                vessel_overlay_names=vessel_overlay_names,
                show_legend=False,
                label_overlays=False,
                show_contact=False,
                show_frustum=False,
                context=self.context,
            )
            context_panel = render_preset(
                self.context.manifest.manifest_path,
                state.preset_id,
                approach=state.approach,
                output_path=context_panel_path,
                metadata_path=context_panel_metadata_path,
                engine="localizer",
                width=self.width,
                height=self.height,
                sector_angle_deg=state.sector_angle_deg,
                max_depth_mm=state.max_depth_mm,
                roll_deg=state.roll_deg,
                mode="debug",
                diagnostic_panel=True,
                virtual_ebus=True,
                simulated_ebus=True,
                airway_overlay=state.overlay_airway,
                target_overlay=state.overlay_target,
                contact_overlay=True,
                station_overlay=state.overlay_station,
                vessel_overlay_names=vessel_overlay_names,
                show_legend=False,
                label_overlays=False,
                show_contact=True,
                show_frustum=True,
                context=self.context,
            )
        finally:
            if original_gain is None:
                defaults.pop("gain", None)
            else:
                defaults["gain"] = original_gain
            if original_attenuation is None:
                defaults.pop("attenuation", None)
            else:
                defaults["attenuation"] = original_attenuation

        sector_metadata = render_metadata_to_dict(sector.metadata)
        context_metadata = render_metadata_to_dict(context_panel.metadata)
        warnings = list(dict.fromkeys(sector.metadata.warnings + context_panel.metadata.warnings))
        return PresetBrowserRender(
            preset_id=state.preset_id,
            approach=state.approach,
            engine=state.engine,
            state=state,
            sector_rgb=np.array(sector.image_rgb, copy=True),
            context_rgb=extract_context_tile(context_panel.image_rgb),
            sector_metadata=sector_metadata,
            context_metadata=context_metadata,
            sector_metadata_path=sector.metadata.metadata_path,
            context_metadata_path=context_panel.metadata.metadata_path,
            summary_text=build_render_summary_text(sector_metadata, context_metadata),
            warnings=warnings,
        )


def launch_app(
    manifest_path: str | Path,
    *,
    width: int = DEFAULT_APP_TILE_SIZE,
    height: int = DEFAULT_APP_TILE_SIZE,
    close_after_ms: int | None = None,
    close_on_first_render: bool = False,
) -> int:
    try:
        from PySide6.QtCore import Qt, QTimer
        from PySide6.QtGui import QImage, QPixmap
        from PySide6.QtWidgets import (
            QApplication,
            QCheckBox,
            QComboBox,
            QDoubleSpinBox,
            QFileDialog,
            QFormLayout,
            QHBoxLayout,
            QLabel,
            QMainWindow,
            QPlainTextEdit,
            QPushButton,
            QSplitter,
            QVBoxLayout,
            QWidget,
        )
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "PySide6 is required for launch-app. Install it with "
            "`python -m pip install -e '.[ui]'` or `python -m pip install PySide6`."
        ) from exc

    session = PresetBrowserSession(manifest_path, width=width, height=height)
    atexit.register(session.close)

    def _to_pixmap(image_rgb: np.ndarray) -> QPixmap:
        rgb = np.ascontiguousarray(image_rgb, dtype=np.uint8)
        height, width, _ = rgb.shape
        image = QImage(rgb.data, width, height, width * 3, QImage.Format.Format_RGB888)
        return QPixmap.fromImage(image.copy())

    class PresetBrowserWindow(QMainWindow):
        def __init__(self) -> None:
            super().__init__()
            self._current_render: PresetBrowserRender | None = None
            self._sector_source_pixmap: QPixmap | None = None
            self._context_source_pixmap: QPixmap | None = None
            self._render_timer = QTimer(self)
            self._render_timer.setSingleShot(True)
            self._render_timer.setInterval(150)
            self._render_timer.timeout.connect(self._start_next_render_if_idle)
            self._render_result_timer = QTimer(self)
            self._render_result_timer.setInterval(50)
            self._render_result_timer.timeout.connect(self._poll_render_results)
            self._render_results: SimpleQueue[_RenderOutcome] = SimpleQueue()
            self._render_thread: threading.Thread | None = None
            self._render_state_in_flight: PresetBrowserState | None = None
            self._queued_state: PresetBrowserState | None = None
            self._closing = False
            self._close_on_first_render = bool(close_on_first_render)
            self.setWindowTitle("EBUS Preset Browser")
            self.resize(1480, 900)

            root = QWidget(self)
            root_layout = QHBoxLayout(root)
            root_layout.setContentsMargins(12, 12, 12, 12)
            root_layout.setSpacing(12)
            self.setCentralWidget(root)

            controls = QWidget(root)
            controls.setMinimumWidth(280)
            controls_layout = QVBoxLayout(controls)
            controls_layout.setContentsMargins(0, 0, 0, 0)
            controls_layout.setSpacing(12)
            form = QFormLayout()
            controls_layout.addLayout(form)

            default_state = session.default_state()

            self.preset_combo = QComboBox(controls)
            for option in session.preset_options:
                self.preset_combo.addItem(option.preset_id)
            self.preset_combo.setCurrentText(default_state.preset_id)
            form.addRow("Preset", self.preset_combo)

            self.approach_combo = QComboBox(controls)
            form.addRow("Approach", self.approach_combo)

            self.engine_combo = QComboBox(controls)
            self.engine_combo.addItems(["physics", "localizer"])
            self.engine_combo.setCurrentText(default_state.engine)
            form.addRow("2D Engine", self.engine_combo)

            self.depth_spin = QDoubleSpinBox(controls)
            self.depth_spin.setRange(5.0, 80.0)
            self.depth_spin.setDecimals(1)
            self.depth_spin.setSingleStep(1.0)
            self.depth_spin.setValue(float(default_state.max_depth_mm or 40.0))
            self.depth_spin.setSuffix(" mm")
            form.addRow("Depth", self.depth_spin)

            self.angle_spin = QDoubleSpinBox(controls)
            self.angle_spin.setRange(20.0, 90.0)
            self.angle_spin.setDecimals(1)
            self.angle_spin.setSingleStep(1.0)
            self.angle_spin.setValue(float(default_state.sector_angle_deg or 60.0))
            self.angle_spin.setSuffix(" deg")
            form.addRow("Sector Angle", self.angle_spin)

            self.roll_spin = QDoubleSpinBox(controls)
            self.roll_spin.setRange(-180.0, 180.0)
            self.roll_spin.setDecimals(1)
            self.roll_spin.setSingleStep(1.0)
            self.roll_spin.setValue(float(default_state.roll_deg or 0.0))
            self.roll_spin.setSuffix(" deg")
            form.addRow("Fine Roll", self.roll_spin)

            self.gain_spin = QDoubleSpinBox(controls)
            self.gain_spin.setRange(0.1, 3.0)
            self.gain_spin.setDecimals(2)
            self.gain_spin.setSingleStep(0.05)
            self.gain_spin.setValue(default_state.gain)
            form.addRow("Gain", self.gain_spin)

            self.attenuation_spin = QDoubleSpinBox(controls)
            self.attenuation_spin.setRange(0.0, 1.5)
            self.attenuation_spin.setDecimals(2)
            self.attenuation_spin.setSingleStep(0.01)
            self.attenuation_spin.setValue(default_state.attenuation)
            form.addRow("Attenuation", self.attenuation_spin)

            self.airway_checkbox = QCheckBox("Airway Overlay", controls)
            self.airway_checkbox.setChecked(default_state.overlay_airway)
            controls_layout.addWidget(self.airway_checkbox)

            self.target_checkbox = QCheckBox("Target Overlay", controls)
            self.target_checkbox.setChecked(default_state.overlay_target)
            controls_layout.addWidget(self.target_checkbox)

            self.station_checkbox = QCheckBox("Station Overlay", controls)
            self.station_checkbox.setChecked(default_state.overlay_station)
            controls_layout.addWidget(self.station_checkbox)

            self.vessels_checkbox = QCheckBox("Vessel Overlays", controls)
            self.vessels_checkbox.setChecked(default_state.overlay_vessels)
            controls_layout.addWidget(self.vessels_checkbox)

            button_row = QHBoxLayout()
            self.refresh_button = QPushButton("Refresh", controls)
            self.screenshot_button = QPushButton("Export Screenshot", controls)
            button_row.addWidget(self.refresh_button)
            button_row.addWidget(self.screenshot_button)
            controls_layout.addLayout(button_row)

            self.status_label = QLabel("Ready", controls)
            self.status_label.setWordWrap(True)
            controls_layout.addWidget(self.status_label)

            summary_heading = QLabel("Render Summary", controls)
            controls_layout.addWidget(summary_heading)

            self.summary_box = QPlainTextEdit(controls)
            self.summary_box.setReadOnly(True)
            self.summary_box.setPlaceholderText("Render metadata summary will appear here.")
            controls_layout.addWidget(self.summary_box, stretch=1)

            warning_heading = QLabel("Warnings", controls)
            controls_layout.addWidget(warning_heading)

            self.warning_box = QPlainTextEdit(controls)
            self.warning_box.setReadOnly(True)
            self.warning_box.setPlaceholderText("Render warnings will appear here.")
            controls_layout.addWidget(self.warning_box, stretch=1)

            root_layout.addWidget(controls)

            image_splitter = QSplitter(Qt.Orientation.Horizontal, root)
            image_splitter.addWidget(self._build_image_panel("2D EBUS"))
            image_splitter.addWidget(self._build_image_panel("3D Context"))
            image_splitter.setStretchFactor(0, 1)
            image_splitter.setStretchFactor(1, 1)
            root_layout.addWidget(image_splitter, stretch=1)

            self._populate_approaches(default_state.preset_id, selected_approach=default_state.approach)
            self._connect_signals()
            self.render_current_state()

        def _build_image_panel(self, title: str) -> QWidget:
            panel = QWidget(self)
            layout = QVBoxLayout(panel)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(8)
            heading = QLabel(title, panel)
            layout.addWidget(heading)
            image_label = QLabel(panel)
            image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            image_label.setScaledContents(False)
            image_label.setMinimumSize(320, 320)
            image_label.setStyleSheet("background-color: #111111; border: 1px solid #333333;")
            layout.addWidget(image_label, stretch=1)
            if title == "2D EBUS":
                self.sector_label = image_label
            else:
                self.context_label = image_label
            return panel

        def _connect_signals(self) -> None:
            self.preset_combo.currentTextChanged.connect(self._on_preset_changed)
            self.approach_combo.currentTextChanged.connect(self.schedule_render)
            self.engine_combo.currentTextChanged.connect(self.schedule_render)
            self.depth_spin.valueChanged.connect(self.schedule_render)
            self.angle_spin.valueChanged.connect(self.schedule_render)
            self.roll_spin.valueChanged.connect(self.schedule_render)
            self.gain_spin.valueChanged.connect(self.schedule_render)
            self.attenuation_spin.valueChanged.connect(self.schedule_render)
            self.airway_checkbox.toggled.connect(self.schedule_render)
            self.target_checkbox.toggled.connect(self.schedule_render)
            self.station_checkbox.toggled.connect(self.schedule_render)
            self.vessels_checkbox.toggled.connect(self.schedule_render)
            self.refresh_button.clicked.connect(self.render_current_state)
            self.screenshot_button.clicked.connect(self.export_screenshot)
            self.screenshot_button.setEnabled(False)

        def _on_preset_changed(self, preset_id: str) -> None:
            self._populate_approaches(preset_id)
            self.schedule_render()

        def _populate_approaches(self, preset_id: str, *, selected_approach: str | None = None) -> None:
            approaches = session.available_approaches(preset_id)
            self.approach_combo.blockSignals(True)
            self.approach_combo.clear()
            self.approach_combo.addItems(approaches)
            target_approach = approaches[0] if selected_approach is None or selected_approach not in approaches else selected_approach
            self.approach_combo.setCurrentText(target_approach)
            self.approach_combo.blockSignals(False)

        def schedule_render(self) -> None:
            if self._closing:
                return
            self._queued_state = self._current_state()
            self._render_timer.start()

        def _start_next_render_if_idle(self) -> None:
            if self._closing or self._render_thread is not None:
                if self._render_thread is not None and self._queued_state is not None and self._render_state_in_flight is not None:
                    self.status_label.setText(
                        f"Rendering {self._render_state_in_flight.preset_id} / "
                        f"{self._render_state_in_flight.approach} ({self._render_state_in_flight.engine}); "
                        f"queued {self._queued_state.preset_id} / {self._queued_state.approach}."
                    )
                return

            state = self._queued_state if self._queued_state is not None else self._current_state()
            self._queued_state = None
            self._render_state_in_flight = state
            self.refresh_button.setEnabled(False)
            self.status_label.setText(f"Rendering {state.preset_id} / {state.approach} ({state.engine})...")

            def _run_render_job() -> None:
                try:
                    rendered = session.render(state)
                    self._render_results.put(_RenderOutcome(state=state, rendered=rendered))
                except Exception as exc:  # pragma: no cover - worker error path
                    self._render_results.put(_RenderOutcome(state=state, error_message=str(exc)))

            self._render_thread = threading.Thread(target=_run_render_job, daemon=True)
            self._render_thread.start()
            self._render_result_timer.start()

        def _poll_render_results(self) -> None:
            completed_any = False
            while True:
                try:
                    outcome = self._render_results.get(block=False)
                except Empty:
                    break
                completed_any = True
                self._render_thread = None
                self._render_state_in_flight = None
                self.refresh_button.setEnabled(True)
                if outcome.error_message is not None:
                    self.status_label.setText(f"Render failed: {outcome.error_message}")
                    self.warning_box.setPlainText(outcome.error_message)
                elif outcome.rendered is not None:
                    self._apply_render(outcome.rendered)

            if self._render_thread is None and self._queued_state is not None:
                self._start_next_render_if_idle()
            elif self._render_thread is None and not completed_any:
                self._render_result_timer.stop()

        def _set_panel_pixmap(self, label: QLabel, pixmap: QPixmap | None) -> None:
            if pixmap is None:
                label.clear()
                return
            scaled = pixmap.scaled(
                label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            label.setPixmap(scaled)

        def _refresh_panel_pixmaps(self) -> None:
            self._set_panel_pixmap(self.sector_label, self._sector_source_pixmap)
            self._set_panel_pixmap(self.context_label, self._context_source_pixmap)

        def _apply_render(self, rendered: PresetBrowserRender) -> None:
            self._current_render = rendered
            self._sector_source_pixmap = _to_pixmap(rendered.sector_rgb)
            self._context_source_pixmap = _to_pixmap(rendered.context_rgb)
            self._refresh_panel_pixmaps()
            self.summary_box.setPlainText(rendered.summary_text)
            warnings_text = "\n".join(rendered.warnings) if rendered.warnings else "No render warnings."
            self.warning_box.setPlainText(warnings_text)
            self.screenshot_button.setEnabled(True)
            self.status_label.setText(
                f"{rendered.preset_id} / {rendered.approach} rendered with {rendered.engine}; "
                f"{len(rendered.warnings)} warning(s)."
            )
            if self._close_on_first_render:
                QTimer.singleShot(0, self.close)

        def _current_state(self) -> PresetBrowserState:
            return PresetBrowserState(
                preset_id=self.preset_combo.currentText(),
                approach=self.approach_combo.currentText(),
                engine=self.engine_combo.currentText(),
                max_depth_mm=float(self.depth_spin.value()),
                sector_angle_deg=float(self.angle_spin.value()),
                roll_deg=float(self.roll_spin.value()),
                gain=float(self.gain_spin.value()),
                attenuation=float(self.attenuation_spin.value()),
                overlay_airway=self.airway_checkbox.isChecked(),
                overlay_target=self.target_checkbox.isChecked(),
                overlay_station=self.station_checkbox.isChecked(),
                overlay_vessels=self.vessels_checkbox.isChecked(),
            )

        def render_current_state(self) -> None:
            if self._closing:
                return
            self._render_timer.stop()
            self._queued_state = self._current_state()
            self._start_next_render_if_idle()

        def export_screenshot(self) -> None:
            if self._current_render is None:
                self.status_label.setText("Nothing to export yet.")
                return
            default_name = build_browser_screenshot_name(self._current_render)
            output_path, _ = QFileDialog.getSaveFileName(
                self,
                "Export Browser Screenshot",
                str(Path.cwd() / default_name),
                "PNG Images (*.png)",
            )
            if not output_path:
                return
            screenshot = build_screenshot_strip(self._current_render.sector_rgb, self._current_render.context_rgb)
            Image.fromarray(screenshot).save(output_path)
            self.status_label.setText(f"Screenshot exported to {output_path}")

        def resizeEvent(self, event) -> None:  # pragma: no cover - UI resize behavior
            self._refresh_panel_pixmaps()
            super().resizeEvent(event)

        def closeEvent(self, event) -> None:  # pragma: no cover - UI lifecycle
            self._closing = True
            self._render_timer.stop()
            self._render_result_timer.stop()
            if self._render_thread is None or not self._render_thread.is_alive():
                session.close()
            super().closeEvent(event)

    app = QApplication.instance() or QApplication([])
    window = PresetBrowserWindow()
    window.show()
    if close_after_ms is not None:
        QTimer.singleShot(int(close_after_ms), window.close)
    return app.exec()


def main() -> int:
    parser = argparse.ArgumentParser(description="Launch the desktop linear EBUS preset browser.")
    parser.add_argument("manifest", help="Path to the case manifest YAML file.")
    args = parser.parse_args()

    try:
        return int(launch_app(args.manifest))
    except RuntimeError as exc:
        print(str(exc))
        return 1
