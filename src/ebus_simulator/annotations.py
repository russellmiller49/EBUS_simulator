from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from ebus_simulator.render_state import OverlayLayer


def _to_uint8(image_rgb: np.ndarray) -> np.ndarray:
    return np.clip(image_rgb * 255.0, 0.0, 255.0).astype(np.uint8)


def _add_panel_label(image_rgb: np.ndarray, label: str) -> np.ndarray:
    image = Image.fromarray(_to_uint8(image_rgb), mode="RGB")
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 140, 20), fill=(0, 0, 0))
    draw.text((6, 4), label, fill=(255, 255, 255))
    return np.asarray(image, dtype=np.uint8)


def _compute_contour(binary_mask: np.ndarray) -> np.ndarray:
    if not np.any(binary_mask):
        return np.zeros_like(binary_mask, dtype=bool)
    eroded = ndimage.binary_erosion(binary_mask, border_value=0)
    contour = np.logical_and(binary_mask, np.logical_not(eroded))
    return ndimage.binary_dilation(contour, iterations=1)


def _apply_contour_overlay(image_rgb: np.ndarray, binary_mask: np.ndarray, color_rgb: np.ndarray) -> None:
    contour = _compute_contour(binary_mask)
    if not np.any(contour):
        return
    image_rgb[contour] = image_rgb[contour] * 0.12 + color_rgb[None, :] * 0.88


def _draw_cross_marker(image_rgb: np.ndarray, *, row: int, column: int, color_rgb: np.ndarray, radius: int = 4) -> None:
    height, width = image_rgb.shape[:2]
    for offset in range(-radius, radius + 1):
        current_row = row + offset
        current_column = column + offset
        opposite_column = column - offset
        if 0 <= current_row < height and 0 <= column < width:
            image_rgb[current_row, column] = image_rgb[current_row, column] * 0.10 + color_rgb * 0.90
        if 0 <= row < height and 0 <= current_column < width:
            image_rgb[row, current_column] = image_rgb[row, current_column] * 0.10 + color_rgb * 0.90
        if 0 <= current_row < height and 0 <= current_column < width:
            image_rgb[current_row, current_column] = image_rgb[current_row, current_column] * 0.25 + color_rgb * 0.75
        if 0 <= current_row < height and 0 <= opposite_column < width:
            image_rgb[current_row, opposite_column] = image_rgb[current_row, opposite_column] * 0.25 + color_rgb * 0.75


def _color_tuple(color_rgb: np.ndarray) -> tuple[int, int, int]:
    return tuple(int(np.clip(channel * 255.0, 0.0, 255.0)) for channel in color_rgb.tolist())


def _annotate_legend_and_labels(
    image_rgb: np.ndarray,
    *,
    visible_layers: list[OverlayLayer],
    show_legend: bool,
    label_overlays: bool,
    legend_entries: list[tuple[str, np.ndarray]],
) -> np.ndarray:
    image = Image.fromarray(_to_uint8(image_rgb), mode="RGB")
    draw = ImageDraw.Draw(image)
    width, height = image.size

    if label_overlays:
        label_layers = [layer for layer in visible_layers if layer.label_enabled and np.any(layer.mask)]
        for index, layer in enumerate(label_layers):
            points = np.argwhere(_compute_contour(layer.mask))
            if points.size == 0:
                continue
            center_row, center_col = points.mean(axis=0)
            label_on_left = bool(center_col > (width * 0.55))
            label_x = 10 if label_on_left else max(10, width - 156)
            label_y = 26 + (index * 18)
            text_color = _color_tuple(layer.color_rgb)
            draw.line(
                (
                    float(center_col),
                    float(center_row),
                    float(label_x + (120 if label_on_left else 0)),
                    float(label_y + 8),
                ),
                fill=text_color,
                width=2,
            )
            draw.rectangle((label_x, label_y, label_x + 146, label_y + 16), fill=(0, 0, 0))
            draw.text((label_x + 4, label_y + 2), layer.label, fill=text_color)

    if show_legend and legend_entries:
        legend_height = 8 + (18 * len(legend_entries))
        legend_width = 186
        legend_x = 10
        legend_y = max(10, height - legend_height - 10)
        draw.rectangle((legend_x, legend_y, legend_x + legend_width, legend_y + legend_height), fill=(0, 0, 0))
        for index, (label, color_rgb) in enumerate(legend_entries):
            top = legend_y + 4 + (index * 18)
            text_color = _color_tuple(color_rgb)
            draw.rectangle((legend_x + 6, top + 4, legend_x + 16, top + 14), fill=text_color)
            draw.text((legend_x + 22, top + 1), label, fill=text_color)

    return np.asarray(image, dtype=np.uint8)


def _filter_mask_components(binary_mask: np.ndarray, *, min_area_px: float, min_length_px: float) -> np.ndarray:
    if not np.any(binary_mask):
        return np.zeros_like(binary_mask, dtype=bool)
    labeled, component_count = ndimage.label(binary_mask)
    filtered = np.zeros_like(binary_mask, dtype=bool)
    for component_index in range(1, component_count + 1):
        component = labeled == component_index
        area = float(np.count_nonzero(component))
        if area < min_area_px:
            continue
        contour = _compute_contour(component)
        contour_length = float(np.count_nonzero(contour))
        if contour_length < min_length_px:
            continue
        filtered |= component
    return filtered
