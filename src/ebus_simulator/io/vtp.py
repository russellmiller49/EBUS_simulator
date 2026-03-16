from __future__ import annotations

import base64
from pathlib import Path
from xml.etree import ElementTree as ET
import zlib

import numpy as np

from ebus_simulator.models import PolyData

RAS_TO_LPS_3X3 = np.diag([-1.0, -1.0, 1.0])


def _numeric_dtype(type_name: str) -> np.dtype:
    dtype_map = {
        "Float32": np.float32,
        "Float64": np.float64,
        "Int32": np.int32,
        "Int64": np.int64,
        "UInt32": np.uint32,
        "UInt64": np.uint64,
    }
    dtype = dtype_map.get(type_name)
    if dtype is None:
        raise ValueError(f"Unsupported VTP array type: {type_name}")
    return np.dtype(dtype)


def _decode_binary_payload(payload: str, *, header_type: str, compressor: str | None) -> bytes:
    encoded = "".join(payload.split())
    if not encoded:
        return b""

    header_dtype = _numeric_dtype(header_type)
    header_size = int(header_dtype.itemsize)
    base64_chars = lambda byte_count: 4 * ((byte_count + 2) // 3)

    if compressor is None:
        header_chars = base64_chars(header_size)
        header_raw = base64.b64decode(encoded[:header_chars])
        if len(header_raw) < header_size:
            raise ValueError("Binary VTP payload is shorter than the header.")
        byte_count = int(np.frombuffer(header_raw[:header_size], dtype=header_dtype, count=1)[0])
        data_raw = base64.b64decode(encoded[header_chars:]) if len(encoded) > header_chars else b""
        return data_raw[:byte_count]

    if compressor != "vtkZLibDataCompressor":
        raise ValueError(f"Unsupported VTP compressor: {compressor}")

    prefix_chars = base64_chars(header_size * 3)
    prefix_raw = base64.b64decode(encoded[:prefix_chars])
    if len(prefix_raw) < (header_size * 3):
        raise ValueError("Compressed VTP payload is shorter than the compressor header.")

    header_prefix = np.frombuffer(prefix_raw[: header_size * 3], dtype=header_dtype, count=3)
    block_count = int(header_prefix[0])
    block_size = int(header_prefix[1])
    last_block_size = int(header_prefix[2])
    header_words = 3 + block_count
    header_bytes = header_words * header_size
    header_chars = base64_chars(header_bytes)
    header_raw = base64.b64decode(encoded[:header_chars])
    if len(header_raw) < header_bytes:
        raise ValueError("Compressed VTP payload is shorter than the expected block-size header.")

    header_values = np.frombuffer(header_raw[:header_bytes], dtype=header_dtype, count=header_words)
    compressed_sizes = header_values[3:]
    raw = base64.b64decode(encoded[header_chars:]) if len(encoded) > header_chars else b""
    offset = 0
    decompressed_blocks: list[bytes] = []

    for block_index, compressed_size in enumerate(compressed_sizes.tolist()):
        size = int(compressed_size)
        block = raw[offset:offset + size]
        if len(block) != size:
            raise ValueError(f"Compressed VTP block {block_index} is truncated.")
        decompressed_blocks.append(zlib.decompress(block))
        offset += size

    decompressed = b"".join(decompressed_blocks)
    if block_count <= 0:
        expected_size = 0
    elif block_count == 1:
        expected_size = last_block_size
    else:
        expected_size = (block_size * (block_count - 1)) + last_block_size
    if len(decompressed) != expected_size:
        raise ValueError(
            f"Compressed VTP payload expanded to {len(decompressed)} bytes, expected {expected_size}."
        )
    return decompressed


def _parse_numeric_array(node: ET.Element, *, header_type: str, compressor: str | None) -> np.ndarray:
    dtype = _numeric_dtype(node.attrib.get("type", "Float64"))
    format_name = node.attrib.get("format", "ascii").lower()
    text = node.text or ""

    if format_name == "ascii":
        stripped = text.strip()
        if not stripped:
            return np.asarray([], dtype=dtype)
        return np.fromstring(stripped, sep=" ", dtype=dtype)

    if format_name == "binary":
        decoded = _decode_binary_payload(text, header_type=header_type, compressor=compressor)
        if not decoded:
            return np.asarray([], dtype=dtype)
        return np.frombuffer(decoded, dtype=dtype).copy()

    raise ValueError(f"Unsupported VTP array format: {format_name}")


def _parse_field_array(node: ET.Element, *, header_type: str, compressor: str | None):
    if node.attrib.get("type") != "String":
        return _parse_numeric_array(node, header_type=header_type, compressor=compressor)

    text = (node.text or "").strip()
    values = np.fromstring(text, sep=" ", dtype=np.uint8).tolist() if text else []
    as_bytes = bytes(values)
    parts = [chunk.decode("utf-8") for chunk in as_bytes.split(b"\x00") if chunk]
    tuples = int(node.attrib.get("NumberOfTuples", len(parts)))
    if tuples == 1:
        return parts[0] if parts else ""
    return parts


def _reshape_points(array: np.ndarray) -> np.ndarray:
    if array.size % 3 != 0:
        raise ValueError("VTP point array size is not divisible by 3.")
    return array.reshape((-1, 3))


def _decode_space(field_data: dict[str, object]) -> str:
    raw_space = field_data.get("SPACE", "UNKNOWN")
    if isinstance(raw_space, list):
        raw_space = raw_space[0] if raw_space else "UNKNOWN"
    return str(raw_space).upper()


def _points_to_lps(points: np.ndarray, source_space: str) -> np.ndarray:
    if source_space in {"LPS", "UNKNOWN"}:
        return points.astype(np.float64)
    if source_space == "RAS":
        return (RAS_TO_LPS_3X3 @ points.T).T.astype(np.float64)
    raise ValueError(f"Unsupported or unknown VTP SPACE value: {source_space}")


def _build_cells(cells_node: ET.Element | None, *, header_type: str, compressor: str | None) -> list[np.ndarray]:
    if cells_node is None:
        return []

    connectivity = None
    offsets = None
    for child in cells_node.findall("DataArray"):
        name = child.attrib.get("Name")
        if name == "connectivity":
            connectivity = _parse_numeric_array(child, header_type=header_type, compressor=compressor).astype(np.int64)
        elif name == "offsets":
            offsets = _parse_numeric_array(child, header_type=header_type, compressor=compressor).astype(np.int64)
    if connectivity is None or offsets is None:
        return []

    cells: list[np.ndarray] = []
    start = 0
    for stop in offsets.tolist():
        cell = connectivity[start:int(stop)]
        if cell.size >= 2:
            cells.append(cell)
        start = int(stop)
    return cells


def load_vtp_polydata(path: str | Path) -> PolyData:
    resolved_path = Path(path).expanduser().resolve()
    root = ET.parse(resolved_path).getroot()
    header_type = root.attrib.get("header_type", "UInt32")
    compressor = root.attrib.get("compressor")

    polydata_node = root.find("./PolyData/Piece")
    if polydata_node is None:
        raise ValueError("VTP PolyData/Piece section not found.")

    field_data: dict[str, object] = {}
    for array_node in root.findall("./PolyData/FieldData/Array"):
        field_data[array_node.attrib["Name"]] = _parse_field_array(
            array_node,
            header_type=header_type,
            compressor=compressor,
        )

    source_space = _decode_space(field_data)

    points_node = polydata_node.find("./Points/DataArray")
    if points_node is None:
        raise ValueError("VTP Points/DataArray section not found.")
    points = _reshape_points(
        _parse_numeric_array(points_node, header_type=header_type, compressor=compressor)
    )

    point_data: dict[str, np.ndarray] = {}
    for array_node in polydata_node.findall("./PointData/DataArray"):
        point_data[array_node.attrib.get("Name", "unnamed")] = _parse_numeric_array(
            array_node,
            header_type=header_type,
            compressor=compressor,
        )

    lines = _build_cells(polydata_node.find("./Lines"), header_type=header_type, compressor=compressor)
    polygons = _build_cells(polydata_node.find("./Polys"), header_type=header_type, compressor=compressor)

    return PolyData(
        path=resolved_path,
        points_lps=_points_to_lps(points, source_space),
        lines=lines,
        point_data=point_data,
        field_data=field_data,
        source_space=source_space,
        polygons=polygons,
    )
