"""Image-to-CAD import pipeline: IR extraction, action synthesis, and QA checks."""

from __future__ import annotations

import math
import re
from typing import Any, Callable

from autocad_mcp.agent.safety import SafetyResult, validate_action_plan
from autocad_mcp.llm.lmstudio_client import LMStudioClient

IMAGE_IMPORT_SYSTEM_PROMPT = """
You convert images into CAD-friendly intermediate representation (IR).
Return JSON only with this schema:
{
  "units": "unknown|mm|cm|m|in|ft",
  "layers": [{"name":"string","color":"optional string"}],
  "geometry": [
    {"type":"line","start":[x,y],"end":[x,y],"layer":"optional"},
    {"type":"circle","center":[x,y],"radius":number,"layer":"optional"},
    {"type":"arc","center":[x,y],"radius":number,"start_angle":number,"end_angle":number,"layer":"optional"},
    {"type":"polyline","points":[[x,y],...],"closed":false,"layer":"optional"},
    {"type":"rectangle","corner1":[x,y],"corner2":[x,y],"layer":"optional"},
    {"type":"ellipse","center":[x,y],"major_axis":[dx,dy],"ratio":number,"layer":"optional"}
  ],
  "annotations": [
    {"text":"string","point":[x,y],"height":number,"layer":"optional"}
  ],
  "notes":["string"]
}
Rules:
- Output valid JSON only; no markdown/prose.
- Include only detectable elements; avoid guessing hidden geometry.
- Keep coordinates in a consistent local 2D frame.
- Keep output compact and deterministic.
""".strip()
IMAGE_IMPORT_MAX_JSON_TOKENS = 2000


class ImageImportPipeline:
    """Builds executable CAD plans from image inputs through a strict IR stage."""

    def __init__(self, lm_client: LMStudioClient):
        self.lm_client = lm_client

    def extract_ir(
        self,
        user_prompt: str,
        image_paths: list[str] | None = None,
        image_b64_pngs: list[str] | None = None,
        autocad_context: str | None = None,
        on_token: Callable[[str], None] | None = None,
    ) -> dict[str, Any]:
        if not (image_paths or image_b64_pngs):
            raise ValueError("Image import requires at least one image.")
        prompt = self._compose_extraction_prompt(user_prompt, autocad_context=autocad_context)
        if on_token is not None:
            on_token("\n[image_import_ir]\n")
        raw_ir = self.lm_client.chat_json_with_images(
            IMAGE_IMPORT_SYSTEM_PROMPT,
            prompt,
            image_paths=image_paths,
            image_b64_pngs=image_b64_pngs,
            on_token=on_token,
            max_tokens=IMAGE_IMPORT_MAX_JSON_TOKENS,
        )
        return self._normalize_ir(raw_ir)

    def build_plan_from_ir(
        self,
        ir: dict[str, Any],
        backend_name: str | None = None,
        max_actions: int = 120,
    ) -> tuple[dict[str, Any], SafetyResult, dict[str, Any]]:
        normalized_ir = self._normalize_ir(ir)
        actions = self._ir_to_actions(normalized_ir, max_actions=max_actions)
        plan: dict[str, Any] = {
            "analysis": {
                "source": "image_import",
                "units": normalized_ir.get("units", "unknown"),
                "geometry_count": len(normalized_ir.get("geometry", [])),
                "annotation_count": len(normalized_ir.get("annotations", [])),
                "notes": normalized_ir.get("notes", []),
                "bounds": normalized_ir.get("bounds"),
            },
            "actions": actions,
            "notes": "Generated from image import IR.",
        }
        safety = validate_action_plan(plan, backend_name=backend_name, max_actions=max_actions)
        qa = self.evaluate_plan_qa(
            plan=plan,
            ir=normalized_ir,
            safety=safety,
            max_actions=max_actions,
        )
        return plan, safety, qa

    @staticmethod
    def evaluate_plan_qa(
        plan: dict[str, Any],
        ir: dict[str, Any],
        safety: SafetyResult,
        max_actions: int = 120,
    ) -> dict[str, Any]:
        blocking_errors: list[str] = []
        non_blocking_errors: list[str] = []
        warnings: list[str] = []
        actions = plan.get("actions")
        geometry = ir.get("geometry")
        annotations = ir.get("annotations")
        bounds = ir.get("bounds")

        if not isinstance(geometry, list) or len(geometry) == 0:
            message = "No geometry was extracted from the image."
            non_blocking_errors.append(message)
            warnings.append(message)
        if not isinstance(actions, list) or len(actions) == 0:
            blocking_errors.append("No CAD actions were generated from extracted IR.")
        if isinstance(actions, list) and len(actions) > max_actions:
            blocking_errors.append(f"Generated {len(actions)} actions; exceeds limit {max_actions}.")
        if not safety.ok:
            blocking_errors.extend(safety.errors)

        if isinstance(bounds, dict):
            min_x = _as_float(bounds.get("min_x"))
            min_y = _as_float(bounds.get("min_y"))
            max_x = _as_float(bounds.get("max_x"))
            max_y = _as_float(bounds.get("max_y"))
            if None not in (min_x, min_y, max_x, max_y):
                span_x = float(max_x) - float(min_x)
                span_y = float(max_y) - float(min_y)
                if span_x <= 0 or span_y <= 0:
                    warnings.append("Extracted geometry bounds are degenerate.")
                if span_x > 1_000_000 or span_y > 1_000_000:
                    warnings.append("Extracted geometry bounds are unusually large.")
                if not all(math.isfinite(v) for v in (span_x, span_y)):
                    blocking_errors.append("Extracted bounds contain non-finite values.")

        if isinstance(annotations, list) and len(annotations) > 80:
            warnings.append("Large text annotation count detected; review before execution.")
        all_errors = [*blocking_errors, *non_blocking_errors]

        return {
            "ok": len(blocking_errors) == 0,
            "errors": all_errors,
            "blocking_errors": blocking_errors,
            "non_blocking_errors": non_blocking_errors,
            "warnings": warnings,
            "stats": {
                "geometry_count": len(geometry) if isinstance(geometry, list) else 0,
                "annotation_count": len(annotations) if isinstance(annotations, list) else 0,
                "action_count": len(actions) if isinstance(actions, list) else 0,
            },
        }

    def _compose_extraction_prompt(self, user_prompt: str, autocad_context: str | None = None) -> str:
        prompt = (user_prompt or "").strip()
        if not prompt:
            prompt = "Convert this image into accurate CAD geometry and text."
        if not autocad_context:
            return prompt
        trimmed_context = autocad_context.strip()
        if len(trimmed_context) > 2500:
            trimmed_context = trimmed_context[:2500] + "\n...[truncated]"
        return (
            f"User objective:\n{prompt}\n\n"
            "Current drawing context (optional grounding):\n"
            f"{trimmed_context}\n\n"
            "Extract image geometry into the required IR schema."
        )

    def _ir_to_actions(self, ir: dict[str, Any], max_actions: int = 120) -> list[dict[str, Any]]:
        actions: list[dict[str, Any]] = []
        layer_specs = self._merge_layer_specs(ir)
        for layer in layer_specs:
            actions.append(
                {
                    "tool": "layer",
                    "operation": "create",
                    "data": {
                        "name": layer["name"],
                        "color": layer.get("color") or "white",
                        "linetype": "CONTINUOUS",
                    },
                }
            )

        geometry_items = ir.get("geometry")
        if isinstance(geometry_items, list):
            for item in geometry_items:
                if len(actions) >= max_actions:
                    break
                mapped = self._map_geometry_item_to_action(item)
                if mapped is not None:
                    actions.append(mapped)

        annotation_items = ir.get("annotations")
        if isinstance(annotation_items, list):
            for item in annotation_items:
                if len(actions) >= max_actions:
                    break
                mapped = self._map_annotation_item_to_action(item)
                if mapped is not None:
                    actions.append(mapped)

        return actions

    def _merge_layer_specs(self, ir: dict[str, Any]) -> list[dict[str, str]]:
        merged: dict[str, dict[str, str]] = {}
        layers = ir.get("layers")
        if isinstance(layers, list):
            for layer in layers:
                if not isinstance(layer, dict):
                    continue
                name = _sanitize_layer_name(layer.get("name"))
                if not name:
                    continue
                entry = merged.setdefault(name, {"name": name, "color": "white"})
                color = layer.get("color")
                if isinstance(color, str) and color.strip():
                    entry["color"] = color.strip()

        for item in ir.get("geometry", []):
            if not isinstance(item, dict):
                continue
            name = _sanitize_layer_name(item.get("layer"))
            if not name:
                continue
            merged.setdefault(name, {"name": name, "color": "white"})
        for item in ir.get("annotations", []):
            if not isinstance(item, dict):
                continue
            name = _sanitize_layer_name(item.get("layer"))
            if not name:
                continue
            merged.setdefault(name, {"name": name, "color": "white"})
        return sorted(merged.values(), key=lambda layer: layer["name"])

    def _map_geometry_item_to_action(self, item: Any) -> dict[str, Any] | None:
        if not isinstance(item, dict):
            return None
        item_type = str(item.get("type") or "").strip().lower()
        layer = _sanitize_layer_name(item.get("layer"))

        if item_type == "line":
            start = _coerce_point(item.get("start"))
            end = _coerce_point(item.get("end"))
            if start is None or end is None:
                return None
            return {
                "tool": "entity",
                "operation": "create_line",
                "data": {
                    "x1": start[0],
                    "y1": start[1],
                    "x2": end[0],
                    "y2": end[1],
                    "layer": layer,
                },
                "expected_effects": {"entity_count_delta": 1, "entity_type_created": "LINE"},
            }

        if item_type == "circle":
            center = _coerce_point(item.get("center"))
            radius = _as_float(item.get("radius"))
            if center is None or radius is None or radius <= 0:
                return None
            return {
                "tool": "entity",
                "operation": "create_circle",
                "data": {"cx": center[0], "cy": center[1], "radius": radius, "layer": layer},
                "expected_effects": {"entity_count_delta": 1, "entity_type_created": "CIRCLE"},
            }

        if item_type == "arc":
            center = _coerce_point(item.get("center"))
            radius = _as_float(item.get("radius"))
            start_angle = _as_float(item.get("start_angle"))
            end_angle = _as_float(item.get("end_angle"))
            if center is None or radius is None or radius <= 0 or start_angle is None or end_angle is None:
                return None
            return {
                "tool": "entity",
                "operation": "create_arc",
                "data": {
                    "cx": center[0],
                    "cy": center[1],
                    "radius": radius,
                    "start_angle": start_angle,
                    "end_angle": end_angle,
                    "layer": layer,
                },
                "expected_effects": {"entity_count_delta": 1, "entity_type_created": "ARC"},
            }

        if item_type == "polyline":
            points_raw = item.get("points")
            if not isinstance(points_raw, list):
                return None
            points = [point for point in (_coerce_point(value) for value in points_raw) if point is not None]
            if len(points) < 2:
                return None
            closed = bool(item.get("closed", False))
            return {
                "tool": "entity",
                "operation": "create_polyline",
                "data": {
                    "points": [[point[0], point[1]] for point in points],
                    "closed": closed,
                    "layer": layer,
                },
                "expected_effects": {"entity_count_delta": 1, "entity_type_created": "POLYLINE"},
            }

        if item_type == "rectangle":
            corner1 = _coerce_point(item.get("corner1"))
            corner2 = _coerce_point(item.get("corner2"))
            if corner1 is None or corner2 is None:
                return None
            return {
                "tool": "entity",
                "operation": "create_rectangle",
                "data": {
                    "x1": corner1[0],
                    "y1": corner1[1],
                    "x2": corner2[0],
                    "y2": corner2[1],
                    "layer": layer,
                },
                "expected_effects": {"entity_count_delta": 1, "entity_type_created": "LWPOLYLINE"},
            }

        if item_type == "ellipse":
            center = _coerce_point(item.get("center"))
            major_axis = _coerce_point(item.get("major_axis"))
            ratio = _as_float(item.get("ratio"))
            if center is None or major_axis is None or ratio is None:
                return None
            if ratio <= 0 or ratio > 1:
                return None
            return {
                "tool": "entity",
                "operation": "create_ellipse",
                "data": {
                    "cx": center[0],
                    "cy": center[1],
                    "major_x": major_axis[0],
                    "major_y": major_axis[1],
                    "ratio": ratio,
                    "layer": layer,
                },
                "expected_effects": {"entity_count_delta": 1, "entity_type_created": "ELLIPSE"},
            }

        return None

    def _map_annotation_item_to_action(self, item: Any) -> dict[str, Any] | None:
        if not isinstance(item, dict):
            return None
        text_value = item.get("text")
        if not isinstance(text_value, str) or not text_value.strip():
            return None
        point = _coerce_point(item.get("point"))
        if point is None:
            return None
        layer = _sanitize_layer_name(item.get("layer"))
        height = _as_float(item.get("height"))
        if height is None or height <= 0:
            height = 2.5
        return {
            "tool": "annotation",
            "operation": "create_text",
            "data": {
                "x": point[0],
                "y": point[1],
                "text": text_value.strip(),
                "height": height,
                "rotation": 0.0,
                "layer": layer,
            },
            "expected_effects": {"entity_count_delta": 1, "entity_type_created": "TEXT"},
        }

    def _normalize_ir(self, raw_ir: Any) -> dict[str, Any]:
        if not isinstance(raw_ir, dict):
            return {
                "units": "unknown",
                "layers": [],
                "geometry": [],
                "annotations": [],
                "notes": ["image_ir_non_object"],
                "bounds": None,
            }

        units_raw = str(raw_ir.get("units") or "unknown").strip().lower()
        units = units_raw if units_raw in {"unknown", "mm", "cm", "m", "in", "ft"} else "unknown"

        layers: list[dict[str, str]] = []
        if isinstance(raw_ir.get("layers"), list):
            for layer in raw_ir["layers"]:
                if not isinstance(layer, dict):
                    continue
                name = _sanitize_layer_name(layer.get("name"))
                if not name:
                    continue
                color = layer.get("color")
                entry = {"name": name}
                if isinstance(color, str) and color.strip():
                    entry["color"] = color.strip()
                layers.append(entry)

        geometry: list[dict[str, Any]] = []
        points_for_bounds: list[tuple[float, float]] = []
        raw_geometry_items = _extract_items(
            raw_ir,
            ("geometry", "geometries", "entities", "objects", "shapes", "primitives"),
        )
        for item in raw_geometry_items:
            if not isinstance(item, dict):
                continue
            normalized = _normalize_geometry_item(item)
            if normalized is None:
                continue
            geometry.append(normalized)
            points_for_bounds.extend(_collect_points_from_geometry_item(normalized))

        annotations: list[dict[str, Any]] = []
        raw_annotation_items = _extract_items(
            raw_ir,
            ("annotations", "annotation", "texts", "labels"),
        )
        for item in raw_annotation_items:
            if not isinstance(item, dict):
                continue
            normalized = _normalize_annotation_item(item)
            if normalized is None:
                continue
            annotations.append(normalized)
            point = _coerce_point(normalized.get("point"))
            if point is not None:
                points_for_bounds.append(point)

        notes: list[str] = []
        raw_notes = raw_ir.get("notes")
        if isinstance(raw_notes, list):
            for value in raw_notes:
                if isinstance(value, str) and value.strip():
                    notes.append(value.strip())

        bounds = _compute_bounds(points_for_bounds)
        return {
            "units": units,
            "layers": layers,
            "geometry": geometry,
            "annotations": annotations,
            "notes": notes,
            "bounds": bounds,
        }


def _as_float(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        numeric = float(value)
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            numeric = float(text)
        except ValueError:
            match = re.search(r"-?\d+(?:\.\d+)?", text)
            if not match:
                return None
            try:
                numeric = float(match.group(0))
            except ValueError:
                return None
    else:
        return None
    if not math.isfinite(numeric):
        return None
    return numeric


def _coerce_point(value: Any) -> tuple[float, float] | None:
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        x_val = _as_float(value[0])
        y_val = _as_float(value[1])
        if x_val is not None and y_val is not None:
            return x_val, y_val
    if isinstance(value, dict):
        key_pairs = (
            ("x", "y"),
            ("cx", "cy"),
            ("center_x", "center_y"),
            ("x1", "y1"),
            ("start_x", "start_y"),
            ("from_x", "from_y"),
        )
        for x_key, y_key in key_pairs:
            x_val = _as_float(value.get(x_key))
            y_val = _as_float(value.get(y_key))
            if x_val is not None and y_val is not None:
                return x_val, y_val
    if isinstance(value, str):
        match = re.search(r"\(?\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)?", value)
        if match:
            x_val = _as_float(match.group(1))
            y_val = _as_float(match.group(2))
            if x_val is not None and y_val is not None:
                return x_val, y_val
    return None


def _sanitize_layer_name(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    cleaned = "".join(ch for ch in text if ch.isalnum() or ch in ("_", "-", "."))
    if not cleaned:
        return None
    return cleaned[:64]


def _extract_items(raw_ir: dict[str, Any], keys: tuple[str, ...]) -> list[Any]:
    for key in keys:
        value = raw_ir.get(key)
        if isinstance(value, list):
            return value
        if isinstance(value, dict):
            return [value]
    return []


def _first_present_value(item: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in item and item.get(key) is not None:
            return item.get(key)
    return None


def _normalize_geometry_type(value: Any) -> str:
    raw = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "segment": "line",
        "edge": "line",
        "lin": "line",
        "rect": "rectangle",
        "box": "rectangle",
        "square": "rectangle",
        "oval": "ellipse",
        "poly": "polyline",
        "polygon": "polyline",
        "path": "polyline",
    }
    return aliases.get(raw, raw)


def _normalize_geometry_item(item: dict[str, Any]) -> dict[str, Any] | None:
    item_type = _normalize_geometry_type(
        _first_present_value(item, ("type", "kind", "shape", "entity_type", "geometry_type", "object_type"))
    )
    if not item_type:
        return None

    layer = _sanitize_layer_name(
        _first_present_value(item, ("layer", "layer_name", "target_layer", "entity_layer"))
    )
    normalized: dict[str, Any] = {"type": item_type}
    if layer:
        normalized["layer"] = layer

    if item_type == "line":
        start = _coerce_point(_first_present_value(item, ("start", "from", "p1", "point1", "a")))
        end = _coerce_point(_first_present_value(item, ("end", "to", "p2", "point2", "b")))
        if (start is None or end is None) and isinstance(item.get("points"), list):
            points = [_coerce_point(value) for value in item.get("points", [])]
            points = [point for point in points if point is not None]
            if len(points) >= 2:
                start = start or points[0]
                end = end or points[1]
        if start is None or end is None:
            return None
        normalized["start"] = [start[0], start[1]]
        normalized["end"] = [end[0], end[1]]
        return normalized

    if item_type == "circle":
        center = _coerce_point(_first_present_value(item, ("center", "origin", "point", "position", "c")))
        radius = _as_float(_first_present_value(item, ("radius", "r")))
        if radius is None:
            diameter = _as_float(_first_present_value(item, ("diameter", "d")))
            if diameter is not None:
                radius = diameter / 2.0
        if center is None or radius is None or radius <= 0:
            return None
        normalized["center"] = [center[0], center[1]]
        normalized["radius"] = radius
        return normalized

    if item_type == "arc":
        center = _coerce_point(_first_present_value(item, ("center", "origin", "point", "position", "c")))
        radius = _as_float(_first_present_value(item, ("radius", "r")))
        if radius is None:
            diameter = _as_float(_first_present_value(item, ("diameter", "d")))
            if diameter is not None:
                radius = diameter / 2.0
        start_angle = _as_float(_first_present_value(item, ("start_angle", "angle_start", "from_angle", "start")))
        end_angle = _as_float(_first_present_value(item, ("end_angle", "angle_end", "to_angle", "end")))
        if end_angle is None:
            sweep = _as_float(_first_present_value(item, ("sweep_angle", "delta_angle")))
            if sweep is not None and start_angle is not None:
                end_angle = start_angle + sweep
        if center is None or radius is None or radius <= 0 or start_angle is None or end_angle is None:
            return None
        normalized["center"] = [center[0], center[1]]
        normalized["radius"] = radius
        normalized["start_angle"] = start_angle
        normalized["end_angle"] = end_angle
        return normalized

    if item_type == "polyline":
        points_raw = _first_present_value(item, ("points", "vertices", "path", "polyline", "coords"))
        points: list[tuple[float, float]] = []
        if isinstance(points_raw, list):
            points = [point for point in (_coerce_point(value) for value in points_raw) if point is not None]
        if len(points) < 2:
            return None
        normalized["points"] = [[point[0], point[1]] for point in points]
        normalized["closed"] = bool(_first_present_value(item, ("closed", "is_closed", "isClosed")) or False)
        return normalized

    if item_type == "rectangle":
        corner1 = _coerce_point(_first_present_value(item, ("corner1", "start", "p1", "point1", "min", "lower_left")))
        corner2 = _coerce_point(_first_present_value(item, ("corner2", "end", "p2", "point2", "max", "upper_right")))
        if corner1 is None or corner2 is None:
            origin = _coerce_point(_first_present_value(item, ("origin", "insertion_point", "insert_point")))
            width = _as_float(_first_present_value(item, ("width", "w")))
            height = _as_float(_first_present_value(item, ("height", "h")))
            if origin is not None and width is not None and height is not None:
                corner1 = origin
                corner2 = (origin[0] + width, origin[1] + height)
        if corner1 is None or corner2 is None:
            return None
        normalized["corner1"] = [corner1[0], corner1[1]]
        normalized["corner2"] = [corner2[0], corner2[1]]
        return normalized

    if item_type == "ellipse":
        center = _coerce_point(_first_present_value(item, ("center", "origin", "point", "position", "c")))
        major_axis = _coerce_point(_first_present_value(item, ("major_axis", "axis", "major", "axis_vector")))
        ratio = _as_float(_first_present_value(item, ("ratio", "minor_major_ratio")))
        if ratio is None:
            major_radius = _as_float(_first_present_value(item, ("major_radius", "a")))
            minor_radius = _as_float(_first_present_value(item, ("minor_radius", "b")))
            if major_radius is not None and minor_radius is not None and major_radius != 0:
                ratio = minor_radius / major_radius
        if center is None or major_axis is None or ratio is None or ratio <= 0 or ratio > 1:
            return None
        normalized["center"] = [center[0], center[1]]
        normalized["major_axis"] = [major_axis[0], major_axis[1]]
        normalized["ratio"] = ratio
        return normalized

    return None


def _normalize_annotation_item(item: dict[str, Any]) -> dict[str, Any] | None:
    text_value = _first_present_value(item, ("text", "content", "label", "value", "string"))
    if not isinstance(text_value, str) or not text_value.strip():
        return None
    point = _coerce_point(
        _first_present_value(item, ("point", "position", "location", "at", "insert", "insertion_point", "origin"))
    )
    if point is None:
        return None
    layer = _sanitize_layer_name(
        _first_present_value(item, ("layer", "layer_name", "target_layer", "entity_layer"))
    )
    height = _as_float(_first_present_value(item, ("height", "text_height", "size")))
    annotation: dict[str, Any] = {
        "text": text_value.strip(),
        "point": [point[0], point[1]],
    }
    if layer:
        annotation["layer"] = layer
    if height is not None and height > 0:
        annotation["height"] = height
    return annotation


def _collect_points_from_geometry_item(item: dict[str, Any]) -> list[tuple[float, float]]:
    item_type = str(item.get("type") or "").strip().lower()
    points: list[tuple[float, float]] = []
    if item_type == "line":
        start = _coerce_point(item.get("start"))
        end = _coerce_point(item.get("end"))
        if start is not None:
            points.append(start)
        if end is not None:
            points.append(end)
    elif item_type in {"circle", "arc", "ellipse"}:
        center = _coerce_point(item.get("center"))
        if center is not None:
            points.append(center)
    elif item_type == "polyline":
        points_raw = item.get("points")
        if isinstance(points_raw, list):
            for value in points_raw:
                point = _coerce_point(value)
                if point is not None:
                    points.append(point)
    elif item_type == "rectangle":
        corner1 = _coerce_point(item.get("corner1"))
        corner2 = _coerce_point(item.get("corner2"))
        if corner1 is not None:
            points.append(corner1)
        if corner2 is not None:
            points.append(corner2)
    return points


def _compute_bounds(points: list[tuple[float, float]]) -> dict[str, float] | None:
    if not points:
        return None
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return {
        "min_x": min(xs),
        "min_y": min(ys),
        "max_x": max(xs),
        "max_y": max(ys),
    }
