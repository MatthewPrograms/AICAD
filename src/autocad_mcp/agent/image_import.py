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
- Handle diverse image styles (technical drawing, sketch, logo, and photo) by prioritizing high-confidence structural geometry.
- Prefer stable primitives over noisy fragments and avoid duplicate entities describing the same shape.
- Preserve likely orthogonality for near-horizontal and near-vertical edges when visually clear.
- Keep output compact and deterministic.
""".strip()

IMAGE_IMPORT_REFINEMENT_SYSTEM_PROMPT = """
You refine an existing image-to-CAD IR into a more accurate CAD-friendly IR.
Return JSON only using the same schema as the extraction stage.
Refinement priorities:
- Keep high-confidence geometry and key annotations.
- Remove obvious noise, malformed entities, and duplicates.
- Preserve shape/layout fidelity over maximum entity count.
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
        primary_raw_ir = self.lm_client.chat_json_with_images(
            IMAGE_IMPORT_SYSTEM_PROMPT,
            prompt,
            image_paths=image_paths,
            image_b64_pngs=image_b64_pngs,
            on_token=on_token,
            max_tokens=IMAGE_IMPORT_MAX_JSON_TOKENS,
        )
        primary_ir = self._normalize_ir(primary_raw_ir)
        if not self._should_attempt_refinement(primary_ir):
            return primary_ir

        if on_token is not None:
            on_token("\n[image_import_ir_refine]\n")
        refinement_prompt = self._compose_refinement_prompt(
            user_prompt=user_prompt,
            candidate_ir=primary_ir,
            autocad_context=autocad_context,
        )
        try:
            refined_raw_ir = self.lm_client.chat_json_with_images(
                IMAGE_IMPORT_REFINEMENT_SYSTEM_PROMPT,
                refinement_prompt,
                image_paths=image_paths,
                image_b64_pngs=image_b64_pngs,
                on_token=on_token,
                max_tokens=IMAGE_IMPORT_MAX_JSON_TOKENS,
            )
        except Exception:
            return primary_ir
        refined_ir = self._normalize_ir(refined_raw_ir)
        return self._select_better_ir(primary_ir, refined_ir)

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

    def _compose_refinement_prompt(
        self,
        user_prompt: str,
        candidate_ir: dict[str, Any],
        autocad_context: str | None = None,
    ) -> str:
        objective = (user_prompt or "").strip() or "Convert this image into accurate CAD geometry and text."
        candidate_preview = str(candidate_ir)
        if len(candidate_preview) > 8000:
            candidate_preview = candidate_preview[:8000] + "...[truncated]"
        if not autocad_context:
            return (
                f"User objective:\n{objective}\n\n"
                "Candidate IR to refine:\n"
                f"{candidate_preview}\n\n"
                "Refine this IR to improve geometric accuracy and remove noisy artifacts."
            )
        trimmed_context = autocad_context.strip()
        if len(trimmed_context) > 2500:
            trimmed_context = trimmed_context[:2500] + "\n...[truncated]"
        return (
            f"User objective:\n{objective}\n\n"
            "Current drawing context (optional grounding):\n"
            f"{trimmed_context}\n\n"
            "Candidate IR to refine:\n"
            f"{candidate_preview}\n\n"
            "Refine this IR to improve geometric accuracy and remove noisy artifacts."
        )

    def _should_attempt_refinement(self, ir: dict[str, Any]) -> bool:
        geometry = ir.get("geometry")
        geometry_count = len(geometry) if isinstance(geometry, list) else 0
        quality_score = self._score_ir_quality(ir)
        return geometry_count < 4 or quality_score < 8.0

    def _select_better_ir(self, primary_ir: dict[str, Any], refined_ir: dict[str, Any]) -> dict[str, Any]:
        if self._score_ir_quality(refined_ir) > self._score_ir_quality(primary_ir):
            return refined_ir
        return primary_ir

    def _score_ir_quality(self, ir: dict[str, Any]) -> float:
        geometry_items = ir.get("geometry") if isinstance(ir.get("geometry"), list) else []
        annotation_items = ir.get("annotations") if isinstance(ir.get("annotations"), list) else []
        note_items = ir.get("notes") if isinstance(ir.get("notes"), list) else []
        bounds = ir.get("bounds")

        score = 0.0
        geometry_count = len(geometry_items)
        score += min(60, geometry_count) * 0.8
        score += min(20, len(annotation_items)) * 0.15
        if geometry_count == 0:
            score -= 20.0
        elif geometry_count > 220:
            score -= (geometry_count - 220) * 0.15

        geometry_types = {
            str(item.get("type")).strip().lower()
            for item in geometry_items
            if isinstance(item, dict) and item.get("type")
        }
        score += min(8, len(geometry_types)) * 1.25
        score -= _estimate_duplicate_geometry_count(geometry_items) * 1.5

        if isinstance(bounds, dict):
            min_x = _as_float(bounds.get("min_x"))
            min_y = _as_float(bounds.get("min_y"))
            max_x = _as_float(bounds.get("max_x"))
            max_y = _as_float(bounds.get("max_y"))
            if None not in (min_x, min_y, max_x, max_y):
                span_x = float(max_x) - float(min_x)
                span_y = float(max_y) - float(min_y)
                if span_x > 0 and span_y > 0 and math.isfinite(span_x) and math.isfinite(span_y):
                    score += 3.0
                else:
                    score -= 6.0
            else:
                score -= 4.0

        if any(isinstance(note, str) and "non_object" in note.lower() for note in note_items):
            score -= 15.0
        return score

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
        geometry = _post_process_geometry(geometry)

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
        annotations = _post_process_annotations(annotations)

        notes: list[str] = []
        raw_notes = raw_ir.get("notes")
        if isinstance(raw_notes, list):
            for value in raw_notes:
                if isinstance(value, str) and value.strip():
                    notes.append(value.strip())

        points_for_bounds: list[tuple[float, float]] = []
        for item in geometry:
            points_for_bounds.extend(_collect_points_from_geometry_item(item))
        for item in annotations:
            point = _coerce_point(item.get("point"))
            if point is not None:
                points_for_bounds.append(point)
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


def _post_process_geometry(geometry: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cleaned: list[dict[str, Any]] = []
    for item in geometry:
        normalized = _sanitize_geometry_for_accuracy(item)
        if normalized is not None:
            cleaned.append(normalized)

    deduped: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for item in cleaned:
        signature = _geometry_signature(item)
        if signature in seen:
            continue
        seen.add(signature)
        deduped.append(item)
    return deduped


def _post_process_annotations(annotations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cleaned: list[dict[str, Any]] = []
    seen: set[tuple[str, float, float]] = set()
    for item in annotations:
        text_value = item.get("text")
        point = _coerce_point(item.get("point"))
        if not isinstance(text_value, str) or not text_value.strip() or point is None:
            continue
        normalized_text = text_value.strip()
        if len(normalized_text) > 160:
            normalized_text = normalized_text[:160]
        height = _as_float(item.get("height"))
        if height is None or height <= 0:
            height = 2.5
        height = max(0.1, min(height, 1000.0))
        signature = (normalized_text.lower(), _quantize(point[0]), _quantize(point[1]))
        if signature in seen:
            continue
        seen.add(signature)
        normalized_item: dict[str, Any] = {
            "text": normalized_text,
            "point": [_round4(point[0]), _round4(point[1])],
            "height": _round4(height),
        }
        layer = _sanitize_layer_name(item.get("layer"))
        if layer:
            normalized_item["layer"] = layer
        cleaned.append(normalized_item)
    return cleaned


def _sanitize_geometry_for_accuracy(item: dict[str, Any]) -> dict[str, Any] | None:
    item_type = str(item.get("type") or "").strip().lower()
    layer = _sanitize_layer_name(item.get("layer"))
    base: dict[str, Any] = {"type": item_type}
    if layer:
        base["layer"] = layer

    if item_type == "line":
        start = _coerce_point(item.get("start"))
        end = _coerce_point(item.get("end"))
        if start is None or end is None:
            return None
        x1, y1 = start
        x2, y2 = end
        dx = x2 - x1
        dy = y2 - y1
        length = math.hypot(dx, dy)
        if length <= 1e-4:
            return None
        if abs(dx) <= length * 0.015:
            x = (x1 + x2) / 2.0
            x1 = x
            x2 = x
        elif abs(dy) <= length * 0.015:
            y = (y1 + y2) / 2.0
            y1 = y
            y2 = y
        if math.hypot(x2 - x1, y2 - y1) <= 1e-4:
            return None
        base["start"] = [_round4(x1), _round4(y1)]
        base["end"] = [_round4(x2), _round4(y2)]
        return base

    if item_type == "circle":
        center = _coerce_point(item.get("center"))
        radius = _as_float(item.get("radius"))
        if center is None or radius is None or radius <= 1e-4:
            return None
        base["center"] = [_round4(center[0]), _round4(center[1])]
        base["radius"] = _round4(radius)
        return base

    if item_type == "arc":
        center = _coerce_point(item.get("center"))
        radius = _as_float(item.get("radius"))
        start_angle = _as_float(item.get("start_angle"))
        end_angle = _as_float(item.get("end_angle"))
        if center is None or radius is None or radius <= 1e-4 or start_angle is None or end_angle is None:
            return None
        sweep = (end_angle - start_angle) % 360.0
        if sweep < 0.5:
            return None
        base["center"] = [_round4(center[0]), _round4(center[1])]
        base["radius"] = _round4(radius)
        base["start_angle"] = _round4(start_angle)
        base["end_angle"] = _round4(end_angle)
        return base

    if item_type == "polyline":
        points_raw = item.get("points")
        if not isinstance(points_raw, list):
            return None
        points: list[tuple[float, float]] = []
        for value in points_raw:
            point = _coerce_point(value)
            if point is None:
                continue
            if points and _points_close(point, points[-1], tolerance=1e-4):
                continue
            points.append(point)
        if len(points) < 2:
            return None
        closed = bool(item.get("closed", False))
        if closed and _points_close(points[0], points[-1], tolerance=1e-4):
            points = points[:-1]
        if len(points) < 2:
            return None
        if closed and len(points) < 3:
            return None
        base["points"] = [[_round4(point[0]), _round4(point[1])] for point in points]
        base["closed"] = closed
        return base

    if item_type == "rectangle":
        corner1 = _coerce_point(item.get("corner1"))
        corner2 = _coerce_point(item.get("corner2"))
        if corner1 is None or corner2 is None:
            return None
        min_x = min(corner1[0], corner2[0])
        min_y = min(corner1[1], corner2[1])
        max_x = max(corner1[0], corner2[0])
        max_y = max(corner1[1], corner2[1])
        if abs(max_x - min_x) <= 1e-4 or abs(max_y - min_y) <= 1e-4:
            return None
        base["corner1"] = [_round4(min_x), _round4(min_y)]
        base["corner2"] = [_round4(max_x), _round4(max_y)]
        return base

    if item_type == "ellipse":
        center = _coerce_point(item.get("center"))
        major_axis = _coerce_point(item.get("major_axis"))
        ratio = _as_float(item.get("ratio"))
        if center is None or major_axis is None or ratio is None:
            return None
        if math.hypot(major_axis[0], major_axis[1]) <= 1e-4:
            return None
        if ratio <= 1e-4 or ratio > 1.0:
            return None
        base["center"] = [_round4(center[0]), _round4(center[1])]
        base["major_axis"] = [_round4(major_axis[0]), _round4(major_axis[1])]
        base["ratio"] = _round4(ratio)
        return base

    return None


def _geometry_signature(item: dict[str, Any]) -> tuple[Any, ...]:
    item_type = str(item.get("type") or "").strip().lower()
    if item_type == "line":
        start = _coerce_point(item.get("start"))
        end = _coerce_point(item.get("end"))
        if start is None or end is None:
            return ("line", "invalid")
        a = (_quantize(start[0]), _quantize(start[1]))
        b = (_quantize(end[0]), _quantize(end[1]))
        ordered = tuple(sorted((a, b)))
        return ("line", *ordered, str(item.get("layer") or ""))
    if item_type == "circle":
        center = _coerce_point(item.get("center"))
        radius = _as_float(item.get("radius"))
        if center is None or radius is None:
            return ("circle", "invalid")
        return (
            "circle",
            _quantize(center[0]),
            _quantize(center[1]),
            _quantize(radius),
            str(item.get("layer") or ""),
        )
    if item_type == "arc":
        center = _coerce_point(item.get("center"))
        radius = _as_float(item.get("radius"))
        start_angle = _as_float(item.get("start_angle"))
        end_angle = _as_float(item.get("end_angle"))
        if center is None or radius is None or start_angle is None or end_angle is None:
            return ("arc", "invalid")
        return (
            "arc",
            _quantize(center[0]),
            _quantize(center[1]),
            _quantize(radius),
            _quantize(start_angle),
            _quantize(end_angle),
            str(item.get("layer") or ""),
        )
    if item_type == "rectangle":
        corner1 = _coerce_point(item.get("corner1"))
        corner2 = _coerce_point(item.get("corner2"))
        if corner1 is None or corner2 is None:
            return ("rectangle", "invalid")
        x1, y1 = corner1
        x2, y2 = corner2
        return (
            "rectangle",
            _quantize(min(x1, x2)),
            _quantize(min(y1, y2)),
            _quantize(max(x1, x2)),
            _quantize(max(y1, y2)),
            str(item.get("layer") or ""),
        )
    if item_type == "polyline":
        points_raw = item.get("points")
        if not isinstance(points_raw, list):
            return ("polyline", "invalid")
        points = [point for point in (_coerce_point(value) for value in points_raw) if point is not None]
        encoded = tuple((_quantize(point[0]), _quantize(point[1])) for point in points)
        reverse_encoded = tuple(reversed(encoded))
        canonical = min(encoded, reverse_encoded) if reverse_encoded else encoded
        return ("polyline", canonical, bool(item.get("closed", False)), str(item.get("layer") or ""))
    if item_type == "ellipse":
        center = _coerce_point(item.get("center"))
        major_axis = _coerce_point(item.get("major_axis"))
        ratio = _as_float(item.get("ratio"))
        if center is None or major_axis is None or ratio is None:
            return ("ellipse", "invalid")
        return (
            "ellipse",
            _quantize(center[0]),
            _quantize(center[1]),
            _quantize(major_axis[0]),
            _quantize(major_axis[1]),
            _quantize(ratio),
            str(item.get("layer") or ""),
        )
    return (item_type or "unknown", str(item))


def _estimate_duplicate_geometry_count(geometry: list[dict[str, Any]]) -> int:
    seen: set[tuple[Any, ...]] = set()
    duplicates = 0
    for item in geometry:
        if not isinstance(item, dict):
            continue
        signature = _geometry_signature(item)
        if signature in seen:
            duplicates += 1
            continue
        seen.add(signature)
    return duplicates


def _points_close(a: tuple[float, float], b: tuple[float, float], tolerance: float = 1e-4) -> bool:
    return math.hypot(a[0] - b[0], a[1] - b[1]) <= tolerance


def _quantize(value: float, precision: int = 3) -> float:
    return float(round(value, precision))


def _round4(value: float) -> float:
    return float(round(value, 4))
