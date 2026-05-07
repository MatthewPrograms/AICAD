"""Natural-language to structured action planner."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import random
import re
from typing import Any, Callable

from autocad_mcp.agent.safety import SafetyResult, validate_action_plan
from autocad_mcp.llm.lmstudio_client import LMStudioClient

CANONICAL_OPERATION_GUIDE = """
Use canonical operations only from this allowlist:
- drawing: create, open, info, save, save_as_dxf, plot_pdf, purge, get_variables, undo, redo
- entity: create_line, create_circle, create_polyline, create_rectangle, create_arc, create_ellipse, create_mtext, create_hatch, list, count, get, copy, move, rotate, scale, mirror, offset, array, fillet, chamfer, erase
- layer: list, create, set_current, set_properties, freeze, thaw, lock, unlock
- block: list, insert, insert_with_attributes, get_attributes, update_attribute, define
- annotation: create_text, create_dimension_linear, create_dimension_aligned, create_dimension_angular, create_dimension_radius, create_leader
- pid: setup_layers, insert_symbol, list_symbols, draw_process_line, connect_equipment, add_flow_arrow, add_equipment_tag, add_line_number, insert_valve, insert_instrument, insert_pump, insert_tank
- view: zoom_extents, zoom_window, get_screenshot
- system: status, health, get_backend, runtime, init
""".strip()

ANALYSIS_SYSTEM_PROMPT = f"""You are a CAD scene-analysis assistant.
Return JSON only in this schema:
{{
  "goal": "string",
  "intent_summary": "string",
  "constraints": ["string"],
  "target_selectors": [{{"type":"optional","layer":"optional","text":"optional","region":"optional","proximity":{{"x":0,"y":0,"max_distance":0}}}}],
  "suggested_operations": ["tool.operation"],
  "risks": ["string"]
}}
Rules:
- Output valid JSON only (no markdown/prose).
- Never output tokenizer artifacts like <unused24> or <unused...>.
- Keep this compact and practical for immediate CAD execution.
- Do not include system.execute_lisp.
- Select operations only from the canonical allowlist.
{CANONICAL_OPERATION_GUIDE}
"""

PLAN_SYSTEM_PROMPT = f"""You are a CAD execution planner.
Return JSON only in this schema:
{{
  "analysis": {{}},
  "actions": [
    {{
      "tool":"entity|drawing|layer|block|annotation|pid|view|system",
      "operation":"string",
      "data":{{}},
      "selector":{{"type":"optional","layer":"optional","text":"optional","region":"optional","proximity":{{"x":0,"y":0,"max_distance":0}}}},
      "expected_effects":{{"entity_count_delta":0,"entity_type_created":"optional","target_exists":"optional"}}
    }}
  ],
  "notes":"optional"
}}
Rules:
- Output valid JSON only (no markdown/prose).
- Never output tokenizer artifacts like <unused24> or <unused...>.
- Do not use system.execute_lisp.
- Use canonical operations only.
- Prefer explicit numeric coordinates and values.
- Include expected_effects for mutable operations whenever feasible.
- Keep plans short: usually <= 8 actions unless the request clearly needs more.
- For high-impact operations (erase, broad transforms), include explicit target handles or confirmation in data.confirm_high_impact=true.
{CANONICAL_OPERATION_GUIDE}
"""


@dataclass
class PlanResult:
    plan: dict[str, Any]
    safety: SafetyResult
    analysis: dict[str, Any] | None = None


class ActionPlanner:
    """Generates and validates action plans from user prompts."""

    def __init__(self, lm_client: LMStudioClient):
        self.lm_client = lm_client

    def create_plan(
        self,
        user_prompt: str,
        autocad_context: str | None = None,
        on_planning_token: Callable[[str], None] | None = None,
        backend_name: str | None = None,
    ) -> PlanResult:
        analysis = self._generate_analysis(
            user_prompt,
            autocad_context=autocad_context,
            on_planning_token=on_planning_token,
        )
        plan = self._generate_plan_from_analysis(
            user_prompt,
            analysis,
            autocad_context=autocad_context,
            on_planning_token=on_planning_token,
        )
        normalized_plan = self._normalize_plan_envelope(plan, analysis)
        safety = validate_action_plan(normalized_plan, backend_name=backend_name)
        return PlanResult(plan=normalized_plan, safety=safety, analysis=analysis)

    def create_noop_fallback_plan(
        self,
        user_prompt: str,
        backend_name: str | None = None,
    ) -> PlanResult:
        analysis = self._build_deterministic_analysis(user_prompt)
        plan = {
            "analysis": analysis,
            "actions": [],
            "notes": (
                "Planner fallback could not derive executable operations from the prompt "
                "after LM JSON failures."
            ),
        }
        safety = validate_action_plan(plan, backend_name=backend_name)
        return PlanResult(plan=plan, safety=safety, analysis=analysis)

    def create_plan_with_vision(
        self,
        user_prompt: str,
        image_paths: list[str] | None = None,
        image_b64_pngs: list[str] | None = None,
        autocad_context: str | None = None,
        on_planning_token: Callable[[str], None] | None = None,
        backend_name: str | None = None,
    ) -> PlanResult:
        analysis = self._generate_analysis(
            user_prompt,
            autocad_context=autocad_context,
            image_paths=image_paths,
            image_b64_pngs=image_b64_pngs,
            on_planning_token=on_planning_token,
        )
        plan = self._generate_plan_from_analysis(
            user_prompt,
            analysis,
            autocad_context=autocad_context,
            on_planning_token=on_planning_token,
        )
        normalized_plan = self._normalize_plan_envelope(plan, analysis)
        safety = validate_action_plan(normalized_plan, backend_name=backend_name)
        return PlanResult(plan=normalized_plan, safety=safety, analysis=analysis)

    def create_fallback_plan(
        self,
        user_prompt: str,
        autocad_context: str | None = None,
        backend_name: str | None = None,
    ) -> PlanResult | None:
        del autocad_context  # Fallback generation is deterministic and prompt-driven.
        analysis = self._build_deterministic_analysis(user_prompt)
        plan = self._build_deterministic_plan(user_prompt)
        if plan is None:
            return None
        normalized_plan = self._normalize_plan_envelope(plan, analysis)
        safety = validate_action_plan(normalized_plan, backend_name=backend_name)
        return PlanResult(plan=normalized_plan, safety=safety, analysis=analysis)

    def _generate_analysis(
        self,
        user_prompt: str,
        autocad_context: str | None = None,
        image_paths: list[str] | None = None,
        image_b64_pngs: list[str] | None = None,
        on_planning_token: Callable[[str], None] | None = None,
    ) -> dict[str, Any]:
        if on_planning_token is not None:
            on_planning_token("\n[analysis]\n")
        analysis_prompt = self._compose_analysis_prompt(user_prompt, autocad_context)
        if image_paths or image_b64_pngs:
            analysis_raw = self.lm_client.chat_json_with_images(
                ANALYSIS_SYSTEM_PROMPT,
                analysis_prompt,
                image_paths=image_paths,
                image_b64_pngs=image_b64_pngs,
                on_token=on_planning_token,
            )
        else:
            analysis_raw = self.lm_client.chat_json(
                ANALYSIS_SYSTEM_PROMPT,
                analysis_prompt,
                on_token=on_planning_token,
            )
        return self._sanitize_analysis(analysis_raw)

    def _generate_plan_from_analysis(
        self,
        user_prompt: str,
        analysis: dict[str, Any],
        autocad_context: str | None = None,
        on_planning_token: Callable[[str], None] | None = None,
    ) -> dict[str, Any]:
        if on_planning_token is not None:
            on_planning_token("\n[plan]\n")
        return self.lm_client.chat_json(
            PLAN_SYSTEM_PROMPT,
            self._compose_plan_prompt(user_prompt, analysis, autocad_context),
            on_token=on_planning_token,
        )

    @staticmethod
    def _compose_analysis_prompt(user_prompt: str, autocad_context: str | None = None) -> str:
        return ActionPlanner._compose_user_prompt(user_prompt, autocad_context)

    @staticmethod
    def _compose_plan_prompt(
        user_prompt: str,
        analysis: dict[str, Any],
        autocad_context: str | None = None,
    ) -> str:
        context_block = ActionPlanner._compose_user_prompt(user_prompt, autocad_context)
        return (
            f"{context_block}\n\n"
            "Structured scene/intent analysis (authoritative):\n"
            f"{json.dumps(analysis, ensure_ascii=False)}\n\n"
            "Generate executable actions that satisfy the analysis with minimal risk."
        )

    @staticmethod
    def _compose_user_prompt(user_prompt: str, autocad_context: str | None = None) -> str:
        trimmed_prompt = (user_prompt or "").strip()
        if not autocad_context:
            return trimmed_prompt
        context = autocad_context.strip()
        if len(context) > 3200:
            context = context[:3200] + "\n...[truncated for speed]"
        return (
            f"User request:\n{trimmed_prompt}\n\n"
            "Live AutoCAD drawing context (authoritative current state):\n"
            f"{context}\n\n"
            "Use this context directly when planning."
        )

    @staticmethod
    def _sanitize_analysis(raw_analysis: Any) -> dict[str, Any]:
        if not isinstance(raw_analysis, dict):
            return {
                "goal": "",
                "intent_summary": "",
                "constraints": [],
                "target_selectors": [],
                "suggested_operations": [],
                "risks": ["analysis_non_object"],
            }
        analysis = dict(raw_analysis)
        analysis.setdefault("goal", "")
        analysis.setdefault("intent_summary", "")
        if not isinstance(analysis.get("constraints"), list):
            analysis["constraints"] = []
        if not isinstance(analysis.get("target_selectors"), list):
            analysis["target_selectors"] = []
        if not isinstance(analysis.get("suggested_operations"), list):
            analysis["suggested_operations"] = []
        if not isinstance(analysis.get("risks"), list):
            analysis["risks"] = []
        return analysis

    @staticmethod
    def _normalize_plan_envelope(plan: Any, analysis: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(plan, dict):
            return {
                "analysis": analysis,
                "actions": [],
                "notes": "Planner returned non-object payload.",
            }
        normalized = dict(plan)
        if not isinstance(normalized.get("actions"), list):
            normalized["actions"] = []
        if not isinstance(normalized.get("analysis"), dict):
            normalized["analysis"] = analysis
        for action in normalized["actions"]:
            if not isinstance(action, dict):
                continue
            if "data" not in action or not isinstance(action.get("data"), dict):
                action["data"] = {}
        return normalized

    @staticmethod
    def _extract_primary_instruction(user_prompt: str) -> str:
        text = (user_prompt or "").strip()
        if not text:
            return ""

        lower = text.lower()
        latest_marker = "latest user message:"
        latest_index = lower.rfind(latest_marker)
        if latest_index >= 0:
            latest_block = text[latest_index + len(latest_marker) :].strip()
            if latest_block:
                return latest_block.splitlines()[0].strip()

        request_marker = "user request:"
        context_marker = "live autocad drawing context"
        request_index = lower.find(request_marker)
        if request_index >= 0:
            request_block = text[request_index + len(request_marker) :].strip()
            context_index = request_block.lower().find(context_marker)
            if context_index >= 0:
                request_block = request_block[:context_index]
            request_line = request_block.strip().splitlines()[0].strip() if request_block.strip() else ""
            if request_line:
                return request_line

        first_line = text.splitlines()[0].strip()
        return first_line or text

    @staticmethod
    def _extract_coordinate_pairs(text: str) -> list[tuple[float, float]]:
        matches = re.findall(r"\(?\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)?", text or "")
        pairs: list[tuple[float, float]] = []
        for x_raw, y_raw in matches:
            try:
                pairs.append((float(x_raw), float(y_raw)))
            except ValueError:
                continue
        return pairs

    @staticmethod
    def _extract_layer_name(text: str) -> str | None:
        match = re.search(r"\bon\s+layer\s+([A-Za-z0-9_.-]+)\b", text or "", flags=re.IGNORECASE)
        if not match:
            return None
        layer = match.group(1).strip()
        return layer or None

    @staticmethod
    def _extract_scalar_hint(text: str, keys: tuple[str, ...]) -> float | None:
        if not text:
            return None
        key_pattern = "|".join(re.escape(key) for key in keys)
        pattern = re.compile(
            rf"\b(?:{key_pattern})\b\s*(?:=|:|of)?\s*(-?\d+(?:\.\d+)?)",
            flags=re.IGNORECASE,
        )
        match = pattern.search(text)
        if not match:
            return None
        try:
            return float(match.group(1))
        except ValueError:
            return None

    @staticmethod
    def _extract_dimensions_hint(text: str) -> tuple[float, float] | None:
        if not text:
            return None
        dimension_patterns = (
            re.compile(r"\b(-?\d+(?:\.\d+)?)\s*(?:x|×)\s*(-?\d+(?:\.\d+)?)\b", flags=re.IGNORECASE),
            re.compile(r"\b(-?\d+(?:\.\d+)?)\s+by\s+(-?\d+(?:\.\d+)?)\b", flags=re.IGNORECASE),
            re.compile(
                r"\bwidth\b\s*(?:=|:|of)?\s*(-?\d+(?:\.\d+)?)\b.{0,80}\bheight\b\s*(?:=|:|of)?\s*(-?\d+(?:\.\d+)?)\b",
                flags=re.IGNORECASE,
            ),
            re.compile(
                r"\bheight\b\s*(?:=|:|of)?\s*(-?\d+(?:\.\d+)?)\b.{0,80}\bwidth\b\s*(?:=|:|of)?\s*(-?\d+(?:\.\d+)?)\b",
                flags=re.IGNORECASE,
            ),
        )
        for idx, pattern in enumerate(dimension_patterns):
            match = pattern.search(text)
            if not match:
                continue
            try:
                first = float(match.group(1))
                second = float(match.group(2))
            except ValueError:
                continue
            if idx == 3:
                first, second = second, first
            if first > 0 and second > 0:
                return first, second
        return None

    @staticmethod
    def _extract_shape_count_hint(text: str) -> int | None:
        if not text:
            return None
        match = re.search(r"\b(\d+)\s+(?:random\s+)?shapes?\b", text, flags=re.IGNORECASE)
        if not match:
            return None
        try:
            parsed = int(match.group(1))
        except ValueError:
            return None
        return max(1, min(8, parsed))

    @staticmethod
    def _extract_handle_hint(text: str, fallback_last: bool = True) -> str | None:
        handle_pattern = re.compile(
            r"\b(?:entity|handle|id)\s*[:=]?\s*([A-Za-z0-9]+)\b",
            flags=re.IGNORECASE,
        )
        match = handle_pattern.search(text or "")
        if match:
            return match.group(1).strip()
        if fallback_last and "last" in (text or "").lower():
            return "last"
        return None

    @staticmethod
    def _extract_handle_pair(text: str) -> tuple[str, str] | None:
        pair_pattern = re.compile(
            r"\b(?:between|with)\s+([A-Za-z0-9]+)\s+(?:and|&)\s+([A-Za-z0-9]+)\b",
            flags=re.IGNORECASE,
        )
        match = pair_pattern.search(text or "")
        if not match:
            return None
        return match.group(1).strip(), match.group(2).strip()

    @staticmethod
    def _build_deterministic_analysis(user_prompt: str) -> dict[str, Any]:
        instruction = ActionPlanner._extract_primary_instruction(user_prompt)
        return {
            "goal": instruction,
            "intent_summary": "Deterministic fallback intent extraction from user prompt.",
            "constraints": [],
            "target_selectors": [],
            "suggested_operations": [],
            "risks": ["fallback_mode"],
        }

    @staticmethod
    def _build_smiley_plan(cx: float, cy: float) -> dict[str, Any]:
        face_radius = 10.0
        eye_radius = 1.2
        return {
            "actions": [
                {
                    "tool": "entity",
                    "operation": "create_circle",
                    "data": {"cx": cx, "cy": cy, "radius": face_radius},
                    "expected_effects": {"entity_count_delta": 1, "entity_type_created": "CIRCLE"},
                },
                {
                    "tool": "entity",
                    "operation": "create_circle",
                    "data": {"cx": cx - 3.2, "cy": cy + 3.2, "radius": eye_radius},
                    "expected_effects": {"entity_count_delta": 1, "entity_type_created": "CIRCLE"},
                },
                {
                    "tool": "entity",
                    "operation": "create_circle",
                    "data": {"cx": cx + 3.2, "cy": cy + 3.2, "radius": eye_radius},
                    "expected_effects": {"entity_count_delta": 1, "entity_type_created": "CIRCLE"},
                },
                {
                    "tool": "entity",
                    "operation": "create_polyline",
                    "data": {
                        "points": [
                            [cx - 5.0, cy - 1.0],
                            [cx - 3.0, cy - 3.2],
                            [cx, cy - 4.1],
                            [cx + 3.0, cy - 3.2],
                            [cx + 5.0, cy - 1.0],
                        ],
                        "closed": False,
                    },
                    "expected_effects": {"entity_count_delta": 1, "entity_type_created": "LWPOLYLINE"},
                },
            ],
            "notes": "Local deterministic fallback plan generated because LM JSON output was unusable.",
        }

    @staticmethod
    def _build_line_plan(
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        layer: str | None = None,
    ) -> dict[str, Any]:
        data: dict[str, float | str] = {
            "x1": float(x1),
            "y1": float(y1),
            "x2": float(x2),
            "y2": float(y2),
        }
        if layer:
            data["layer"] = layer
        return {
            "actions": [
                {
                    "tool": "entity",
                    "operation": "create_line",
                    "data": data,
                    "expected_effects": {"entity_count_delta": 1, "entity_type_created": "LINE"},
                }
            ],
            "notes": "Local deterministic fallback plan generated because LM JSON output was unusable.",
        }

    @staticmethod
    def _build_circle_plan(
        cx: float,
        cy: float,
        radius: float,
        layer: str | None = None,
    ) -> dict[str, Any]:
        safe_radius = max(0.001, float(radius))
        data: dict[str, float | str] = {
            "cx": float(cx),
            "cy": float(cy),
            "radius": safe_radius,
        }
        if layer:
            data["layer"] = layer
        return {
            "actions": [
                {
                    "tool": "entity",
                    "operation": "create_circle",
                    "data": data,
                    "expected_effects": {"entity_count_delta": 1, "entity_type_created": "CIRCLE"},
                }
            ],
            "notes": "Local deterministic fallback plan generated because LM JSON output was unusable.",
        }

    @staticmethod
    def _build_rectangle_plan(
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        layer: str | None = None,
    ) -> dict[str, Any]:
        data: dict[str, float | str] = {
            "x1": float(x1),
            "y1": float(y1),
            "x2": float(x2),
            "y2": float(y2),
        }
        if layer:
            data["layer"] = layer
        return {
            "actions": [
                {
                    "tool": "entity",
                    "operation": "create_rectangle",
                    "data": data,
                    "expected_effects": {"entity_count_delta": 1, "entity_type_created": "LWPOLYLINE"},
                }
            ],
            "notes": "Local deterministic fallback plan generated because LM JSON output was unusable.",
        }

    @staticmethod
    def _build_arc_plan(
        cx: float,
        cy: float,
        radius: float,
        start_angle: float,
        end_angle: float,
        layer: str | None = None,
    ) -> dict[str, Any]:
        data: dict[str, float | str] = {
            "cx": float(cx),
            "cy": float(cy),
            "radius": max(0.001, float(radius)),
            "start_angle": float(start_angle),
            "end_angle": float(end_angle),
        }
        if layer:
            data["layer"] = layer
        return {
            "actions": [
                {
                    "tool": "entity",
                    "operation": "create_arc",
                    "data": data,
                    "expected_effects": {"entity_count_delta": 1, "entity_type_created": "ARC"},
                }
            ],
            "notes": "Local deterministic fallback plan generated because LM JSON output was unusable.",
        }

    @staticmethod
    def _build_ellipse_plan(
        cx: float,
        cy: float,
        major_x: float,
        major_y: float,
        ratio: float,
        layer: str | None = None,
    ) -> dict[str, Any]:
        data: dict[str, float | str] = {
            "cx": float(cx),
            "cy": float(cy),
            "major_x": float(major_x),
            "major_y": float(major_y),
            "ratio": max(0.01, min(0.99, float(ratio))),
        }
        if layer:
            data["layer"] = layer
        return {
            "actions": [
                {
                    "tool": "entity",
                    "operation": "create_ellipse",
                    "data": data,
                    "expected_effects": {"entity_count_delta": 1, "entity_type_created": "ELLIPSE"},
                }
            ],
            "notes": "Local deterministic fallback plan generated because LM JSON output was unusable.",
        }

    @staticmethod
    def _build_offset_plan(entity_id: str, distance: float) -> dict[str, Any]:
        return {
            "actions": [
                {
                    "tool": "entity",
                    "operation": "offset",
                    "data": {"entity_id": entity_id, "distance": float(distance)},
                    "expected_effects": {"entity_count_delta": 1},
                }
            ],
            "notes": "Local deterministic fallback plan generated because LM JSON output was unusable.",
        }

    @staticmethod
    def _build_fillet_plan(id1: str, id2: str, radius: float) -> dict[str, Any]:
        return {
            "actions": [
                {
                    "tool": "entity",
                    "operation": "fillet",
                    "data": {"id1": id1, "id2": id2, "radius": float(radius)},
                    "expected_effects": {"target_exists": id1},
                }
            ],
            "notes": "Local deterministic fallback plan generated because LM JSON output was unusable.",
        }

    @staticmethod
    def _build_chamfer_plan(id1: str, id2: str, dist1: float, dist2: float) -> dict[str, Any]:
        return {
            "actions": [
                {
                    "tool": "entity",
                    "operation": "chamfer",
                    "data": {"id1": id1, "id2": id2, "dist1": float(dist1), "dist2": float(dist2)},
                    "expected_effects": {"target_exists": id1},
                }
            ],
            "notes": "Local deterministic fallback plan generated because LM JSON output was unusable.",
        }

    @staticmethod
    def _build_random_shapes_plan(
        instruction: str,
        width: float,
        height: float,
        origin: tuple[float, float],
        shape_count: int,
    ) -> dict[str, Any]:
        safe_width = max(10.0, float(width))
        safe_height = max(10.0, float(height))
        ox, oy = float(origin[0]), float(origin[1])
        requested_shapes = max(3, min(8, int(shape_count)))
        seed_material = f"{instruction}|{safe_width}|{safe_height}|{ox}|{oy}"
        seed = int(hashlib.sha256(seed_material.encode("utf-8")).hexdigest()[:16], 16)
        rng = random.Random(seed)

        def rand_between(low: float, high: float) -> float:
            if high <= low:
                return (low + high) / 2.0
            return rng.uniform(low, high)

        actions: list[dict[str, Any]] = [
            {
                "tool": "layer",
                "operation": "create",
                "data": {"name": "RAND_AREA", "color": 8},
            },
            {
                "tool": "entity",
                "operation": "create_rectangle",
                "data": {
                    "x1": ox,
                    "y1": oy,
                    "x2": ox + safe_width,
                    "y2": oy + safe_height,
                    "layer": "RAND_AREA",
                },
                "expected_effects": {"entity_count_delta": 1, "entity_type_created": "LWPOLYLINE"},
            },
        ]
        palette = [1, 2, 3, 4, 5, 6, 30, 140]
        rng.shuffle(palette)
        shape_types = ["create_circle", "create_rectangle", "create_line", "create_arc"]
        margin = max(1.0, min(safe_width, safe_height) * 0.06)
        for idx in range(requested_shapes):
            layer_name = f"RAND_SHAPE_{idx + 1}"
            color = palette[idx % len(palette)]
            actions.append(
                {
                    "tool": "layer",
                    "operation": "create",
                    "data": {"name": layer_name, "color": color},
                }
            )
            shape_type = rng.choice(shape_types)
            if shape_type == "create_circle":
                radius = rand_between(
                    max(1.0, min(safe_width, safe_height) * 0.04),
                    max(2.0, min(safe_width, safe_height) * 0.16),
                )
                cx = rand_between(ox + margin + radius, ox + safe_width - margin - radius)
                cy = rand_between(oy + margin + radius, oy + safe_height - margin - radius)
                actions.append(
                    {
                        "tool": "entity",
                        "operation": "create_circle",
                        "data": {"cx": cx, "cy": cy, "radius": radius, "layer": layer_name},
                        "expected_effects": {"entity_count_delta": 1, "entity_type_created": "CIRCLE"},
                    }
                )
                continue
            if shape_type == "create_rectangle":
                rect_w = rand_between(max(2.0, safe_width * 0.12), max(3.0, safe_width * 0.28))
                rect_h = rand_between(max(2.0, safe_height * 0.12), max(3.0, safe_height * 0.32))
                x1 = rand_between(ox + margin, ox + safe_width - margin - rect_w)
                y1 = rand_between(oy + margin, oy + safe_height - margin - rect_h)
                actions.append(
                    {
                        "tool": "entity",
                        "operation": "create_rectangle",
                        "data": {"x1": x1, "y1": y1, "x2": x1 + rect_w, "y2": y1 + rect_h, "layer": layer_name},
                        "expected_effects": {"entity_count_delta": 1, "entity_type_created": "LWPOLYLINE"},
                    }
                )
                continue
            if shape_type == "create_arc":
                radius = rand_between(
                    max(1.0, min(safe_width, safe_height) * 0.05),
                    max(2.0, min(safe_width, safe_height) * 0.14),
                )
                cx = rand_between(ox + margin + radius, ox + safe_width - margin - radius)
                cy = rand_between(oy + margin + radius, oy + safe_height - margin - radius)
                start_angle = rand_between(0.0, 180.0)
                sweep = rand_between(45.0, 210.0)
                actions.append(
                    {
                        "tool": "entity",
                        "operation": "create_arc",
                        "data": {
                            "cx": cx,
                            "cy": cy,
                            "radius": radius,
                            "start_angle": start_angle,
                            "end_angle": start_angle + sweep,
                            "layer": layer_name,
                        },
                        "expected_effects": {"entity_count_delta": 1, "entity_type_created": "ARC"},
                    }
                )
                continue
            x1 = rand_between(ox + margin, ox + safe_width - margin)
            y1 = rand_between(oy + margin, oy + safe_height - margin)
            x2 = rand_between(ox + margin, ox + safe_width - margin)
            y2 = rand_between(oy + margin, oy + safe_height - margin)
            actions.append(
                {
                    "tool": "entity",
                    "operation": "create_line",
                    "data": {"x1": x1, "y1": y1, "x2": x2, "y2": y2, "layer": layer_name},
                    "expected_effects": {"entity_count_delta": 1, "entity_type_created": "LINE"},
                }
            )

        return {
            "actions": actions,
            "notes": (
                "Local deterministic fallback generated random-style colored shapes "
                "because LM JSON output was unusable."
            ),
        }

    def _build_deterministic_plan(self, user_prompt: str) -> dict[str, Any] | None:
        instruction = self._extract_primary_instruction(user_prompt)
        normalized = instruction.lower()
        if not instruction:
            return None
        pairs = self._extract_coordinate_pairs(instruction)

        if "smiley" in normalized or "smiley face" in normalized:
            cx, cy = pairs[0] if pairs else (0.0, 0.0)
            return self._build_smiley_plan(cx, cy)
        if "random" in normalized and any(keyword in normalized for keyword in ("shape", "shapes")):
            dimensions = self._extract_dimensions_hint(instruction) or (100.0, 50.0)
            origin = pairs[0] if pairs else (0.0, 0.0)
            shape_count = self._extract_shape_count_hint(instruction) or 5
            return self._build_random_shapes_plan(
                instruction,
                width=dimensions[0],
                height=dimensions[1],
                origin=origin,
                shape_count=shape_count,
            )

        layer = self._extract_layer_name(instruction)

        if any(keyword in normalized for keyword in ("fillet", "chamfer")):
            handle_pair = self._extract_handle_pair(instruction)
            if handle_pair:
                id1, id2 = handle_pair
                if "fillet" in normalized:
                    radius = self._extract_scalar_hint(instruction, ("radius", "r")) or 0.0
                    return self._build_fillet_plan(id1, id2, radius)
                dist1 = self._extract_scalar_hint(instruction, ("dist1", "distance1", "d1")) or 0.0
                dist2 = self._extract_scalar_hint(instruction, ("dist2", "distance2", "d2"))
                if dist2 is None:
                    dist2 = dist1
                return self._build_chamfer_plan(id1, id2, dist1, dist2)

        if "offset" in normalized:
            distance = self._extract_scalar_hint(instruction, ("distance", "offset", "d"))
            entity_id = self._extract_handle_hint(instruction, fallback_last=True)
            if distance is not None and entity_id:
                return self._build_offset_plan(entity_id, distance)

        if any(keyword in normalized for keyword in ("ellipse", "draw ellipse", "create ellipse")):
            center = pairs[0] if pairs else None
            if center is None:
                return None
            ratio = self._extract_scalar_hint(instruction, ("ratio", "minor_major_ratio")) or 0.5
            if len(pairs) >= 2:
                major_x, major_y = pairs[1]
            else:
                major_x, major_y = center[0] + 10.0, center[1]
            return self._build_ellipse_plan(center[0], center[1], major_x, major_y, ratio, layer=layer)

        if any(keyword in normalized for keyword in ("arc", "draw arc", "create arc")):
            center = pairs[0] if pairs else None
            if center is None:
                return None
            radius = self._extract_scalar_hint(instruction, ("radius", "r"))
            diameter = self._extract_scalar_hint(instruction, ("diameter", "d"))
            if radius is None and diameter is not None:
                radius = diameter / 2.0
            if radius is None and len(pairs) >= 2:
                radius = math.hypot(pairs[1][0] - center[0], pairs[1][1] - center[1])
            if radius is None:
                radius = 10.0
            start_angle = self._extract_scalar_hint(instruction, ("start_angle", "start"))
            end_angle = self._extract_scalar_hint(instruction, ("end_angle", "end"))
            sweep = self._extract_scalar_hint(instruction, ("sweep_angle", "delta_angle", "sweep"))
            if start_angle is None:
                start_angle = 0.0
            if end_angle is None:
                end_angle = start_angle + (sweep if sweep is not None else 90.0)
            return self._build_arc_plan(
                center[0],
                center[1],
                radius,
                start_angle,
                end_angle,
                layer=layer,
            )

        if any(keyword in normalized for keyword in ("rectangle", "draw rectangle", "create rectangle", "box")):
            if len(pairs) >= 2:
                (x1, y1), (x2, y2) = pairs[0], pairs[1]
                return self._build_rectangle_plan(x1, y1, x2, y2, layer=layer)
            if pairs:
                x1, y1 = pairs[0]
                width = self._extract_scalar_hint(instruction, ("width", "w")) or 10.0
                height = self._extract_scalar_hint(instruction, ("height", "h")) or width
                return self._build_rectangle_plan(x1, y1, x1 + width, y1 + height, layer=layer)

        if any(keyword in normalized for keyword in ("circle", "draw circle", "create circle")):
            radius = self._extract_scalar_hint(instruction, ("radius", "r"))
            diameter = self._extract_scalar_hint(instruction, ("diameter", "d"))
            if radius is None and diameter is not None:
                radius = diameter / 2.0

            center: tuple[float, float] | None = pairs[0] if pairs else None
            if radius is None and len(pairs) >= 2:
                (x1, y1), (x2, y2) = pairs[0], pairs[1]
                distance = math.hypot(x2 - x1, y2 - y1)
                if ("diameter" in normalized) or ("from" in normalized and "to" in normalized):
                    center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
                    radius = distance / 2.0
                else:
                    center = (x1, y1)
                    radius = distance

            if center is not None:
                if radius is None:
                    radius = 10.0
                return self._build_circle_plan(center[0], center[1], radius, layer=layer)

        if any(keyword in normalized for keyword in ("line", "draw line", "create line")):
            if len(pairs) >= 2:
                (x1, y1), (x2, y2) = pairs[0], pairs[1]
                return self._build_line_plan(x1, y1, x2, y2, layer=layer)

        return None
