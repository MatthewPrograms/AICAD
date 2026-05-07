"""Scene graph extraction and summarization utilities for CAD planning."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import math
from typing import Any

from autocad_mcp.backends.base import AutoCADBackend


@dataclass
class SceneEntity:
    """Typed scene entity representation used by planner/executor."""

    handle: str
    entity_type: str
    layer: str
    anchors: list[tuple[float, float]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SceneGraph:
    """Compact scene graph for architectural reasoning and target selection."""

    entities: list[SceneEntity]
    entity_count: int
    type_counts: dict[str, int]
    layer_counts: dict[str, int]
    text_sample: list[str]
    bounds: dict[str, float] | None
    region_counts: dict[str, int]
    sample_truncated: bool

    def to_planner_context_payload(self) -> dict[str, Any]:
        entity_sample = [
            {
                "handle": entity.handle,
                "type": entity.entity_type,
                "layer": entity.layer,
                "anchors": [[round(x, 4), round(y, 4)] for x, y in entity.anchors[:4]],
                "metadata": entity.metadata,
            }
            for entity in self.entities[:40]
        ]
        payload: dict[str, Any] = {
            "entity_count": self.entity_count,
            "entity_type_counts": self.type_counts,
            "entity_layer_counts": self.layer_counts,
            "text_entity_sample": self.text_sample,
            "scene_bounds": self.bounds,
            "scene_region_counts": self.region_counts,
            "entity_sample": entity_sample,
            "entity_sample_truncated": self.sample_truncated,
        }
        return payload


@dataclass
class SceneGraphCache:
    """Simple cache + invalidation hooks for scene extraction."""

    graph: SceneGraph | None = None
    entity_count_hint: int | None = None
    version: int = 0

    def invalidate(self) -> None:
        self.version += 1
        self.graph = None
        self.entity_count_hint = None

    def maybe_get(self, current_count_hint: int | None) -> SceneGraph | None:
        if self.graph is None:
            return None
        if current_count_hint is None:
            return self.graph
        if self.entity_count_hint is None:
            return self.graph
        if current_count_hint == self.entity_count_hint:
            return self.graph
        return None

    def put(self, graph: SceneGraph, entity_count_hint: int | None) -> None:
        self.graph = graph
        self.entity_count_hint = entity_count_hint


class SceneGraphBuilder:
    """Progressive scene extraction from backend entity list/get operations."""

    def __init__(self, backend: AutoCADBackend):
        self.backend = backend

    async def build(
        self,
        detail_budget: int = 140,
        sample_budget: int = 160,
        text_sample_budget: int = 20,
        layer_hint: str | None = None,
    ) -> SceneGraph:
        list_result = await self.backend.entity_list(layer_hint)
        if not list_result.ok or not isinstance(list_result.payload, dict):
            return SceneGraph(
                entities=[],
                entity_count=0,
                type_counts={},
                layer_counts={},
                text_sample=[],
                bounds=None,
                region_counts={},
                sample_truncated=False,
            )

        entities_raw = list_result.payload.get("entities")
        if not isinstance(entities_raw, list):
            entities_raw = []

        entities: list[SceneEntity] = []
        type_counts: Counter[str] = Counter()
        layer_counts: Counter[str] = Counter()
        text_sample: list[str] = []
        all_points: list[tuple[float, float]] = []

        sampled_entries = entities_raw[:sample_budget]
        handles_for_detail: list[str] = []
        for raw in sampled_entries:
            if not isinstance(raw, dict):
                continue
            handle = str(raw.get("handle") or "").strip()
            entity_type = str(raw.get("type") or "UNKNOWN").upper()
            layer = str(raw.get("layer") or "UNSPECIFIED")
            type_counts[entity_type] += 1
            layer_counts[layer] += 1
            metadata: dict[str, Any] = {}
            text_value = raw.get("text")
            if isinstance(text_value, str):
                candidate = text_value.strip()
                if candidate and candidate not in text_sample and len(text_sample) < text_sample_budget:
                    text_sample.append(candidate)
            if handle:
                handles_for_detail.append(handle)
            entity = SceneEntity(
                handle=handle or "unknown",
                entity_type=entity_type,
                layer=layer,
                anchors=self._extract_anchor_points(raw),
                metadata=metadata,
            )
            all_points.extend(entity.anchors)
            entities.append(entity)

        if len(entities_raw) > sample_budget:
            step = max(1, len(entities_raw) // sample_budget)
            for raw in entities_raw[sample_budget::step]:
                if not isinstance(raw, dict):
                    continue
                entity_type = str(raw.get("type") or "UNKNOWN").upper()
                layer = str(raw.get("layer") or "UNSPECIFIED")
                type_counts[entity_type] += 1
                layer_counts[layer] += 1

        if handles_for_detail:
            if len(handles_for_detail) > detail_budget:
                step = max(1, len(handles_for_detail) // detail_budget)
                handles_for_detail = handles_for_detail[::step][:detail_budget]
            entity_by_handle: dict[str, SceneEntity] = {
                entity.handle: entity for entity in entities if entity.handle and entity.handle != "unknown"
            }
            for handle in handles_for_detail:
                detail = await self.backend.entity_get(handle)
                if not detail.ok or not isinstance(detail.payload, dict):
                    continue
                detail_payload = detail.payload
                entity = entity_by_handle.get(handle)
                if entity is None:
                    entity = SceneEntity(
                        handle=handle,
                        entity_type=str(detail_payload.get("type") or "UNKNOWN").upper(),
                        layer=str(detail_payload.get("layer") or "UNSPECIFIED"),
                    )
                    entities.append(entity)
                    type_counts[entity.entity_type] += 1
                    layer_counts[entity.layer] += 1
                    entity_by_handle[handle] = entity
                self._merge_entity_detail(entity, detail_payload, text_sample, text_sample_budget)
                all_points.extend(entity.anchors)

        bounds = self._compute_bounds(all_points)
        regions = self._compute_region_counts(entities, bounds)

        return SceneGraph(
            entities=entities,
            entity_count=len(entities_raw),
            type_counts=dict(type_counts.most_common(30)),
            layer_counts=dict(layer_counts.most_common(30)),
            text_sample=text_sample,
            bounds=bounds,
            region_counts=regions,
            sample_truncated=len(entities_raw) > sample_budget,
        )

    @staticmethod
    def _merge_entity_detail(
        entity: SceneEntity,
        detail_payload: dict[str, Any],
        text_sample: list[str],
        text_sample_budget: int,
    ) -> None:
        anchors = SceneGraphBuilder._extract_anchor_points(detail_payload)
        if anchors:
            deduped = list(entity.anchors)
            seen = {(round(x, 6), round(y, 6)) for x, y in deduped}
            for point in anchors:
                key = (round(point[0], 6), round(point[1], 6))
                if key in seen:
                    continue
                seen.add(key)
                deduped.append(point)
            entity.anchors = deduped[:14]

        for key in (
            "radius",
            "diameter",
            "start_angle",
            "end_angle",
            "closed",
            "width",
            "height",
            "ratio",
            "rotation",
            "points",
            "major_axis",
            "minor_axis",
            "vertex_count",
            "dim_text",
            "measurement",
            "block_name",
        ):
            if key in detail_payload and key not in entity.metadata:
                entity.metadata[key] = detail_payload.get(key)

        for text_key in ("text", "contents", "mtext", "label"):
            value = detail_payload.get(text_key)
            if isinstance(value, str):
                candidate = value.strip()
                if candidate and candidate not in text_sample and len(text_sample) < text_sample_budget:
                    text_sample.append(candidate)
                if candidate:
                    entity.metadata.setdefault("text", candidate)

    @staticmethod
    def _extract_anchor_points(payload: dict[str, Any]) -> list[tuple[float, float]]:
        points: list[tuple[float, float]] = []
        for key in (
            "center",
            "start",
            "end",
            "insert",
            "point",
            "location",
            "text_point",
            "dim_point",
        ):
            parsed = SceneGraphBuilder._parse_point(payload.get(key))
            if parsed is not None:
                points.append(parsed)

        for key in ("points", "vertices"):
            raw_points = payload.get(key)
            if not isinstance(raw_points, list):
                continue
            for value in raw_points[:24]:
                parsed = SceneGraphBuilder._parse_point(value)
                if parsed is not None:
                    points.append(parsed)

        for pair_key in (
            ("x", "y"),
            ("cx", "cy"),
            ("center_x", "center_y"),
            ("text_x", "text_y"),
        ):
            x_key, y_key = pair_key
            x_val = payload.get(x_key)
            y_val = payload.get(y_key)
            if isinstance(x_val, (int, float)) and isinstance(y_val, (int, float)):
                points.append((float(x_val), float(y_val)))

        deduped: list[tuple[float, float]] = []
        seen: set[tuple[float, float]] = set()
        for x_val, y_val in points:
            key = (round(float(x_val), 6), round(float(y_val), 6))
            if key in seen:
                continue
            seen.add(key)
            deduped.append((float(x_val), float(y_val)))
        return deduped

    @staticmethod
    def _parse_point(value: Any) -> tuple[float, float] | None:
        if isinstance(value, (list, tuple)) and len(value) >= 2:
            if isinstance(value[0], (int, float)) and isinstance(value[1], (int, float)):
                return float(value[0]), float(value[1])
        if isinstance(value, dict):
            for pair in (("x", "y"), ("cx", "cy"), ("center_x", "center_y")):
                x_key, y_key = pair
                x_val = value.get(x_key)
                y_val = value.get(y_key)
                if isinstance(x_val, (int, float)) and isinstance(y_val, (int, float)):
                    return float(x_val), float(y_val)
        return None

    @staticmethod
    def _compute_bounds(points: list[tuple[float, float]]) -> dict[str, float] | None:
        if not points:
            return None
        min_x = min(point[0] for point in points)
        max_x = max(point[0] for point in points)
        min_y = min(point[1] for point in points)
        max_y = max(point[1] for point in points)
        return {
            "min_x": float(min_x),
            "max_x": float(max_x),
            "min_y": float(min_y),
            "max_y": float(max_y),
            "width": float(max_x - min_x),
            "height": float(max_y - min_y),
            "diagonal": float(math.hypot(max_x - min_x, max_y - min_y)),
        }

    @staticmethod
    def _compute_region_counts(
        entities: list[SceneEntity],
        bounds: dict[str, float] | None,
    ) -> dict[str, int]:
        if not bounds:
            return {}
        min_x = bounds["min_x"]
        min_y = bounds["min_y"]
        width = bounds["width"] if bounds["width"] > 0 else 1.0
        height = bounds["height"] if bounds["height"] > 0 else 1.0

        counts: Counter[str] = Counter()
        for entity in entities:
            if not entity.anchors:
                continue
            center_x = sum(point[0] for point in entity.anchors) / len(entity.anchors)
            center_y = sum(point[1] for point in entity.anchors) / len(entity.anchors)
            nx = (center_x - min_x) / width
            ny = (center_y - min_y) / height
            if nx < 1 / 3:
                x_region = "left"
            elif nx < 2 / 3:
                x_region = "center"
            else:
                x_region = "right"
            if ny < 1 / 3:
                y_region = "bottom"
            elif ny < 2 / 3:
                y_region = "middle"
            else:
                y_region = "top"
            counts[f"{y_region}_{x_region}"] += 1
        return dict(counts)
