"""Tests for image import IR extraction and plan synthesis."""

import pytest

from autocad_mcp.agent.image_import import ImageImportPipeline


class _DummyLMClient:
    def __init__(self, payload):
        self.payload = payload

    def chat_json_with_images(self, *_args, **_kwargs):
        return self.payload


class _DummyLMClientSequence:
    def __init__(self, payloads):
        self.payloads = list(payloads)
        self.calls = 0

    def chat_json_with_images(self, *_args, **_kwargs):
        self.calls += 1
        if not self.payloads:
            return {}
        if len(self.payloads) == 1:
            return self.payloads[0]
        return self.payloads.pop(0)


def test_extract_ir_requires_at_least_one_image():
    pipeline = ImageImportPipeline(_DummyLMClient({}))  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        pipeline.extract_ir("import this", image_paths=[], image_b64_pngs=[])


def test_build_plan_from_ir_maps_geometry_and_annotations():
    ir = {
        "units": "mm",
        "layers": [{"name": "OUTLINE", "color": "green"}],
        "geometry": [
            {"type": "line", "start": [0, 0], "end": [10, 0], "layer": "OUTLINE"},
            {"type": "circle", "center": [5, 5], "radius": 2.5, "layer": "OUTLINE"},
            {"type": "polyline", "points": [[0, 0], [0, 4], [4, 4]], "closed": False},
        ],
        "annotations": [{"text": "A1", "point": [1, 1], "height": 2.5, "layer": "TEXT"}],
        "notes": ["from test"],
    }
    pipeline = ImageImportPipeline(_DummyLMClient(ir))  # type: ignore[arg-type]
    plan, safety, qa = pipeline.build_plan_from_ir(ir, backend_name="file_ipc")
    actions = plan.get("actions")
    assert isinstance(actions, list) and actions
    operations = {f"{a.get('tool')}.{a.get('operation')}" for a in actions}
    assert "layer.create" in operations
    assert "entity.create_line" in operations
    assert "entity.create_circle" in operations
    assert "entity.create_polyline" in operations
    assert "annotation.create_text" in operations
    assert safety.ok is True
    assert qa.get("ok") is True


def test_extract_ir_uses_refinement_when_primary_is_low_quality():
    primary = {
        "units": "unknown",
        "geometry": [{"type": "line", "start": [0, 0], "end": [0, 0]}],
        "annotations": [],
        "notes": [],
    }
    refined = {
        "units": "unknown",
        "geometry": [{"type": "line", "start": [0, 0], "end": [10, 0], "layer": "0"}],
        "annotations": [],
        "notes": [],
    }
    lm = _DummyLMClientSequence([primary, refined])
    pipeline = ImageImportPipeline(lm)  # type: ignore[arg-type]
    ir = pipeline.extract_ir("import", image_paths=["example.png"], image_b64_pngs=None)
    assert lm.calls == 2
    assert len(ir["geometry"]) == 1
    assert ir["geometry"][0]["type"] == "line"


def test_build_plan_from_ir_cleans_duplicate_and_degenerate_entities():
    pipeline = ImageImportPipeline(_DummyLMClient({}))  # type: ignore[arg-type]
    plan, safety, qa = pipeline.build_plan_from_ir(
        {
            "units": "unknown",
            "geometry": [
                {"type": "line", "start": [0, 0], "end": [100, 0.4], "layer": "EDGE"},
                {"type": "line", "start": [100, 0.4], "end": [0, 0], "layer": "EDGE"},
                {"type": "line", "start": [5, 5], "end": [5, 5], "layer": "EDGE"},
                {
                    "type": "polyline",
                    "points": [[0, 0], [0, 0], [5, 0], [5, 5], [5, 5]],
                    "closed": False,
                    "layer": "EDGE",
                },
            ],
            "annotations": [
                {"text": "NOTE", "point": [10, 10], "height": 2.5, "layer": "TEXT"},
                {"text": "NOTE", "point": [10, 10], "height": 2.5, "layer": "TEXT"},
            ],
            "notes": [],
        },
        backend_name="file_ipc",
    )
    assert safety.ok is True
    assert qa.get("ok") is True
    actions = plan.get("actions")
    assert isinstance(actions, list)

    line_actions = [
        action for action in actions if action.get("tool") == "entity" and action.get("operation") == "create_line"
    ]
    assert len(line_actions) == 1
    line_data = line_actions[0]["data"]
    assert line_data["y1"] == line_data["y2"]

    polyline_actions = [
        action
        for action in actions
        if action.get("tool") == "entity" and action.get("operation") == "create_polyline"
    ]
    assert len(polyline_actions) == 1
    assert len(polyline_actions[0]["data"]["points"]) == 3

    text_actions = [
        action
        for action in actions
        if action.get("tool") == "annotation" and action.get("operation") == "create_text"
    ]
    assert len(text_actions) == 1


def test_extract_ir_normalizes_non_object_model_output():
    pipeline = ImageImportPipeline(_DummyLMClient("not-json-object"))  # type: ignore[arg-type]
    ir = pipeline.extract_ir("import", image_paths=["example.png"], image_b64_pngs=None)
    assert ir["units"] == "unknown"
    assert ir["geometry"] == []
    assert "image_ir_non_object" in ir["notes"]


def test_build_plan_from_ir_qa_fails_without_geometry():
    pipeline = ImageImportPipeline(_DummyLMClient({}))  # type: ignore[arg-type]
    plan, safety, qa = pipeline.build_plan_from_ir(
        {"units": "unknown", "layers": [], "geometry": [], "annotations": [], "notes": []},
        backend_name="file_ipc",
    )
    assert isinstance(plan.get("actions"), list)
    assert safety.ok is True
    assert qa.get("ok") is False
    assert any("No geometry was extracted" in err for err in qa.get("errors", []))


def test_build_plan_from_ir_annotation_only_is_non_blocking():
    pipeline = ImageImportPipeline(_DummyLMClient({}))  # type: ignore[arg-type]
    plan, safety, qa = pipeline.build_plan_from_ir(
        {
            "units": "unknown",
            "layers": [{"name": "TEXT"}],
            "geometry": [],
            "annotations": [{"text": "NOTE", "point": [10, 20], "layer": "TEXT"}],
            "notes": [],
        },
        backend_name="file_ipc",
    )
    actions = plan.get("actions")
    assert isinstance(actions, list) and actions
    assert any(
        action.get("tool") == "annotation" and action.get("operation") == "create_text"
        for action in actions
    )
    assert safety.ok is True
    assert qa.get("ok") is True
    assert any("No geometry was extracted" in warning for warning in qa.get("warnings", []))


def test_build_plan_from_ir_accepts_common_alias_shapes_and_texts():
    pipeline = ImageImportPipeline(_DummyLMClient({}))  # type: ignore[arg-type]
    plan, safety, qa = pipeline.build_plan_from_ir(
        {
            "units": "mm",
            "objects": [
                {"kind": "segment", "from": [0, 0], "to": [25, 0], "layer_name": "OUTLINE"},
                {"shape": "circle", "center": {"x": 10, "y": 8}, "d": "6.0", "layer": "OUTLINE"},
            ],
            "texts": [
                {"content": "R1", "position": {"x": 2, "y": 2}, "size": "2.5", "layer_name": "TEXT"}
            ],
            "notes": ["alias test"],
        },
        backend_name="file_ipc",
    )
    actions = plan.get("actions")
    assert isinstance(actions, list) and actions
    operations = {f"{a.get('tool')}.{a.get('operation')}" for a in actions}
    assert "entity.create_line" in operations
    assert "entity.create_circle" in operations
    assert "annotation.create_text" in operations
    assert safety.ok is True
    assert qa.get("ok") is True
