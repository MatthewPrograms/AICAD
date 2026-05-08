"""Tests for image import IR extraction and plan synthesis."""

import pytest

from autocad_mcp.agent.image_import import ImageImportPipeline


class _DummyLMClient:
    def __init__(self, payload):
        self.payload = payload

    def chat_json_with_images(self, *_args, **_kwargs):
        return self.payload


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
