"""Tests for agent planner fallback behavior."""

from autocad_mcp.agent.planner import ActionPlanner


class _DummyLMClient:
    """Placeholder LM client for deterministic fallback tests."""


def test_fallback_arc_plan_includes_expected_effects():
    planner = ActionPlanner(_DummyLMClient())  # type: ignore[arg-type]
    result = planner.create_fallback_plan("Create arc at (10,20) radius 5 start 0 end 180")
    assert result is not None
    assert result.safety.ok is True
    actions = result.plan.get("actions")
    assert isinstance(actions, list) and actions
    action = actions[0]
    assert action["tool"] == "entity"
    assert action["operation"] == "create_arc"
    assert isinstance(action.get("expected_effects"), dict)
    assert action["expected_effects"].get("entity_type_created") == "ARC"


def test_fallback_ellipse_plan():
    planner = ActionPlanner(_DummyLMClient())  # type: ignore[arg-type]
    result = planner.create_fallback_plan("Draw ellipse centered at (0,0) through (12,0) ratio 0.4")
    assert result is not None
    actions = result.plan.get("actions")
    assert isinstance(actions, list) and actions
    action = actions[0]
    assert action["operation"] == "create_ellipse"
    assert action["data"]["ratio"] == 0.4


def test_fallback_random_shapes_plan_uses_colored_layers_and_area():
    planner = ActionPlanner(_DummyLMClient())  # type: ignore[arg-type]
    result = planner.create_fallback_plan(
        "Create random shapes that is 100x50 feet. Make sure the shapes are in different colours, randomly."
    )
    assert result is not None
    actions = result.plan.get("actions")
    assert isinstance(actions, list) and actions
    boundary = actions[1]
    assert boundary["tool"] == "entity"
    assert boundary["operation"] == "create_rectangle"
    assert boundary["data"]["x2"] == 100.0
    assert boundary["data"]["y2"] == 50.0
    colored_shape_layers = [
        action["data"]["color"]
        for action in actions
        if action.get("tool") == "layer"
        and action.get("operation") == "create"
        and str(action.get("data", {}).get("name", "")).startswith("RAND_SHAPE_")
    ]
    assert len(colored_shape_layers) >= 3
    assert len(set(colored_shape_layers)) >= 3
