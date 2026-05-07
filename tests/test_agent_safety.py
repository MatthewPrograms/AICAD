"""Tests for action-plan safety validation."""

from autocad_mcp.agent.safety import validate_action_plan


def test_allowlisted_operation_passes():
    plan = {
        "actions": [
            {
                "tool": "entity",
                "operation": "create_line",
                "data": {"x1": 0, "y1": 0, "x2": 10, "y2": 10},
            }
        ]
    }
    safety = validate_action_plan(plan)
    assert safety.ok is True


def test_unknown_tool_blocked():
    plan = {"actions": [{"tool": "unknown_tool", "operation": "noop", "data": {}}]}
    safety = validate_action_plan(plan)
    assert safety.ok is False
    assert any("not allowlisted" in err for err in safety.errors)


def test_backend_specific_block_define_for_file_ipc():
    plan = {
        "actions": [
            {
                "tool": "block",
                "operation": "define",
                "data": {"name": "B1", "entities": []},
            }
        ]
    }
    safety = validate_action_plan(plan, backend_name="file_ipc")
    assert safety.ok is False
    assert any("unsupported for backend" in err for err in safety.errors)


def test_high_impact_requires_explicit_target():
    plan = {
        "actions": [
            {
                "tool": "entity",
                "operation": "erase",
                "data": {"selector": {"layer": "A-WALL"}},
            }
        ]
    }
    safety = validate_action_plan(plan)
    assert safety.ok is False
    assert any("requires an explicit target" in err for err in safety.errors)


def test_high_impact_with_handle_passes():
    plan = {
        "actions": [
            {
                "tool": "entity",
                "operation": "erase",
                "data": {"entity_id": "1A2B"},
            }
        ]
    }
    safety = validate_action_plan(plan)
    assert safety.ok is True
