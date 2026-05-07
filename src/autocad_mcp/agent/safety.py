"""Safety validation for LLM-generated action plans."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

TOOL_ALIASES: dict[str, str] = {
    "draw": "entity",
    "entities": "entity",
    "annotate": "annotation",
    "annotations": "annotation",
    "document": "drawing",
    "file": "drawing",
}

ENTITY_CREATE_BASES = {
    "line",
    "circle",
    "polyline",
    "rectangle",
    "arc",
    "ellipse",
    "mtext",
    "hatch",
}

GLOBAL_OPERATION_ALIASES: dict[str, str] = {
    "draw_line": "create_line",
    "line": "create_line",
    "draw_circle": "create_circle",
    "circle": "create_circle",
    "draw_polyline": "create_polyline",
    "polyline": "create_polyline",
    "draw_rectangle": "create_rectangle",
    "rectangle": "create_rectangle",
    "draw_arc": "create_arc",
    "arc": "create_arc",
    "draw_ellipse": "create_ellipse",
    "ellipse": "create_ellipse",
    "text": "create_text",
    "zoomextents": "zoom_extents",
    "modify_line": "create_line",
    "update_line": "create_line",
    "edit_line": "create_line",
    "change_line": "create_line",
    "entity_get_properties": "get",
    "get_entity_properties": "get",
    "get_properties": "get",
    "get_entity_info": "get",
    "entity_get": "get",
    "get_entity": "get",
    "query_entity": "get",
    "find_entity": "get",
    "list_entities": "list",
    "entity_list": "list",
    "entity_count": "count",
    "count_entities": "count",
    "get_entity_count": "count",
    "erase_entity": "erase",
    "remove_entity": "erase",
    "entity_erase": "erase",
    "move_entity": "move",
    "entity_move": "move",
    "copy_entity": "copy",
    "entity_copy": "copy",
    "rotate_entity": "rotate",
    "entity_rotate": "rotate",
    "scale_entity": "scale",
    "entity_scale": "scale",
    "mirror_entity": "mirror",
    "entity_mirror": "mirror",
    "offset_entity": "offset",
    "entity_offset": "offset",
    "array_entity": "array",
    "entity_array": "array",
    "entity_fillet": "fillet",
    "entity_chamfer": "chamfer",
    "layer_list": "list",
    "layer_create": "create",
    "layer_set_current": "set_current",
    "layer_set_properties": "set_properties",
    "layer_freeze": "freeze",
    "layer_thaw": "thaw",
    "layer_lock": "lock",
    "layer_unlock": "unlock",
    "block_list": "list",
    "block_insert": "insert",
    "block_insert_with_attributes": "insert_with_attributes",
    "block_get_attributes": "get_attributes",
    "block_update_attribute": "update_attribute",
    "block_define": "define",
    "drawing_save_as_dxf": "save_as_dxf",
    "drawing_plot_pdf": "plot_pdf",
    "drawing_purge": "purge",
    "drawing_get_variables": "get_variables",
    "drawing_open": "open",
    "drawing_undo": "undo",
    "drawing_redo": "redo",
}

ALLOWED_OPERATIONS: dict[str, set[str]] = {
    "drawing": {
        "create",
        "open",
        "info",
        "save",
        "save_as_dxf",
        "plot_pdf",
        "purge",
        "get_variables",
        "undo",
        "redo",
    },
    "entity": {
        "create_line",
        "create_circle",
        "create_polyline",
        "create_rectangle",
        "create_arc",
        "create_ellipse",
        "create_mtext",
        "create_hatch",
        "list",
        "count",
        "get",
        "copy",
        "move",
        "rotate",
        "scale",
        "mirror",
        "offset",
        "array",
        "fillet",
        "chamfer",
        "erase",
    },
    "layer": {
        "list",
        "create",
        "set_current",
        "set_properties",
        "freeze",
        "thaw",
        "lock",
        "unlock",
    },
    "block": {
        "list",
        "insert",
        "insert_with_attributes",
        "get_attributes",
        "update_attribute",
        "define",
    },
    "annotation": {
        "create_text",
        "create_dimension_linear",
        "create_dimension_aligned",
        "create_dimension_angular",
        "create_dimension_radius",
        "create_leader",
    },
    "pid": {
        "setup_layers",
        "insert_symbol",
        "list_symbols",
        "draw_process_line",
        "connect_equipment",
        "add_flow_arrow",
        "add_equipment_tag",
        "add_line_number",
        "insert_valve",
        "insert_instrument",
        "insert_pump",
        "insert_tank",
    },
    "view": {
        "zoom_extents",
        "zoom_window",
        "get_screenshot",
    },
    "system": {
        "status",
        "health",
        "get_backend",
        "runtime",
        "init",
    },
}

BACKEND_OPERATION_DENYLIST: dict[str, dict[str, set[str]]] = {
    "file_ipc": {
        "block": {"define"},
    },
    "ezdxf": {
        "drawing": {"plot_pdf", "undo", "redo"},
        "entity": {"offset", "fillet", "chamfer"},
        "view": {"zoom_extents", "zoom_window"},
    },
}

HIGH_IMPACT_ENTITY_OPERATIONS = {
    "erase",
    "move",
    "copy",
    "rotate",
    "scale",
    "mirror",
    "offset",
    "array",
}


@dataclass
class SafetyResult:
    ok: bool
    errors: list[str]


def _norm(value: str) -> str:
    return value.strip().lower().replace("-", "_").replace(" ", "_").replace(".", "_")


def _normalize_action_in_place(action: dict[str, Any]) -> None:
    """Normalize tool/operation aliases to canonical values in-place."""
    tool_raw = action.get("tool")
    op_raw = action.get("operation")

    if isinstance(tool_raw, str) and tool_raw.strip():
        tool = TOOL_ALIASES.get(_norm(tool_raw), _norm(tool_raw))
    else:
        tool = tool_raw

    if isinstance(op_raw, str) and op_raw.strip():
        operation = _norm(op_raw)
    else:
        operation = op_raw

    if isinstance(operation, str):
        operation = GLOBAL_OPERATION_ALIASES.get(operation, operation)

        if operation.startswith("drawing_"):
            candidate = operation.removeprefix("drawing_")
            if candidate in ENTITY_CREATE_BASES:
                tool = "entity"
                operation = f"create_{candidate}"

    if isinstance(tool, str) and isinstance(operation, str):
        for prefix in ("modify_", "update_", "edit_", "change_"):
            if operation.startswith(prefix):
                candidate = operation.removeprefix(prefix)
                if candidate in ENTITY_CREATE_BASES:
                    tool = "entity"
                    operation = f"create_{candidate}"
                    break
        if tool == "drawing" and operation in ENTITY_CREATE_BASES:
            tool = "entity"
            operation = f"create_{operation}"
        elif (
            tool == "drawing"
            and operation.startswith("create_")
            and operation.removeprefix("create_") in ENTITY_CREATE_BASES
        ):
            tool = "entity"
        elif tool == "entity" and operation in ENTITY_CREATE_BASES:
            operation = f"create_{operation}"
        elif tool == "annotation" and operation == "text":
            operation = "create_text"

    action["tool"] = tool
    action["operation"] = operation


def _contains_any_key(payload: dict[str, Any], keys: tuple[str, ...]) -> bool:
    return any(key in payload for key in keys)


def _selector_is_explicit(selector: dict[str, Any] | None) -> bool:
    if not isinstance(selector, dict):
        return False
    explicit_keys = ("handle", "entity_id", "id")
    for key in explicit_keys:
        value = selector.get(key)
        if isinstance(value, str) and value.strip() and value.strip().lower() not in {"all", "*"}:
            return True
    return False


def _action_has_explicit_target(data: dict[str, Any], selector: dict[str, Any] | None) -> bool:
    target_keys = ("entity_id", "handle", "id", "id1", "id2", "entity_id1", "entity_id2")
    for key in target_keys:
        value = data.get(key)
        if isinstance(value, str) and value.strip() and value.strip().lower() not in {"all", "*"}:
            return True
    if _selector_is_explicit(selector):
        return True
    return False


def _validate_selector(idx: int, selector: Any, errors: list[str]) -> None:
    if selector is None:
        return
    if not isinstance(selector, dict):
        errors.append(f"actions[{idx}].selector must be an object when provided.")
        return
    for key in ("layer", "type", "text", "region", "handle", "entity_id", "id"):
        value = selector.get(key)
        if isinstance(value, str) and value.strip() in {"*", "all"}:
            errors.append(f"actions[{idx}].selector.{key} cannot be wildcard '*' or 'all'.")

    proximity = selector.get("proximity")
    if proximity is not None and not isinstance(proximity, dict):
        errors.append(f"actions[{idx}].selector.proximity must be an object when provided.")


def _validate_operation_payload(
    idx: int,
    tool: str,
    operation: str,
    data: dict[str, Any],
    selector: dict[str, Any] | None,
    errors: list[str],
) -> None:
    if tool == "drawing" and operation in {"open", "save_as_dxf", "plot_pdf"}:
        path = data.get("path")
        if not isinstance(path, str) or not path.strip():
            errors.append(f"actions[{idx}] requires data.path for drawing.{operation}.")

    if tool == "entity" and operation == "create_hatch":
        if not _action_has_explicit_target(data, selector):
            errors.append(f"actions[{idx}] create_hatch requires an explicit target entity.")

    if tool == "entity" and operation in {"erase", "move", "copy", "rotate", "scale", "mirror", "offset", "array"}:
        if not _action_has_explicit_target(data, selector):
            errors.append(
                f"actions[{idx}] {operation} requires an explicit target (handle/entity_id) "
                "or selector handle/entity_id."
            )
        confirm = data.get("confirm_high_impact")
        confirmation_block = data.get("confirmation")
        has_confirmation = bool(confirm) or (
            isinstance(confirmation_block, dict) and bool(confirmation_block.get("high_impact"))
        )
        if (not _action_has_explicit_target(data, selector)) and not has_confirmation:
            errors.append(
                f"actions[{idx}] high-impact entity.{operation} requires explicit confirmation "
                "(data.confirm_high_impact=true)."
            )

    if tool == "entity" and operation == "erase":
        entity_id = data.get("entity_id")
        if isinstance(entity_id, str) and entity_id.strip().lower() in {"all", "*"}:
            errors.append(f"actions[{idx}] erase cannot target wildcard entity_id.")

    if tool == "entity" and operation in {"fillet", "chamfer"}:
        has_pair = _contains_any_key(data, ("id1", "entity_id1")) and _contains_any_key(data, ("id2", "entity_id2"))
        if not has_pair:
            errors.append(f"actions[{idx}] entity.{operation} requires both id1 and id2 (or entity_id1/entity_id2).")

    expected_effects = data.get("expected_effects")
    if expected_effects is not None and not isinstance(expected_effects, dict):
        errors.append(f"actions[{idx}].data.expected_effects must be an object when provided.")


def validate_action_plan(
    plan: dict[str, Any],
    backend_name: str | None = None,
    max_actions: int = 40,
) -> SafetyResult:
    """Validate plan structure, allowlisted operations, and payload safety constraints."""
    errors: list[str] = []
    actions = plan.get("actions")

    if not isinstance(actions, list):
        return SafetyResult(ok=False, errors=["Plan must include an 'actions' list."])
    if len(actions) > max_actions:
        errors.append(f"Plan has {len(actions)} actions; maximum allowed is {max_actions}.")

    backend_key = _norm(backend_name) if isinstance(backend_name, str) and backend_name.strip() else None
    backend_denied = BACKEND_OPERATION_DENYLIST.get(backend_key or "", {})

    for idx, action in enumerate(actions):
        if not isinstance(action, dict):
            errors.append(f"actions[{idx}] must be an object.")
            continue

        _normalize_action_in_place(action)
        tool = action.get("tool")
        operation = action.get("operation")
        data = action.get("data", {})
        selector = action.get("selector")
        expected_effects = action.get("expected_effects")

        if not isinstance(tool, str) or not tool:
            errors.append(f"actions[{idx}].tool must be a non-empty string.")
            continue
        if not isinstance(operation, str) or not operation:
            errors.append(f"actions[{idx}].operation must be a non-empty string.")
            continue
        if not isinstance(data, dict):
            errors.append(f"actions[{idx}].data must be an object when provided.")
            continue
        if expected_effects is not None and not isinstance(expected_effects, dict):
            errors.append(f"actions[{idx}].expected_effects must be an object when provided.")

        if "execute_lisp" in _norm(operation):
            errors.append(f"actions[{idx}] operation '{tool}.{operation}' is blocked for safety.")
            continue

        if tool not in ALLOWED_OPERATIONS:
            errors.append(f"actions[{idx}] tool '{tool}' is not allowlisted.")
            continue
        if operation not in ALLOWED_OPERATIONS[tool]:
            errors.append(f"actions[{idx}] operation '{tool}.{operation}' is not allowlisted.")
            continue

        denied_ops = backend_denied.get(tool, set())
        if operation in denied_ops:
            errors.append(
                f"actions[{idx}] operation '{tool}.{operation}' is unsupported for backend '{backend_key}'."
            )
            continue

        _validate_selector(idx, selector, errors)
        _validate_operation_payload(
            idx=idx,
            tool=tool,
            operation=operation,
            data=data,
            selector=selector if isinstance(selector, dict) else None,
            errors=errors,
        )

        if expected_effects is not None and isinstance(expected_effects, dict):
            action.setdefault("data", {})
            if "expected_effects" not in action["data"]:
                action["data"]["expected_effects"] = expected_effects

    return SafetyResult(ok=len(errors) == 0, errors=errors)
