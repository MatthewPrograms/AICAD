"""Tests for LM Studio JSON parsing resilience."""

import json

from autocad_mcp.llm.lmstudio_client import _loads_json


def test_loads_json_extracts_markdown_fenced_payload():
    parsed = _loads_json("```json\n{\"actions\": [{\"tool\": \"entity\"}]}\n```")
    assert parsed["actions"][0]["tool"] == "entity"


def test_loads_json_extracts_embedded_balanced_object():
    parsed = _loads_json(
        "Model note: here is your result -> {\"analysis\": {}, \"actions\": [{\"tool\": \"layer\", \"operation\": \"create\"}]}"
    )
    assert parsed["actions"][0]["operation"] == "create"


def test_loads_json_rejects_empty_content():
    try:
        _loads_json("")
    except json.JSONDecodeError:
        return
    assert False, "Expected JSONDecodeError for empty content"
