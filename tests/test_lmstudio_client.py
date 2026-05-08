"""Tests for LM Studio JSON parsing resilience."""

import json
import pytest

from autocad_mcp.llm.lmstudio_client import LMStudioClient, LMStudioConfig, _loads_json


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


def test_loads_json_rejects_truncated_root_instead_of_inner_fragment():
    payload = (
        "```json\n"
        "{\n"
        '  "units": "unknown",\n'
        '  "layers": [{"name": "0"}],\n'
        '  "geometry": [\n'
        '    {"type": "polyline", "points": [[0, 0], [10, 0]], "closed": false, "layer": "0"},\n'
        '    {"type": "arc", "center": [5, 5], "radius": 3, "start_angle": 0\n'
    )
    with pytest.raises(json.JSONDecodeError):
        _loads_json(payload)

def test_loads_json_salvages_orphan_string_member_inside_object():
    payload = (
        "```json\n"
        "{\n"
        '  "units": "unknown",\n'
        '  "layers": [{"name": "0"}],\n'
        '  "geometry": [\n'
        '    {"type": "circle", "center": [785, 375], "radius": 185, "0"},\n'
        '    {"type": "line", "start": [0, 0], "end": [10, 10], "layer": "0"}\n'
        "  ],\n"
        '  "annotations": [],\n'
        '  "notes": []\n'
        "}\n"
        "```"
    )
    parsed = _loads_json(payload)
    assert parsed["geometry"][0]["type"] == "circle"
    assert parsed["geometry"][0]["radius"] == 185
    assert parsed["geometry"][1]["type"] == "line"


def test_chat_json_uses_model_repair_fallback(monkeypatch):
    client = LMStudioClient(LMStudioConfig(base_url="http://127.0.0.1:1234/v1"))
    responses = [
        {
            "choices": [
                {
                    "message": {
                        "content": (
                            "```json\n"
                            "{\n"
                            '  "units": "unknown",\n'
                            '  "geometry": [\n'
                            '    {"type": "circle", "center": [0, 0], "radius": 5}\n'
                            "```"
                        )
                    }
                }
            ]
        },
        {
            "choices": [
                {
                    "message": {
                        "content": (
                            "{"
                            "\"units\": \"unknown\","
                            "\"layers\": [{\"name\": \"0\"}],"
                            "\"geometry\": [{\"type\": \"circle\", \"center\": [0, 0], \"radius\": 5, \"layer\": \"0\"}],"
                            "\"annotations\": [],"
                            "\"notes\": []"
                            "}"
                        )
                    }
                }
            ]
        },
    ]

    monkeypatch.setattr(client, "_resolve_model", lambda: "local-model")

    def fake_post_chat_completion(_payload, _allow_response_format_fallback=None, **_kwargs):
        return responses.pop(0)

    monkeypatch.setattr(client, "_post_chat_completion", fake_post_chat_completion)

    parsed = client.chat_json("system", "user")
    assert parsed["geometry"][0]["type"] == "circle"
    assert parsed["geometry"][0]["layer"] == "0"
    client.close()
