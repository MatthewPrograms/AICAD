# AICAD (AutoCAD LT AI Assistant)
AICAD is a local-first AI automation project for **AutoCAD LT 2024+ on Windows**.

It combines:
- a Python MCP server (`autocad-mcp`)
- an AutoLISP dispatcher bridge (`lisp-code/mcp_dispatch.lsp`)
- an LM Studio client for local model planning
- a desktop GUI for chat-driven CAD execution and visual grounding

This project is designed around **AutoCAD LT + AutoLISP**, not .NET plug-ins.

## What this project does
- Converts natural-language requests into structured CAD action plans.
- Executes plans in AutoCAD LT through a file-based IPC bridge.
- Supports vision/context-aware planning using screenshots/reference images.
- Includes a dedicated Image→CAD import pipeline with preview-before-execute flow.
- Includes safety checks and backend capability checks before execution.
- Falls back to deterministic plans when model JSON output is unusable.
- Also supports headless DXF workflows via `ezdxf` backend.

## Core components
- `src/autocad_mcp/server.py`: MCP server with consolidated tools:
  - `drawing`, `entity`, `layer`, `block`, `annotation`, `pid`, `view`, `system`
- `src/autocad_mcp/gui/app.py`: standalone AI planning/execution GUI
- `src/autocad_mcp/agent/planner.py`: two-stage analysis + plan generation, deterministic fallback
- `src/autocad_mcp/agent/image_import.py`: image IR extraction + deterministic IR→CAD action mapping
- `src/autocad_mcp/agent/safety.py`: action allowlist and safety policy validation
- `src/autocad_mcp/llm/lmstudio_client.py`: LM Studio OpenAI-compatible client + JSON parsing resilience
- `src/autocad_mcp/backends/file_ipc.py`: AutoCAD LT bridge backend
- `src/autocad_mcp/backends/ezdxf_backend.py`: headless backend
- `lisp-code/mcp_dispatch.lsp`: AutoLISP dispatcher loaded in AutoCAD LT

## Architecture
```text
User / MCP Client / GUI
          |
          v
   Python planner + safety
          |
          v
   Backend selector
     |            |
     |            +--> ezdxf (headless)
     |
     +--> file_ipc (AutoCAD LT)
              |
              v
      C:/temp JSON command/result files
              |
              v
      mcp_dispatch.lsp in AutoCAD LT
```

## Requirements
- Windows 10/11
- AutoCAD LT 2024+ (Windows)
- Python 3.10+
- `uv` (recommended) or `pip`
- LM Studio running locally for AI planning

## Quick start (GUI workflow)
1) Clone and install dependencies
```powershell
git clone https://github.com/MatthewPrograms/AICAD.git
cd AICAD
uv sync
```

2) Load the AutoLISP dispatcher in AutoCAD LT
- Open AutoCAD LT
- Run `APPLOAD`
- Load `lisp-code/mcp_dispatch.lsp`
- Keep a drawing open

3) Start LM Studio
- Load/select your local model
- Ensure API endpoint is available (default expected by app: `http://127.0.0.1:1234/v1`)

4) Launch GUI
```powershell
uv run python -m autocad_mcp.gui_main
```
or
```powershell
start_gui.bat
```

## MCP server mode (for Claude Desktop / other MCP clients)
Run server over stdio:
```powershell
uv run python -m autocad_mcp
```

## Image→CAD workflow (GUI)
1) Open the GUI and choose your reference image(s).
2) Run **Preview Image→CAD** to extract image IR and generate CAD actions.
3) Review generated actions and warnings in the preview output.
4) Run **Execute Image→CAD** to apply actions.

Execution gating behavior:
- The GUI blocks execution only on **blocking** QA errors.
- Non-blocking QA issues (for example, annotation-only extraction) are shown as warnings and can proceed with confirmation.
- If LM output is near-valid but malformed (e.g., stray quoted member tokens), local JSON salvage/recovery is attempted before fallback repair.

Example MCP server config (Windows):
```json
{
  "mcpServers": {
    "autocad-mcp": {
      "command": "C:\\\\path\\\\to\\\\AICAD\\\\.venv\\\\Scripts\\\\python.exe",
      "args": ["-m", "autocad_mcp"],
      "env": {
        "AUTOCAD_MCP_BACKEND": "auto"
      }
    }
  }
}
```

## Backend modes
- `file_ipc`: AutoCAD LT bridge (requires AutoCAD running)
- `ezdxf`: headless/no AutoCAD
- `auto`: tries `file_ipc`, falls back to `ezdxf`

## Safety model
- Plan actions are normalized and validated against allowlisted tools/operations.
- High-impact operations (erase/transforms) require explicit targets or confirmation patterns.
- `system.execute_lisp` is blocked in planner-generated flows by safety policy.
- Backend-specific unsupported operations are rejected before execution.

## Environment variables
- `AUTOCAD_MCP_BACKEND` = `auto | file_ipc | ezdxf`
- `AUTOCAD_MCP_IPC_DIR` (default `C:/temp`)
- `AUTOCAD_MCP_IPC_TIMEOUT` (default `10.0`)
- `AUTOCAD_MCP_ONLY_TEXT` (`true/false`)
- `AUTOCAD_MCP_LMSTUDIO_BASE_URL` (default `http://127.0.0.1:1234/v1`)
- `AUTOCAD_MCP_LMSTUDIO_MODEL`
- `AUTOCAD_MCP_LMSTUDIO_TIMEOUT`
- `AUTOCAD_MCP_LMSTUDIO_TIMEOUT_RETRIES`
- `AUTOCAD_MCP_LMSTUDIO_TIMEOUT_BACKOFF`
- `AUTOCAD_MCP_LMSTUDIO_MAX_JSON_TOKENS`
- `AUTOCAD_MCP_LMSTUDIO_MAX_TEXT_TOKENS`

## Development
```powershell
uv sync
uv run pytest tests -v
```

## Troubleshooting
- **AutoCAD not detected**: start AutoCAD LT, open a drawing, reload `mcp_dispatch.lsp`.
- **IPC timeouts**: verify both Python and LISP use the same IPC directory (`C:/temp` by default).
- **LM returns non-JSON/empty content**: fallback planning is enabled, but verify LM Studio model/API settings.
- **Image import preview fails due to malformed JSON**: retry preview first; the parser now attempts fenced-JSON extraction, balanced-root recovery, malformed-member cleanup, and model-assisted JSON repair.
- **Image import says no CAD actions were extracted**: confirm IR contains geometry/annotation objects (aliases like `entities`, `objects`, and `labels` are supported).
- **Push/pull mismatch after repo migration**: verify `git remote -v` points to `MatthewPrograms/AICAD`.

## License
MIT