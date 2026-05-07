@echo off
setlocal
cd /d "%~dp0"

python -m autocad_mcp.gui_main

if errorlevel 1 (
  echo.
  echo Failed to launch the GUI.
  echo If needed, install dependencies with:
  echo   python -m pip install -e .
  pause
)

endlocal
