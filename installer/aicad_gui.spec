# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path

project_root = Path(globals().get("SPECPATH", Path.cwd())).resolve().parent
src_dir = project_root / "src"

datas = []
for folder_name in ("lisp-code", "assets"):
    folder = project_root / folder_name
    if folder.exists():
        datas.append((str(folder), folder_name))

a = Analysis(
    [str(src_dir / "autocad_mcp" / "gui_main.py")],
    pathex=[str(src_dir)],
    binaries=[],
    datas=datas,
    hiddenimports=["PIL._tkinter_finder"],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="AICAD",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="AICAD",
)
