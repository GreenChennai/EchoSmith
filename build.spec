# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller 打包：EchoSmith onefile。引擎整合包外置（首运行向导下载/导入）。"""
from PyInstaller.utils.hooks import collect_all

datas = []
binaries = []
hiddenimports = []

d, b, h = collect_all("dearpygui")
datas += d
binaries += b
hiddenimports += h
d2, b2, h2 = collect_all("py7zr")
datas += d2
binaries += b2
hiddenimports += h2

datas.append(("assets/fonts/HarmonyOS_Sans_SC_Regular.ttf", "assets/fonts"))
datas.append(("assets/engine_manifest.json", "assets"))

a = Analysis(
    ["main.py"],
    pathex=["src"],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    excludes=["tkinter", "matplotlib"],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="EchoSmith",
    debug=False,
    strip=False,
    upx=False,
    console=False,
    icon=None,
)
