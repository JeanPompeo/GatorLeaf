# -*- mode: python ; coding: utf-8 -*-
import sys
from pathlib import Path

# Simple data files
datas=[
    ('config.JSON', '.'),
    ('astrobotany_calibration_card.py', '.'),
    ('astrobotany_airisquare.py', '.'),
]

a = Analysis(
    ['GatorLeaf.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=['cv2', 'cv2.aruco', 'PIL', 'PIL.Image', 'PIL.ImageOps', 'numpy', 'json', 'csv', 'calendar'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=['qt_plugin_path.py'],
    excludes=[],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='GatorLeaf',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=True,
    target_arch='x86_64',  # Older Macs (Intel chip)
    codesign_identity=None,
    entitlements_file=None,
    icon='icon.icns' if Path('icon.icns').exists() else None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='GatorLeaf',
)

app = BUNDLE(
    coll,
    name='GatorLeaf.app',
    icon='icon.icns' if Path('icon.icns').exists() else None,
    bundle_identifier='com.ufl.gatorleaf',
    info_plist={
        'NSHighResolutionCapable': 'True',
        'CFBundleShortVersionString': '1.0.0',
        'LSMinimumSystemVersion': '11.0',
    },
)