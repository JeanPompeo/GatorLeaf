# -*- mode: python ; coding: utf-8 -*-

import sys
from pathlib import Path
from PyInstaller.utils.hooks import collect_submodules
from PyInstaller.building.datastruct import Tree

# Hidden imports for cv2 and its submodules
hiddenimports = collect_submodules('cv2')
hiddenimports += ['astrobotany_calibration_card', 'astrobotany_airisquare']

# Data files to bundle
datas=[
    ('config.JSON', '.'),
    ('astrobotany_calibration_card.py', '.'),
    ('astrobotany_airisquare.py', '.'),
]
# On macOS, bundle all Qt plugins (platforms, imageformats, styles, etc.) so opencv-python can find them

if sys.platform == 'darwin':
    venv_root = Path(sys.executable).resolve().parents[1]
    qt_plugins_dir = venv_root / 'lib' / f'python{sys.version_info.major}.{sys.version_info.minor}' / 'site-packages' / 'PyQt5' / 'Qt5' / 'plugins'
    if qt_plugins_dir.exists():
        # Use Tree to copy entire plugins directory structure into qt_plugins inside the bundle
        qt_tree = Tree(str(qt_plugins_dir), prefix='qt_plugins')
        print(f"[INFO] Bundling Qt plugins from: {qt_plugins_dir}")
    else:
        qt_tree = []
        print(f"[WARNING] Qt plugins folder not found: {qt_plugins_dir}")
else:
    qt_tree = []


a = Analysis(
    ['GatorLeaf.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=['qt_plugin_path.py'],
    excludes=[],
    noarchive=False,
    optimize=0,
)

# Add Qt plugins after Analysis creation
a.datas += qt_tree


pyz = PYZ(a.pure)

# Use onedir mode (recommended for macOS .app bundles)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='GatorLeaf',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,  # Disable UPX to avoid notarization issues on macOS
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=True,  # Better argument handling on macOS
    target_arch='x86_64',  # Apple Silicon (arm64), Apple Intel (x86_64), Opens on both? (universal2)
    icon='icon.icns',  
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
    icon='icon.icns',  
    bundle_identifier='com.gatorleaf.app',
    info_plist={
        'NSHighResolutionCapable': 'True',
        'CFBundleShortVersionString': '1.0.0',
        'NSRequiresAquaSystemAppearance': False,
        'LSEnvironment': {
            'QT_MAC_WANTS_LAYER': '1',
        },
    },
)
