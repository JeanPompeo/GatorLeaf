# This runtime hook sets Qt plugin search paths at startup so OpenCV (cv2) can find
# the Cocoa platform plugin when bundled by PyInstaller on macOS.
# It is safe on other platforms and no-ops there.

import os
import sys

def _set_qt_plugin_path():
    # Only relevant on macOS
    if sys.platform != 'darwin':
        return

    candidates = []

    # If running from a bundled app, prefer the packaged qt_plugins directory
    meipass = getattr(sys, '_MEIPASS', None)
    if meipass:
        bundled_plugins = os.path.join(meipass, 'qt_plugins')
        if os.path.isdir(os.path.join(bundled_plugins, 'platforms')):
            candidates.append(bundled_plugins)

    # Try to use cv2's own qt/plugins folder if present
    try:
        import cv2
        base = os.path.dirname(cv2.__file__)
        cv2_qt_plugins = os.path.join(base, 'qt', 'plugins')
        if os.path.isdir(os.path.join(cv2_qt_plugins, 'platforms')):
            candidates.append(cv2_qt_plugins)
    except Exception:
        pass

    # Try PyQt/PySide if present (best-effort)
    for mod in ('PyQt6', 'PyQt5', 'PySide6', 'PySide2'):
        try:
            QtCore = __import__(f'{mod}.QtCore', fromlist=['QtCore'])
            qtcore_dir = os.path.dirname(QtCore.__file__)
            for rel in ('plugins', os.path.join('Qt', 'plugins')):
                p = os.path.join(qtcore_dir, rel)
                if os.path.isdir(os.path.join(p, 'platforms')):
                    candidates.append(p)
                    break
        except Exception:
            continue

    # Apply the first discovered candidate
    for p in candidates:
        os.environ.setdefault('QT_QPA_PLATFORM_PLUGIN_PATH', os.path.join(p, 'platforms'))
        os.environ.setdefault('QT_PLUGIN_PATH', p)
        break

_set_qt_plugin_path()

