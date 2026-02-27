# GatorLeaf - Leaf Area Analysis Pipeline

import os
import re
import sys
import time

if getattr(sys, "frozen", False):
    exe_dir = os.path.dirname(sys.executable)
    # Be flexible about Qt plugin locations inside .app bundles
    qt_plugin_candidates = [
        os.path.join(exe_dir, "platforms"),
        os.path.join(exe_dir, "PyQt5", "Qt", "plugins", "platforms"),
        os.path.abspath(os.path.join(exe_dir, "..", "PlugIns", "platforms")),
    ]
    for _p in qt_plugin_candidates:
        if os.path.isdir(_p):
            os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = _p
            break

import cv2 as cv
import numpy as np
import csv
import calendar
import datetime as dt
import json
import shutil
from typing import Optional
import subprocess
import re


# Global variables for persistent settings
_persistent_date = None
_persistent_px_per_cm = None
_persistent_labels = {}


# Global variables to track current selections for display in Image Label window
_current_month = None           # e.g., "01"
_current_day = None             # e.g., "15"
_current_year = None            # e.g., "2025"

# Fallback defaults when neither filename nor folder provide labels
DEFAULT_DATE_YYYY_MM_DD = "__/__/__"  # placeholder for missing date; displayed as MM/DD/YY


# ------------------------------------------------------------------------------------------
# ENVIRONMENT/FILE DETECTION UTILITIES 
# ------------------------------------------------------------------------------------------


# Helper: zero-pad numbers based on the largest value in a range
def format_number_with_padding(num, min_range, max_range):
    """
    Return the number as a string, zero-padded to the width of the largest value in the range.
    Example: format_number_with_padding(3, 1, 120) -> '003'
             format_number_with_padding(345, 300, 80000) -> '00345'
    """
    width = max(len(str(min_range)), len(str(max_range)))
    return str(num).zfill(width)

# Detect if running as a compiled executable

def safe_join(*parts):
    """
    Join path parts but raise a clear ValueError if any part is None.
    Converts non-None parts to str before joining so Path-like objects work.
    """
    if any(p is None for p in parts):
        raise ValueError(f"safe_join received None in parts: {parts}")
    return os.path.join(*[str(p) for p in parts])

def is_frozen():
    """Return True if running as a compiled executable (e.g., from PyInstaller)."""
    return getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS')

def read_image_with_orientation(path):
    """
    Robust image reader with fallbacks.
    - Tries OpenCV imread first (fast path)
    - Falls back to Pillow with EXIF orientation correction and RGB→BGR conversion
    - Finally attempts OpenCV imdecode from raw bytes
    Returns BGR image or raises FileNotFoundError if unreadable.
    """
    # 1) Fast path: OpenCV
    try:
        img = cv.imread(path, cv.IMREAD_COLOR)
        if img is not None:
            return img
    except Exception:
        pass

    # 2) Fallback: Pillow with EXIF orientation handling

        pil_available = True
    except Exception:
        pil_available = False

    if pil_available:
        try:
            with Image.open(path) as im:
                # Correct orientation using EXIF, if present
                try:
                    im = ImageOps.exif_transpose(im)
                except Exception:
                    pass
                # Normalize to RGB
                if im.mode not in ("RGB",):
                    im = im.convert("RGB")
                arr = np.array(im)
                # Convert RGB (Pillow) -> BGR (OpenCV)
                img_bgr = cv.cvtColor(arr, cv.COLOR_RGB2BGR)
                if img_bgr is not None:
                    return img_bgr
        except Exception:
            pass

    # 3) Final fallback: OpenCV imdecode from raw bytes
    try:
        with open(path, "rb") as f:
            data = np.frombuffer(f.read(), dtype=np.uint8)
        img2 = cv.imdecode(data, cv.IMREAD_COLOR)
        if img2 is not None:
            return img2
    except Exception:
        pass

    # If all loaders failed, raise
    raise FileNotFoundError(f"Failed to read image with available loaders: {path}")

def is_supported_image_type(path):
    """Return True if file bytes indicate JPEG, PNG, or TIFF, regardless of extension.
    Helps avoid trying to read HEIF/HEIC files mislabeled as .jpg.
    """
    try:
        with open(path, 'rb') as f:
            header = f.read(16)
        # JPEG: starts with 0xFF 0xD8 0xFF
        if len(header) >= 3 and header[0] == 0xFF and header[1] == 0xD8 and header[2] == 0xFF:
            return True
        # PNG: 89 50 4E 47 0D 0A 1A 0A
        if header.startswith(b"\x89PNG\r\n\x1a\n"):
            return True
        # TIFF: II*\x00 or MM\x00*
        if header.startswith(b"II*\x00") or header.startswith(b"MM\x00*"):
            return True
        return False
    except Exception:
        return False
    
def get_working_directory():
    """Get the appropriate working directory for file operations.
    For bundled apps (.app), only looks in the directory containing the .app.
    For development, uses the script directory.
    """
    if is_frozen():
        executable_dir = os.path.dirname(sys.executable)
        
        # For macOS .app bundles, navigate to the directory containing the .app
        if sys.platform == "darwin" and ".app" in executable_dir:
            # From Contents/MacOS/GatorLeaf, go up to GatorLeaf.app's parent
            app_parent_dir = os.path.abspath(os.path.join(executable_dir, '../../..'))
            print(f"📱 Detected .app bundle, using directory: {app_parent_dir}")
            return app_parent_dir
        
        # For Windows .exe or other frozen executables, use the executable directory
        return executable_dir
    else:
        return os.path.dirname(os.path.abspath(__file__))

def find_input_directory():
    """Find the input directory.
    For .app bundles: ONLY looks in the directory containing the .app file.
    For .exe/development: looks in script directory.
    """
    # For .app bundles, ONLY check the .app parent directory
    if is_frozen() and sys.platform == "darwin":
        executable_dir = os.path.dirname(sys.executable)
        if ".app" in executable_dir:
            app_parent_dir = os.path.abspath(os.path.join(executable_dir, '../../..'))
            inputs_path = os.path.join(app_parent_dir, "Inputs")
            if os.path.isdir(inputs_path):
                print(f"✅ Found Inputs directory: {inputs_path}")
                return inputs_path
            else:
                print(f"❌ ERROR: Inputs folder not found in {app_parent_dir}")
                print(f"   Please place an 'Inputs' folder in the same directory as GatorLeaf.app")
                return None
    
    # For Windows .exe or development: check multiple locations
    possible_names = ["Inputs", "inputs"]
    possible_locations = [
        SCRIPT_DIR,                 # Script directory
        os.getcwd(),                # Current working directory
        os.path.dirname(SCRIPT_DIR) # Parent directory
    ]
    
    for location in possible_locations:
        for name in possible_names:
            candidate_path = os.path.join(location, name)
            if os.path.exists(candidate_path) and os.path.isdir(candidate_path):
                return candidate_path
    # If nothing found, return the default (will be created if needed)
    return os.path.join(SCRIPT_DIR, "Inputs")

def move_outputs_to_current_directory():  
    """Move output files from temp directory to the app/exe directory when running as executable."""
    if not is_frozen():
        return
    if hasattr(sys, 'frozen'):
        executable_dir = os.path.dirname(sys.executable)
        if sys.platform == "darwin" and ".app" in executable_dir:
            executable_dir = os.path.abspath(os.path.join(executable_dir, '../../..'))
    else:
        executable_dir = os.path.dirname(os.path.abspath(__file__))

    temp_outputs_dir = CONFIG["PATHS"]["DIR_OUTPUTS"]
    target_outputs_dir = os.path.join(executable_dir, "Outputs")

    if os.path.exists(temp_outputs_dir) and os.path.abspath(temp_outputs_dir) != os.path.abspath(target_outputs_dir):
        try:
            print(f"\n=== Moving Output Files ===")
            print(f"From temp location: {temp_outputs_dir}")
            print(f"To current directory: {target_outputs_dir}")
            os.makedirs(target_outputs_dir, exist_ok=True)
            files_moved = 0
            for item_name in os.listdir(temp_outputs_dir):
                src_path = os.path.join(temp_outputs_dir, item_name)
                dst_path = os.path.join(target_outputs_dir, item_name)
                try:
                    if os.path.isfile(src_path):
                        shutil.move(src_path, dst_path)
                        files_moved += 1
                        print(f"✓ Moved: {item_name}")
                    elif os.path.isdir(src_path):
                        if os.path.exists(dst_path):
                            for sub_item in os.listdir(src_path):
                                sub_src = os.path.join(src_path, sub_item)
                                sub_dst = os.path.join(dst_path, sub_item)
                                shutil.move(sub_src, sub_dst)
                                files_moved += 1
                            try:
                                os.rmdir(src_path)
                            except:
                                pass
                        else:
                            shutil.move(src_path, dst_path)
                            files_moved += 1
                        print(f"✓ Moved directory: {item_name}")
                except Exception as e:
                    print(f"⚠ Warning: Could not move {item_name}: {e}")
            print(f"\n🎉 Successfully moved {files_moved} items to: {target_outputs_dir}")
            print(f"Your results are now accessible at: {os.path.abspath(target_outputs_dir)}")
        except Exception as e:
            print(f"❌ Error moving output files: {e}")
            print(f"Files may still be accessible at: {temp_outputs_dir}")

# Find the directory where this Python script is located
SCRIPT_DIR = get_working_directory()

# ------------------------------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------------------------------
# Key sections to configure BEFORE running:
# - PATHS: Input/output directories and file names
    # - INPUT: UI selection options (dynamic labels, date ranges)
# - CALIB: Calibration settings (ruler length, padding)
# - SEG: Segmentation parameters (HSV thresholds, morphology, size filters)
# - RUN: Runtime behavior (debug, overlays, training data)
# - ML_TRAINING_OUTPUTS: Which AI training files to generate
#
# Most settings can be overridden via config.JSON without editing this file
# ------------------------------------------------------------------------------------------


CONFIG = {
    # ======================================================================================
    # PATHS: File and Directory Configuration
    # ======================================================================================
    "PATHS": {
        # Input/Output Directories
        "DIR_SAMPLE_IMAGES": find_input_directory(),  # Auto-detects "Inputs" folder
        "DIR_OUTPUTS": os.path.join(SCRIPT_DIR, "Outputs"),  # Where all results are saved
        
        # Output Subdirectories (created inside DIR_OUTPUTS)
        "OVERLAY_SUBDIR": "Overlays",            # Annotated images with leaf outlines
        "DEBUG_SUBDIR": "Debug",                 # Two-panel review images (original | mask)
        "RENAMED_SUBDIR": "Renamed_Images",      # Copies of images with standardized names
        
        # AI Training Data Subdirectories
        "TRAINING_SUBDIR": "ML_Training_Data",  # Root folder for all training artifacts
        "MASKS_SUBDIR": "Binary_Masks",         # Binary leaf masks (PNG)
        "CONTOURS_SUBDIR": "Leaf_Contours",      # Individual leaf contours (JSON + visualization)
        "OBJECTS_SUBDIR": "Object_Annotations",  # Calibration card and label annotations
        "YOLO_SUBDIR": "YOLO",                   # YOLO format bounding boxes (TXT)
        "COCO_SUBDIR": "COCO",                   # COCO format annotations (JSON)
        
        # CSV Output Filenames
        "LEAF_AREA_CSV_NAME": "Leaf_Area.csv",             # Main results: total leaf area per sample
        "LEAF_DIST_CSV_NAME": "Leaf_Distribution.csv",     # Individual leaf areas (L1, L2, L3...)
        "LEAF_LENGTH_CSV_NAME": "Leaf_Length.csv",         # Individual leaf lengths (L1, L2, L3...)
        "LEAF_WIDTH_CSV_NAME": "Leaf_Width.csv",           # Individual leaf widths (L1, L2, L3...)
    },
    
    # ======================================================================================
    # INPUTS: Dynamic Label Schema (L1..Ln)
    # ======================================================================================
    "INPUTS": {
        "LABELS":  [],
        "YEAR": [2024, 2025],
        "MONTH": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        "DAY": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31],
        "HOUR": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
        "MINUTE": [0, 15, 30, 45],
        "SECOND": [0, 15, 30, 45],
        "LEAF_NUMBER_COLUMNS": 120
    },

    # ======================================================================================
    # FILE_FORMATS: Patterns and Date Formats
    # ======================================================================================
    "FILE_FORMATS": {
        "DATE_DATA_FORMAT" : "YYYY-MM-DD",
        "DATETIME_DATA_FORMAT" : "YYYY-MM-DD HH:MM:SS",

        "INPUT_DATE_FORMAT": ["YYYY_MM_DD"],
        "INPUT_FILENAME_PATTERN": "{L1}_{L2}_{L3}_{L4}_{L5}.jpg",

        "OUTPUT_DATE_FORMAT": ["YYYY-MM-DD"],
        "OUTPUT_FOLDER_DATE_FORMAT": "YYYY_MM_DD",
        "OUTPUT_FOLDER_PATTERN": "{L1}_{L2}_{L3}",
        "OUTPUT_FILE_DATE_FORMAT": "YYYY_MM_DD",
        "OUTPUT_FILE_NAME_PATTERN": "{L1}_{L2}_{L3}_{L4}_{L5}.jpg"
    },

    # ======================================================================================
    # COLORS: Optional per-label color settings
    # ======================================================================================
    "COLORS": {
        "Labels": ["#3c78b4", "#c88d28", "#646464", "#559655", "#b45050", "#e06ca7", "#bdb937"],
        "L2": ["#3c78b4", "#c88d28", "#646464", "#559655", "#b45050", "#e06ca7"],
        "L4": ["#3c78b4", "#c88d28", "#646464", "#559655", "#b45050", "#e06ca7"]
    },
    
    # ======================================================================================
    # CALIB: Calibration Settings
    # ======================================================================================
    "CALIB": {
        "MANUAL_CALIBRATION_CM": 5.0,       # Known length on ruler/card (cm). Larger = more accurate. Default: 5cm
        "EXCLUDE_PAD_CM": 0.0,              # Extra padding around calibration card to exclude (cm). Increase if card shadow causes issues
        "PERSISTENT_CALIBRATION": False,    # If true, reuse last calibration for subsequent images (skips manual step)
        "QR_CODE_CALIBRATION": False,       # Use QR/Astrobotany square for auto calibration; fallback to manual if not detected
    },
    
    # ======================================================================================
    # SEG: Segmentation Parameters (MOST IMPORTANT SECTION TO TUNE)
    # ======================================================================================
    "SEG": {
        # ==================================================================================
        # Base HSV Thresholds (Applied to ALL pixels before hue filtering)
        # Now using min/max pairs for all relevant thresholds
        # ==================================================================================
        "HSV_S_MIN": 45,              # (Legacy, for backward compatibility)
        "HSV_S_MAX": 255,             # Max saturation (0-255)
        "HSV_V_MIN": 65,              # (Legacy, for backward compatibility)
        "HSV_V_MAX": 255,             # Max brightness (0-255)
        "BLACK_V_MIN": 0,             # Min brightness for black exclusion (0-255)
        "BLACK_V_MAX": 35,            # Max brightness for black exclusion (0-255)
        "WHITE_S_MIN": 0,             # Min saturation for white exclusion (0-255)
        "WHITE_S_MAX": 25,            # Max saturation for white exclusion (0-255)
        "WHITE_V_MIN": 200,           # Min brightness for white exclusion (0-255)
        "WHITE_V_MAX": 255,           # Max brightness for white exclusion (0-255)
        "BLUE_EXCLUDE": True,         # Exclude blue hues (leaves are rarely blue)
        "BLUE_H_MIN": 90,             # Blue hue start (0-180)
        "BLUE_H_MAX": 130,            # Blue hue end (0-180)
        
        # ==================================================================================
        # Hue Band Filtering (Defines which COLORS are considered leaf tissue)
        # OpenCV Hue range: 0-180 (Red=0/180, Yellow=30, Green=60, Cyan=90, Blue=120, Magenta=150)
        # ==================================================================================
        "HSV_USE_BANDS": True,        # Enable hue-based filtering. False=any color passes (will include background!)
        
        # Green Leaves (most common)
        "GREEN_H_MIN": 38,            # Green hue start (0-180). Lower=includes yellow-green, Higher=purer green only
        "GREEN_H_MAX": 78,            # Green hue end (0-180). Lower=excludes blue-green, Higher=includes cyan/teal
        "GREEN_V_MIN": 60,            # Min brightness for green (0-255). Lower=includes darker greens, Higher=only bright greens
        
        # Yellow-Green Leaves
        "YELLOW_H_MIN": 20,           # Yellow hue start (0-180). Lower=includes orange tones, Higher=purer yellow
        "YELLOW_H_MAX": 40,           # Yellow hue end (0-180). Lower=less yellow-green, Higher=more yellow-green
        
        # Brown/Dead Leaves
        "BROWN_H_MIN": 8,             # Brown hue start (0-180). Lower=includes red-brown, Higher=purer brown
        "BROWN_H_MAX": 28,            # Brown hue end (0-180). Lower=less brown, Higher=includes yellow-brown
        "BROWN_S_MIN": 45,            # Min saturation for brown (0-255). Lower=includes dull browns, Higher=only vivid browns
        "BROWN_V_MIN": 100,            # Min brightness for brown (0-255). Lower=includes darker browns, Higher=only brighter browns 
        
        # Purple/Magenta Leaves (with STRICTER thresholds to block black background)
        "PURPLE_H_MIN": 110,          # Purple hue start (0-180). Lower=includes blue-purple, Higher=purer magenta
        "PURPLE_H_MAX": 160,          # Purple hue end (0-180). Lower=less purple, Higher=includes pink/red
        "PURPLE_S_MIN": 40,           # Min saturation for purple (0-255). HIGHER than HSV_S_MIN to block dark background
        "PURPLE_V_MIN": 55,           # Min brightness for purple (0-255). HIGHER than HSV_V_MIN to block dark background
        
        # ==================================================================================
        # Lab Color Space Gating (Additional color filtering in L*a*b* space)
        # ==================================================================================
        "LAB_GATE": True,             # Enable Lab filtering. False=skip Lab checks (faster but less selective)
        "LAB_A_MAX": 15,              # Max a* value (green-red axis, -128 to +127). Lower=greener only, Higher=allows red/purple tones
        "LAB_B_MAX": 140,             # Max b* value (blue-yellow axis, -128 to +127). Lower=bluer, Higher=allows yellow tones
        "LAB_NEUTRAL_EXCLUDE": True,  # Exclude near-neutral Lab values to reduce background bleed
        "LAB_NEUTRAL_A_ABS_MAX": 3,   # Neutral a* abs max
        "LAB_NEUTRAL_B_ABS_MAX": 3,   # Neutral b* abs max
        
        # ==================================================================================
        # Morphological Operations (Cleanup and hole filling)
        # ==================================================================================
        "OPEN_DIAMETER_CM": 0.045,   # Opening kernel size (cm). Smaller=keeps thin stems, Larger=removes small noise/specks
        "CLOSE_DIAMETER_CM": 0.015,   # Closing kernel size (cm). Smaller=preserves fine details, Larger=bridges gaps in leaves
        "CLOSE_BRIDGE_CM": 0.015,     # Bridge-closing kernel (cm). Smaller=fewer connections, Larger=connects separated leaf parts
        "REOPEN_CM": 0.015,           # Final opening to restore edges (cm). Smaller=sharper edges, Larger=smoother edges
        "FILL_HOLES": True,           # Fill enclosed holes in leaves. False=keeps interior voids (useful for some leaf types)
        "HOLE_MAX_CM2": 0.50,         # Max hole size to fill (cm²). Smaller=only tiny holes filled, Larger=fills big interior gaps. 0=fill all holes
        
        # ==================================================================================
        # Size Thresholds (Component filtering by area)
        # CRITICAL: Must follow hierarchy: NOISE_CM2 < TINY_FRAGMENT_CM2 < MIN_LEAF_CM2
        # ==================================================================================
        "NOISE_CM2": 0.038,           # Immediate removal threshold (cm²). Components smaller than this are deleted as noise
                                      # Lower=keeps more small bits, Higher=removes more specks
        
        "TINY_FRAGMENT_CM2": 0.039,   # Mergeable fragment size (cm²). Fragments this size can merge into nearby large leaves
                                      # Lower=merges smaller pieces, Higher=only merges bigger fragments
                                      # MUST BE: NOISE_CM2 < TINY_FRAGMENT_CM2 < MIN_LEAF_CM2
        
        "MIN_LEAF_CM2": 0.04,         # Minimum standalone leaf size (cm²). Final leaves must be at least this big
                                      # Lower=keeps small leaves, Higher=only large leaves in final output
                                      # MUST BE LARGER THAN TINY_FRAGMENT_CM2
        
        "MERGE_SMALL_WITHIN_CM": 0.04,  # Search distance for merging fragments (cm). Larger=merges fragments farther away, Smaller=only very close fragments merge
        
        # ==================================================================================
        # Performance
        # ==================================================================================
        "FAST_MAX_WIDTH": 5000,        # Downscale images wider than this for speed (pixels). 0=disable downscaling (slower but more accurate)
        "SKIP_MEDIAN": True,           # Skip median blur on HSV channels. True=faster, False=smoother but slower
    },
    
    # ======================================================================================
    # RUN: Runtime Behavior Flags
    # ======================================================================================
    "RUN": {
        "TROUBLESHOOT_MODE": 3,           # Debug verbosity (0-3). 0=silent, 1=errors, 2=warnings, 3=all info
        "SAVE_DEBUG_IMAGES": True,        # Save two-panel review images (original | mask) to Debug/
        "SAVE_OVERLAYS": True,            # Save annotated images with leaf outlines to Overlays/
        "REVIEW_SEGMENTATION": True,      # Shows interactive window for reviewing segmentation. False=auto-accept all segmentations
        "HEADLESS": False,                # Run without UI windows. True=no display (for servers), False=show windows
        "INTERACTIVE_CALIB": True,        # Use manual calibration workflow. False=requires pre-calibrated images
        "SELECT_MASKS": True,             # Enable manual label/bag masking. False=skip mask selection
        "SAVE_RENAMED_COPIES": False,     # Save renamed image copies. True=duplicate to Renamed_Images/, False=rename in place
        "MANUAL_LABEL_EXCLUSION": True,   # Allow manual exclusion of labels/bags. False=auto-exclude calibration card only
        "SAVE_ML_TRAINING_DATA": True,    # Generate AI training files. False=skip all training data generation (faster)
    },
    
    # ======================================================================================
    # ML_TRAINING_OUTPUTS: AI Training Data Generation Control
    # Only applies if RUN.SAVE_ML_TRAINING_DATA is True
    # ======================================================================================
    "ML_TRAINING_OUTPUTS": {
        "COCO": True,                     # Generate COCO format JSON. True=creates comprehensive annotations for object detection models
        "Leaf_Contours": True,            # Generate leaf contour files. True=saves individual leaf outlines + visualizations
        "Object_Annotations": True,       # Generate calibration/label annotations. True=saves bounding boxes for non-leaf objects
        "Segmentation_Masks": True,       # Generate binary segmentation masks. True=saves leaf masks as PNG images
        "YOLO": True,                     # Generate YOLO format labels. True=creates bounding box TXT files for YOLO models
    },
    "UI": {
        # === DYNAMIC WINDOW SIZING ===
        "REFERENCE_IMAGE_SIZE": (1200, 900),
        "WINDOW_SPACING": 15,
        "PANEL_MIN_WIDTH": 400,
        "PANEL_MAX_WIDTH": 600,
        "CONTENT_PADDING_MULTIPLIER": 4,
        "LABEL_BUTTON_GAP": 10,            # space between last label and top button row
        "BUTTON_INSTRUCTION_GAP": 30,      # button→instruction gap (all panels)
        "INSTRUCTION_BOTTOM_MARGIN": 18,    # instruction→bottom edge
        "LABEL_LINE_HEIGHT": 18,            # estimated per-label line height
        "INSTRUCTION_TEXT_BLOCK_HEIGHT": 30,

        # === REVIEW WINDOW SIZING  ===
        # Width for segmentation review now follows REFERENCE_IMAGE_SIZE only;
        # these controls are limited to header height and spacing.
        "REVIEW_HEADER_HEIGHT": 40,       # Header strip height
        "REVIEW_PANEL_SPACING": 12,       # Gap between the two panels

        # === WINDOW POSITIONING ===
        "DEFAULT_WINDOW_POSITION": (0, 0),
        "REFERENCE_WINDOW_POSITION": (0, 0),
        "PANEL_OFFSET_X": 15,
        "UPDATE_WINDOW_GAP": 45,

        # === BUTTON DIMENSIONS ===
        "BUTTON_HEIGHT": 30,
        "BUTTON_SPACING": 8,
        "INPUT_TOP_OFFSET": 80,              # Top Y offset for numeric input panels
        "BUTTON_STACK_TOP_OFFSET": 130,      # Start Y for Image Label buttons
        "GRID_BUTTON_HEIGHT": 30,
        
        "GRID_SPACING": 10,
        "GRID_SPACING_TIGHT": 5,
        "MIN_BUTTON_WIDTH": 50,
        "MAX_GRID_BUTTON_WIDTH": 100,
        "MAX_GRID_PANEL_WIDTH": 400,

        # Background color for high-DPI UI panels (separate from VISUALIZATION panels)
        "PANEL_BG_COLOR": (240, 240, 240),

        # === LAYOUT CONFIGURATIONS ===
        "MONTH_GRID_COLS": 3,
        "DAY_GRID_COLS": 7,

        # === WINDOW LAYOUTS (cols + layout type) ===
        # These can be overridden in user config to customize grid/list behavior per selection type.
        "WINDOW_LAYOUTS": {
            "YEAR_SELECTION": {"cols": 2, "layout_type": "list"},
            "MONTH_SELECTION": {"cols": 3, "layout_type": "grid"},
            "DAY_SELECTION": {"cols": 7, "layout_type": "grid"},
            "HOUR_SELECTION": {"cols": 6, "layout_type": "grid"},
            "MINUTE_SELECTION": {"cols": 4, "layout_type": "grid"},
            "SECOND_SELECTION": {"cols": 4, "layout_type": "grid"},
        },

        # === INPUT VALIDATION ===
        "MAX_SAMPLE_INPUT_DIGITS": 3,
        "INPUT_ERROR_DISPLAY_MS": 1500,    
    },
    "VISUALIZATION": {
        # === OVERLAY PANEL DIMENSIONS ===
        "HEADER_PANEL_WIDTH": 400,
        "HEADER_PANEL_HEIGHT": 180,
        "STATS_PANEL_WIDTH": 250,
        "STATS_PANEL_HEIGHT": 120,
        "PANEL_MARGIN": 10,
        
        # === TEXT STYLING ===
        # Centralized text styling and colors. All text-related options live here.
        # Overlay header/labels
        "HEADER_FONT_SCALE": 1,
        "HEADER_THICKNESS": 1,
        "HEADER_LINE_HEIGHT": 20,
        # Stats panel
        "STATS_FONT_SCALE": 0.4,
        "STATS_THICKNESS": 1,
        "STATS_LINE_SPACING": 20,
        # Leaf labels
        "LABEL_FONT_SCALE": 0.4,
        "LABEL_THICKNESS": 1,

        # High-DPI UI panels (manual input)
        "TITLE_FONT_SIZE": 1.0,
        "TITLE_THICKNESS": 3,

        "SUBTITLE_FONT_SIZE": 0.6,
        "SUBTITLE_THICKNESS": 2,
        
        "BUTTON_FONT_SIZE": 0.5,
        "BUTTON_THICKNESS": 2,
        
        "INSTRUCTION_FONT_SIZE": 0.4,
        "INSTRUCTION_THICKNESS": 1,
        "REFERENCE_INSTRUCTION_SCALE": 1.3,
        "REVIEW_HEADER_FONT_SCALE": 0.4,
        "REVIEW_HEADER_THICKNESS": 1,

        # Base text colors for panels and buttons (BGR for OpenCV)
        "TEXT_PRIMARY": (51, 51, 51),
        "TEXT_SECONDARY": (119, 119, 119),
        "TEXT_WHITE": (255, 255, 255),
        # Font family for all UI text (OpenCV Hershey fonts)
        # Options: HERSHEY_SIMPLEX, HERSHEY_PLAIN, HERSHEY_DUPLEX, HERSHEY_COMPLEX,
        #          HERSHEY_TRIPLEX, HERSHEY_COMPLEX_SMALL, HERSHEY_SCRIPT_SIMPLEX,
        #          HERSHEY_SCRIPT_COMPLEX
        "FONT_FACE": "HERSHEY_SIMPLEX",

        # === COMPONENT VISUALIZATION ===
        "BBOX_THICKNESS": 2,
        "CIRCLE_RADIUS": 12,
        "CIRCLE_THICKNESS": 2,
        "TEXT_BACKGROUND_PADDING": 1,
        "TEXT_BORDER_THICKNESS": 1,

        # === SIZE-BASED COLORS (BGR format) ===
        "SIZE_COLORS": {
            "small": (0, 250, 250),         # Green for < 10 cm²
            "medium": (0, 250, 250),         # Orange for 10-30 cm²
            "large": (0, 250, 250),         # Blue for >= 30 cm²
        },
        # === PANEL COLORS ===
        "PANEL_ALPHA": 0.8,
        "PANEL_BG_COLOR": (50, 50, 50),        # Dark gray
        "STATS_PANEL_BG_COLOR": (40, 40, 40),  # Darker gray
        "PANEL_BORDER_COLOR": (255, 255, 255), # White
        "PANEL_BORDER_THICKNESS": 2,
        # === TEXT COLORS ===
        "TITLE_COLOR": (0, 255, 255),       # Cyan
        "SUCCESS_COLOR": (0, 255, 0),       # Green
        "WARNING_COLOR": (0, 0, 255),       # Red
        "INFO_COLOR": (255, 255, 0),        # Yellow
        "SECONDARY_COLOR": (200, 200, 255), # Light purple
        "TEXT_COLOR": (255, 255, 255),      # White
    }
}


# Centralized OpenCV font mapping and access helper

# =====================
# Helper: min/max range check
def _in_range(val, minv, maxv):
    """Helper: check if val is in [minv, maxv] inclusive. Supports numpy arrays."""
    import numpy as np
    return np.logical_and(val >= minv, val <= maxv)

# Example: segmentation mask creation using min/max pairs
def segment_leaf_mask(hsv_img, config):
    """Segment leaf mask using min/max pairs from config['SEG']."""
    import numpy as np
    import cv2 as cv
    seg = config["SEG"]
    # HSV base thresholds
    s_min = seg.get("HSV_S_MIN", 0)
    s_max = seg.get("HSV_S_MAX", 255)
    v_min = seg.get("HSV_V_MIN", 0)
    v_max = seg.get("HSV_V_MAX", 255)
    # Black exclusion
    black_v_min = seg.get("BLACK_V_MIN", 0)
    black_v_max = seg.get("BLACK_V_MAX", 35)
    # White exclusion
    white_s_min = seg.get("WHITE_S_MIN", 0)
    white_s_max = seg.get("WHITE_S_MAX", 25)
    white_v_min = seg.get("WHITE_V_MIN", 200)
    white_v_max = seg.get("WHITE_V_MAX", 255)

    # Create masks for each exclusion
    h, s, v = cv.split(hsv_img)
    # Main HSV mask (in S/V range)
    mask_hsv = np.logical_and(_in_range(s, s_min, s_max), _in_range(v, v_min, v_max))
    # Black exclusion mask
    mask_black = _in_range(v, black_v_min, black_v_max)
    # White exclusion mask
    mask_white = np.logical_and(_in_range(s, white_s_min, white_s_max), _in_range(v, white_v_min, white_v_max))
    # Combine masks: keep only pixels in main HSV mask, not in black/white exclusion
    mask = np.logical_and(mask_hsv, np.logical_not(mask_black))
    mask = np.logical_and(mask, np.logical_not(mask_white))
    return mask.astype(np.uint8) * 255
_FONT_MAP = {
    "HERSHEY_SIMPLEX": cv.FONT_HERSHEY_SIMPLEX,
    "HERSHEY_PLAIN": cv.FONT_HERSHEY_PLAIN,
    "HERSHEY_DUPLEX": cv.FONT_HERSHEY_DUPLEX,
    "HERSHEY_COMPLEX": cv.FONT_HERSHEY_COMPLEX,
    "HERSHEY_TRIPLEX": cv.FONT_HERSHEY_TRIPLEX,
    "HERSHEY_COMPLEX_SMALL": cv.FONT_HERSHEY_COMPLEX_SMALL,
    "HERSHEY_SCRIPT_SIMPLEX": cv.FONT_HERSHEY_SCRIPT_SIMPLEX,
    "HERSHEY_SCRIPT_COMPLEX": cv.FONT_HERSHEY_SCRIPT_COMPLEX,
}


# ------------------------------------------------------------------------------------------
# CONFIG LOADING / USER OVERRIDE LOGIC
# ------------------------------------------------------------------------------------------

def load_config_from_json():
    """Load configuration from a JSON file if it exists."""
    config_names = ["config.JSON"]
    config_locations = [
        SCRIPT_DIR,
        os.getcwd(),
        os.path.expanduser("~"),
    ]

    config_path = None
    for location in config_locations:
        for name in config_names:
            candidate_path = os.path.join(location, name)
            if os.path.exists(candidate_path) and os.path.isfile(candidate_path):
                config_path = candidate_path
                break
        if config_path:
            break

    if not config_path:
        print("No configuration file found. Using default settings.")
        return {}

    try:
        print(f"Loading configuration from: {config_path}")
        with open(config_path, 'r') as f:
            user_config = json.load(f)
        print("Configuration loaded successfully.")
        return user_config
    except Exception as e:
        print(f"Error loading configuration: {e}")
        print("Using default settings instead.")
        return {}

def deep_update(base_dict, update_dict):
    """Recursively merge update_dict into base_dict.
    Safeguards: if update_dict is not a dict, return base_dict unchanged.
    """
    if not isinstance(update_dict, dict):
        return base_dict
    for key, value in update_dict.items():
        if isinstance(base_dict.get(key), dict) and isinstance(value, dict):
            deep_update(base_dict[key], value)
        else:
            base_dict[key] = value
    return base_dict

def normalize_date_string(date_str):
    """Normalize arbitrary date strings to canonical 'YYYY_MM_DD'.

    Tries formats from CONFIG['FILE_FORMATS']['INPUT_DATE_FORMAT'] first
    (accepts string or list), then falls back to common patterns.
    Returns 'YYYY_MM_DD' or None if parsing fails.
    """
    if not date_str:
        return None
    s = str(date_str).strip()
    file_formats = CONFIG.get("FILE_FORMATS", {}) or {}
    cfg = file_formats.get("INPUT_DATE_FORMAT") or file_formats.get("INPUT_DATE_FORMATS")
    formats = []
    if isinstance(cfg, str):
        formats = [str(cfg)]
    elif isinstance(cfg, (list, tuple)):
        formats = [str(x) for x in cfg if x]
    # Add robust fallbacks
    formats += [
        "YYYY_MM_DD", "YY_MM_DD",
        "MM_DD_YYYY", "MM_DD_YY",
        "YYYY-MM-DD", "YY-MM-DD", "MM-DD-YYYY", "MM-DD-YY",
        "YYYY.MM.DD", "YY.MM.DD", "MM.DD.YYYY", "MM.DD.YY",
        "MM/DD/YYYY", "MM/DD/YY"
    ]
    for f in formats:
        try:
            dt_obj = dt.datetime.strptime(s, _tokens_to_strptime(f))
            return f"{dt_obj.year:04d}_{dt_obj.month:02d}_{dt_obj.day:02d}"
        except Exception:
            continue
    return None

def format_date_with_pattern(date_input, out_format):
    """Format arbitrary date strings to a configured output date string.

    Accepts various input formats (underscores, dashes, slashes, dots) using tokens
    defined by config; outputs based on tokenized `out_format` (e.g., 'MM.DD.YY').
    """
    canonical = normalize_date_string(date_input)
    if canonical is None:
        return date_input
    try:
        date = dt.datetime.strptime(canonical, "%Y_%m_%d")
        return date.strftime(_tokens_to_strptime(out_format))
    except Exception:
        return canonical

def _format_time_value(value):
    if isinstance(value, dt.time):
        return value.strftime("%H:%M:%S")
    if isinstance(value, dt.datetime):
        return value.strftime("%H:%M:%S")
    if isinstance(value, str) and value:
        m = re.match(r"^\s*(\d{1,2}):(\d{1,2})(?::(\d{1,2}))?\s*$", value)
        if m:
            h, mnt, sec = m.groups()
            sec = sec or "00"
            return f"{int(h):02d}:{int(mnt):02d}:{int(sec):02d}"
    return value

def format_label_value_for_csv(label_name, value, date_data_fmt):
    if value is None:
        return ""
    field_cfg = next((f for f in CONFIG.get("SCHEMA_FIELDS", []) or [] if f.get("name") == label_name), None)
    ftype = str(field_cfg.get("type") if field_cfg else "").upper()
    if ftype == "DATE":
        if isinstance(value, str) and value:
            canon = normalize_date_string(value)
            if canon:
                try:
                    return dt.datetime.strptime(canon, "%Y_%m_%d").strftime(_tokens_to_strptime(date_data_fmt))
                except Exception:
                    return value
        return value
    if ftype == "TIME":
        return _format_time_value(value)
    if ftype == "NUMERIC":
        try:
            return float(value)
        except Exception:
            return value
    if ftype == "SELECT":
        return "" if value is None else str(value)
    return value

def _tokens_to_strptime(fmt):
    """
    Map tokenized formats like 'YYYY_MM_DD' or 'MM.DD.YY' to Python strptime/strftime.
    Supported tokens: YYYY, YY, MM, DD and time pattern 'HH:MM:SS'. Non-alnum separators are preserved.
    """
    # Special-case time pattern to avoid ambiguity of 'MM' (month vs minutes)
    if isinstance(fmt, str):
        fmt = fmt.replace("HH:MM:SS", "%H:%M:%S").replace("HH:MM", "%H:%M")

    mapping = {
        "YYYY": "%Y",
        "YY": "%y",
        "MM": "%m",
        "DD": "%d",
    }
    out = []
    i = 0
    while i < len(fmt):
        if fmt[i:i+4] == "YYYY":
            out.append(mapping["YYYY"]); i += 4
        elif fmt[i:i+2] == "YY":
            out.append(mapping["YY"]); i += 2
        elif fmt[i:i+2] == "MM":
            out.append(mapping["MM"]); i += 2
        elif fmt[i:i+2] == "DD":
            out.append(mapping["DD"]); i += 2
        else:
            out.append(fmt[i]); i += 1
    return "".join(out)

def parse_date_token_with_formats(token, formats):
    """Try parsing a single token against a list of configured formats; return 'YYYY_MM_DD' if successful."""
    token = str(token).strip()
    for f in formats:
        try:
            date = dt.datetime.strptime(token, _tokens_to_strptime(f))
            return f"{date.year:04d}_{date.month:02d}_{date.day:02d}"
        except Exception:
            continue
    return None

def build_filename_from_pattern(date_yyyy_mm_dd, label_info, pattern):
    """Deprecated wrapper: use dynamic labels only via build_filename_generic."""
    return build_filename_generic(date_yyyy_mm_dd, label_info or {}, pattern)


# Note: unified OUTPUT_FILE_NAME_PATTERN is required and used throughout to build

def build_folder_name_from_pattern(date_yyyy_mm_dd, pattern=None, label_info=None):
    """Build an output subfolder name using only config-defined tokens.

    Falls back to provided `date_yyyy_mm_dd` if `label_info` is missing.
    Ignores any file extension in the pattern.
    """
    file_formats = CONFIG.get("FILE_FORMATS", {})
    # Prefer unified OUTPUT_DATE_FORMAT (string or first element of list); fallback to legacy keys
    out_date_conf = file_formats.get("OUTPUT_DATE_FORMAT")
    if isinstance(out_date_conf, (list, tuple)) and len(out_date_conf) > 0:
        out_date_fmt = str(out_date_conf[0])
    elif isinstance(out_date_conf, str):
        out_date_fmt = out_date_conf
    else:
        out_date_fmt = file_formats.get("OUTPUT_FOLDER_DATE_FORMAT", "YYYY_MM_DD")
    pat = pattern if pattern is not None else file_formats.get("OUTPUT_FOLDER_PATTERN", "{DATE}")

    # Base values always include DATE
    # Prefer label_info['Date'] if available; else use date_yyyy_mm_dd param
    if isinstance(label_info, dict) and label_info.get("Date"):
        date_src = str(label_info.get("Date"))
    elif isinstance(label_info, dict) and label_info.get("Original_Stem"):
        date_src = str(label_info.get("Original_Stem"))
    else:
        date_src = date_yyyy_mm_dd
    date_src_norm = normalize_date_string(date_src)
    if date_src_norm:
        date_out = format_date_with_pattern(date_src_norm, out_date_fmt)
    else:
        date_out = str(date_src) if date_src else "Output"

    # Assemble dynamic values
    vals = {"DATE": date_out}
    def _is_date_label_name(nm):
        try:
            for f in CONFIG.get("SCHEMA_FIELDS", []) or []:
                if f.get("name") == nm and str(f.get("type")).upper() == "DATE":
                    return True
        except Exception:
            pass
        return False
    # Copy all label fields so custom tokens can resolve
    if isinstance(label_info, dict):
        for k, v in label_info.items():
            # Check if this label is NUMERIC and has min/max in schema
            field_cfg = next((f for f in CONFIG.get("SCHEMA_FIELDS", []) if f.get("name") == k), None)
            if field_cfg and str(field_cfg.get("type")).upper() == "NUMERIC":
                minv, maxv = field_cfg.get("min", 1), field_cfg.get("max", 1)
                try:
                    vals[k] = format_number_with_padding(int(v), minv, maxv)
                except Exception:
                    vals[k] = v
            elif _is_date_label_name(k):
                canon = normalize_date_string(v)
                vals[k] = format_date_with_pattern(canon, out_date_fmt) if canon else v
            else:
                vals[k] = v

    # Map Ln tokens from label_info order (INPUTS.LABELS)
    try:
        labels = CONFIG.get("INPUTS", {}).get("LABELS") or CONFIG.get("INPUTS", {}).get("NAMES") or []
        if isinstance(label_info, dict) and labels:
            for i, nm in enumerate(labels, start=1):
                if nm in label_info:
                    v = label_info.get(nm)
                    field_cfg = next((f for f in CONFIG.get("SCHEMA_FIELDS", []) if f.get("name") == nm), None)
                    if field_cfg and str(field_cfg.get("type")).upper() == "NUMERIC":
                        minv, maxv = field_cfg.get("min", 1), field_cfg.get("max", 1)
                        try:
                            vals[f"L{i}"] = format_number_with_padding(int(v), minv, maxv)
                        except Exception:
                            vals[f"L{i}"] = v
                    elif _is_date_label_name(nm):
                        canon = normalize_date_string(v)
                        vals[f"L{i}"] = format_date_with_pattern(canon, out_date_fmt) if canon else v
                    else:
                        vals[f"L{i}"] = v
    except Exception:
        pass

    try:
        name = pat.format(**vals)
    except Exception:
        name = f"{date_out}"

    # Strip any trailing image extension if user accidentally included one
    for ext in (".jpg", ".jpeg", ".png", ".tif", ".tiff"):
        if name.lower().endswith(ext):
            name = name[: -len(ext)]
            break

    # Sanitize path separators if present
    name = name.replace(os.sep, "_").replace("/", "_").strip()
    return name or "Output"

def build_output_image_path(date_yyyy_mm_dd, image_pattern, output_subdir, outputs_root, label_info=None):
    """Build full path for output image (overlay, debug, etc.) using only config-driven naming.

    Folder path honors OUTPUT_FOLDER_PATTERN and may use dynamic tokens from `label_info`.
    File base honors OUTPUT_FILE_NAME_PATTERN resolved via dynamic label fields.
    """
    # Build experiment folder using dynamic label info when available
    experiment_folder = build_folder_name_from_pattern(date_yyyy_mm_dd, label_info=label_info)
    flat_output = False
    if isinstance(label_info, dict):
        if not label_info.get("Date") and label_info.get("Original_Stem"):
            flat_output = True

    # Special behavior for Debug and Overlays roots
    debug_root = CONFIG["PATHS"].get("DEBUG_SUBDIR", "Debug")
    overlay_root = CONFIG["PATHS"].get("OVERLAY_SUBDIR", "Overlays")
    if output_subdir == debug_root:
        folder_path = os.path.join(outputs_root, debug_root) if flat_output else os.path.join(outputs_root, debug_root, experiment_folder)
    elif output_subdir == overlay_root:
        folder_path = os.path.join(outputs_root, overlay_root) if flat_output else os.path.join(outputs_root, overlay_root, experiment_folder)
    else:
        folder_path = os.path.join(outputs_root, output_subdir) if flat_output else os.path.join(outputs_root, experiment_folder, output_subdir)

    os.makedirs(folder_path, exist_ok=True)

    # Build filename from unified OUTPUT_FILE_NAME_PATTERN using dynamic label_info
    file_formats = CONFIG.get("FILE_FORMATS", {})
    base_pattern = file_formats.get("OUTPUT_FILE_NAME_PATTERN", "{DATE}")

    # Prefer generic builder with label_info; fallback to fixed mapping
    base_with_ext = build_filename_generic(date_yyyy_mm_dd, label_info or {}, base_pattern)
    base_no_ext = os.path.splitext(base_with_ext)[0]

    # Derive suffix from image_pattern
    suffix = ""
    try:
        if isinstance(image_pattern, str) and base_pattern in image_pattern:
            suffix = image_pattern.replace(base_pattern, "")
        elif isinstance(image_pattern, str):
            suffix = image_pattern
        else:
            suffix = "_overlay.jpg"
    except Exception:
        suffix = "_overlay.jpg"

    if not os.path.splitext(suffix)[1]:
        suffix = suffix + ".jpg"

    image_filename = base_no_ext + suffix
    return os.path.join(folder_path, image_filename)

# Deprecated: dynamic filename building now handled by build_filename_generic

def build_filename_generic(date_yyyy_mm_dd, label_info, pattern):
    """Build filename from a generic pattern using dynamic label_info keys.
    - Always provides 'DATE' formatted via NAMING.OUTPUT_FILE_DATE_FORMAT (or fallback).
    - Includes all other keys from label_info to satisfy patterns like {Location}, {Replicate}, {Genotype}.
    """
    label_info = dict(label_info or {})
    file_formats = CONFIG.get("FILE_FORMATS", {})
    out_date_conf = file_formats.get("OUTPUT_DATE_FORMAT")
    if isinstance(out_date_conf, (list, tuple)) and len(out_date_conf) > 0:
        out_date_fmt = str(out_date_conf[0])
    elif isinstance(out_date_conf, str):
        out_date_fmt = out_date_conf
    else:
        out_date_fmt = file_formats.get("OUTPUT_FILE_DATE_FORMAT") or file_formats.get("OUTPUT_FOLDER_DATE_FORMAT") or "YYYY_MM_DD"
    canonical = normalize_date_string(date_yyyy_mm_dd)
    if not canonical and label_info.get("Original_Stem"):
        date_out = str(label_info.get("Original_Stem"))
    else:
        canonical = canonical or "1970_01_01"
        date_out = format_date_with_pattern(canonical, out_date_fmt)

    # Base values always include DATE
    vals = {"DATE": date_out}
    def _is_date_label_name(nm):
        try:
            for f in CONFIG.get("SCHEMA_FIELDS", []) or []:
                if f.get("name") == nm and str(f.get("type")).upper() == "DATE":
                    return True
        except Exception:
            pass
        return False

    # Map Ln tokens according to INPUTS.LABELS ordering
    try:
        labels = CONFIG.get("INPUTS", {}).get("LABELS") or CONFIG.get("INPUTS", {}).get("NAMES") or []
        if labels:
            for i, nm in enumerate(labels, start=1):
                if nm in label_info:
                    v = label_info.get(nm)
                    field_cfg = next((f for f in CONFIG.get("SCHEMA_FIELDS", []) if f.get("name") == nm), None)
                    if field_cfg and str(field_cfg.get("type")).upper() == "NUMERIC":
                        minv, maxv = field_cfg.get("min", 1), field_cfg.get("max", 1)
                        try:
                            vals[f"L{i}"] = format_number_with_padding(int(v), minv, maxv)
                        except Exception:
                            vals[f"L{i}"] = v
                    elif _is_date_label_name(nm):
                        canon = normalize_date_string(v)
                        vals[f"L{i}"] = format_date_with_pattern(canon, out_date_fmt) if canon else v
                    else:
                        vals[f"L{i}"] = v
    except Exception:
        pass

    # Copy all other label fields to vals so pattern tokens can resolve
    for k, v in label_info.items():
        field_cfg = next((f for f in CONFIG.get("SCHEMA_FIELDS", []) if f.get("name") == k), None)
        if field_cfg and str(field_cfg.get("type")).upper() == "NUMERIC":
            minv, maxv = field_cfg.get("min", 1), field_cfg.get("max", 1)
            try:
                vals[k] = format_number_with_padding(int(v), minv, maxv)
            except Exception:
                vals[k] = v
        elif _is_date_label_name(k):
            canon = normalize_date_string(v)
            vals[k] = format_date_with_pattern(canon, out_date_fmt) if canon else v
        else:
            vals[k] = v

    # Attempt formatting directly with dynamic keys
    try:
        base = pattern.format(**vals)
    except Exception:
        # Fallback: ensure at least a sensible filename with DATE
        base = f"{date_out}.jpg"

    if not base.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff")):
        base += ".jpg"
    return base

# Load and apply user configuration from JSON file
_user_config = load_config_from_json()
deep_update(CONFIG, _user_config)

def normalize_dynamic_schema():
    """Normalize dynamic field schema from CONFIG['INPUTS'] using LABELS + L1..Ln definitions.

    Produces CONFIG['SCHEMA_FIELDS'] = [
        { 'name': str, 'type': 'DATE'|'SELECT'|'NUMERIC'|'TIME', 'options': [...], 'min': int, 'max': int, 'colors': {option: (B,G,R)} }
    ]

    Resolves each label name from `INPUTS.LABELS`, then maps ordinal L-index to a definition in `INPUTS.L{i}`.
    """
    inputs = CONFIG.get("INPUTS", {}) or {}
    names = inputs.get("LABELS") or inputs.get("NAMES") or []
    color_cfg = CONFIG.get("COLORS", {}) or {}

    def _hex_to_bgr_local(s):
        try:
            s = str(s).lstrip('#')
            if len(s) == 6:
                r = int(s[0:2], 16); g = int(s[2:4], 16); b = int(s[4:6], 16)
                return (b, g, r)
        except Exception:
            pass
        return None

    fields = []
    for idx, name in enumerate(names):
        fcfg = inputs.get(name, None)
        if fcfg is None:
            fcfg = inputs.get(f"L{idx+1}", None)
        entry = {"name": name, "type": None, "options": [], "min": None, "max": None, "colors": {}}

        if isinstance(fcfg, (list, tuple)) and len(fcfg) > 0:
            type_token = str(fcfg[0]).strip().upper()
            entry["type"] = type_token
            if type_token == "SELECT":
                opts = [o for o in fcfg[1:]]
                entry["options"] = opts
                # Support COLORS keyed by label name or by Ln ordinal (e.g., L2, L4)
                opt_colors = color_cfg.get(name)
                if not opt_colors:
                    opt_colors = color_cfg.get(f"L{idx+1}")
                if isinstance(opt_colors, (list, tuple)):
                    mapped = {}
                    for i, opt in enumerate(opts):
                        c = opt_colors[i] if i < len(opt_colors) else None
                        bc = _hex_to_bgr_local(c) if c else None
                        if bc:
                            mapped[opt] = bc
                    entry["colors"] = mapped
            elif type_token == "NUMERIC":
                try:
                    if len(fcfg) >= 3:
                        entry["min"] = int(fcfg[1]); entry["max"] = int(fcfg[2])
                    elif len(fcfg) == 2:
                        entry["min"] = 1; entry["max"] = int(fcfg[1])
                    else:
                        entry["min"] = 1; entry["max"] = 120
                except Exception:
                    entry["min"] = 1; entry["max"] = 120
            elif type_token == "DATE":
                months_key = any(str(t).upper() == "MONTH" for t in fcfg[1:])
                years_key  = any(str(t).upper() == "YEAR" for t in fcfg[1:])
                days_key   = any(str(t).upper() == "DAY" for t in fcfg[1:])
                if months_key:
                    entry["months"] = inputs.get("MONTH") or list(range(1, 13))
                if years_key:
                    entry["years"]  = inputs.get("YEAR") or [2024, 2025, 2026]
                if days_key:
                    entry["days"]   = inputs.get("DAY") or list(range(1, 32))
            elif type_token == "TIME":
                hours = inputs.get("HOUR") or list(range(0, 24))
                mins  = inputs.get("MINUTE") or [0, 15, 30, 45]
                secs  = inputs.get("SECOND") or [0, 15, 30, 45]
                opts = [f"{int(h):02d}:{int(m):02d}:{int(s):02d}" for h in hours for m in mins for s in secs]
                entry["options"] = opts
        elif isinstance(fcfg, dict):
            ftype = (fcfg.get("TYPE") or fcfg.get("type") or "").strip().upper()
            entry["type"] = ftype
            if ftype == "SELECT":
                entry["options"] = fcfg.get("OPTIONS") or fcfg.get("VALUES") or []
                entry["colors"] = fcfg.get("COLORS") or {}
            elif ftype == "NUMERIC":
                if fcfg.get("MIN") is not None:
                    entry["min"] = int(fcfg.get("MIN"))
                if fcfg.get("MAX") is not None:
                    entry["max"] = int(fcfg.get("MAX"))
            elif ftype == "DATE":
                entry["months"] = fcfg.get("MONTH_RANGE") or list(range(1, 13))
                entry["years"]  = fcfg.get("YEAR_RANGE") or [2024, 2025, 2026]
                entry["days"]   = fcfg.get("DAY_RANGE") or list(range(1, 32))
        else:
            lname = str(name).lower()
            if "date" in lname:
                entry["type"] = "DATE"
                entry["months"] = inputs.get("MONTH") or list(range(1, 13))
                entry["years"]  = inputs.get("YEAR") or [2024, 2025, 2026]
                entry["days"]   = inputs.get("DAY") or list(range(1, 32))

        fields.append(entry)
    CONFIG["SCHEMA_FIELDS"] = fields

normalize_dynamic_schema()

print("[INFO] Effective CONFIG['RUN']:", CONFIG["RUN"])

def _get_window_layout(name):
    """Return a layout dict for a given selection window name.
    Always returns a dict with at least keys: 'layout_type' and 'cols'.
    Falls back to a safe list layout if the key is missing or misconfigured.
    """
    ui = CONFIG.get("UI", {})
    layouts = ui.get("WINDOW_LAYOUTS") or {}
    layout = layouts.get(name)
    if isinstance(layout, dict) and layout.get("layout_type") in ("grid", "list"):
        # Ensure cols is an int >=1
        cols = layout.get("cols", 1)
        try:
            cols = int(cols)
            if cols < 1:
                cols = 1
        except Exception:
            cols = 1

        # If a layout is configured as 'list' but asks for more than one
        # column, treat it as a grid so WINDOW_LAYOUTS['...']['cols']
        # always drives the actual column count.
        layout_type = layout.get("layout_type")
        if layout_type == "list" and cols > 1:
            layout_type = "grid"

        return {"layout_type": layout_type, "cols": cols}
    # Fallback: safe list layout
    return {"layout_type": "list", "cols": 1}

def _hex_to_bgr(hex_str):
    hex_str = str(hex_str).lstrip('#')
    if len(hex_str) != 6:
        return (128, 128, 128)  # fallback gray if malformed
    r = int(hex_str[0:2], 16)
    g = int(hex_str[2:4], 16)
    b = int(hex_str[4:6], 16)
    return (b, g, r)

# Seaborn "colorblind" palette (10 colors)
_COLORBLIND_HEX = [
    "#0173b2", "#de8f05", "#029e73", "#d55e00", "#cc78bc",
    "#ca9161", "#fbafe4", "#949494", "#ece133", "#56b4e9"
]
_COLORBLIND_BGR = [_hex_to_bgr(h) for h in _COLORBLIND_HEX]

def palette_color(i):
    """Return color i cycling through the seaborn colorblind palette (BGR for OpenCV)."""
    return _COLORBLIND_BGR[i % len(_COLORBLIND_BGR)]

def sync_ui_style_from_config():
    """Sync text styling keys from VISUALIZATION to UI for compatibility.
    Users should modify VISUALIZATION['TEXT STYLING'] only; UI mirrors those values.
    """
    try:
        viz = CONFIG.get("VISUALIZATION", {})
        ui = CONFIG.setdefault("UI", {})
        # Colors
        for k in ("TEXT_PRIMARY", "TEXT_SECONDARY", "TEXT_WHITE"):
            if k in viz:
                ui[k] = viz[k]
        # Thickness
        for k in ("TITLE_THICKNESS", "SUBTITLE_THICKNESS", "BUTTON_THICKNESS", "INSTRUCTION_THICKNESS"):
            if k in viz:
                ui[k] = viz[k]
        # Font sizes
        for k in ("TITLE_FONT_SIZE", "SUBTITLE_FONT_SIZE", "BUTTON_FONT_SIZE", "INSTRUCTION_FONT_SIZE"):
            if k in viz:
                ui[k] = viz[k]
        # Reference instruction scale (for overlays) mirrors VISUALIZATION
        if "REFERENCE_INSTRUCTION_SCALE" in viz:
            ui["REFERENCE_INSTRUCTION_SCALE"] = viz["REFERENCE_INSTRUCTION_SCALE"]
        # Map VISUALIZATION font sizes to UI scales used by header/button renderers
        ui["TITLE_SCALE"] = float(viz.get("TITLE_FONT_SIZE", ui.get("TITLE_FONT_SIZE", 0.85)))
        ui["SUBTITLE_SCALE"] = float(viz.get("SUBTITLE_FONT_SIZE", ui.get("SUBTITLE_FONT_SIZE", 0.7)))
        # Some UI code reads BUTTON_SIZE/INSTRUCTION_SIZE; keep them in sync
        ui["BUTTON_SIZE"] = float(viz.get("BUTTON_FONT_SIZE", ui.get("BUTTON_FONT_SIZE", 0.6)))
        ui["INSTRUCTION_SIZE"] = float(viz.get("INSTRUCTION_FONT_SIZE", ui.get("INSTRUCTION_FONT_SIZE", 0.45)))
        # Font face
        if "FONT_FACE" in viz:
            ui["FONT_FACE"] = viz["FONT_FACE"]
    except Exception:
        pass

# Call sync now that function is defined and CONFIG is loaded
sync_ui_style_from_config()

# ----------------------------------------------------------------------------------------------------------------------------------
# 3) HELPER FUNCTIONS - UI Creation and Text Sizing
# ----------------------------------------------------------------------------------------------------------------------------------
def get_allow_skip(field):
    """Return True if skipping the given field is allowed.

    The config `INPUT.ALLOW_SKIP` is deprecated/ignored — skip buttons are
    always available in the UI. This keeps the skip behavior consistent and
    simplifies configuration.
    """
    return True

def _instruction_style(img):
    """Return instruction text scales for reference/zoom overlays.

    UI panels (Image Label, selection grids, etc.) use `_add_instructions`
    and are controlled by `UI.INSTRUCTION_FONT_SIZE` directly.
    Reference-image overlays (calibration ROI, zoomed card, label/bag
    exclusion) use this helper, which applies an independent scale so
    that their text can be a different size from panel text.
    Returns: (TITLE_FONT_SIZE, title_thick, sub_scale, sub_thick)
    """
    ui = CONFIG.get("UI", {})
    viz = CONFIG.get("VISUALIZATION", {})

    # Base instruction font size/thickness come directly from CONFIG.
    # For reference/zoom overlays, the ONLY size controls are:
    #   - VISUALIZATION.INSTRUCTION_FONT_SIZE
    #   - VISUALIZATION.REFERENCE_INSTRUCTION_SCALE (or UI mirror)
    base_size = float(viz.get("INSTRUCTION_FONT_SIZE", ui.get("INSTRUCTION_FONT_SIZE", 0.45)))
    base_thick = int(viz.get("INSTRUCTION_THICKNESS", ui.get("INSTRUCTION_THICKNESS", 1)))

    # Independent multiplier for reference-image instructions
    ref_scale = float(viz.get("REFERENCE_INSTRUCTION_SCALE",
                              ui.get("REFERENCE_INSTRUCTION_SCALE", 1.0)))
    ref_size = base_size * ref_scale

    # Both title and subtitle use the same CONFIG-driven reference size;
    # no additional hard-coded factors or per-call overrides.
    TITLE_FONT_SIZE = ref_size
    sub_scale = ref_size
    title_thick = max(1, base_thick)
    sub_thick = max(1, base_thick)
    return TITLE_FONT_SIZE, title_thick, sub_scale, sub_thick

def _load_qr_modules(debug: Optional["Debugger"] = None):
    """Load Astrobotany QR modules from QR_Code folder if present."""
    # When frozen, modules are bundled at _MEIPASS root; ensure it's on sys.path
    if is_frozen():
        meipass = getattr(sys, '_MEIPASS', None)
        if meipass and meipass not in sys.path:
            sys.path.insert(0, meipass)
    else:
        try:
            qr_dir = os.path.join(os.path.dirname(__file__), "QR_Code")
        except Exception:
            qr_dir = None
        if qr_dir and os.path.isdir(qr_dir) and qr_dir not in sys.path:
            sys.path.append(qr_dir)

    qr_calib = None
    qr_square = None
    try:
        import astrobotany_calibration_card as qr_calib
    except Exception as e:
        if debug:
            debug.log(f"QR calibration module load failed: {e}", level=2)

    try:
        import astrobotany_airisquare as qr_square
    except Exception as e:
        if debug:
            debug.log(f"QR square module load failed: {e}", level=2)

    return qr_calib, qr_square

def _count_valid_markers(img, debug: Optional["Debugger"] = None):
    """Count valid Astrobotany ArUco markers (IDs 46-49)."""
    valid_ids = {46, 47, 48, 49}
    if not hasattr(cv, "aruco"):
        if debug:
            debug.log("cv2.aruco not available; install opencv-contrib-python", level=1)
        return 0, []
    try:
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        aruco_dict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_100)
        parameters = cv.aruco.DetectorParameters()
        detector = cv.aruco.ArucoDetector(aruco_dict, parameters)
        corners, ids, _ = detector.detectMarkers(gray)
        if ids is None or len(ids) == 0:
            return 0, []
        ids_list = [int(i) for i in ids.flatten() if int(i) in valid_ids]
        return len(ids_list), ids_list
    except Exception as e:
        if debug:
            debug.log(f"Marker counting failed: {e}", level=2)
        return 0, []

def _aruco_bbox_from_image(img, debug: Optional["Debugger"] = None):
    """Compute QR bbox directly from ArUco corners when available."""
    valid_ids = {46, 47, 48, 49}
    if not hasattr(cv, "aruco"):
        return None
    try:
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        aruco_dict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_100)
        parameters = cv.aruco.DetectorParameters()
        detector = cv.aruco.ArucoDetector(aruco_dict, parameters)
        corners, ids, _ = detector.detectMarkers(gray)
        if ids is None or len(ids) == 0:
            return None
        valid_indices = [i for i, idv in enumerate(ids.flatten()) if int(idv) in valid_ids]
        if not valid_indices:
            return None
        all_pts = []
        for i in valid_indices:
            pts = np.array(corners[i]).reshape(-1, 2)
            all_pts.append(pts)
        if not all_pts:
            return None
        all_pts = np.vstack(all_pts)
        x, y, w, h = cv.boundingRect(all_pts.astype(np.float32))
        return (int(x), int(y), int(w), int(h))
    except Exception as e:
        if debug:
            debug.log(f"Aruco bbox detection failed: {e}", level=2)
        return None

def _qr_calibration(img, debug: Optional["Debugger"] = None):
    """Attempt QR-based calibration. Returns (px_per_cm, exclude_mask, qr_info)."""
    qr_info = {
        "QR_detected": False,
        "QR_count": 0,
        "QR_marker_ids": [],
        "bbox": None,
        "marker_corners": None,
    }

    qr_calib, qr_square = _load_qr_modules(debug)
    if not qr_calib or not qr_square:
        return None, None, qr_info

    marker_corners = []
    marker_ids = []
    try:
        marker_corners, marker_ids = qr_square.get_validate_square_ids(img)
    except Exception as e:
        if debug:
            debug.log(f"QR marker detection failed: {e}", level=2)

    # Use batch_astrobotany-style marker counting
    marker_count, counted_ids = _count_valid_markers(img, debug)

    if marker_ids:
        try:
            marker_ids = [int(i) for i in marker_ids]
        except Exception:
            marker_ids = list(marker_ids)

    qr_info["QR_marker_ids"] = counted_ids or marker_ids or []
    qr_info["QR_count"] = int(marker_count)

    bbox = None
    if marker_corners:
        try:
            all_pts = []
            for marker in marker_corners:
                pts = np.array(marker).reshape(-1, 2)
                all_pts.append(pts)
            if all_pts:
                all_pts = np.vstack(all_pts)
                x, y, w, h = cv.boundingRect(all_pts.astype(np.float32))
                bbox = (int(x), int(y), int(w), int(h))
        except Exception:
            bbox = None
    if bbox is None:
        bbox = _aruco_bbox_from_image(img, debug=debug)

    qr_info["bbox"] = bbox
    qr_info["marker_corners"] = marker_corners

    px_per_cm = None
    try:
        _, px_per_cm, marker_detected = qr_calib.process_image(img, calculate_black_area=False)
    except Exception as e:
        if debug:
            debug.log(f"QR calibration processing failed: {e}", level=2)
        marker_detected = False

    qr_detected = (qr_info["QR_count"] > 1) and (px_per_cm is not None) and bool(marker_detected)
    qr_info["QR_detected"] = qr_detected

    if debug and not qr_detected:
        debug.log(
            f"QR calibration not detected: markers={qr_info['QR_count']}, px/cm={px_per_cm}",
            level=2,
        )

    if not qr_detected or px_per_cm is None:
        exclude_mask = _build_qr_exclude_mask(img, bbox, px_per_cm)
        return None, exclude_mask, qr_info

    # Build calibration exclusion mask around the QR square
    exclude_mask = np.zeros(img.shape[:2], np.uint8)
    if bbox is not None:
        x, y, w, h = bbox
        pad_cm = float(CONFIG.get("CALIB", {}).get("EXCLUDE_PAD_CM", 0.0))
        pad_px = int(round(pad_cm * float(px_per_cm)))
        x0 = max(0, x - pad_px)
        y0 = max(0, y - pad_px)
        x1 = min(img.shape[1], x + w + pad_px)
        y1 = min(img.shape[0], y + h + pad_px)
        cv.rectangle(exclude_mask, (x0, y0), (x1, y1), 255, -1)

    return float(px_per_cm), exclude_mask, qr_info

def _build_qr_exclude_mask(img, bbox, px_per_cm=None):
    """Build an exclusion mask from a QR bbox, with optional padding in cm."""
    if bbox is None:
        return None
    x, y, w, h = bbox
    try:
        pad_cm = float(CONFIG.get("CALIB", {}).get("EXCLUDE_PAD_CM", 0.0))
    except Exception:
        pad_cm = 0.0
    if px_per_cm is not None:
        try:
            pad_px = int(round(pad_cm * float(px_per_cm)))
        except Exception:
            pad_px = 0
    else:
        pad_px = 0
    x0 = max(0, x - pad_px)
    y0 = max(0, y - pad_px)
    x1 = min(img.shape[1], x + w + pad_px)
    y1 = min(img.shape[0], y + h + pad_px)
    mask = np.zeros(img.shape[:2], np.uint8)
    cv.rectangle(mask, (x0, y0), (x1, y1), 255, -1)
    return mask

def show_startup_window(message="Starting GatorLeaf..."):
    img = np.ones((200, 600, 3), dtype=np.uint8) * 255
    cv.putText(img, message, (30, 100), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,0), 2, cv.LINE_AA)
    cv.imshow("GatorLeaf", img)
    cv.waitKey(1000)  # Show splash for 1 second or until user closes it
    cv.destroyAllWindows()

def create_window_pair(img, panel_title):
    """Create and position reference image + panel window pair using CONFIG."""
    image_window = f"Reference Image - {panel_title}"
    panel_window = f"{panel_title} Window"
    
    # Display reference image
    display_img = _fit_for_reference(img)
    cv.namedWindow(image_window, cv.WINDOW_NORMAL)
    cv.imshow(image_window, display_img)
    cv.resizeWindow(image_window, display_img.shape[1], display_img.shape[0])
    
    # Position windows using CONFIG (clamped to screen)
    ref_pos_cfg = CONFIG["UI"]["REFERENCE_WINDOW_POSITION"]
    ref_w, ref_h = display_img.shape[1], display_img.shape[0]
    try:
        ref_x, ref_y = _clamp_position(ref_pos_cfg[0], ref_pos_cfg[1], ref_w, ref_h)
    except Exception:
        ref_x, ref_y = ref_pos_cfg
    panel_pos = (ref_x + ref_w + CONFIG["UI"]["PANEL_OFFSET_X"], ref_y)

    cv.moveWindow(image_window, ref_x, ref_y)
    cv.waitKey(1)  # Allow window manager to process
    
    return image_window, panel_window, display_img, panel_pos

def _get_screen_size():
    """Return (screen_width, screen_height) with safe cross-platform defaults.
    On macOS, attempt Quartz/CoreGraphics; on Linux, try xrandr; else fallback.
    """
    # macOS: Quartz/CoreGraphics
    try:
        if sys.platform == 'darwin':
            try:
                import Quartz
                display_id = Quartz.CGMainDisplayID()
                w = Quartz.CGDisplayPixelsWide(display_id)
                h = Quartz.CGDisplayPixelsHigh(display_id)
                if int(w) > 0 and int(h) > 0:
                    return int(w), int(h)
            except Exception:
                pass
    except Exception:
        pass

    # Linux/X11: xrandr
    try:
        if sys.platform.startswith('linux'):
            out = subprocess.check_output(['xrandr', '--current'], stderr=subprocess.DEVNULL).decode(errors='ignore')
            m = re.search(r'current\s+(\d+)\s*x\s*(\d+)', out)
            if m:
                return int(m.group(1)), int(m.group(2))
            m2 = re.search(r"(\d+)x(\d+)\s+\d+\.\d+\*\+", out)
            if m2:
                return int(m2.group(1)), int(m2.group(2))
    except Exception:
        pass

    # Fallback conservative default
    return 1920, 1080

def _clamp_y_only(y, panel_h):
    """Clamp Y so the panel stays fully on screen; X is left unchanged elsewhere."""
    _sw, sh = _get_screen_size()
    return max(0, min(int(y), max(0, sh - int(panel_h))))

def _clamp_position(x, y, w, h):
    """Clamp window top-left (x,y) so the window of size (w,h) fits on screen."""
    sw, sh = _get_screen_size()
    max_x = max(0, sw - int(w))
    max_y = max(0, sh - int(h))
    return max(0, min(int(x), max_x)), max(0, min(int(y), max_y))


def _get_hidpi_scale():
    """Internal high-DPI rendering scale for crisp text.
    
    This is intentionally not user-configurable; window sizing now adapts
    to the current screen dimensions instead of a global UI.SCALE_FACTOR.
    Return as integer to avoid float shape errors in NumPy.
    """
    return 2

def _get_reference_max_size():
    ui = CONFIG.get("UI", {})
    max_width, max_height = ui.get("REFERENCE_IMAGE_SIZE", (1200, 900))
    try:
        sw, sh = _get_screen_size()
        print(f"DEBUG: Detected screen size: {sw}×{sh}")  # Add this
        print(f"DEBUG: Requested size: {max_width}×{max_height}")  # Add this
        max_width = min(int(max_width), max(200, int(sw * 0.95)))  # Your change
        max_height = min(int(max_height), max(150, int(sh * 0.95)))  # Your change
        print(f"DEBUG: Final constrained size: {max_width}×{max_height}")  # Add this
    except Exception as e:
        print(f"DEBUG: Screen size detection failed: {e}")  # Add this
        pass
    return max_width, max_height
"""
def _get_reference_max_size():
    """ """Return (max_width, max_height) for reference-style image windows.

    All reference images (Image Label, calibration preview, mask review,
    zoomed calibration) should use the same effective maximum size derived
    from UI.REFERENCE_IMAGE_SIZE and clamped to a safe fraction of the
    current screen. """
"""
    ui = CONFIG.get("UI", {})
    max_width, max_height = ui.get("REFERENCE_IMAGE_SIZE", (2400, 1800))
    try:
        sw, sh = _get_screen_size()
        max_width = min(int(max_width), max(200, int(sw * 0.95)))
        max_height = min(int(max_height), max(150, int(sh * 0.95)))
    except Exception:
        pass
    return max_width, max_height
"""
def _fit_for_reference(img, max_width=None, max_height=None):
    """Fit image for reference display using CONFIG settings."""
    # Use CONFIG settings if no parameters provided
    if max_width is None or max_height is None:
        max_width, max_height = _get_reference_max_size()
    else:
        # When explicit sizes are provided, still respect screen bounds.
        try:
            sw, sh = _get_screen_size()
            max_width = min(int(max_width), max(200, int(sw * 0.95)))
            max_height = min(int(max_height), max(150, int(sh * 0.95)))
        except Exception:
            pass
    
    h, w = img.shape[:2]
    scale = min(max_width / w, max_height / h)
    
    if scale < 1.0:
        new_w = int(w * scale)
        new_h = int(h * scale)
        return cv.resize(img, (new_w, new_h), interpolation=cv.INTER_AREA)
    return img.copy()

def _get_hidpi_scale():
    """Internal high-DPI rendering scale for crisp text.
    
    Not user-configurable; window sizing adapts to screen rather than a global scale.
    Return as integer to avoid float shape errors in NumPy.
    """
    return 2

def _get_font_face():
    viz = CONFIG.get("VISUALIZATION", {})
    font_face_str = str(viz.get("FONT_FACE", "HERSHEY_COMPLEX")).upper()
    return _FONT_MAP.get(font_face_str, cv.FONT_HERSHEY_COMPLEX)

def _fit_panel_to_screen(width, height, margin_ratio=0.9):
    """Shrink width/height uniformly if larger than a fraction of screen size."""
    try:
        sw, sh = _get_screen_size()
        max_w = int(sw * float(margin_ratio))
        max_h = int(sh * float(margin_ratio))
        shrink = min(max_w / float(max(1, width)), max_h / float(max(1, height)), 1.0)
        if shrink < 1.0:
            return max(200, int(width * shrink)), max(150, int(height * shrink))
    except Exception:
        pass
    return int(width), int(height)
def calculate_text_size(text, font_size, scale_factor=1):
    """Calculate text size with current font settings."""
    viz = CONFIG.get("VISUALIZATION", {})
    # font_size here is a base scale (e.g., TITLE_SIZE); when callers pass a literal, use as-is
    scaled_font_size = float(font_size) * float(scale_factor)
    # Determine thickness by context token (callers pass one of the configured sizes when computing)
    # Provide fallbacks if direct thickness not provided
    default_thick = 2
    scaled_thickness = max(1, int(default_thick * scale_factor))
    font_face = _get_font_face()
    text_size, baseline = cv.getTextSize(text, font_face, scaled_font_size, scaled_thickness)
    return text_size[0], text_size[1] + baseline

def calculate_dynamic_panel_size(title, subtitle, items, extra_content_height=0, scale_factor=3):
    """Calculate panel dimensions dynamically based on actual content needs."""
    # Calculate text heights at actual scale (the bold text takes more vertical space)
    viz = CONFIG.get("VISUALIZATION", {})
    TITLE_FONT_SIZE = float(viz.get("TITLE_FONT_SIZE", 0.85))
    subTITLE_FONT_SIZE = float(viz.get("SUBTITLE_FONT_SIZE", 0.7))
    BUTTON_FONT_SIZE = float(viz.get("BUTTON_FONT_SIZE", 0.6))
    ui = CONFIG.get("UI", {})
    margin = int(ui.get("MARGIN", 18))
    title_spacing = int(ui.get("TITLE_SPACING", 30))

    title_width, title_height = calculate_text_size(title, TITLE_FONT_SIZE, 1)
    subtitle_height = 0
    if subtitle:
        subtitle_width, subtitle_height = calculate_text_size(subtitle, subTITLE_FONT_SIZE, 1)
    
    # Calculate button text widths to determine panel width
    max_content_width = max(title_width, subtitle_width if subtitle else 0)
    if items:
        for item in items:
            item_width, _ = calculate_text_size(str(item), BUTTON_FONT_SIZE, 1)
            max_content_width = max(max_content_width, item_width)
    
    # Window width: content + margins + padding
    panel_width = max_content_width + (margin * 4)  # Extra margin for button padding
    panel_width = max(ui.get("PANEL_MIN_WIDTH", 400), min(panel_width, ui.get("PANEL_MAX_WIDTH", 600)))
    
    # Calculate height dynamically
    # Header space (title + subtitle with proper spacing)
    header_height = margin * 2  # Top margin
    header_height += max(35, title_height + 10)  # Title with padding
    if subtitle:
        header_height += title_spacing + subtitle_height
    
    # Items space (buttons)
    items_height = 0
    button_height = int(ui.get("BUTTON_HEIGHT", 30))
    if items:
        btn_gap = int(ui.get("BUTTON_SPACING", 10))
        items_height = len(items) * button_height
        items_height += (len(items) - 1) * btn_gap  # Spacing between buttons
    
    # Extra content space
    extra_height = extra_content_height
    
    # Instructions space (shared rule for list-style panels):
    # bottom margin plus a fixed button-to-instruction gap from config.
    bottom_margin = int(ui.get("INSTRUCTION_BOTTOM_MARGIN", margin))
    gap_unscaled = _get_button_instruction_gap_unscaled()
    bottom_safety = int(ui.get("BUTTON_BOTTOM_SAFETY", 4))
    instruction_height = bottom_margin + gap_unscaled + bottom_safety
    
    # Total height
    panel_height = header_height + items_height + extra_height + instruction_height
    
    return int(panel_width), int(panel_height)

def calculate_grid_panel_dimensions(title, subtitle, items, cols, scale_factor=3):
    """Calculate dimensions for grid-based panels (Month/Day/Time) with proper bottom spacing."""
    # Header text sizing (unscaled coordinates)
    viz = CONFIG.get("VISUALIZATION", {})
    ui = CONFIG.get("UI", {})
    TITLE_FONT_SIZE = float(viz.get("TITLE_FONT_SIZE", 0.85))
    subTITLE_FONT_SIZE = float(viz.get("SUBTITLE_FONT_SIZE", 0.7))
    margin = int(ui.get("MARGIN", 18))
    TITLE_FONT_SIZE = float(ui.get("TITLE_FONT_SIZE", 0.85))
    subTITLE_FONT_SIZE = float(ui.get("SUBTITLE_FONT_SIZE", 0.7))
    title_width, title_height = calculate_text_size(title, TITLE_FONT_SIZE, 1)
    subtitle_width = 0
    if subtitle:
        subtitle_width, _ = calculate_text_size(subtitle, subTITLE_FONT_SIZE, 1)

    # Grid geometry
    rows = (len(items) + max(cols, 1) - 1) // max(cols, 1)
    button_width = int(CONFIG["UI"].get("MIN_BUTTON_WIDTH", 60))
    button_height = int(CONFIG["UI"].get("GRID_BUTTON_HEIGHT", 30))
    grid_gap = int(CONFIG["UI"].get("GRID_SPACING", 10))
    grid_width = max(cols, 1) * button_width + (max(cols, 1) - 1) * grid_gap
    content_width = max(title_width, subtitle_width, grid_width)
    max_width = int(CONFIG["UI"].get("MAX_GRID_PANEL_WIDTH", 450))
    panel_width = max(int(CONFIG["UI"].get("PANEL_MIN_WIDTH", 400)),
                      min(int(content_width + margin * 2), max_width))

    # Header height to match draw_hires_panel_header behavior
    header_height = margin * 2 + max(35, title_height + 10)
    if subtitle:
        header_height += int(ui.get("TITLE_SPACING", 30))

    grid_height = rows * button_height + (rows - 1) * grid_gap if rows > 0 else 0

    # Reserve bottom margin plus the same fixed button-to-instruction gap used
    # for list panels so DATE grids follow the exact same rule.
    instruction_bottom_margin = int(ui.get("INSTRUCTION_BOTTOM_MARGIN", margin))
    gap_unscaled = _get_button_instruction_gap_unscaled()
    bottom_safety = int(ui.get("BUTTON_BOTTOM_SAFETY", 4))
    panel_height = header_height + grid_height + instruction_bottom_margin + gap_unscaled + bottom_safety
    return int(panel_width), int(panel_height)

def create_high_dpi_panel(width, height, scale_factor=2):
    """Create a high-DPI panel that's rendered at higher resolution then scaled down."""
    hires_width = int(width) * int(scale_factor)
    hires_height = int(height) * int(scale_factor)
    panel_bg = tuple(CONFIG.get("UI", {}).get("PANEL_BG_COLOR", (240, 240, 240)))
    hires_panel = np.full((hires_height, hires_width, 3), panel_bg, dtype=np.uint8)
    return hires_panel, scale_factor

def show_hires_panel(window_name, hires_panel, target_width, target_height):
    """Display high-DPI panel scaled down for crisp rendering."""
    # Scale down using INTER_AREA for best quality (ensure integer sizes)
    tw = int(max(1, target_width))
    th = int(max(1, target_height))
    display_panel = cv.resize(hires_panel, (tw, th), interpolation=cv.INTER_AREA)
    cv.namedWindow(window_name, cv.WINDOW_NORMAL)
    cv.imshow(window_name, display_panel)
    cv.resizeWindow(window_name, tw, th)
    return display_panel

def draw_hires_panel_header(img, title, subtitle, scale_factor, y_pos=None):
    """Draw panel header with BOLD title (supports multi-line with \\n) and gray subtitle."""
    ui = CONFIG.get("UI", {})
    if y_pos is None:
        y_pos = int(ui.get("MARGIN", 18))
        
    # Scale positions and sizes
    margin = int(ui.get("MARGIN", 18))
    scaled_margin = int(margin * scale_factor)
    title_scale = float(ui.get("TITLE_SCALE", 0.85))
    subtitle_scale = float(ui.get("SUBTITLE_SCALE", 0.7))
    scaled_title_size = title_scale * scale_factor
    scaled_subtitle_size = subtitle_scale * scale_factor
    scaled_y_pos = int(y_pos * scale_factor)
    scaled_spacing = int(int(ui.get("TITLE_SPACING", 30)) * scale_factor)
    
    # BOLD title with increased thickness
    scaled_title_thickness = max(1, int(int(ui.get("TITLE_THICKNESS", 3)) * scale_factor))
    font_face = _get_font_face()
    title_y = scaled_y_pos + int(35 * scale_factor)
    
    # Support multi-line titles
    title_lines = title.split('\n')
    line_spacing = int(30 * scale_factor)
    for i, line in enumerate(title_lines):
        text_primary = tuple(ui.get("TEXT_PRIMARY", (51, 51, 51)))
        cv.putText(img, line, (scaled_margin, title_y + i * line_spacing), font_face,
               scaled_title_size, text_primary, scaled_title_thickness, cv.LINE_AA)
    
    current_y = title_y + (len(title_lines) - 1) * line_spacing
    
    # Gray subtitle with normal thickness (not bold)
    if subtitle:
        current_y += scaled_spacing
        scaled_subtitle_thickness = max(1, int(int(ui.get("SUBTITLE_THICKNESS", 2)) * scale_factor))
        text_secondary = tuple(ui.get("TEXT_SECONDARY", (119, 119, 119)))
        cv.putText(img, subtitle, (scaled_margin, current_y), font_face,
               scaled_subtitle_size, text_secondary, scaled_subtitle_thickness, cv.LINE_AA)
    
    return (current_y + scaled_spacing) // scale_factor  # Return unscaled coordinate

def draw_hires_styled_button(img, text, rect, color, scale_factor, text_color=None, font_size=None):
    """Draw a styled button on high-DPI canvas with configurable font and proper thickness."""
    x1, y1, x2, y2 = rect
    # Scale all coordinates
    x1, y1, x2, y2 = [int(coord * scale_factor) for coord in [x1, y1, x2, y2]]
    
    ui = CONFIG.get("UI", {})
    if text_color is None:
        text_color = tuple(ui.get("TEXT_WHITE", (255, 255, 255)))
    if font_size is None:
        font_size = float(ui.get("BUTTON_FONT_SIZE", ui.get("BUTTON_SIZE", 0.6)))
    
    # Scale font size and use proper button thickness
    scaled_font_size = float(font_size) * float(scale_factor)
    scaled_thickness = max(1, int(int(ui.get("BUTTON_THICKNESS", 2)) * scale_factor))
    shadow_offset = int(2 * scale_factor)
    border_thickness = max(1, int(2 * scale_factor))
    
    # Draw shadow
    shadow_rect = (x1 + shadow_offset, y1 + shadow_offset, x2 + shadow_offset, y2 + shadow_offset)
    cv.rectangle(img, shadow_rect[:2], shadow_rect[2:], (200, 200, 200), -1)
    
    # Draw main button
    cv.rectangle(img, (x1, y1), (x2, y2), color, -1)
    
    # Add border
    cv.rectangle(img, (x1, y1), (x2, y2), (180, 180, 180), border_thickness)
    
    # Center text using configurable font and proper thickness
    font_face = _get_font_face()
    text_size = cv.getTextSize(text, font_face, scaled_font_size, scaled_thickness)[0]
    text_y = y1 + (y2 - y1 + text_size[1]) // 2
    text_x = x1 + (x2 - x1 - text_size[0]) // 2
    cv.putText(img, text, (text_x, text_y), font_face, scaled_font_size,
               text_color, scaled_thickness, cv.LINE_AA)
    
def _add_instructions(hires_panel, instruction_text, y_pos, scale_factor):
    """Add instruction text at specified position."""
    ui = CONFIG.get("UI", {})
    margin = int(ui.get("MARGIN", 18))
    scaled_margin = int(margin * scale_factor)
    scaled_INSTRUCTION_FONT_SIZE = float(ui.get("INSTRUCTION_FONT_SIZE", 0.45)) * float(scale_factor)
    scaled_instruction_thickness = max(1, int(int(ui.get("INSTRUCTION_THICKNESS", 1)) * scale_factor))
    text_secondary = tuple(ui.get("TEXT_SECONDARY", (119, 119, 119)))
    font_face = _get_font_face()
    cv.putText(hires_panel, instruction_text, (scaled_margin, y_pos), font_face,
               scaled_INSTRUCTION_FONT_SIZE, text_secondary, scaled_instruction_thickness, cv.LINE_AA)

def _instruction_text_block_height(scale_factor):
    """Compute unscaled instruction text block height to reserve above bottom margin.

    By default, we reserve at least the shared button-to-instruction gap so
    that panels which compute button stacks top-down (like the main Image
    Label window) still leave enough room between the last button and the
    instructional text. If INSTRUCTION_TEXT_BLOCK_HEIGHT is set, it overrides
    this default.
    """
    try:
        ui = CONFIG.get("UI", {})
        if "INSTRUCTION_TEXT_BLOCK_HEIGHT" in ui:
            fixed = int(ui["INSTRUCTION_TEXT_BLOCK_HEIGHT"])
            return max(0, fixed)
        # Fallback: use the same gap as other panels
        return _get_button_instruction_gap_unscaled()
    except Exception:
        return _get_button_instruction_gap_unscaled()

def _get_button_instruction_gap_unscaled():
    """Return the unscaled gap between the bottom button and instructions.

    This is shared by all panel styles (SELECT, NUMERIC, DATE grid) so that the
    distance from the last button to the instruction text is consistent.
    Controlled primarily by UI.BUTTON_INSTRUCTION_GAP, falling back to the
    older LIST_BOTTOM_GAP / GRID_BOTTOM_GAP settings if not present.
    """
    ui = CONFIG.get("UI", {})
    try:
        if "BUTTON_INSTRUCTION_GAP" in ui:
            return max(0, int(ui["BUTTON_INSTRUCTION_GAP"]))
        if "LIST_BOTTOM_GAP" in ui:
            return max(0, int(ui["LIST_BOTTOM_GAP"]))
        if "GRID_BOTTOM_GAP" in ui:
            return max(0, int(ui["GRID_BOTTOM_GAP"]))
        return 16
    except Exception:
        return 16

def format_date_for_display(date_str):
    """Convert stored or raw date strings to MM/DD/YY for display.

    Accepts inputs like 'YYYY_MM_DD', 'YY-MM-DD', 'MM.DD.YY', 'MM/DD/YYYY', etc.
    Falls back gracefully for partial dates (YYYY_MM -> MM/YY, MM_DD -> MM/DD).
    """
    if not date_str:
        return "Not Set"

    s = str(date_str).strip()
    # Try full normalization first
    canonical = normalize_date_string(s)
    if isinstance(canonical, str):
        try:
            y, m, d = canonical.split('_')
            return f"{int(m):02d}/{int(d):02d}/{str(y)[-2:]}"
        except Exception:
            pass

    # Partial or non-normalized: split by common separators
    tokens = re.split(r"[\-_.\/]", s)
    tokens = [t for t in tokens if t]
    try:
        if len(tokens) == 3:
            # Heuristic: if first looks like year (4 digits), treat as YYYY_MM_DD
            a, b, c = tokens
            if len(a) == 4:
                return f"{int(b):02d}/{int(c):02d}/{str(a)[-2:]}"
            # Else assume MM_DD_YY
            return f"{int(a):02d}/{int(b):02d}/{str(c)[-2:]}"
        if len(tokens) == 2:
            a, b = tokens
            if len(a) == 4:
                # YYYY_MM -> MM/YY
                return f"{int(b):02d}/{str(a)[-2:]}"
            return f"{int(a):02d}/{int(b):02d}"
        if len(tokens) == 1:
            return f"{int(tokens[0]):02d}"
    except Exception:
        pass
    return s

def draw_selection_rectangle(img, start_point, end_point, color=(255, 255, 0), thickness=2):
    """Draw a selection rectangle with custom color (cyan by default)"""
    cv.rectangle(img, start_point, end_point, color, thickness)


def custom_roi_selection(window_name, img, instruction_text="Click & drag box around calibration card or ruler, then press ENTER"):
    """ROI selection for calibration:
    - Click & drag to draw box
    - U: undo and restart selection
    - ENTER: accept selection (works even if cursor went off-screen)
    - ESC/Q: quit the app
    Instructions remain visible throughout, with dynamically sized text."""
    
    # Dynamic instruction styling based on current image size
    t_scale, t_thick, s_scale, s_thick = _instruction_style(img)
    
    # Baseline image with instructions (persist during interaction)
    img_base_with_text = img.copy()
    # Use CONFIG-driven font face
    ui = CONFIG.get("UI", {})
    _font_face_cfg = _get_font_face()
    # Add black background rectangles for better readability
    (tw1, th1), _ = cv.getTextSize(instruction_text, _font_face_cfg, t_scale, t_thick)
    cv.rectangle(img_base_with_text, (8, 30 - th1 - 2), (12 + tw1, 32), (0, 0, 0), -1)
    cv.putText(img_base_with_text, instruction_text,
               (10, 30), _font_face_cfg, t_scale, (255, 255, 255), t_thick, cv.LINE_AA)
    sub_text = " U=Undo, ENTER=Accept, ESC/Q=Quit"
    (tw2, th2), _ = cv.getTextSize(sub_text, _font_face_cfg, s_scale, s_thick)
    cv.rectangle(img_base_with_text, (8, 55 - th2 - 2), (12 + tw2, 57), (0, 0, 0), -1)
    cv.putText(img_base_with_text, sub_text,
               (10, 55), _font_face_cfg, s_scale, (255, 255, 255), s_thick, cv.LINE_AA)
    cv.imshow(window_name, img_base_with_text)
    
    drawing = False
    start_point = None
    current_rect = None
    last_mouse_pos = None  # Track last known mouse position
    
    def mouse_callback(event, x, y, flags, param):
        nonlocal drawing, start_point, current_rect, last_mouse_pos
        
        # Always update last known position when mouse is in window
        last_mouse_pos = (x, y)
        
        if event == cv.EVENT_LBUTTONDOWN:
            drawing = True
            start_point = (x, y)
            current_rect = None
            
        elif event == cv.EVENT_MOUSEMOVE and drawing:
            img_copy = img_base_with_text.copy()
            if start_point:
                draw_selection_rectangle(img_copy, start_point, (x, y), (255, 255, 0), 2)
                cv.imshow(window_name, img_copy)
                
        elif event == cv.EVENT_LBUTTONUP:
            if drawing and start_point:
                drawing = False
                end_point = (x, y)
                x1, y1 = start_point
                x2, y2 = end_point
                roi_x = min(x1, x2)
                roi_y = min(y1, y2)
                roi_w = abs(x2 - x1)
                roi_h = abs(y2 - y1)
                if roi_w > 5 and roi_h > 5:
                    current_rect = (roi_x, roi_y, roi_w, roi_h)
                    img_copy = img_base_with_text.copy()
                    draw_selection_rectangle(img_copy, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (255, 255, 0), 3)
                    cv.imshow(window_name, img_copy)
    
    cv.setMouseCallback(window_name, mouse_callback)
    
    while True:
        key = cv.waitKey(30)
        
        if key in [10, 13]:  # ENTER
            # If currently drawing (cursor went off-screen), complete with last known position
            if drawing and start_point and last_mouse_pos:
                drawing = False
                x1, y1 = start_point
                x2, y2 = last_mouse_pos
                roi_x = min(x1, x2)
                roi_y = min(y1, y2)
                roi_w = abs(x2 - x1)
                roi_h = abs(y2 - y1)
                if roi_w > 5 and roi_h > 5:
                    current_rect = (roi_x, roi_y, roi_w, roi_h)
            
            # Accept current selection if valid
            if current_rect and current_rect[2] > 0 and current_rect[3] > 0:
                return current_rect
                
        elif key in [ord('u'), ord('U')]:  # Undo
            current_rect = None
            drawing = False
            start_point = None
            last_mouse_pos = None
            cv.imshow(window_name, img_base_with_text)
            
        elif key in [ord('q'), 27]:  # q or ESC -> quit app
            print("User requested exit by q or ESC.")
            cv.destroyAllWindows()
            sys.exit(0)
            return (0, 0, 0, 0)
        
        
def custom_multi_roi_selection(window_name, img, instruction_text="Select areas, press SPACE after each, ENTER when done"):
    """Multi-ROI selection for label/bag exclusion:
    - SPACE: add current selection (even if cursor is off-screen)
    - U: undo previous selection
    - ENTER: accept all selections
    - ESC/Q: quit the app
    Instructions remain visible throughout, with dynamically sized text."""
    
    # Dynamic instruction styling
    t_scale, t_thick, s_scale, s_thick = _instruction_style(img)
    img_base = img.copy()
    img_base_with_text = img_base.copy()
    ui = CONFIG.get("UI", {})
    _font_face_cfg = _get_font_face()
    (tw1, th1), _ = cv.getTextSize(instruction_text, _font_face_cfg, t_scale, t_thick)
    cv.rectangle(img_base_with_text, (8, 30 - th1 - 2), (12 + tw1, 32), (0, 0, 0), -1)
    cv.putText(img_base_with_text, instruction_text,
               (10, 30), _font_face_cfg, t_scale, (255, 255, 255), t_thick, cv.LINE_AA)
    sub_text = "SPACE=Select,  U=Undo, ENTER=Accept, ESC/Q=Quit"
    (tw2, th2), _ = cv.getTextSize(sub_text, _font_face_cfg, s_scale, s_thick)
    cv.rectangle(img_base_with_text, (8, 55 - th2 - 2), (12 + tw2, 57), (0, 0, 0), -1)
    cv.putText(img_base_with_text, sub_text,
               (10, 55), _font_face_cfg, s_scale, (255, 255, 255), s_thick, cv.LINE_AA)
    cv.imshow(window_name, img_base_with_text)
    
    drawing = False
    start_point = None
    current_rect = None
    last_mouse_pos = None  # Track last known mouse position
    selected_rects = []
    
    def redraw_all_selections():
        img_copy = img_base_with_text.copy()
        for i, rect in enumerate(selected_rects):
            rx, ry, rw, rh = rect
            draw_selection_rectangle(img_copy, (rx, ry), (rx + rw, ry + rh), (0, 255, 0), 2)
            cv.putText(img_copy, f"{i+1}", (rx + 5, ry + 20), _font_face_cfg, 0.6, (0, 255, 0), 2, cv.LINE_AA)
        
        # Draw current in-progress selection
        if drawing and start_point and last_mouse_pos:
            draw_selection_rectangle(img_copy, start_point, last_mouse_pos, (255, 255, 0), 2)
        elif current_rect:
            rx, ry, rw, rh = current_rect
            draw_selection_rectangle(img_copy, (rx, ry), (rx + rw, ry + rh), (0, 255, 255), 3)
        
        status_text = f"Selected: {len(selected_rects)} areas"
        if drawing:
            status_text += " | Drawing... (SPACE to finish)"
        cv.putText(img_copy, status_text, (10, img_copy.shape[0] - 10), _font_face_cfg, s_scale, (255, 255, 255), s_thick, cv.LINE_AA)
        return img_copy
    
    def mouse_callback(event, x, y, flags, param):
        nonlocal drawing, start_point, current_rect, last_mouse_pos
        
        # Always update last known position
        last_mouse_pos = (x, y)
        
        if event == cv.EVENT_LBUTTONDOWN:
            drawing = True
            start_point = (x, y)
            current_rect = None
            redraw = redraw_all_selections()
            cv.imshow(window_name, redraw)
            
        elif event == cv.EVENT_MOUSEMOVE:
            if drawing and start_point:
                redraw = redraw_all_selections()
                cv.imshow(window_name, redraw)
                
        elif event == cv.EVENT_LBUTTONUP:
            if drawing and start_point:
                drawing = False
                end_point = (x, y)
                x1, y1 = start_point
                x2, y2 = end_point
                roi_x = min(x1, x2)
                roi_y = min(y1, y2)
                roi_w = abs(x2 - x1)
                roi_h = abs(y2 - y1)
                if roi_w > 5 and roi_h > 5:
                    current_rect = (roi_x, roi_y, roi_w, roi_h)
                else:
                    current_rect = None
                    start_point = None
                redraw = redraw_all_selections()
                cv.imshow(window_name, redraw)
    
    cv.setMouseCallback(window_name, mouse_callback)
    
    while True:
        key = cv.waitKey(30)
        
        if key == ord(' '):  # SPACE - finalize current selection
            # If currently drawing, finish the selection with last known position
            if drawing and start_point and last_mouse_pos:
                x1, y1 = start_point
                x2, y2 = last_mouse_pos
                roi_x = min(x1, x2)
                roi_y = min(y1, y2)
                roi_w = abs(x2 - x1)
                roi_h = abs(y2 - y1)
                if roi_w > 5 and roi_h > 5:
                    current_rect = (roi_x, roi_y, roi_w, roi_h)
                    drawing = False
                    start_point = None
            
            # Add current_rect to selected list
            if current_rect and current_rect[2] > 0 and current_rect[3] > 0:
                selected_rects.append(current_rect)
                current_rect = None
                img_copy = redraw_all_selections()
                cv.imshow(window_name, img_copy)
                
        elif key in [10, 13]:  # ENTER - finish
            # Auto-add any current selection before finishing
            if current_rect and current_rect[2] > 0 and current_rect[3] > 0:
                selected_rects.append(current_rect)
            return selected_rects
            
        elif key in [ord('u'), ord('U')]:  # Undo last
            if selected_rects:
                selected_rects.pop()
                img_copy = redraw_all_selections()
                cv.imshow(window_name, img_copy)
                
        elif key in [ord('q'), 27]:  # q or ESC -> quit app
            print("User requested exit by q or ESC.")
            cv.destroyAllWindows()
            sys.exit(0)
            return []
        

# ----------------------------------------------------------------------------------------------------------------------------------
# 4) UNIFIED SELECTION INTERFACES
# ----------------------------------------------------------------------------------------------------------------------------------
def create_selection_interface(img, title, subtitle, items, layout_config,
                               format_func=None, add_skip=False,
                               suppress_reference=False, panel_position=None,
                               panel_width_override=None):
    """Unified interface for selections with optional Skip.
    New:
      - suppress_reference: show panel only (no 'Reference Image' window).
      - panel_position: (x, y) position for the panel window.
      - panel_width_override: minimum panel width (still dynamic).
    """
    layout_type = layout_config["layout_type"]
    cols = layout_config.get("cols", 1)
    ui = CONFIG.get("UI", {})
    # Internal high-DPI scale; window size now adapts to screen, not CONFIG['UI']['SCALE_FACTOR']
    scale_factor = _get_hidpi_scale()

    window_colors_map = {
        "Select Day":   palette_color(0),
        "Select Month": palette_color(1),
        "Select Year":  palette_color(4),
    }

    window_color = palette_color(7)
    for key, col in window_colors_map.items():
        if key in title:
            window_color = col
            break

    SKIP_SENTINEL = "__SKIP__"
    local_items = list(items)
    if add_skip:
        # Always append a simple sentinel; layout renderers handle styling
        local_items.append(SKIP_SENTINEL)

    def display_formatter(x):
        if x == SKIP_SENTINEL or (isinstance(x, tuple) and x and x[0] == SKIP_SENTINEL):
            return "Skip"
        return format_func(x) if format_func else (str(x[0]) if isinstance(x, tuple) else str(x))

    # Dynamic size + optional width override
    if layout_type == "grid":
        calc_w, calc_h = calculate_grid_panel_dimensions(title, subtitle, local_items, cols, scale_factor)
    else:
        # Add extra height for all list layouts to prevent overlap with instructions
        extra_height = 50
        calc_w, calc_h = calculate_dynamic_panel_size(title, subtitle, local_items, extra_height, scale_factor)

    # Base panel width from content, then optionally clamp to a maximum
    panel_width = calc_w
    if panel_width_override is not None:
        try:
            panel_width = min(panel_width, int(panel_width_override))
        except Exception:
            pass
    # Respect global UI width bounds
    try:
        min_w = int(ui.get("PANEL_MIN_WIDTH", 300))
    except Exception:
        min_w = 300
    try:
        max_w = int(ui.get("PANEL_MAX_WIDTH", 500))
    except Exception:
        max_w = 500
    panel_width = max(min_w, min(panel_width, max_w))
    panel_height = calc_h

    # Auto-fit panel to current screen dimensions (shrink uniformly if too large)
    panel_width, panel_height = _fit_panel_to_screen(panel_width, panel_height, margin_ratio=0.9)

    if suppress_reference:
        # Window-only mode (no reference image window); sanitize window name to avoid issues with newlines
        safe_title = title.replace("\n", " ") if isinstance(title, str) else str(title)
        panel_window = f"{safe_title} Window"
        hires_panel, _ = create_high_dpi_panel(panel_width, panel_height, scale_factor)
        current_y = draw_hires_panel_header(hires_panel, title, subtitle, scale_factor, CONFIG.get("UI", {}).get("MARGIN", 18))

        if layout_type == "grid":
            click_zones = _create_grid_layout(
                hires_panel, local_items, cols, panel_width, current_y,
                scale_factor, display_formatter, window_color
            )
        else:
            click_zones = _create_list_layout(
                hires_panel, local_items, panel_width, current_y,
                scale_factor, display_formatter, window_color
            )

        # Place instruction text a fixed config-driven distance below the
        # bottom-most button, clamped by the bottom margin.
        ui = CONFIG.get("UI", {})
        bottom_margin_unscaled = int(ui.get("INSTRUCTION_BOTTOM_MARGIN", ui.get("MARGIN", 18)))
        gap_unscaled = _get_button_instruction_gap_unscaled()
        if click_zones:
            max_bottom = max(rect[3] for rect in click_zones.values())
            desired_instruction_unscaled = max_bottom + gap_unscaled
        else:
            desired_instruction_unscaled = panel_height - bottom_margin_unscaled - gap_unscaled
        max_instruction_unscaled = panel_height - bottom_margin_unscaled
        instruction_y_unscaled = min(desired_instruction_unscaled, max_instruction_unscaled)
        instruction_y = int(instruction_y_unscaled * scale_factor)
        _add_instructions(hires_panel, "Click or S=Skip / ESC=Quit", instruction_y, scale_factor)

        display_panel = show_hires_panel(panel_window, hires_panel, panel_width, panel_height)
        # In panel-only mode, treat the provided X as a hard anchor so
        # child panels (DATE/TIME/etc.) line up horizontally with the
        # Image Label window. Clamp only Y to keep the panel on-screen.
        desired_pos = panel_position if (panel_position and isinstance(panel_position, tuple)) else CONFIG["UI"]["DEFAULT_WINDOW_POSITION"]
        px, py = desired_pos
        pw, ph = display_panel.shape[1], display_panel.shape[0]
        try:
            py = _clamp_y_only(py, ph)
        except Exception:
            pass
        cv.moveWindow(panel_window, px, py)

        # Inline mouse handling for macOS reliability
        selected_item = None
        def _mouse_cb(event, x, y, flags, param):
            nonlocal selected_item
            if event in (cv.EVENT_LBUTTONDOWN, cv.EVENT_LBUTTONUP):
                for item, (x1, y1, x2, y2) in click_zones.items():
                    if (x1 - 1) <= x <= (x2 + 1) and (y1 - 1) <= y <= (y2 + 1):
                        selected_item = item
                        try:
                            print(f"[UI] Click detected: {item} @ ({x},{y})")
                        except Exception:
                            pass
                        return

        cv.setMouseCallback(panel_window, _mouse_cb)

        while selected_item is None:
            key = cv.waitKey(30)
            if key in (27, ord('q'), ord('Q')):
                print("User requested exit by ESC/Q. Quitting.")
                try:
                    cv.destroyAllWindows()
                except Exception:
                    pass
                sys.exit(0)
            if key in (ord('s'), ord('S')):
                try:
                    print("[UI] Keyboard Skip detected (S); closing panel.")
                except Exception:
                    pass
                selected_item = SKIP_SENTINEL
                break

        try:
            cv.destroyWindow(panel_window)
        except Exception:
            pass
        return None if selected_item == SKIP_SENTINEL else selected_item
    else:
        # Original two-window mode
        image_window, panel_window, display_img, panel_pos = create_window_pair(img, title)
        try:
            hires_panel, _ = create_high_dpi_panel(panel_width, panel_height, scale_factor)
            current_y = draw_hires_panel_header(hires_panel, title, subtitle, scale_factor, CONFIG.get("UI", {}).get("MARGIN", 18))

            if layout_type == "grid":
                click_zones = _create_grid_layout(
                    hires_panel, local_items, cols, panel_width, current_y,
                    scale_factor, display_formatter, window_color
                )
            else:
                click_zones = _create_list_layout(
                    hires_panel, local_items, panel_width, current_y,
                    scale_factor, display_formatter, window_color
                )

            ui = CONFIG.get("UI", {})
            bottom_margin_unscaled = int(ui.get("INSTRUCTION_BOTTOM_MARGIN", ui.get("MARGIN", 18)))
            gap_unscaled = _get_button_instruction_gap_unscaled()
            if click_zones:
                max_bottom = max(rect[3] for rect in click_zones.values())
                desired_instruction_unscaled = max_bottom + gap_unscaled
            else:
                desired_instruction_unscaled = panel_height - bottom_margin_unscaled - gap_unscaled
            max_instruction_unscaled = panel_height - bottom_margin_unscaled
            instruction_y_unscaled = min(desired_instruction_unscaled, max_instruction_unscaled)
            instruction_y = int(instruction_y_unscaled * scale_factor)
            _add_instructions(hires_panel, "Click or S=Skip / ESC=Quit", instruction_y, scale_factor)

            display_panel = show_hires_panel(panel_window, hires_panel, panel_width, panel_height)
            # Clamp panel window next to reference image within screen
            pw, ph = display_panel.shape[1], display_panel.shape[0]
            try:
                clamped_x, clamped_y = _clamp_position(panel_pos[0], panel_pos[1], pw, ph)
            except Exception:
                clamped_x, clamped_y = panel_pos
            cv.moveWindow(panel_window, clamped_x, clamped_y)
            cv.waitKey(1)
            selected = _handle_selection_separate_window(display_panel, click_zones, panel_window, skip_sentinel=SKIP_SENTINEL)
            cv.destroyWindow(image_window)
            cv.destroyWindow(panel_window)
            return selected
        except Exception as e:
            print(f"Error in {title}: {e}")
            try:
                cv.destroyWindow(image_window)
                cv.destroyWindow(panel_window)
            except:
                pass
            return None        

def _create_list_layout(hires_panel, items, panel_width, current_y, scale_factor, format_func=None, window_color=None):
    """Create vertical list layout with colorblind palette; 'Skip' is white with black text."""
    click_zones = {}
    ui = CONFIG.get("UI", {})
    margin = int(ui.get("MARGIN", 18))
    text_white = tuple(ui.get("TEXT_WHITE", (255, 255, 255)))
    BUTTON_FONT_SIZE = float(ui.get("BUTTON_FONT_SIZE", ui.get("BUTTON_SIZE", 0.6)))
    button_width = panel_width - (margin * 2)
    button_height = int(ui.get("BUTTON_HEIGHT", 30))
    button_spacing = int(ui.get("BUTTON_SPACING", 10))

    # Ensure content respects a fixed button-to-instruction gap at bottom
    panel_height_unscaled = int(hires_panel.shape[0] // max(scale_factor, 1))
    bottom_margin_unscaled = int(ui.get("INSTRUCTION_BOTTOM_MARGIN", margin))
    gap_unscaled = _get_button_instruction_gap_unscaled()
    bottom_safety = int(ui.get("BUTTON_BOTTOM_SAFETY", 4))
    rows = len(items)
    total_height = rows * button_height + max(0, rows - 1) * button_spacing
    desired_bottom_y = panel_height_unscaled - bottom_margin_unscaled - gap_unscaled - bottom_safety
    header_bottom_y = current_y  # incoming current_y is just below header
    computed_y = desired_bottom_y - total_height
    # If there is enough vertical space, float the list so the bottom button
    # sits exactly `gap_unscaled` above the instruction text; otherwise
    # clamp to the header so content never overlaps the title.
    if computed_y < header_bottom_y:
        current_y = header_bottom_y
    else:
        current_y = computed_y

    for i, item in enumerate(items):
        y_pos = current_y + i * (button_height + button_spacing)
        button_rect = (margin, y_pos, margin + button_width, y_pos + button_height)

        # Determine display text and color
        text_color = text_white
        # Tuple items may carry a color; keep their label via actual_item
        if isinstance(item, tuple) and len(item) == 2:
            actual_item, provided_color = item
            display_text = format_func(actual_item) if format_func else str(actual_item)
            # Skip button special styling
            if isinstance(actual_item, str) and actual_item == "__SKIP__":
                bg_color = (255, 255, 255)   # white
                text_color = (0, 0, 0)       # black
            else:
                # Use given color if provided; otherwise palette
                bg_color = provided_color if provided_color is not None else palette_color(i)
        else:
            actual_item = item
            display_text = format_func(item) if format_func else str(item)
            # Skip sentinel passed as string in some grids/lists
            if isinstance(item, str) and item == "__SKIP__":
                bg_color = (255, 255, 255)
                text_color = (0, 0, 0)
            else:
                bg_color = palette_color(i)

        draw_hires_styled_button(hires_panel, display_text, button_rect, bg_color, scale_factor,
                     text_color, BUTTON_FONT_SIZE)
        click_zones[actual_item] = button_rect
    return click_zones


def _create_grid_layout(hires_panel, items, cols, panel_width, current_y, scale_factor, format_func=None, window_color=None):
    """Create grid layout with SINGLE COLOR per window; Skip is white with black text."""
    click_zones = {}
    spacing = CONFIG["UI"]["GRID_SPACING"]
    button_height = CONFIG["UI"]["GRID_BUTTON_HEIGHT"]

    # Use provided single color or a palette fallback
    single_color = window_color if window_color is not None else palette_color(0)

    # Compute button dimensions
    ui = CONFIG.get("UI", {})
    margin = int(ui.get("MARGIN", 18))
    text_white = tuple(ui.get("TEXT_WHITE", (255, 255, 255)))
    BUTTON_FONT_SIZE = float(ui.get("BUTTON_FONT_SIZE", ui.get("BUTTON_SIZE", 0.6)))
    available_width = panel_width - (margin * 2)
    calculated_width = (available_width - (cols - 1) * spacing) // cols
    max_width = CONFIG["UI"].get("MAX_GRID_BUTTON_WIDTH", 120)
    button_width = min(calculated_width, max_width)

    # Center grid
    grid_width = cols * button_width + (cols - 1) * spacing
    start_x = (panel_width - grid_width) // 2

    # Ensure the grid respects the same fixed button-to-instruction gap
    panel_height_unscaled = int(hires_panel.shape[0] // max(scale_factor, 1))
    bottom_margin_unscaled = int(ui.get("INSTRUCTION_BOTTOM_MARGIN", margin))
    gap_unscaled = _get_button_instruction_gap_unscaled()
    bottom_safety = int(ui.get("BUTTON_BOTTOM_SAFETY", 4))
    rows = (len(items) + max(cols, 1) - 1) // max(cols, 1)
    grid_total_height = rows * button_height + max(0, rows - 1) * spacing
    desired_bottom_y = panel_height_unscaled - bottom_margin_unscaled - gap_unscaled - bottom_safety
    header_bottom_y = current_y
    computed_y = desired_bottom_y - grid_total_height
    if computed_y < header_bottom_y:
        current_y = header_bottom_y
    else:
        current_y = computed_y

    for idx, item in enumerate(items):
        row = idx // cols
        col = idx % cols
        x_pos = start_x + col * (button_width + spacing)
        y_pos = current_y + row * (button_height + spacing)
        button_rect = (x_pos, y_pos, x_pos + button_width, y_pos + button_height)

        # Label and colors
        display_text = format_func(item) if format_func else (str(item[0]) if isinstance(item, tuple) else str(item))
        if (isinstance(item, str) and item == "__SKIP__") or (isinstance(item, tuple) and item[0] == "__SKIP__"):
            bg_color = (255, 255, 255)  # white
            text_color = (0, 0, 0)      # black
        else:
            bg_color = single_color
            text_color = text_white

        draw_hires_styled_button(hires_panel, display_text, button_rect, bg_color,
                                 scale_factor, text_color, BUTTON_FONT_SIZE)
        # For tuples, click key is the actual item (first element)
        click_key = item[0] if isinstance(item, tuple) else item
        click_zones[click_key] = button_rect
    return click_zones


def _handle_selection_separate_window(panel_bg, click_zones, window_name, skip_sentinel="__SKIP__"):
    """Generic handler for clickable selections in separate windows.
    - Click: returns the clicked item
    - Clicking skip_sentinel returns None
    - ESC/Q quits the app
    """
    selected_item = None

    def mouse_callback(event, x, y, flags, param):
        nonlocal selected_item
        # Some macOS builds deliver LBUTTONUP rather than LBUTTONDOWN; accept either
        if event in (cv.EVENT_LBUTTONDOWN, cv.EVENT_LBUTTONUP):
            for item, (x1, y1, x2, y2) in click_zones.items():
                # Slight tolerance to account for rounding
                if (x1 - 1) <= x <= (x2 + 1) and (y1 - 1) <= y <= (y2 + 1):
                    selected_item = item
                    try:
                        print(f"[UI] Click detected: {item} @ ({x},{y})")
                    except Exception:
                        pass
                    return

    cv.setMouseCallback(window_name, mouse_callback)

    while selected_item is None:
        key = cv.waitKey(30)
        if key in (27, ord('q'), ord('Q')):  # ESC or Q -> quit app
            print("User requested exit by ESC/Q. Quitting.")
            try:
                cv.destroyAllWindows()
            except Exception:
                pass
            sys.exit(0)
        # Keyboard fallback for Skip to ensure progress even if mouse events fail
        if key in (ord('s'), ord('S')):
            try:
                print("[UI] Keyboard Skip detected (S); closing panel.")
            except Exception:
                pass
            selected_item = skip_sentinel
            break

    # Map skip sentinel to None
    if selected_item == skip_sentinel:
        try:
            print("[UI] Skip selected; returning None and closing panel.")
        except Exception:
            pass
        return None
    return selected_item

# ----------------------------------------------------------------------------------------------------------------------------------
# 5) DATE AND EXPERIMENT INPUT FUNCTIONS
# ----------------------------------------------------------------------------------------------------------------------------------
def _get_date_visual(img, filename, base_anchor=None, panel_width_override=None):
    """
    Month → Day → Year selection, all at the SAME screen position (directly below Image Label).
    Each window replaces the previous one when the user makes a selection.
    """
    allow_skip = get_allow_skip("DATE")

    # Starting anchor - all windows will use this same Y position
    base_x, base_y = base_anchor if base_anchor else CONFIG["UI"].get("DEFAULT_WINDOW_POSITION", (0, 0))

    def _size_for(title, subtitle, items, layout_key):
        # This helper was previously used to pre-compute sizes with SCALE_FACTOR;
        # the actual sizing is now handled inside create_selection_interface.
        layout = _get_window_layout(layout_key) or {}
        if panel_width_override:
            return int(panel_width_override), 0
        return 0, 0

    # 1) Month
    months_range = CONFIG["INPUTS"].get("MONTH", list(range(1, 13)))
    months = [(i, f"{i} {calendar.month_name[i][:3]}") for i in months_range if 1 <= i <= 12]
    m_w, m_h = _size_for("Select Month", "", months, "MONTH_SELECTION")
    month = create_selection_interface(
        img, "Select Month", "", months,
        _get_window_layout("MONTH_SELECTION"), lambda x: x[1],
        add_skip=allow_skip, suppress_reference=True,
        panel_position=(base_x, base_y), panel_width_override=400
    )
    if month is None:
        return None

    # 2) Day
    month_num = month
    #month_num = month[0] if isinstance(month, tuple) else month
    days_in_month = calendar.monthrange(2024, month_num)[1]  # Use arbitrary year for day count
    day_range = CONFIG["INPUTS"].get("DAY", list(range(1, 32)))
    days = [d for d in day_range if 1 <= d <= days_in_month] or list(range(1, days_in_month + 1))
    day_sel = create_selection_interface(
        img, "Select Day", "",
        days, _get_window_layout("DAY_SELECTION"), add_skip=allow_skip,
        suppress_reference=True,
        panel_position=(base_x, base_y),
        panel_width_override=600
    )
    # day_sel may be None (user skipped day) — continue to Year selection in that case

    # 3) Year
    years = CONFIG["INPUTS"].get("YEAR", [2024, 2025])
    y_w, y_h = _size_for("Select Year", "", years, "YEAR_SELECTION")
    year_sel = create_selection_interface(
        img, "Select Year", "", years, _get_window_layout("YEAR_SELECTION"),
        add_skip=allow_skip, suppress_reference=True,
        panel_position=(base_x, base_y), 
        panel_width_override=300
    )
    # year_sel may be None (user skipped year)

    # Track selections globally for Image Label display
    global _current_month, _current_day, _current_year
    _current_month = str(month_num)
    _current_day = str(day_sel) if day_sel is not None else None
    _current_year = str(year_sel) if year_sel is not None else None

    # Build an appropriate return string depending on which parts were provided
    # Priority: full YYYY_MM_DD when all present. If two parts present, return
    # either YYYY_MM (year+month) or MM_DD (month+day). If only month present,
    # return MM. Downstream display function handles partial formats.
    if (year_sel is not None) and (day_sel is not None):
        return f"{year_sel:04d}_{month_num:02d}_{day_sel:02d}"
    if (year_sel is not None) and (day_sel is None):
        # Year + Month -> return YYYY_MM so display shows MM/YY
        return f"{year_sel:04d}_{month_num:02d}"
    if (year_sel is None) and (day_sel is not None):
        # Month + Day -> return MM_DD so display shows MM/DD
        return f"{month_num:02d}_{day_sel:02d}"
    # Only month selected
    return f"{month_num:02d}"

def _get_Sample_Num_visual(img, field_name, image_window=None, panel_pos=None,
                           close_image_window=True, max_panel_width=None):
    """Generic numeric NUMERIC input with optional Skip; supports a min-max range and a persistent warning below Skip.

    Args:
        img: The image to display
        field_name: Title base for the window (e.g., "Replicate" or "Plot")
        image_window: Optional existing reference image window name to reuse
        panel_pos: Optional panel position tuple (x, y) to match Image Label location
        close_image_window: If True, close the image window when done; if False, leave it open
    """
    allow_skip = get_allow_skip(field_name)
    # Read Valid Range from config-driven schema
    min_allowed = None
    max_allowed = None
    try:
        field_cfg = next((f for f in CONFIG.get("SCHEMA_FIELDS", []) if f.get("name") == field_name), None)
        if field_cfg and str(field_cfg.get("type")).upper() == "NUMERIC":
            min_allowed = field_cfg.get("min")
            max_allowed = field_cfg.get("max")
    except Exception:
        pass
    if min_allowed is None or max_allowed is None:
        min_allowed, max_allowed = 1, 1
    if min_allowed > max_allowed:
        min_allowed, max_allowed = max_allowed, min_allowed

    panel_window = f"{field_name} Window"
    # Use provided panel_pos or fall back to default
    if panel_pos is None:
        panel_pos = CONFIG["UI"].get("DEFAULT_WINDOW_POSITION", (0, 0))
    scale_factor = _get_hidpi_scale()
    # Tighten extra content space: reserve only what's needed for input + optional skip.
    # We already anchor controls from the bottom; this extra height only needs to
    # cover the input + optional Skip and a small error text region.
    extra_error_space = 8 if allow_skip else 6
    extra_content_height = CONFIG["UI"]["BUTTON_HEIGHT"] * (1 + (1 if allow_skip else 0)) + extra_error_space
    panel_width, panel_height = calculate_dynamic_panel_size(
        field_name, f"Valid Range: {min_allowed}-{max_allowed}",
        [], extra_content_height, scale_factor
    )
    # Screen-fit to avoid oversized panels
    panel_width, panel_height = _fit_panel_to_screen(panel_width, panel_height)
    # Never allow this NUMERIC panel to exceed an optional maximum (e.g.,
    # the width of the Image Label window) while still respecting
    # global UI min/max bounds enforced in calculate_dynamic_panel_size.
    try:
        if max_panel_width is not None:
            panel_width = min(int(panel_width), int(max_panel_width))
    except Exception:
        pass

    try:
        current_input = ""
        persistent_warning_msg = None  # keep warning visible until input changes or is valid
        max_digits = max(CONFIG["UI"]["MAX_SAMPLE_INPUT_DIGITS"], len(str(abs(max_allowed))))
        # Button rects (unscaled coordinates)
        ui = CONFIG.get("UI", {})
        margin = int(ui.get("MARGIN", 18))
        input_width = panel_width - (margin * 2)
        input_height = CONFIG["UI"]["BUTTON_HEIGHT"]
        button_spacing = int(CONFIG["UI"].get("BUTTON_SPACING", ui.get("BUTTON_SPACING", 10)))

        # Anchor controls from the bottom so the bottom-most control sits at
        # a fixed gap above the instruction text, just like list/grid panels.
        bottom_margin_unscaled = int(ui.get("INSTRUCTION_BOTTOM_MARGIN", margin))
        gap_unscaled = _get_button_instruction_gap_unscaled()
        bottom_safety = int(ui.get("BUTTON_BOTTOM_SAFETY", 4))
        desired_bottom_y = panel_height - bottom_margin_unscaled - gap_unscaled - bottom_safety

        skip_rect = None
        if allow_skip:
            # Layout: input above Skip, both anchored from the bottom.
            skip_bottom_y = desired_bottom_y
            skip_top_y = skip_bottom_y - input_height
            input_bottom_y = skip_top_y - button_spacing
            input_top_y = input_bottom_y - input_height
            skip_rect = (margin, skip_top_y, margin + input_width, skip_bottom_y)
        else:
            # Only an input field; anchor it directly from the bottom.
            input_bottom_y = desired_bottom_y
            input_top_y = input_bottom_y - input_height

        input_rect = (margin, input_top_y, margin + input_width, input_top_y + input_height)

        def render_panel(msg=None, field_color=(100, 100, 100)):
            hires_panel, _ = create_high_dpi_panel(panel_width, panel_height, scale_factor)
            # Header
            draw_hires_panel_header(hires_panel, field_name,
                                    f"Valid Range: {min_allowed}-{max_allowed}",
                                    scale_factor, margin)
            # Input field
            input_text = f"# {current_input}" if current_input else "# ___"
            text_white = tuple(ui.get("TEXT_WHITE", (255, 255, 255)))
            BUTTON_FONT_SIZE = float(ui.get("BUTTON_FONT_SIZE", ui.get("BUTTON_SIZE", 0.6)))
            draw_hires_styled_button(hires_panel, input_text, input_rect, field_color,
                                     scale_factor, text_white, BUTTON_FONT_SIZE)
            # Optional Skip button
            if allow_skip and skip_rect:
                draw_hires_styled_button(hires_panel, "Skip", skip_rect, (255, 255, 255),
                         scale_factor, (0, 0, 0), BUTTON_FONT_SIZE)
            # Instructions placed a fixed config-driven distance below the
            # bottom-most control (Skip if present, else the input field).
            bottom_margin_unscaled = int(ui.get("INSTRUCTION_BOTTOM_MARGIN", margin))
            gap_unscaled = _get_button_instruction_gap_unscaled()
            # By construction, the bottom-most control sits at desired_bottom_y
            # above; use that directly as the anchor.
            bottom_y_unscaled = desired_bottom_y
            desired_instruction_unscaled = bottom_y_unscaled + gap_unscaled
            max_instruction_unscaled = panel_height - bottom_margin_unscaled
            instruction_y_unscaled = min(desired_instruction_unscaled, max_instruction_unscaled)
            instruction_y = int(instruction_y_unscaled * scale_factor)
            _add_instructions(hires_panel, "ENTER=Accept / ESC=Quit", instruction_y, scale_factor)
            # Persistent warning (below Skip if present, else below input)
            if msg:
                if allow_skip and skip_rect:
                    err_y_unscaled = skip_rect[3] + 24
                else:
                    err_y_unscaled = input_top_y + input_height + 24
                err_y = int(err_y_unscaled * scale_factor)
                err_x = int(margin * scale_factor)
                # Font and styling from centralized font helper
                _font_face_cfg = _get_font_face()
                subTITLE_FONT_SIZE = float(ui.get("SUBTITLE_FONT_SIZE", 0.7))
                subtitle_thick = max(1, int(int(ui.get("SUBTITLE_THICKNESS", 2)) * scale_factor))
                error_color = tuple(ui.get("ERROR_COLOR", (0, 0, 255)))  # BGR red default
                cv.putText(hires_panel, msg, (err_x, err_y), _font_face_cfg,
                           subTITLE_FONT_SIZE * scale_factor, error_color,
                           subtitle_thick, cv.LINE_AA)
            # Show
            display_panel = show_hires_panel(panel_window, hires_panel, panel_width, panel_height)
            # Keep the X anchor fixed (to match the Image Label's left
            # edge) and clamp only Y so the NUMERIC panel stays on-screen.
            pw, ph = display_panel.shape[1], display_panel.shape[0]
            px, py = panel_pos
            try:
                py = _clamp_y_only(py, ph)
            except Exception:
                pass
            cv.moveWindow(panel_window, px, py)
            cv.waitKey(1)  # pump events
            return display_panel

        display_panel = render_panel(persistent_warning_msg)

        # Mouse handling for Skip
        selected_skip = False
        def inside_rect(x, y, rect):
            x1, y1, x2, y2 = rect
            return x1 <= x <= x2 and y1 <= y <= y2
        def on_mouse(event, mx, my, flags, param):
            nonlocal selected_skip
            if allow_skip and skip_rect and event == cv.EVENT_LBUTTONDOWN:
                if inside_rect(mx, my, skip_rect):
                    selected_skip = True
        cv.setMouseCallback(panel_window, on_mouse)

        # Continuous event loop keeps window responsive
        while True:
            if selected_skip:
                if close_image_window and image_window:
                    cv.destroyWindow(image_window)
                cv.destroyWindow(panel_window)
                return None

            key = cv.waitKey(30)  # non-blocking; keeps UI responsive
            if key == -1:
                continue
            if key in (ord('q'), 27):  # ESC or 'q'
                if close_image_window and image_window:
                    cv.destroyWindow(image_window)
                cv.destroyWindow(panel_window)
                sys.exit(0)
                return None
            elif key in (10, 13):  # ENTER
                if current_input:
                    try:
                        selected_num = int(current_input)
                        if min_allowed <= selected_num <= max_allowed:
                            if close_image_window and image_window:
                                cv.destroyWindow(image_window)
                            cv.destroyWindow(panel_window)
                            # Return zero-padded string for numeric label
                            return format_number_with_padding(selected_num, min_allowed, max_allowed)
                        else:
                            persistent_warning_msg = f"Must be {min_allowed}-{max_allowed}"
                            render_panel(persistent_warning_msg, field_color=tuple(CONFIG.get("UI", {}).get("BTN_RED", (0, 0, 255))))
                            # mouse callback stays attached; no need to reattach
                    except ValueError:
                        persistent_warning_msg = "Invalid number"
                        render_panel(persistent_warning_msg, field_color=tuple(CONFIG.get("UI", {}).get("BTN_RED", (0, 0, 255))))
                else:
                    if allow_skip:
                        if close_image_window and image_window:
                            cv.destroyWindow(image_window)
                        cv.destroyWindow(panel_window)
                        return None
                    else:
                        persistent_warning_msg = "Enter a number"
                        render_panel(persistent_warning_msg, field_color=tuple(CONFIG.get("UI", {}).get("BTN_RED", (0, 0, 255))))
            # Cross-platform Backspace/Delete (8 on Windows/Linux, 127 on macOS)
            elif key in (8, 127):
                if current_input:
                    current_input = current_input[:-1]
                persistent_warning_msg = None  # clear warning when editing
                render_panel(persistent_warning_msg)
            # Digits 0-9
            elif ord('0') <= key <= ord('9'):
                if len(current_input) < max_digits:
                    current_input += chr(key)
                    persistent_warning_msg = None  # clear warning when editing
                    render_panel(persistent_warning_msg)
            # Ignore other keys; keep loop alive
    except Exception as e:
        print(f"Error in sample number input: {e}")
        try:
            if close_image_window and image_window:
                cv.destroyWindow(image_window)
            cv.destroyWindow(panel_window)
        except:
            pass
        return None

def _get_generic_select_visual(img, field_name, options,
                               colors=None, base_anchor=None,
                               panel_width_override=None,
                               layout_key=None):
    """Generic SELECT input panel using styled buttons; returns selected option or None.

    Note: Does NOT open a new reference image window; reuses the existing one by positioning panel at base_anchor.
    """
    scale_factor = _get_hidpi_scale()
    panel_window = f"{field_name}"
    panel_pos = base_anchor if base_anchor else CONFIG["UI"].get("DEFAULT_WINDOW_POSITION", (0, 0))
    button_texts = list(options)[:]
    # Ensure a Skip option is always present
    if "__SKIP__" not in button_texts:
        button_texts.append("__SKIP__")
    # Keep standalone SELECT panels compact: rely on dynamic sizing plus a
    # modest padding instead of a large default to avoid excessive space
    # between the title and the first row of buttons.
    extra_content_height = CONFIG["UI"].get("PANEL_CONTENT_PADDING", 20)
    panel_width, panel_height = calculate_dynamic_panel_size(
        f"{field_name}", "", button_texts, extra_content_height, scale_factor
    )
    panel_width, panel_height = _fit_panel_to_screen(panel_width, panel_height)
    # Treat panel_width_override as an upper bound so that
    # child SELECT panels (driven from the Image Label window)
    # are never wider than their parent, while still ensuring
    # a reasonable minimum width from calculate_dynamic_panel_size.
    if panel_width_override is not None:
        try:
            panel_width = min(int(panel_width), int(panel_width_override))
        except Exception:
            pass
    ui = CONFIG.get("UI", {})
    # Optional layout override from CONFIG['UI']['WINDOW_LAYOUTS']
    layout_cfg = _get_window_layout(layout_key) if layout_key else None
    margin = int(ui.get("MARGIN", 18))
    button_height = CONFIG["UI"]["BUTTON_HEIGHT"]
    button_spacing = CONFIG["UI"]["BUTTON_SPACING"]
    update_window_gap = int(CONFIG["UI"].get("UPDATE_WINDOW_GAP", 15))

    SKIP_SENTINEL = "__SKIP__"

    def render_panel():
        hires_panel, _ = create_high_dpi_panel(panel_width, panel_height, scale_factor)
        current_y = draw_hires_panel_header(hires_panel, f"{field_name}", "", scale_factor, margin)
        click_zones = {}
        # Determine number of columns: prefer layout config if provided, otherwise
        # fall back to the original heuristic (1/2/3 columns by option count).
        n = len(button_texts)
        if layout_cfg and isinstance(layout_cfg, dict):
            try:
                cfg_cols = int(layout_cfg.get("cols", 1))
                cols = max(1, cfg_cols)
            except Exception:
                cols = 1 if n <= 8 else (2 if n <= 16 else 3)
        else:
            cols = 1 if n <= 8 else (2 if n <= 16 else 3)
        col_gap = int(button_spacing)
        total_inner_w = panel_width - (margin * 2)
        button_w = int((total_inner_w - (col_gap * (cols - 1))) / max(cols, 1))
        # Compute rows
        rows = int(np.ceil(n / max(cols, 1)))

        # Anchor the grid from the bottom so that the bottom-most row of
        # buttons sits a fixed config-driven distance above the instruction
        # text, matching other SELECT/DATE/NUMERIC panels.
        panel_height_unscaled = int(hires_panel.shape[0] // max(scale_factor, 1))
        bottom_margin_unscaled = int(ui.get("INSTRUCTION_BOTTOM_MARGIN", margin))
        gap_unscaled = _get_button_instruction_gap_unscaled()
        bottom_safety = int(ui.get("BUTTON_BOTTOM_SAFETY", 4))
        grid_total_height = rows * button_height + max(0, rows - 1) * button_spacing
        desired_bottom_y = panel_height_unscaled - bottom_margin_unscaled - gap_unscaled - bottom_safety
        header_bottom_y = current_y
        base_y = desired_bottom_y - grid_total_height
        if base_y < header_bottom_y:
            base_y = header_bottom_y

        for idx, opt in enumerate(button_texts):
            r = idx // cols
            c = idx % cols
            x1 = margin + c * (button_w + col_gap)
            y_pos = base_y + r * (button_height + button_spacing)
            rect = (x1, y_pos, x1 + button_w, y_pos + button_height)
            # Special styling for Skip
            if isinstance(opt, str) and opt == "__SKIP__":
                col = (255, 255, 255); txt_col = (0, 0, 0)
                draw_hires_styled_button(hires_panel, "Skip", rect, col, scale_factor, txt_col,
                                         float(ui.get("BUTTON_FONT_SIZE", ui.get("BUTTON_SIZE", 0.6))))
                click_zones[opt] = rect
            else:
                col = tuple(ui.get("BTN_GREEN", (0, 180, 0)))
                if colors and opt in colors and isinstance(colors[opt], (list, tuple)):
                    try:
                        col = tuple(int(c) for c in colors[opt])
                    except Exception:
                        pass
                draw_hires_styled_button(hires_panel, str(opt), rect, col, scale_factor,
                                         tuple(ui.get("TEXT_WHITE", (255, 255, 255))),
                                         float(ui.get("BUTTON_FONT_SIZE", ui.get("BUTTON_SIZE", 0.6))))
                click_zones[opt] = rect

        # Place instructions a fixed distance below the bottom-most button,
        # using the same config-driven gap as other panels.
        if click_zones:
            max_bottom = max(r[3] for r in click_zones.values())
            desired_instruction_unscaled = max_bottom + gap_unscaled
        else:
            desired_instruction_unscaled = panel_height - bottom_margin_unscaled - gap_unscaled
        max_instruction_unscaled = panel_height - bottom_margin_unscaled
        instruction_y_unscaled = min(desired_instruction_unscaled, max_instruction_unscaled)
        instruction_y = int(instruction_y_unscaled * scale_factor)
        _add_instructions(hires_panel, "CLICK=Select / S=Skip / ESC=Quit", instruction_y, scale_factor)
        show_hires_panel(panel_window, hires_panel, panel_width, panel_height)
        cv.moveWindow(panel_window, *panel_pos)
        cv.waitKey(1)
        return click_zones

    selected = None
    try:
        zones = render_panel()
        def mouse_cb(event, mx, my, flags, param):
            nonlocal selected
            if event in (cv.EVENT_LBUTTONDOWN, cv.EVENT_LBUTTONUP):
                # 1px tolerance helps on some macOS OpenCV builds
                for opt, (x1, y1, x2, y2) in zones.items():
                    if (x1 - 1) <= mx <= (x2 + 1) and (y1 - 1) <= my <= (y2 + 1):
                        selected = SKIP_SENTINEL if (isinstance(opt, str) and opt == SKIP_SENTINEL) else opt
                        return
        cv.setMouseCallback(panel_window, mouse_cb)
        while selected is None:
            key = cv.waitKey(30)
            if key in (27, ord('q'), ord('Q')):
                try: cv.destroyWindow(panel_window)
                except Exception: pass
                return None
            if key in (ord('s'), ord('S')):
                selected = SKIP_SENTINEL
        cv.destroyWindow(panel_window)
        return None if selected == SKIP_SENTINEL else selected
    except Exception:
        try: cv.destroyWindow(panel_window)
        except Exception: pass
        return None

def _get_generic_range_visual(img, field_name, min_allowed, max_allowed,
                              image_window=None, panel_pos=None,
                              close_image_window=True, max_panel_width=None):
    """Generic NUMERIC input panel (numeric); returns int or None.

    The optional max_panel_width can be used to ensure the NUMERIC window is
    never wider than a parent panel (e.g., the Image Label review window).
    """
    return _get_Sample_Num_visual(
        img,
        field_name,
        image_window=image_window,
        panel_pos=panel_pos,
        close_image_window=close_image_window,
        max_panel_width=max_panel_width,
    )

def _get_time_visual(img, field_name, base_anchor=None, panel_width_override=None):
    """Three-step time selection using grid-style panels (Hour → Minute → Second)."""
    try:
        inputs = CONFIG.get("INPUTS", {}) or {}
        hours = inputs.get("HOUR") or list(range(0, 24))
        minutes = inputs.get("MINUTE") or [0, 15, 30, 45]
        seconds = inputs.get("SECOND") or [0, 15, 30, 45]

        base_x, base_y = base_anchor if base_anchor else CONFIG["UI"].get("DEFAULT_WINDOW_POSITION", (0, 0))

        # Hour selection (grid)
        hour_items = [f"{int(x):02d}" for x in hours]
        h = create_selection_interface(
            img, f"Select {field_name} Hour", "", hour_items,
            _get_window_layout("HOUR_SELECTION"), add_skip=True,
            suppress_reference=True, panel_position=(base_x, base_y),
            panel_width_override=panel_width_override
        )
        if h is None:
            return None

        # Minute selection (grid)
        minute_items = [f"{int(x):02d}" for x in minutes]
        m = create_selection_interface(
            img, f"Select {field_name} Minute", "", minute_items,
            _get_window_layout("MINUTE_SELECTION"), add_skip=True,
            suppress_reference=True, panel_position=(base_x, base_y),
            panel_width_override=panel_width_override
        )
        if m is None:
            return None

        # Second selection (grid)
        second_items = [f"{int(x):02d}" for x in seconds]
        s = create_selection_interface(
            img, f"Select {field_name} Second", "", second_items,
            _get_window_layout("SECOND_SELECTION"), add_skip=True,
            suppress_reference=True, panel_position=(base_x, base_y),
            panel_width_override=panel_width_override
        )
        if s is None:
            return None

        return f"{h}:{m}:{s}"
    except Exception as e:
        print(f"Error in time selection: {e}")
        return None

def _check_update_persistent_data_dynamic(img):
    """Dynamic Image Label dialog: renders fields from CONFIG['SCHEMA_FIELDS'] with Update buttons."""
    global _persistent_labels
    scale_factor = _get_hidpi_scale()
    image_window, panel_window, display_img, panel_pos = create_window_pair(img, "Image Label (Dynamic)")
    # Build button list using only the field names; we will render the
    # buttons in a column to the LEFT of the label values instead of in
    # a vertical stack below them. This makes the window wider than it
    # is tall, leaving more vertical space for child windows.
    fields = CONFIG.get("SCHEMA_FIELDS", [])
    button_texts = ["Done"] + [f["name"] for f in fields]

    ui = CONFIG.get("UI", {})
    viz = CONFIG.get("VISUALIZATION", {})
    margin = int(ui.get("MARGIN", 18))
    button_height = int(ui.get("BUTTON_HEIGHT", 30))
    button_spacing = int(ui.get("BUTTON_SPACING", 8))
    # Gap between the label block and the first row of buttons/values
    label_button_gap = int(ui.get("LABEL_BUTTON_GAP", button_spacing * 2))

    # Estimate header height (unscaled) similar to calculate_dynamic_panel_size
    TITLE_FONT_SIZE = float(viz.get("TITLE_FONT_SIZE", 0.85))
    title_w, title_h = calculate_text_size("Image Label", TITLE_FONT_SIZE, 1)
    header_height_est = margin * 2 + max(35, title_h + 10)

    # Each field occupies one row containing a button and its label/value
    rows = max(1, len(fields))
    rows_block_h = rows * button_height + max(0, rows - 1) * button_spacing

    # Add an extra row for the "Done" button underneath the field rows
    instruction_block = _instruction_text_block_height(scale_factor)
    bottom_margin_unscaled = int(ui.get("INSTRUCTION_BOTTOM_MARGIN", margin))
    panel_height = (
        header_height_est
        + label_button_gap
        + rows_block_h
        + label_button_gap
        + button_height  # Done row
        + instruction_block
        + bottom_margin_unscaled
    )

    # Width is still driven by dynamic sizing based on the header and
    # button text; reuse calculate_dynamic_panel_size for that part.
    panel_width, _ = calculate_dynamic_panel_size("Image Label", "", button_texts, 0, scale_factor)

    def render_panel_and_get_anchor():
        hires_panel, _ = create_high_dpi_panel(panel_width, panel_height, scale_factor)
        current_y = draw_hires_panel_header(hires_panel, "Image Label", "", scale_factor, margin)
        # Layout parameters for rows and columns
        scaled_margin = int(margin * scale_factor)
        # Left column: buttons (make these relatively wide so text is clear)
        inner_width = max(1, panel_width - 2 * margin)
        try:
            frac = float(ui.get("IMAGE_LABEL_BUTTON_FRACTION", 0.45))
        except Exception:
            frac = 0.45
        # Clamp fraction to a reasonable range
        frac = max(0.25, min(frac, 0.75))
        button_col_width = int(max(ui.get("MIN_BUTTON_WIDTH", 80), inner_width * frac))
        col_gap = int(ui.get("COLUMN_GAP", 18))
        # Value text starts just to the right of the button column
        value_x_unscaled = margin + button_col_width + col_gap

        # Text styling for values
        scaled_subtitle_size = float(ui.get("SUBTITLE_FONT_SIZE", 0.7)) * scale_factor
        scaled_subtitle_thickness = max(1, int(int(ui.get("SUBTITLE_THICKNESS", 2)) * scale_factor))
        _font_face_cfg = _get_font_face()

        # Build value strings only (DATE fields formatted for display)
        values = []
        for f in fields:
            raw = _persistent_labels.get(f["name"], "...")
            if f.get("type") == "DATE":
                disp = format_date_for_display(str(raw)) if raw not in (None, "...") else "..."
                values.append(disp)
            else:
                values.append(str(raw))

        # Vertical placement: start just below the header
        first_row_top_unscaled = current_y + label_button_gap
        click_zones = {}

        # Per-field rows
        for idx, (val, f_cfg) in enumerate(zip(values, fields)):
            row_top_unscaled = first_row_top_unscaled + idx * (button_height + button_spacing)
            row_bottom_unscaled = row_top_unscaled + button_height
            row_center_unscaled = row_top_unscaled + button_height // 2

            # Button rectangle for this field (left column)
            btn_rect = (
                margin,
                row_top_unscaled,
                margin + button_col_width,
                row_bottom_unscaled,
            )

            # Button text is just the field name (no "Update" prefix)
            btn_text = f_cfg["name"]
            colors_cfg = CONFIG.get("COLORS", {}) or {}
            col_hex = None
            if isinstance(colors_cfg.get("Labels"), dict):
                col_hex = colors_cfg["Labels"].get(f_cfg["name"])
            elif isinstance(colors_cfg.get("Labels"), (list, tuple)):
                # Map by index when using a simple list of colors
                if idx < len(colors_cfg["Labels"]):
                    col_hex = colors_cfg["Labels"][idx]

            def _hex_to_bgr_local(s):
                try:
                    s = str(s).lstrip('#')
                    if len(s) == 6:
                        r = int(s[0:2], 16); g = int(s[2:4], 16); b = int(s[4:6], 16)
                        return (b, g, r)
                except Exception:
                    return None
                return None

            btn_color = _hex_to_bgr_local(col_hex) or palette_color(idx + 1)
            draw_hires_styled_button(
                hires_panel,
                btn_text,
                btn_rect,
                btn_color,
                scale_factor,
                tuple(ui.get("TEXT_WHITE", (255, 255, 255))),
                float(ui.get("BUTTON_FONT_SIZE", ui.get("BUTTON_SIZE", 0.6))),
            )
            click_zones[f_cfg["name"]] = btn_rect

            # Value text to the right of the button (no label name here) - properly centered
            val_x = int(value_x_unscaled * scale_factor)

            # Calculate text size to properly center it vertically
            text_size, baseline = cv.getTextSize(
                val, 
                _font_face_cfg, 
                scaled_subtitle_size, 
                scaled_subtitle_thickness
            )
            text_height = text_size[1]

            # Position text so it's vertically centered in the button
            button_center_y = int(row_center_unscaled * scale_factor)
            text_y = button_center_y + text_height // 2

            cv.putText(
                hires_panel,
                val,
                (val_x, text_y),
                _font_face_cfg,
                scaled_subtitle_size,
                tuple(ui.get("TEXT_PRIMARY", (51, 51, 51))),
                scaled_subtitle_thickness,
                cv.LINE_AA,
            )

        # "Done" button row directly beneath the field rows, aligned with
        # the left button column so it does not increase the overall width.
        done_top_unscaled = first_row_top_unscaled + rows_block_h + label_button_gap
        done_rect = (
            margin,
            done_top_unscaled,
            margin + button_col_width,
            done_top_unscaled + button_height,
        )
        draw_hires_styled_button(
            hires_panel,
            "Done",
            done_rect,
            palette_color(0),
            scale_factor,
            tuple(ui.get("TEXT_WHITE", (255, 255, 255))),
            float(ui.get("BUTTON_FONT_SIZE", ui.get("BUTTON_SIZE", 0.6))),
        )
        click_zones["DONE"] = done_rect

        # Use CONFIG-driven bottom margin for instruction placement for consistency
        bottom_margin_unscaled = ui.get("INSTRUCTION_BOTTOM_MARGIN", margin)
        instruction_y = int((panel_height - bottom_margin_unscaled) * scale_factor)
        _add_instructions(hires_panel, "ENTER=Done / ESC=Quit", instruction_y, scale_factor)
        show_hires_panel(panel_window, hires_panel, panel_width, panel_height)
        cv.moveWindow(panel_window, *panel_pos)
        cv.waitKey(1)
        # Anchor child windows directly *below* the Image Label window,
        # with an optional configurable vertical gap so they do not
        # overlap the label panel.
        anchor_x = panel_pos[0]
        try:
            update_gap = int(ui.get("UPDATE_WINDOW_GAP", 0))
        except Exception:
            update_gap = 0
        anchor_y = panel_pos[1] + panel_height + update_gap
        return click_zones, (anchor_x, anchor_y)

    try:
        while True:
            click_zones, anchor_pos = render_panel_and_get_anchor()
            selected_action = None
            def mouse_cb(event, mx, my, flags, param):
                nonlocal selected_action
                if event == cv.EVENT_LBUTTONDOWN:
                    for key, (x1, y1, x2, y2) in click_zones.items():
                        if x1 <= mx <= x2 and y1 <= my <= y2:
                            selected_action = key; return
            cv.setMouseCallback(panel_window, mouse_cb)
            while selected_action is None:
                key = cv.waitKey(30)
                if key in (10, 13):
                    selected_action = "DONE"
                elif key in (27, ord('q'), ord('Q')):
                    try: cv.destroyAllWindows()
                    except Exception: pass
                    sys.exit(0)
            if selected_action == "DONE":
                cv.destroyWindow(panel_window)
                cv.destroyWindow(image_window)
                return None, anchor_pos
            # Find field schema
            f = next((fs for fs in fields if fs["name"] == selected_action), None)
            if not f:
                continue
            if f["type"] == "SELECT":
                sel = _get_generic_select_visual(img, f["name"], f.get("options", []), f.get("colors", {}), base_anchor=anchor_pos, panel_width_override=panel_width)
                if sel is not None:
                    _persistent_labels[f["name"]] = sel
            elif f["type"] == "NUMERIC":
                rng = _get_generic_range_visual(
                    img,
                    f["name"],
                    f.get("min", 1),
                    f.get("max", 9999),
                    image_window=image_window,
                    panel_pos=anchor_pos,
                    close_image_window=False,
                    max_panel_width=panel_width,
                )
                if rng is not None:
                    _persistent_labels[f["name"]] = rng
            elif f["type"] == "DATE":
                date_str = _get_date_visual(img, "", base_anchor=anchor_pos, panel_width_override=panel_width)
                if date_str is not None:
                    _persistent_labels[f["name"]] = date_str
            elif f["type"] == "TIME":
                tval = _get_time_visual(img, f["name"], base_anchor=anchor_pos, panel_width_override=panel_width)
                if tval is not None:
                    _persistent_labels[f["name"]] = tval
    except Exception as e:
        print(f"Error in dynamic settings update: {e}")
        try:
            cv.destroyWindow(panel_window); cv.destroyWindow(image_window)
        except Exception: pass
        return None, None
                            
# ----------------------------------------------------------------------------------------------------------------------------------
# 6) DEBUG/LOG CONTROLLER
# ----------------------------------------------------------------------------------------------------------------------------------
class Debugger:
    def __init__(self, mode=None, headless=None, save_dir=None, view_max=None, auto_save=None):
        self.mode = mode if mode is not None else CONFIG["RUN"]["TROUBLESHOOT_MODE"]
        self.headless = headless if headless is not None else CONFIG["RUN"]["HEADLESS"]
        self.save_dir = save_dir if save_dir is not None else None
        # Default debug view size follows REFERENCE_IMAGE_SIZE unless
        # explicitly overridden, keeping a single primary size setting.
        ui = CONFIG.get("UI", {})
        default_view = ui.get("REFERENCE_IMAGE_SIZE", (1200, 900))
        self.view_max = view_max if view_max is not None else default_view
        # Accept both SAVE_DEBUG_IMAGES (code default) and SAVE_DEBUG (user config)
        default_auto = CONFIG.get("RUN", {}).get("SAVE_DEBUG_IMAGES", CONFIG.get("RUN", {}).get("SAVE_DEBUG", False))
        self.auto_save = auto_save if auto_save is not None else default_auto

    def log(self, msg, level=2):
        if self.mode >= level:
            print(msg)

    def _fit(self, img):
        h, w = img.shape[:2]
        maxw, maxh = self.view_max
        # Never upscale debug views; clamp scale to <= 1.0
        s = min(maxw / max(w, 1), maxh / max(h, 1), 1.0)
        if abs(s - 1.0) > 1e-6:
            interp = cv.INTER_AREA if s < 1.0 else cv.INTER_LINEAR
            img = cv.resize(img, (max(1, int(w * s)), max(1, int(h * s))), interpolation=interp)
        return img

    def show(self, title, img, level=3, block=True, save_name=None):
        if self.mode < level:
            return
        img_disp = self._fit(img)
        if not self.headless:
            try:
                cv.namedWindow(title, cv.WINDOW_NORMAL)
                cv.imshow(title, img_disp)
                if block:
                    cv.waitKey(0)
                else:
                    cv.waitKey(1)
                cv.destroyWindow(title)
            except Exception:
                pass
        if (self.auto_save or self.headless) and save_name and self.save_dir:
            try:
                os.makedirs(self.save_dir, exist_ok=True)
                cv.imwrite(os.path.join(self.save_dir, save_name), img_disp)
            except Exception:
                pass

# ----------------------------------------------------------------------------------------------------------------------------------
# 7) MANUAL INPUT COORDINATION
# ----------------------------------------------------------------------------------------------------------------------------------
def manual_input_label_fields(image_path, debug: Optional[Debugger] = None):
    """Visual input using unified interfaces, fully driven by config-defined labels.
    When `RUN.LABEL_IMAGES` is False, parse labels from filename only. Otherwise, open the dynamic panel.
    """
    file_formats = CONFIG.get("FILE_FORMATS", {})
    filename = os.path.basename(image_path)
    original_stem = os.path.splitext(filename)[0]

    label_images_enabled = bool(CONFIG.get("RUN", {}).get("LABEL_IMAGES", True))
    if not label_images_enabled:
        parsed = parse_metadata_from_filename(filename)
        dyn = parsed.get("labels", {})
        # Determine date string for naming
        date_str = None
        if isinstance(dyn.get("Date"), str):
            date_str = normalize_date_string(dyn.get("Date"))
        else:
            df = next((f for f in CONFIG.get("SCHEMA_FIELDS", []) if f.get("type") == "DATE"), None)
            if df:
                v = dyn.get(df["name"]) or parsed.get("date")
                if isinstance(v, str):
                    date_str = normalize_date_string(v)
        # Build New_File using dynamic pattern
        New_File = None
        try:
            pattern = file_formats.get("OUTPUT_FILE_NAME_PATTERN", "{DATE}_{L1}")
            if date_str:
                New_File = build_filename_generic(date_str, dyn, pattern)
        except Exception:
            New_File = None
        result = dict(dyn)
        result["Date"] = result.get("Date") or parsed.get("date")
        result["Original_File"] = filename
        result["Original_Stem"] = original_stem
        result["raw_text"] = f"(Label_Images disabled) Parsed from filename: {filename}"
        result["New_File"] = New_File or filename
        return result

    # UI enabled path
    img = read_image_with_orientation(image_path)
    if img is None:
        return {"raw_text": "", "New_File": None}

    parsed = parse_metadata_from_filename(filename)
    # Seed dynamic labels from filename
    try:
        lbls = parsed.get("labels") or {}
        for k, v in lbls.items():
            _persistent_labels[k] = v
        if parsed.get("date"):
            date_field_name = next((f["name"] for f in CONFIG.get("SCHEMA_FIELDS", []) if f.get("type") == "DATE"), "Date")
            _persistent_labels[date_field_name] = parsed["date"]
    except Exception:
        pass

    _windows_closed, _anchor_pos = _check_update_persistent_data_dynamic(img)
    dyn = dict(_persistent_labels)

    # Determine date string for naming
    date_str = None
    if isinstance(dyn.get("Date"), str):
        date_str = normalize_date_string(dyn.get("Date"))
    else:
        df = next((f for f in CONFIG.get("SCHEMA_FIELDS", []) if f.get("type") == "DATE"), None)
        if df:
            v = dyn.get(df["name"]) or parsed.get("date")
            if isinstance(v, str):
                date_str = normalize_date_string(v)

    # Build New_File using dynamic pattern
    New_File = None
    try:
        pattern = file_formats.get("OUTPUT_FILE_NAME_PATTERN", "{DATE}_{L1}")
        if date_str:
            New_File = build_filename_generic(date_str, dyn, pattern)
    except Exception:
        New_File = None

    result = dict(dyn)
    result["Date"] = result.get("Date") or parsed.get("date")
    result["Original_File"] = filename
    result["Original_Stem"] = original_stem
    result["raw_text"] = f"Labels: {dyn}"
    result["New_File"] = New_File
    if debug:
        debug.log(f"Visual input (dynamic) result: {result}", level=2)
    return result

def generate_New_File(date_str, labels_dict):
    """Generate filename using OUTPUT_FILE_NAME_PATTERN with dynamic labels only."""
    if date_str is None:
        return None
    file_formats = CONFIG.get("FILE_FORMATS", {})
    base_pattern = file_formats.get("OUTPUT_FILE_NAME_PATTERN", "{DATE}")
    label_info = dict(labels_dict or {})
    return build_filename_generic(date_str, label_info, base_pattern)

def parse_metadata_from_filename(file_name):
    """
    Parse metadata from filename using config-driven pattern first.
    1) Try CONFIG['FILE_FORMATS']['INPUT_FILENAME_PATTERN'] (supports tokens like {DATE}, {L1}..{Ln}).
    Returns meta dict with canonical fields plus any dynamic label mapping when pattern succeeds.
    """
    # Pattern-driven path
    try:
        file_formats = CONFIG.get("FILE_FORMATS", {}) or {}
        pattern = file_formats.get("INPUT_FILENAME_PATTERN")
        labels_conf = CONFIG.get("INPUTS", {}).get("LABELS") or CONFIG.get("INPUTS", {}).get("NAMES") or []
        if isinstance(pattern, str) and pattern.strip():
            base = os.path.basename(file_name)
            stem = os.path.splitext(base)[0]
            # Extract tokens between braces in order
            tokens = []
            buf = ""; in_brace = False
            for ch in pattern:
                if ch == '{':
                    in_brace = True; buf = ""; continue
                if ch == '}':
                    in_brace = False; tokens.append(buf.strip()); continue
                if in_brace:
                    buf += ch

            # Helper: group pattern per token based on INPUTS schema
            inputs_cfg = CONFIG.get("INPUTS", {}) or {}
            labels_conf = inputs_cfg.get("LABELS") or inputs_cfg.get("NAMES") or []

            def _get_group_pattern(tok):
                t = tok.strip()
                # Explicit DATE token: allow various separators, normalize later
                if t.upper() == "DATE":
                    return r"([0-9]{2,4}[\-_.\/][0-9]{1,2}[\-_.\/][0-9]{1,2})"
                # Ln tokens: inspect schema
                m = re.fullmatch(r"L(\d+)", t)
                if m:
                    n = int(m.group(1))
                    name = labels_conf[n-1] if 1 <= n <= len(labels_conf) else None
                    fcfg = inputs_cfg.get(name) or inputs_cfg.get(f"L{n}")
                    if isinstance(fcfg, (list, tuple)) and len(fcfg) > 0:
                        typ = str(fcfg[0]).strip().upper()
                        if typ == "SELECT":
                            opts = [str(o) for o in fcfg[1:]]
                            if opts:
                                alts = "|".join([re.escape(o) for o in opts])
                                return f"({alts})"
                        if typ == "NUMERIC":
                            return r"(\d+)"
                        if typ == "DATE":
                            return r"([0-9]{2,4}[\-_.\/][0-9]{1,2}[\-_.\/][0-9]{1,2})"
                    # Fallback: non-greedy until separator
                    return r"(.+?)"
                # Named tokens (treat as generic label strings)
                return r"(.+?)"

            # Build regex: replace each {token} with a typed group; escape other chars
            regex_parts = []
            i = 0
            while i < len(pattern):
                if pattern[i] == '{':
                    j = pattern.find('}', i + 1)
                    if j == -1:
                        break
                    tok = pattern[i+1:j]
                    regex_parts.append(_get_group_pattern(tok))
                    i = j + 1
                else:
                    regex_parts.append(re.escape(pattern[i]))
                    i += 1
            rx = re.compile("^" + "".join(regex_parts) + "$")
            m = rx.match(base)
            if not m:
                m = rx.match(stem)
            if m:
                groups = list(m.groups())
                dyn = {}
                meta = {"date": None, "labels": dyn}
                for idx, tok in enumerate(tokens):
                    val = groups[idx] if idx < len(groups) else None
                    if val is None:
                        continue
                    t = tok.strip()
                    # Map special tokens
                    if t.upper() == "DATE":
                        meta["date"] = normalize_date_string(val)
                    elif re.fullmatch(r"L\d+", t):
                        # Map Ln to label name if available
                        try:
                            n = int(t[1:])
                            if 1 <= n <= len(labels_conf):
                                dyn[labels_conf[n-1]] = val
                            else:
                                dyn[t] = val
                        except Exception:
                            dyn[t] = val
                    else:
                        # Treat any non-DATE token as a dynamic label name from the pattern
                        dyn[t] = val
                # If canonical date is still unset, derive it from the configured Date label (L1 mapping)
                if not meta.get("date"):
                    date_label_name = labels_conf[0] if labels_conf else None
                    if isinstance(date_label_name, str):
                        date_val = dyn.get(date_label_name)
                        if isinstance(date_val, str) and date_val.strip():
                            formats = []
                            def _add_fmt(f):
                                if isinstance(f, str):
                                    formats.append(f)
                                elif isinstance(f, (list, tuple)):
                                    formats.extend(list(f))
                            _add_fmt(file_formats.get("INPUT_DATE_FORMAT"))
                            _add_fmt(file_formats.get("INPUT_DATE_FORMATS"))
                            naming = CONFIG.get("NAMING", {}) or {}
                            _add_fmt(naming.get("INPUT_DATE_FORMAT"))
                            _add_fmt(naming.get("INPUT_DATE_FORMATS"))
                            if not formats:
                                formats = ["YYYY_MM_DD", "YY_MM_DD", "MM.DD.YYYY", "MM.DD.YY"]
                            meta["date"] = parse_date_token_with_formats(date_val, formats)
                return meta
    except Exception:
        pass

    meta: dict = {"date": None, "labels": {}}
    try:
        base = os.path.basename(file_name)
        stem, ext = os.path.splitext(base)
        # Validate extension (optional, since we already have a filename)
        ext_ok = ext.lower() in (".jpg", ".jpeg", ".png", ".tif", ".tiff")
        # Fallback: treat entire stem as loose input for date parsing only
        date_raw = stem

        # Try multiple date formats for date_raw
        parsed_date = None
        # 1) MM.DD.YY or MM.DD.YYYY (dots)
        m = re.fullmatch(r'\s*(\d{1,2})\.(\d{1,2})\.(\d{2}|\d{4})\s*', date_raw)
        if m:
            mm = int(m.group(1)); dd = int(m.group(2)); yy = m.group(3)
            yyyy = int(yy) if len(yy) == 4 else 2000 + int(yy)
            try:
                dt.datetime(yyyy, mm, dd)
                parsed_date = f"{yyyy:04d}_{mm:02d}_{dd:02d}"
            except Exception:
                parsed_date = None

        # 2) YYYY_MM_DD or YYYY-MM-DD
        if parsed_date is None:
            m2 = re.fullmatch(r"\s*(\d{4})[_\-](\d{1,2})[_\-](\d{1,2})\s*", date_raw)
            if m2:
                yyyy = int(m2.group(1)); mm = int(m2.group(2)); dd = int(m2.group(3))
                try:
                    dt.datetime(yyyy, mm, dd)
                    parsed_date = f"{yyyy:04d}_{mm:02d}_{dd:02d}"
                except Exception:
                    parsed_date = None
        # 2b) Search anywhere for YYYY[_-]MM[_-]DD (handles extra tokens after date)
        if parsed_date is None:
            m2b = re.search(r"(\d{4})[_\-](\d{1,2})[_\-](\d{1,2})", date_raw)
            if m2b:
                yyyy = int(m2b.group(1)); mm = int(m2b.group(2)); dd = int(m2b.group(3))
                try:
                    dt.datetime(yyyy, mm, dd)
                    parsed_date = f"{yyyy:04d}_{mm:02d}_{dd:02d}"
                except Exception:
                    parsed_date = None

        # 3) Fallback: try to find a date anywhere in date_raw using loose regex
        if parsed_date is None:
            m3 = re.search(r'(\d{1,2})[.\-_/](\d{1,2})[.\-_/](\d{2,4})', date_raw)
            if m3:
                mm = int(m3.group(1)); dd = int(m3.group(2)); yy = m3.group(3)
                yyyy = int(yy) if len(yy) == 4 else 2000 + int(yy)
                try:
                    dt.datetime(yyyy, mm, dd)
                    parsed_date = f"{yyyy:04d}_{mm:02d}_{dd:02d}"
                except Exception:
                    parsed_date = None

        if parsed_date:
            meta["date"] = parsed_date
    except Exception:
        # Continue to loose inference below
        pass

    return meta

# ----------------------------------------------------------------------------------------------------------------------------------
# 8) CALIBRATION FUNCTIONS
# ----------------------------------------------------------------------------------------------------------------------------------
def manual_calibration(img, debug: Optional[Debugger] = None, **kwargs):
    """Manual calibration using CONFIG settings with optional overrides."""
    ui = CONFIG.get("UI", {})
    # Prefer a single base size driven by REFERENCE_IMAGE_SIZE and
    # clamped to the screen, shared with other reference windows.
    base_view = _get_reference_max_size()
    view_max = kwargs.get('view_max', base_view)
    # Tie the default zoomed calibration width to the same base size so
    # the zoomed-in window scales consistently with the reference image.
    default_zoom_w = base_view[0]
    desired_zoom_width = kwargs.get('desired_zoom_width', default_zoom_w)
    # Use internal sane defaults for zoom bounds; not user-configurable.
    min_zoom = kwargs.get('min_zoom', 2)
    max_zoom = kwargs.get('max_zoom', 8)
    calib_cfg = CONFIG.get("CALIB", {})
    bar_len_cm = float(calib_cfg.get("BAR_LENGTH_CM", calib_cfg.get("MANUAL_CALIBRATION_CM", 5.0)))
    exclude_pad_cm = float(calib_cfg.get("EXCLUDE_PAD_CM", 0.0))

    H, W = img.shape[:2]
    sx = view_max[0] / max(W, 1)
    sy = view_max[1] / max(H, 1)
    # Never upscale calibration images; clamp scale to <= 1.0
    s = min(sx, sy, 1.0)
    interp = cv.INTER_LINEAR if s > 1.0 else cv.INTER_AREA
    disp = cv.resize(img, (max(1, int(W * s)), max(1, int(H * s))), interpolation=interp)

    win1 = "Select calibration card (scaled)"
    try:
        cv.namedWindow(win1, cv.WINDOW_NORMAL)
        cv.imshow(win1, disp)
        cv.resizeWindow(win1, disp.shape[1], disp.shape[0])
        cv.moveWindow(win1, *CONFIG["UI"]["REFERENCE_WINDOW_POSITION"])
        cv.waitKey(1)
        r = custom_roi_selection(win1, disp, "Click & drag box around calibration card or ruler, then press ENTER")
    except Exception as e:
        if debug: debug.log(f"manual_calibration ROI setup: {e}", level=1)
        try: cv.destroyWindow(win1)
        except Exception: pass
        return None, None, None
    finally:
        try: cv.destroyWindow(win1)
        except Exception: pass

    x_s, y_s, w_s, h_s = map(int, r)
    if w_s <= 0 or h_s <= 0:
        return None, None, None

    x = int(round(x_s / max(s, 1e-6))); y = int(round(y_s / max(s, 1e-6)))
    w = int(round(w_s / max(s, 1e-6))); h = int(round(h_s / max(s, 1e-6)))
    x = max(0, min(x, W - 1)); y = max(0, min(y, H - 1))
    w = max(1, min(w, W - x)); h = max(1, min(h, H - y))
    roi = img[y:y + h, x:x + w]
    if roi.size == 0:
        return None, None, None

    if desired_zoom_width:
        # Choose a zoom factor based on the *original* ROI width so that
        # the drawn zoomed image is close to the desired width driven by
        # REFERENCE_IMAGE_SIZE, but do not let this depend on subsequent
        # window shrinking. This keeps instruction text sizing stable.
        z = int(np.clip(np.ceil(desired_zoom_width / max(1, w)), min_zoom, max_zoom))
    else:
        z = min_zoom

    def make_zoomed(zf):
        return cv.resize(roi, (roi.shape[1] * zf, roi.shape[0] * zf), interpolation=cv.INTER_NEAREST)

    clicks = []
    win2 = "Zoomed card (click two ends)"
    try:
        zoomed = make_zoomed(z)
        # Fit zoomed image to the same reference window size (no window scaling)
        maxw, maxh = _get_reference_max_size()
        zw, zh = zoomed.shape[1], zoomed.shape[0]
        display_scale = min(maxw / float(max(1, zw)), maxh / float(max(1, zh)), 1.0)
        if display_scale < 1.0:
            zoomed_display = cv.resize(
                zoomed,
                (int(zw * display_scale), int(zh * display_scale)),
                interpolation=cv.INTER_AREA,
            )
        else:
            zoomed_display = zoomed
        cv.namedWindow(win2, cv.WINDOW_NORMAL)
        # Use CONFIG-driven instruction styling on the reference-scaled ROI (not the zoomed version)
        # to get consistent text sizing with other reference windows
        t_scale, t_thick, s_scale, s_thick = _instruction_style(roi)
        zoomed_instr = zoomed_display.copy()
        # Centralized font face
        _font_face_cfg = _get_font_face()
        # Add black background rectangles for better readability
        main_text = f"Click two ends of the {bar_len_cm:g} cm ruler"
        (tw1, th1), _ = cv.getTextSize(main_text, _font_face_cfg, t_scale, t_thick)
        cv.rectangle(zoomed_instr, (8, 30 - th1 - 2), (12 + tw1, 32), (0, 0, 0), -1)
        cv.putText(zoomed_instr, main_text,
                   (10, 30), _font_face_cfg, t_scale, (255, 255, 255), t_thick, cv.LINE_AA)
        sub_text = "ENTER=Accept (2 points), U=Undo, ESC/Q=Quit"
        (tw2, th2), _ = cv.getTextSize(sub_text, _font_face_cfg, s_scale, s_thick)
        cv.rectangle(zoomed_instr, (8, 55 - th2 - 2), (12 + tw2, 57), (0, 0, 0), -1)
        cv.putText(zoomed_instr, sub_text,
                   (10, 55), _font_face_cfg, s_scale, (255, 255, 255), s_thick, cv.LINE_AA)
        cv.imshow(win2, zoomed_instr)
        cv.resizeWindow(win2, zoomed_display.shape[1], zoomed_display.shape[0])
        cv.moveWindow(win2, *CONFIG["UI"]["REFERENCE_WINDOW_POSITION"])
        cv.waitKey(1)
    except Exception as e:
        if debug: debug.log(f"manual_calibration zoom setup: {e}", level=1)
        try: cv.destroyWindow(win2)
        except Exception: pass
        return None, None, None

    def redraw_zoomed():
        base = zoomed_display.copy()
        # Reuse the same CONFIG-driven instruction styling on reference-scale ROI.
        ts, tt, ss, st = _instruction_style(roi)
        main_text = "2 points set - ENTER to accept, U to adjust" if len(clicks) == 2 else f"Click two ends of the {bar_len_cm:g} cm ruler"
        # Add black background rectangles for better readability
        (tw1, th1), _ = cv.getTextSize(main_text, _font_face_cfg, ts, tt)
        cv.rectangle(base, (8, 30 - th1 - 2), (12 + tw1, 32), (0, 0, 0), -1)
        cv.putText(base, main_text, (10, 30), _font_face_cfg, ts, (255, 255, 255), tt, cv.LINE_AA)
        sub_text = "ENTER=Accept (2 points), U=Undo, ESC/Q=Quit"
        (tw2, th2), _ = cv.getTextSize(sub_text, _font_face_cfg, ss, st)
        cv.rectangle(base, (8, 55 - th2 - 2), (12 + tw2, 57), (0, 0, 0), -1)
        cv.putText(base, sub_text, (10, 55), _font_face_cfg, ss, (255, 255, 255), st, cv.LINE_AA)
        for (mx, my) in clicks:
            cv.circle(base, (mx, my), 12, (255, 255, 0), -1)
            cv.circle(base, (mx, my), 12, (0, 0, 0), 2)
        if len(clicks) >= 2:
            cv.line(base, clicks[0], clicks[1], (255, 255, 0), 2)
        cv.imshow(win2, base)
    def on_mouse(event, mx, my, flags, param):
        nonlocal clicks
        if event == cv.EVENT_LBUTTONDOWN and len(clicks) < 2:
            clicks.append((mx, my))
            redraw_zoomed()

    cv.setMouseCallback(win2, on_mouse)
    redraw_zoomed()

    while True:
        key = cv.waitKey(10)
        if key in (ord('u'), ord('U')):
            if clicks:
                clicks.pop()
                redraw_zoomed()
        elif key in (10, 13):
            if len(clicks) == 2:
                scale_denom = max(z * display_scale, 1e-6)
                p1 = np.array(clicks[0], np.float32) / scale_denom
                p2 = np.array(clicks[1], np.float32) / scale_denom
                dist_px = float(np.linalg.norm(p1 - p2))
                Pixel_ratio = dist_px / max(bar_len_cm, 1e-6)

                full_image_clicks = [(int(p1[0] + x), int(p1[1] + y)), (int(p2[0] + x), int(p2[1] + y))]

                pad_px = int(round(exclude_pad_cm * Pixel_ratio))
                x0 = max(0, x - pad_px); y0 = max(0, y - pad_px)
                x1 = min(W, x + w + pad_px); y1 = min(H, y + h + pad_px)
                exclude_mask = np.zeros((H, W), np.uint8)
                cv.rectangle(exclude_mask, (x0, y0), (x1, y1), 255, -1)
                try: cv.destroyWindow(win2)
                except Exception: pass
                return Pixel_ratio, exclude_mask, full_image_clicks
            else:
                redraw_zoomed()
        elif key in (ord('q'), 27):
            if debug: debug.log("User requested exit by q or ESC.", level=1)
            try: cv.destroyWindow(win2)
            except Exception: pass
            sys.exit(0)        

def manual_label_exclusion(img, debug: Optional[Debugger] = None, force_select: bool = False, calib_exclude_mask=None, qr_detected=False):
    """Enhanced label exclusion using CONFIG settings - supports multiple areas."""
    if not force_select and not CONFIG["RUN"].get("MANUAL_LABEL_EXCLUSION", False):
        return None
    
    H, W = img.shape[:2]
    # Use the same helper as other reference windows to get clamped reference size
    base_w, base_h = _get_reference_max_size()
    view_max = (base_w, base_h)

   # Scale image for selection (limit to reference max, don't upscale)
    sx = view_max[0] / max(W, 1)
    sy = view_max[1] / max(H, 1)
    s = min(sx, sy, 1.0)  

    interp = cv.INTER_LINEAR if s > 1.0 else cv.INTER_AREA
    scaled_img = cv.resize(img, (max(1, int(W * s)), max(1, int(H * s))), interpolation=interp)

    # Show any existing calibration exclusion (e.g., QR code areas) on the image
    # Show any existing calibration exclusion (e.g., QR code areas) on the image
    if 'calib_exclude_mask' in locals() and calib_exclude_mask is not None:
        # Scale the calibration mask to match the scaled image
        scaled_calib_mask = cv.resize(calib_exclude_mask, (scaled_img.shape[1], scaled_img.shape[0]), interpolation=cv.INTER_NEAREST)
        
        # Create a dark overlay for excluded areas
        overlay = scaled_img.copy()
        overlay[scaled_calib_mask > 0] = overlay[scaled_calib_mask > 0] * 0.3  # Darken excluded areas
        
        # Add a GREEN border around excluded areas for clarity (instead of cyan)
        contours, _ = cv.findContours(scaled_calib_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            cv.drawContours(overlay, [contour], -1, (0, 255, 0), 2)  # Green border instead of cyan
            
            # Label with "QR" instead of sequential numbers
            if len(contours) > 0:
                x, y, w, h = cv.boundingRect(contour)

                if qr_detected == True:
                    # Get bounding box for text placement
                    cv.putText(overlay, "QR", (x + 5, y + 18), 
                            cv.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1, cv.LINE_AA)  # White "QR" label
                    cv.putText(overlay, "Calibration", (x + 5, y + 38), 
                            cv.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1, cv.LINE_AA)  # White "Calibration" label
                    cv.putText(overlay, "Successful", (x + 5, y + 58), 
                            cv.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1, cv.LINE_AA)  # White "Successful" label
                else:
                    # Get bounding box for text placement
                    cv.putText(overlay, "Manual", (x + 5, y + 18), 
                            cv.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1, cv.LINE_AA)  # White "Manual" label
                    cv.putText(overlay, "Calibration", (x + 5, y + 38), 
                            cv.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1, cv.LINE_AA)  # White "Calibration" label
                    cv.putText(overlay, "Successful", (x + 5, y + 58), 
                            cv.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1, cv.LINE_AA)  # White "Successful   " label
        
        scaled_img = overlay

   # Add padding for easier edge selection with checkerboard pattern
    edge_padding = 25

    # Create checkerboard pattern for transparency effect
    def create_checkerboard_border(img, padding, checker_size=8):
        h, w = img.shape[:2]
        padded_h, padded_w = h + 2*padding, w + 2*padding
        
        # Create checkerboard pattern
        checkerboard = np.zeros((padded_h, padded_w, 3), dtype=np.uint8)
        
        for y in range(0, padded_h, checker_size):
            for x in range(0, padded_w, checker_size):
                # Alternate between light gray and darker gray
                if (y // checker_size + x // checker_size) % 2 == 0:
                    color = (200, 200, 200)  # Light gray
                else:
                    color = (255, 255, 255)  # White
                
                checkerboard[y:y+checker_size, x:x+checker_size] = color
        
        # Place original image in center
        checkerboard[padding:padding+h, padding:padding+w] = img
        
        return checkerboard

    disp = create_checkerboard_border(scaled_img, edge_padding)

    # Store the original scaled dimensions and padding for coordinate mapping
    scaled_w, scaled_h = scaled_img.shape[1], scaled_img.shape[0]
    
    win_name = "Click & drag a box around any area(s) to be excluded (MULTIPLE)"
    try:
        cv.namedWindow(win_name, cv.WINDOW_NORMAL)
        cv.imshow(win_name, disp)
        # Enforce the reference window size explicitly
        cv.resizeWindow(win_name, disp.shape[1], disp.shape[0])
        cv.moveWindow(win_name, *CONFIG["UI"]["REFERENCE_WINDOW_POSITION"])
        cv.waitKey(1)
        
        if debug:
            debug.log("Click & drag a box around any area(s) to exclude from analysis", level=2)
        
        # Use multi-selection function
        rects = custom_multi_roi_selection(win_name, disp, "Select area(s) to be excluded from analysis") 
        cv.destroyWindow(win_name)
    except Exception as e:
        if debug:
            debug.log(f"Error in label exclusion selection: {e}", level=2)
        try:
            cv.destroyWindow(win_name)
        except:
            pass
        return None
    
    if not rects:
        if debug:
            debug.log("Label exclusion cancelled by user", level=2)
        return None

    # Create combined exclusion mask from all selected rectangles
    exclusion_mask = np.zeros((H, W), np.uint8)  

    for i, r in enumerate(rects):
        x_s, y_s, w_s, h_s = map(int, r)
        if w_s <= 0 or h_s <= 0:
            continue

        # Account for padding when mapping coordinates
        x_s -= edge_padding
        y_s -= edge_padding
        
        # Clamp selection to original image bounds in scaled space
        x_s = max(0, min(x_s, scaled_w))
        y_s = max(0, min(y_s, scaled_h))
        w_s = max(1, min(w_s, scaled_w - x_s))
        h_s = max(1, min(h_s, scaled_h - y_s))
        
        # Map back to full resolution
        x = int(round(x_s / max(s, 1e-6)))
        y = int(round(y_s / max(s, 1e-6)))
        w = int(round(w_s / max(s, 1e-6)))
        h = int(round(h_s / max(s, 1e-6)))

        # Final clamp to image bounds (this now handles edge cases perfectly)
        x = max(0, min(x, W - 1))
        y = max(0, min(y, H - 1))
        w = max(1, min(w, W - x))
        h = max(1, min(h, H - y))

        # Add this rectangle to the exclusion mask
        cv.rectangle(exclusion_mask, (x, y), (x + w, y + h), 255, -1)  
        
        if debug:
            debug.log(f"Added exclusion area {i+1}: ({x}, {y}, {w}, {h})", level=2)
    
    if debug:
        debug.log(f"Created combined exclusion mask with {len(rects)} areas", level=2)
        # debug.show("Combined label exclusion mask", exclusion_mask, level=3, block=False, save_name="label_exclusion_mask.jpg")
    
    return exclusion_mask

# ----------------------------------------------------------------------------------------------------------------------------------
# 9) SEGMENTATION FUNCTIONS
# ----------------------------------------------------------------------------------------------------------------------------------
def lab_green_gate(img, base_mask=None):
    """Return 0/255 mask keeping only 'green enough' pixels in Lab."""
    lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)
    a8 = lab[:, :, 1].astype(np.int16)
    b8 = lab[:, :, 2].astype(np.int16)
    a_signed = a8 - 128
    b_signed = b8 - 128
    
    a_keep = (a_signed <= CONFIG["SEG"]["LAB_A_MAX"])
    if CONFIG["SEG"]["LAB_B_MAX"] is not None:
        b_keep = (b_signed <= CONFIG["SEG"]["LAB_B_MAX"])
        keep = np.logical_and(a_keep, b_keep)
    else:
        keep = a_keep
    
    gate = (keep.astype(np.uint8) * 255)
    if base_mask is not None:
        gate = cv.bitwise_and(gate, base_mask)
    
    return gate

def fill_holes(binary_mask):
    """Fill enclosed holes so leaves are solid shapes."""
    m = binary_mask.copy()
    h, w = m.shape[:2]
    ff = m.copy()
    mask = np.zeros((h + 2, w + 2), np.uint8)
    cv.floodFill(ff, mask, (0, 0), 255)
    holes = cv.bitwise_not(ff)
    return cv.bitwise_or(m, holes)

def compute_leaf_length_width(contour):
    """
    Compute leaf length and width from a contour (Nx1x2 array):

      - length_px: longest distance between any two contour points
      - width_px: maximum distance between contour points along a direction
                  perpendicular to the length vector
      - length_segment: (p1, p2) endpoints of the length line in full-image coords
      - width_segment: (w1, w2) endpoints of the width line in full-image coords

    All points are (x, y) float32.
    """
    pts = contour.reshape(-1, 2).astype(np.float32)
    n = pts.shape[0]
    if n < 2:
        return 0.0, 0.0, None, None

    # --- Find the longest distance between any two contour points (length) ---
    # This is O(n^2) over contour points; contours are usually modest, so OK.
    max_d2 = 0.0
    max_i, max_j = 0, 1
    for i in range(n):
        pi = pts[i]
        diff = pts[i+1:] - pi
        if diff.size == 0:
            continue
        d2 = np.sum(diff * diff, axis=1)
        j_rel = np.argmax(d2)
        if d2[j_rel] > max_d2:
            max_d2 = d2[j_rel]
            max_i = i
            max_j = i + 1 + j_rel

    p1 = pts[max_i]
    p2 = pts[max_j]
    length_px = float(np.sqrt(max_d2))
    if length_px < 1e-6:
        return 0.0, 0.0, None, None

    # --- Compute width as max spread perpendicular to the length direction ---
    v = p2 - p1
    v_unit = v / np.linalg.norm(v)

    # Perpendicular unit vector (rotate by +90 degrees)
    perp = np.array([-v_unit[1], v_unit[0]], dtype=np.float32)

    # Project all points onto perpendicular direction relative to mid-point of length
    mid = 0.5 * (p1 + p2)
    rel = pts - mid
    proj = rel @ perp  # dot product

    proj_min = float(np.min(proj))
    proj_max = float(np.max(proj))
    width_px = proj_max - proj_min

    # Endpoints of width segment in image coords
    w1 = mid + perp * proj_min
    w2 = mid + perp * proj_max

    length_segment = (tuple(p1), tuple(p2))
    width_segment = (tuple(w1), tuple(w2))

    return length_px, width_px, length_segment, width_segment

def annotate_components_with_length_width(mask_full, components, debug: Optional[Debugger] = None):
    """
    For each component in 'components', compute:
      - length_px, width_px
      - length_segment: ((x1, y1), (x2, y2))
      - width_segment:  ((x1, y1), (x2, y2))
    using the component's contour in the full-resolution mask.

    Modifies the component dicts in place.
    """
    if not components:
        return

    # Label the full mask once
    num_labels, labels, stats, _ = cv.connectedComponentsWithStats(mask_full, connectivity=8)

    for comp_idx, comp in enumerate(components):
        x, y, w, h = comp["bbox"]

        # Use labels to isolate connected region inside bbox
        # Assumption: components were derived from this same mask_full
        roi_labels = labels[y:y+h, x:x+w]
        roi_mask = np.zeros((h, w), dtype=np.uint8)
        roi_mask[roi_labels > 0] = 255  # all non-zero labels in this bbox

        # Intersect with original mask to be safe
        roi_mask = cv.bitwise_and(roi_mask, mask_full[y:y+h, x:x+w])

        contours, _ = cv.findContours(roi_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        if not contours:
            comp["length_px"] = 0.0
            comp["width_px"] = 0.0
            comp["length_segment"] = None
            comp["width_segment"] = None
            continue

        # Take largest contour
        contour = max(contours, key=cv.contourArea)

        # Convert contour back to full-image coordinates
        contour_full = contour + np.array([[x, y]], dtype=np.int32)

        length_px, width_px, length_seg, width_seg = compute_leaf_length_width(contour_full)

        comp["length_px"] = length_px
        comp["width_px"] = width_px
        comp["length_segment"] = length_seg  # ((x1,y1),(x2,y2))
        comp["width_segment"] = width_seg    # ((x1,y1),(x2,y2))

def segment_leaves(img, Pixel_ratio, exclude_mask=None, debug: Optional[Debugger] = None, **kwargs):
    """
    Improved HSV segmentation with better handling of dark greens and white stems.
    """
    H, W = img.shape[:2]
    try:
        assert Pixel_ratio is not None and Pixel_ratio > 0, "Calibration required"
        seg = CONFIG.get("SEG", {}) or {}
        
        # ---- Speed knobs ----
        FAST_MAX_WIDTH = int(seg.get("FAST_MAX_WIDTH", 1600))
        SKIP_MEDIAN = bool(seg.get("SKIP_MEDIAN", True))
        
        # ---- Downscale for speed (moved earlier) ----
        scale = 1.0
        if FAST_MAX_WIDTH > 0 and W > FAST_MAX_WIDTH:
            scale = FAST_MAX_WIDTH / float(W)
        
        if scale < 1.0:
            wS, hS = int(round(W * scale)), int(round(H * scale))
            imgS = cv.resize(img, (wS, hS), interpolation=cv.INTER_AREA)
            excludeS = None
            if exclude_mask is not None:
                excludeS = cv.resize(exclude_mask, (wS, hS), interpolation=cv.INTER_NEAREST)
            px_per_cm = float(Pixel_ratio) * scale
        else:
            imgS = img
            excludeS = exclude_mask
            px_per_cm = float(Pixel_ratio)
        
        px_per_cm2 = px_per_cm ** 2
        
        # ---- BACKGROUND-ADAPTIVE HSV THRESHOLDS ----
        # Built-in background presets
        BLACK_BACKGROUND_PRESET = {
            "BLACK_V_MAX": 25,
            "WHITE_S_MAX": 45,
            "WHITE_V_MIN": 180,
            "HSV_S_MIN": 55,
            "HSV_V_MIN": 70,
            "LAB_NEUTRAL_A_ABS_MAX": 2,
            "LAB_NEUTRAL_B_ABS_MAX": 2
        }
        WHITE_BACKGROUND_PRESET = {
            "BLACK_V_MAX": 45,
            "WHITE_S_MAX": 20,
            "WHITE_V_MIN": 220,
            "HSV_S_MIN": 35,
            "HSV_V_MIN": 45,
            "LAB_NEUTRAL_A_ABS_MAX": 4,
            "LAB_NEUTRAL_B_ABS_MAX": 4
        }
        
        def detect_background_type(img_sample):
            """Auto-detect if background is predominantly black or white."""
            # Sample corners and edges to determine background
            h, w = img_sample.shape[:2]
            corners = [
                img_sample[0:h//4, 0:w//4],           # top-left
                img_sample[0:h//4, 3*w//4:w],         # top-right  
                img_sample[3*h//4:h, 0:w//4],         # bottom-left
                img_sample[3*h//4:h, 3*w//4:w]        # bottom-right
            ]
            
            corner_means = [np.mean(corner) for corner in corners]
            avg_corner_brightness = np.mean(corner_means)
            
            # Threshold: < 80 = dark/black, > 175 = bright/white
            if avg_corner_brightness < 80:
                return "BLACK"
            elif avg_corner_brightness > 175:
                return "WHITE"
            else:
                return "MIXED"
        
        background_color = seg.get("BACKGROUND_COLOR", "AUTO")
        # Auto-detect background if needed
        if background_color == "AUTO":
            background_color = detect_background_type(imgS)
        
        # Apply appropriate preset
        if background_color == "BLACK":
            preset = BLACK_BACKGROUND_PRESET
        elif background_color == "WHITE":
            preset = WHITE_BACKGROUND_PRESET
        else:
            preset = {}  # Use config defaults for mixed/unknown backgrounds
        
        # Apply preset values with config fallbacks (min/max pairs)
        HSV_S_MIN = int(preset.get("HSV_S_MIN", seg.get("HSV_S_MIN", 30)))
        HSV_S_MAX = int(preset.get("HSV_S_MAX", seg.get("HSV_S_MAX", 255)))
        HSV_V_MIN = int(preset.get("HSV_V_MIN", seg.get("HSV_V_MIN", 50)))
        HSV_V_MAX = int(preset.get("HSV_V_MAX", seg.get("HSV_V_MAX", 255)))
        BLACK_V_MIN = int(preset.get("BLACK_V_MIN", seg.get("BLACK_V_MIN", 0)))
        BLACK_V_MAX = int(preset.get("BLACK_V_MAX", seg.get("BLACK_V_MAX", 35)))
        WHITE_S_MIN = int(preset.get("WHITE_S_MIN", seg.get("WHITE_S_MIN", 0)))
        WHITE_S_MAX = int(preset.get("WHITE_S_MAX", seg.get("WHITE_S_MAX", 35)))
        WHITE_V_MIN = int(preset.get("WHITE_V_MIN", seg.get("WHITE_V_MIN", 200)))
        WHITE_V_MAX = int(preset.get("WHITE_V_MAX", seg.get("WHITE_V_MAX", 255)))
        
        # Apply Lab settings from preset if available
        if "LAB_NEUTRAL_A_ABS_MAX" in preset:
            LAB_NEUTRAL_A_ABS_MAX = int(preset["LAB_NEUTRAL_A_ABS_MAX"])
        else:
            LAB_NEUTRAL_A_ABS_MAX = int(seg.get("LAB_NEUTRAL_A_ABS_MAX", 3))
            
        if "LAB_NEUTRAL_B_ABS_MAX" in preset:
            LAB_NEUTRAL_B_ABS_MAX = int(preset["LAB_NEUTRAL_B_ABS_MAX"])
        else:
            LAB_NEUTRAL_B_ABS_MAX = int(seg.get("LAB_NEUTRAL_B_ABS_MAX", 3))
        
        BLUE_EXCLUDE = bool(seg.get("BLUE_EXCLUDE", True))
        BLUE_H_MIN = int(seg.get("BLUE_H_MIN", 90))
        BLUE_H_MAX = int(seg.get("BLUE_H_MAX", 130))
        
        # ---- Expanded hue bands ----
        HSV_USE_BANDS = bool(seg.get("HSV_USE_BANDS", True))
        GREEN_H_MIN, GREEN_H_MAX = int(seg.get("GREEN_H_MIN", 30)), int(seg.get("GREEN_H_MAX", 90))
        YELLOW_H_MIN, YELLOW_H_MAX = int(seg.get("YELLOW_H_MIN", 18)), int(seg.get("YELLOW_H_MAX", 48))
        BROWN_H_MIN, BROWN_H_MAX = int(seg.get("BROWN_H_MIN", 8)), int(seg.get("BROWN_H_MAX", 28))
        PURPLE_H_MIN, PURPLE_H_MAX = int(seg.get("PURPLE_H_MIN", 110)), int(seg.get("PURPLE_H_MAX", 160))
        
        # ---- Lab color space gating ----
        LAB_GATE = bool(seg.get("LAB_GATE", True))
        LAB_A_MAX = int(seg.get("LAB_A_MAX", 20))
        LAB_B_MAX = int(seg.get("LAB_B_MAX", 160))
        
        # ---- Morphology parameters ----
        OPEN_DIAMETER_CM = float(seg.get("OPEN_DIAMETER_CM", 0.045))
        CLOSE_DIAMETER_CM = float(seg.get("CLOSE_DIAMETER_CM", 0.015))
        CLOSE_BRIDGE_CM = float(seg.get("CLOSE_BRIDGE_CM", 0.015))
        REOPEN_CM = float(seg.get("REOPEN_CM", 0.030))
        FILL_HOLES_FLAG = bool(seg.get("FILL_HOLES", True))
        HOLE_MAX_CM2 = float(seg.get("HOLE_MAX_CM2", 0.50))
        
        # ---- Size thresholds ----
        MIN_LEAF_CM2 = float(kwargs.get('min_leaf_cm2', seg.get("MIN_LEAF_CM2", 0.04)))
        NOISE_CM2 = float(kwargs.get('tiny_cm2', seg.get("NOISE_CM2", 0.065)))
        TINY_FRAGMENT_CM2 = float(seg.get("TINY_FRAGMENT_CM2", 0.045))
        MERGE_SMALL_WITHIN_CM = float(kwargs.get('merge_small_within_cm', seg.get("MERGE_SMALL_WITHIN_CM", 0.10)))
        
        min_leaf_px = int(round(MIN_LEAF_CM2 * px_per_cm2))
        tiny_px = int(round(NOISE_CM2 * px_per_cm2))
        tiny_frag_px = int(round(TINY_FRAGMENT_CM2 * px_per_cm2))
        merge_r_px = int(round(MERGE_SMALL_WITHIN_CM * px_per_cm))
        
        # ---- HSV conversion and masking ----
        hsv = cv.cvtColor(imgS, cv.COLOR_BGR2HSV)
        h_chan = hsv[:, :, 0]
        s_chan = hsv[:, :, 1] if SKIP_MEDIAN else cv.medianBlur(hsv[:, :, 1], 5)
        v_chan = hsv[:, :, 2] if SKIP_MEDIAN else cv.medianBlur(hsv[:, :, 2], 5)
        
        # Core base mask (strict to block grays/black) using min/max pairs
        colorful = (s_chan >= HSV_S_MIN) & (s_chan <= HSV_S_MAX)
        bright_enough = (v_chan >= HSV_V_MIN) & (v_chan <= HSV_V_MAX)
        not_black = ~((v_chan >= BLACK_V_MIN) & (v_chan <= BLACK_V_MAX))
        not_white = ~((s_chan >= WHITE_S_MIN) & (s_chan <= WHITE_S_MAX) & (v_chan >= WHITE_V_MIN) & (v_chan <= WHITE_V_MAX))
        base_keep_strict = (colorful & bright_enough & not_black & not_white)
        
        # Exclude blue hues (leaves are rarely blue)
        blue_mask = ((h_chan >= BLUE_H_MIN) & (h_chan <= BLUE_H_MAX)) if BLUE_EXCLUDE else np.zeros_like(h_chan, dtype=bool)
        
        # Relaxed path for dark green: allow lower V if hue is green
        GREEN_V_MIN = int(seg.get("GREEN_V_MIN", 60))
        GREEN_V_MAX = int(seg.get("GREEN_V_MAX", 255))
        green_hue_ok = ((h_chan >= GREEN_H_MIN) & (h_chan <= GREEN_H_MAX))
        green_relax = (green_hue_ok & (s_chan >= HSV_S_MIN) & (s_chan <= HSV_S_MAX) & (v_chan >= GREEN_V_MIN) & (v_chan <= GREEN_V_MAX) & not_black & not_white)
        
        # Combine: strict OR relaxed-green
        base_keep = ((base_keep_strict | green_relax) & (~blue_mask)).astype(np.uint8) * 255
        
        # Apply hue bands if enabled
        if HSV_USE_BANDS:
            hue_mask = np.zeros_like(base_keep)
            hue_mask = cv.bitwise_or(hue_mask, ((h_chan >= GREEN_H_MIN) & (h_chan <= GREEN_H_MAX)).astype(np.uint8) * 255)
            hue_mask = cv.bitwise_or(hue_mask, ((h_chan >= YELLOW_H_MIN) & (h_chan <= YELLOW_H_MAX)).astype(np.uint8) * 255)

            # Brown with min/max pairs
            BROWN_S_MIN = int(seg.get("BROWN_S_MIN", 35))
            BROWN_S_MAX = int(seg.get("BROWN_S_MAX", 255))
            BROWN_V_MIN = int(seg.get("BROWN_V_MIN", 75))
            BROWN_V_MAX = int(seg.get("BROWN_V_MAX", 255))
            brown_hue = ((h_chan >= BROWN_H_MIN) & (h_chan <= BROWN_H_MAX))
            brown_colorful = (s_chan >= BROWN_S_MIN) & (s_chan <= BROWN_S_MAX)
            brown_bright = (v_chan >= BROWN_V_MIN) & (v_chan <= BROWN_V_MAX)
            brown_safe = (brown_hue & brown_colorful & brown_bright).astype(np.uint8) * 255

            hue_mask = cv.bitwise_or(hue_mask, brown_safe)

            # Purple with min/max pairs
            PURPLE_S_MIN = int(seg.get("PURPLE_S_MIN", 40))
            PURPLE_S_MAX = int(seg.get("PURPLE_S_MAX", 255))
            PURPLE_V_MIN = int(seg.get("PURPLE_V_MIN", 55))
            PURPLE_V_MAX = int(seg.get("PURPLE_V_MAX", 255))
            purple_hue = ((h_chan >= PURPLE_H_MIN) & (h_chan <= PURPLE_H_MAX))
            purple_colorful = (s_chan >= PURPLE_S_MIN) & (s_chan <= PURPLE_S_MAX)
            purple_bright = (v_chan >= PURPLE_V_MIN) & (v_chan <= PURPLE_V_MAX)
            purple_safe = (purple_hue & purple_colorful & purple_bright).astype(np.uint8) * 255

            hue_mask = cv.bitwise_or(hue_mask, purple_safe)

            maskS = cv.bitwise_and(base_keep, hue_mask)
        else:
            maskS = base_keep
        
        # Apply Lab color space gating
        if LAB_GATE:
            lab = cv.cvtColor(imgS, cv.COLOR_BGR2LAB)
            a_chan = lab[:, :, 1].astype(np.int16) - 128
            b_chan = lab[:, :, 2].astype(np.int16) - 128
            lab_mask = ((a_chan <= LAB_A_MAX) & (b_chan <= LAB_B_MAX)).astype(np.uint8) * 255
            
            # Neutral Lab suppression: exclude near‑gray pixels
            if bool(seg.get("LAB_NEUTRAL_EXCLUDE", True)):
                a_abs_max = LAB_NEUTRAL_A_ABS_MAX
                b_abs_max = LAB_NEUTRAL_B_ABS_MAX
                neutral = ((np.abs(a_chan) <= a_abs_max) & (np.abs(b_chan) <= b_abs_max)).astype(np.uint8) * 255
                lab_mask = cv.bitwise_and(lab_mask, cv.bitwise_not(neutral))
            
            maskS = cv.bitwise_and(maskS, lab_mask)
        
        # Apply exclusions
        if excludeS is not None:
            maskS = cv.bitwise_and(maskS, cv.bitwise_not(excludeS))
    
        
        # ---- Morphological operations ----
        k_open = max(3, int(round(px_per_cm * OPEN_DIAMETER_CM)))
        k_close = max(5, int(round(px_per_cm * CLOSE_DIAMETER_CM)))
        ker_o = cv.getStructuringElement(cv.MORPH_ELLIPSE, (k_open, k_open))
        ker_c = cv.getStructuringElement(cv.MORPH_ELLIPSE, (k_close, k_close))
        
        maskS = cv.morphologyEx(maskS, cv.MORPH_OPEN, ker_o, iterations=1)
        maskS = cv.morphologyEx(maskS, cv.MORPH_CLOSE, ker_c, iterations=1)
        
        # Bridge and fill holes
        bridge_px = max(1, int(round(CLOSE_BRIDGE_CM * px_per_cm)))
        ker_bridge = cv.getStructuringElement(cv.MORPH_ELLIPSE, (2 * bridge_px + 1, 2 * bridge_px + 1))
        bridged = cv.morphologyEx(maskS, cv.MORPH_CLOSE, ker_bridge, iterations=1)
        
        if FILL_HOLES_FLAG:
            filled_all = fill_holes(bridged)
            if HOLE_MAX_CM2 > 0:
                hole_max_px = int(round(HOLE_MAX_CM2 * px_per_cm2))
                new_fill = cv.bitwise_and(filled_all, cv.bitwise_not(bridged))
                n_h, lab_h, st_h, _ = cv.connectedComponentsWithStats(new_fill, connectivity=8)
                keep_fill = np.zeros_like(new_fill)
                for hi in range(1, n_h):
                    if st_h[hi, cv.CC_STAT_AREA] <= hole_max_px:
                        keep_fill = cv.bitwise_or(keep_fill, (lab_h == hi).astype(np.uint8) * 255)
                maskS = cv.bitwise_or(bridged, keep_fill)
            else:
                maskS = filled_all
        else:
            maskS = bridged
        
        # Reopen to restore edges
        reopen_px = max(1, int(round(REOPEN_CM * px_per_cm)))
        ker_reopen = cv.getStructuringElement(cv.MORPH_ELLIPSE, (2 * reopen_px + 1, 2 * reopen_px + 1))
        maskS = cv.morphologyEx(maskS, cv.MORPH_OPEN, ker_reopen, iterations=1)
        
        # Remove tiny noise
        n1, lab1, st1, _ = cv.connectedComponentsWithStats(maskS, connectivity=8)
        if n1 <= 1:
            mask_full = cv.resize(maskS, (W, H), interpolation=cv.INTER_NEAREST) if scale < 1.0 else maskS
            return mask_full, []
        
        keep_idx = [i for i in range(1, n1) if st1[i, cv.CC_STAT_AREA] >= tiny_px]
        maskS = (np.isin(lab1, keep_idx).astype(np.uint8) * 255) if keep_idx else np.zeros_like(maskS)
        
        # Component merging logic
        n2, lab2, st2, cents2 = cv.connectedComponentsWithStats(maskS, connectivity=8)
        big_ids, small_ids = [], []
        for i in range(1, n2):
            if st2[i, cv.CC_STAT_AREA] >= min_leaf_px:
                big_ids.append(i)
            else:
                small_ids.append(i)
        
        bigS = (np.isin(lab2, big_ids).astype(np.uint8) * 255) if big_ids else np.zeros_like(maskS)
        smallS = (np.isin(lab2, small_ids).astype(np.uint8) * 255) if small_ids else np.zeros_like(maskS)
        
        if big_ids and small_ids and merge_r_px > 0:
            ker_merge = cv.getStructuringElement(cv.MORPH_ELLIPSE, (2 * merge_r_px + 1, 2 * merge_r_px + 1))
            big_dil = cv.dilate(bigS, ker_merge, iterations=1)
            small_touch = cv.bitwise_and(smallS, big_dil)
            n3, lab3, st3, _ = cv.connectedComponentsWithStats(small_touch, connectivity=8)
            tiny_close = np.zeros_like(small_touch)
            for si in range(1, n3):
                if st3[si, cv.CC_STAT_AREA] < tiny_frag_px:
                    tiny_close = cv.bitwise_or(tiny_close, (lab3 == si).astype(np.uint8) * 255)
            mergedS = cv.bitwise_or(bigS, tiny_close)
        else:
            mergedS = bigS if big_ids else smallS
        
        # Scale back to original size and extract components
        if scale < 1.0:
            mask_full = cv.resize(mergedS, (W, H), interpolation=cv.INTER_NEAREST)
            inv_s = 1.0 / scale
            nF, labF, stF, cenF = cv.connectedComponentsWithStats(mergedS, connectivity=8)
            components = []
            for i in range(1, nF):
                x, y, w, h = stF[i, 0], stF[i, 1], stF[i, 2], stF[i, 3]
                area_s = stF[i, cv.CC_STAT_AREA]
                cx, cy = cenF[i][0], cenF[i][1]
                X = int(round(x * inv_s))
                Y = int(round(y * inv_s))
                Wb = int(round(w * inv_s))
                Hb = int(round(h * inv_s))
                Ao = int(round(area_s / (scale * scale)))
                Cx = float(cx * inv_s)
                Cy = float(cy * inv_s)
                X = max(0, min(X, W - 1))
                Y = max(0, min(Y, H - 1))
                Wb = max(1, min(Wb, W - X))
                Hb = max(1, min(Hb, H - Y))
                components.append({
                    "label": i,
                    "bbox": (X, Y, Wb, Hb),
                    "area_px": Ao,
                    "centroid": (Cx, Cy)
                })
            return mask_full, components
        else:
            nF, labF, stF, cenF = cv.connectedComponentsWithStats(mergedS, connectivity=8)
            components = []
            for i in range(1, nF):
                components.append({
                    "label": i,
                    "bbox": tuple(stF[i, 0:4]),
                    "area_px": int(stF[i, cv.CC_STAT_AREA]),
                    "centroid": (float(cenF[i][0]), float(cenF[i][1]))
                })
            return mergedS, components
            
    except Exception as e:
        try:
            if debug:
                debug.log(f"segment_leaves error: {e}", level=1)
        except Exception:
            pass
        empty = np.zeros((H, W), np.uint8)
        return empty, []


# ----------------------------------------------------------------------------------------------------------------------------------
# 10) VISUALIZATION FUNCTIONS
# ----------------------------------------------------------------------------------------------------------------------------------
def draw_results(img, components, Pixel_ratio, label_info, calibration_points=None, qr_info=None):
    """Enhanced visualization using CONFIG styling."""
    overlay = img.copy()
    
    # Draw the calibration points first, so they are underneath other labels if they overlap
    if calibration_points:
        for i, point_coords in enumerate(calibration_points):
            # Ensure coordinates are integers for drawing
            x, y = int(point_coords[0]), int(point_coords[1])
            # Draw a visible marker for each point
            cv.circle(overlay, (x, y), 12, (0, 255, 255), -1)  # Bright Cyan/Yellow filled circle
            cv.circle(overlay, (x, y), 12, (0, 0, 0), 2)       # Black border for visibility
            cv.putText(overlay, f"P{i+1}", (x - 10, y + 6), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

     # Add QR square annotation
    if qr_info and qr_info.get("QR_detected") and qr_info.get("bbox"):
        bbox = qr_info.get("bbox")
        if bbox:
            x, y, w, h = bbox
            cv.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green border
            cv.putText(overlay, "QR", (x, max(0, y - 8)), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)  # Green text, just "QR"

    if not components:
        _add_header_info(overlay, 0, 0, Pixel_ratio, label_info, components=None)
        return overlay
    
    Leaf_Num = len(components)
    total_area_cm2 = sum(c["area_px"] for c in components) / (Pixel_ratio ** 2) if Pixel_ratio else 0
    _add_header_info(overlay, Leaf_Num, total_area_cm2, Pixel_ratio, label_info, components=components)
    _draw_leaf_components(overlay, components, Pixel_ratio)
    
    return overlay


def _draw_leaf_components(overlay, components, Pixel_ratio):
    """Draw bounding boxes, indices, area, and length/width lines onto overlay."""
    viz_config = CONFIG["VISUALIZATION"]

    length_color = (0, 0, 255)      # red
    width_color  = (255, 255, 0)    # cyan/yellow

    # Draw annotations on a separate layer, then composite for solid colors
    annot = np.zeros_like(overlay)
    annot_mask = np.zeros(overlay.shape[:2], dtype=np.uint8)

    def _rect(img, mask, pt1, pt2, color, thickness):
        cv.rectangle(img, pt1, pt2, color, thickness)
        cv.rectangle(mask, pt1, pt2, 255, thickness)

    def _circle(img, mask, center, radius, color, thickness):
        cv.circle(img, center, radius, color, thickness)
        cv.circle(mask, center, radius, 255, thickness)

    def _line(img, mask, pt1, pt2, color, thickness):
        cv.line(img, pt1, pt2, color, thickness)
        cv.line(mask, pt1, pt2, 255, thickness)

    def _text(img, mask, text, org, font, scale, color, thickness):
        cv.putText(img, text, org, font, scale, color, thickness, cv.LINE_AA)
        cv.putText(mask, text, org, font, scale, 255, thickness, cv.LINE_AA)

    # Number leaves by descending area to match distribution/metrics ordering
    ordered_components = sorted(components, key=lambda c: c.get("area_px", 0), reverse=True)
    for i, comp in enumerate(ordered_components, start=1):
        x, y, w, h = comp["bbox"]
        area_px = comp["area_px"]

        # Choose colors based on leaf size using CONFIG
        if Pixel_ratio:
            area_cm2 = area_px / (Pixel_ratio ** 2)
            if area_cm2 < 10.0:
                color = viz_config["SIZE_COLORS"]["small"]
            elif area_cm2 < 35.0:
                color = viz_config["SIZE_COLORS"]["medium"]
            else:
                color = viz_config["SIZE_COLORS"]["large"]
        else:
            color = viz_config["SIZE_COLORS"]["large"]  # Default color

        # Draw bounding rectangle
        _rect(annot, annot_mask, (x, y), (x + w, y + h), color, viz_config["BBOX_THICKNESS"] + 3)

        # Leaf index in a circle (top-left of bbox, clamped inside image bounds)
        circle_center = (
            int(max(15, min(x + viz_config["CIRCLE_RADIUS"] + 6, overlay.shape[1] - 15))),
            int(max(15, min(y + viz_config["CIRCLE_RADIUS"] + 6, overlay.shape[0] - 15)))
        )
        number_scale = 0.6        # was 0.5
        number_thickness = 2      # was 1

        circle_radius = int(viz_config["CIRCLE_RADIUS"] * 1.4)
        circle_thickness = int(viz_config["CIRCLE_THICKNESS"] * 1.2)
        _circle(annot, annot_mask, circle_center, circle_radius, color, -1)
        _circle(annot, annot_mask, circle_center, circle_radius, (0, 0, 0), circle_thickness)
        text_size = cv.getTextSize(str(i), cv.FONT_HERSHEY_SIMPLEX, 0.5, number_thickness)[0]
        text_x = circle_center[0] - text_size[0] // 2
        text_y = circle_center[1] + text_size[1] // 2
        _text(annot, annot_mask, str(i), (text_x, text_y), cv.FONT_HERSHEY_SIMPLEX, number_scale, (0, 0, 0), number_thickness)

        # Area label
        if Pixel_ratio:
            area_text = f"{area_cm2:.2f} sqcm"
        else:
            area_text = f"{area_px} px"

        text_x = x
        text_y = y + h + 25
        font_scale = viz_config["LABEL_FONT_SCALE"]
        font_thickness = viz_config["LABEL_THICKNESS"]
        text_size = cv.getTextSize(area_text, cv.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
        padding = viz_config["TEXT_BACKGROUND_PADDING"]
        border_thickness = viz_config["TEXT_BORDER_THICKNESS"]
        _rect(annot, annot_mask,
            (text_x - padding//2, text_y - text_size[1] - padding),
            (text_x + text_size[0] + padding, text_y + padding//2),
            (255, 255, 255), -1)
        _rect(annot, annot_mask,
            (text_x - padding//2, text_y - text_size[1] - padding),
            (text_x + text_size[0] + padding, text_y + padding//2),
            (0, 0, 0), border_thickness)
        _text(annot, annot_mask, area_text, (text_x, text_y),
            cv.FONT_HERSHEY_SIMPLEX, font_scale, color, font_thickness)

        # === NEW: length & width annotations ===
        length_cm = comp.get("length_cm", 0.0)
        width_cm  = comp.get("width_cm", 0.0)
        length_seg = comp.get("length_segment")
        width_seg  = comp.get("width_segment")

        # Area/Length/width text near top of bbox
        dim_text1 = f"A={area_cm2:.2f} sqcm"
        dim_text2 = f"L={length_cm:.2f} cm"
        dim_text3 = f"W={width_cm:.2f} cm"

        dim_y1 = y - 10 if y - 10 > 10 else y + 35
        dim_y2 = y - 30 if y - 30 > -10 else y + 35
        dim_y3 = y - 50 if y - 50 > -30 else y + 35

        _text(annot, annot_mask, dim_text1, (x + 5, dim_y1),
              cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        _text(annot, annot_mask, dim_text2, (x + 5, dim_y2),
              cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        _text(annot, annot_mask, dim_text3, (x + 5, dim_y3),
              cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Draw true length and width segments if available
        if length_seg is not None:
            (lx1, ly1), (lx2, ly2) = length_seg
            _line(annot, annot_mask, (int(lx1), int(ly1)), (int(lx2), int(ly2)), length_color, 5)

        if width_seg is not None:
            (wx1, wy1), (wx2, wy2) = width_seg
            _line(annot, annot_mask, (int(wx1), int(wy1)), (int(wx2), int(wy2)), width_color, 5)

    # Composite annotations with full opacity
    overlay[annot_mask > 0] = annot[annot_mask > 0]
            
def _add_header_info(overlay, Leaf_Num, total_area_cm2, Pixel_ratio, label_info, components=None):
    """Enhanced header info using CONFIG styling with dynamic panel width."""
    viz_config = CONFIG["VISUALIZATION"]
    panel_margin = viz_config["PANEL_MARGIN"]
    # Header text parameters
    header_font = cv.FONT_HERSHEY_SIMPLEX
    header_scale = viz_config["HEADER_FONT_SCALE"] + 0.5
    header_thickness = viz_config["HEADER_THICKNESS"]
    base_line_height = viz_config["HEADER_LINE_HEIGHT"]
    text_h = cv.getTextSize("Ag", header_font, header_scale, header_thickness)[0][1]
    line_height = max(base_line_height, int(text_h * 1.35))

    # Build all header lines first (so we can compute required width)
    lines = []
    text_color = (255, 255, 255)
    lines.append(("LEAF ANALYSIS RESULTS", text_color))
    if Pixel_ratio:
        lines.append((f"Total area: {total_area_cm2:.2f} sqcm", text_color))
        lines.append((f"Calibration: {Pixel_ratio:.2f} px/cm", text_color))
    else:
        lines.append(("Total area: No calibration", text_color))

    if components and Pixel_ratio:
        areas_cm2 = [c["area_px"] / (Pixel_ratio ** 2) for c in components]
        avg_area = sum(areas_cm2) / len(areas_cm2)
        min_area = min(areas_cm2)
        max_area = max(areas_cm2)
        lines.append(("", text_color))
        lines.append(("LEAF STATISTICS", text_color))
        lines.append((f"Average: {avg_area:.2f} sqcm", text_color))
        lines.append((f"Range: {min_area:.2f} - {max_area:.2f} sqcm", text_color))
        lines.append((f"Total: {len(components)} leaves", text_color))

    # Label information from config-defined labels
    if isinstance(label_info, dict):
        label_order = CONFIG.get("INPUTS", {}).get("LABELS") or []
        for nm in label_order:
            val = label_info.get(nm)
            if val is None:
                continue
            lines.append((f"{nm}: {val}", text_color))

    # Compute required panel width dynamically based on the longest line
    max_text_width = 0
    for text, _color in lines:
        size = cv.getTextSize(text, header_font, header_scale, header_thickness)[0]
        max_text_width = max(max_text_width, size[0])

    # Base minimum width, but expand to fit longest text + padding
    base_width = viz_config["HEADER_PANEL_WIDTH"]
    padding = 30  # extra padding around text
    dynamic_width = max(base_width, max_text_width + padding + panel_margin * 2)

    # Window height based on number of lines
    base_height = viz_config["HEADER_PANEL_HEIGHT"]
    # Roughly: top margin + lines * line_height + bottom margin
    dynamic_height = max(base_height, 40 + len(lines) * line_height + 20)

    # Draw background panel using dynamic width/height
    panel_overlay = overlay.copy()
    bg_color = viz_config.get("STATS_PANEL_BG_COLOR", (0, 0, 0))
    cv.rectangle(panel_overlay, (panel_margin, panel_margin),
                 (panel_margin + dynamic_width, panel_margin + dynamic_height),
                 bg_color, -1)
    cv.addWeighted(panel_overlay, viz_config["PANEL_ALPHA"], overlay, 1 - viz_config["PANEL_ALPHA"], 0, overlay)

    # Draw border
    cv.rectangle(overlay, (panel_margin, panel_margin),
                 (panel_margin + dynamic_width, panel_margin + dynamic_height),
                 viz_config["PANEL_BORDER_COLOR"], viz_config["PANEL_BORDER_THICKNESS"])

    # Render lines
    y_pos = panel_margin + line_height
    x_pos = panel_margin + 20
    for text, color in lines:
        if text:
            cv.putText(overlay, text, (x_pos, y_pos), header_font, header_scale, color, header_thickness, cv.LINE_AA)
        y_pos += line_height


def _add_component_stats(overlay, components, Pixel_ratio):
    """Enhanced statistics panel using CONFIG styling."""
    if not components or not Pixel_ratio:
        return
    
    viz_config = CONFIG["VISUALIZATION"]
    
    # Calculate statistics
    areas_cm2 = [c["area_px"] / (Pixel_ratio ** 2) for c in components]
    avg_area = sum(areas_cm2) / len(areas_cm2)
    min_area = min(areas_cm2)
    max_area = max(areas_cm2)
    
    # Position stats panel using CONFIG dimensions
    img_height, img_width = overlay.shape[:2]
    panel_width = viz_config["STATS_PANEL_WIDTH"]
    panel_height = viz_config["STATS_PANEL_HEIGHT"]
    panel_margin = viz_config["PANEL_MARGIN"]
    
    panel_x = img_width - panel_width - panel_margin
    panel_y = panel_margin
    
    # Draw background panel using CONFIG styling
    stats_overlay = overlay.copy()
    cv.rectangle(stats_overlay, (panel_x, panel_y),
                 (panel_x + panel_width, panel_y + panel_height),
                 viz_config["STATS_PANEL_BG_COLOR"], -1)
    cv.addWeighted(stats_overlay, viz_config["PANEL_ALPHA"], overlay, 1 - viz_config["PANEL_ALPHA"], 0, overlay)
    
    # Draw border using CONFIG settings
    cv.rectangle(overlay, (panel_x, panel_y),
                 (panel_x + panel_width, panel_y + panel_height),
                 viz_config["PANEL_BORDER_COLOR"], viz_config["PANEL_BORDER_THICKNESS"])
    
    # Add statistics text using CONFIG styling
    font = cv.FONT_HERSHEY_SIMPLEX
    scale = viz_config["STATS_FONT_SCALE"]
    thickness = viz_config["STATS_THICKNESS"]
    color = viz_config["TEXT_COLOR"]
    line_spacing = viz_config["STATS_LINE_SPACING"]
    
    y_pos = panel_y + 25
    
    def add_stat_line(text):
        nonlocal y_pos
        cv.putText(overlay, text, (panel_x + 10, y_pos), font, scale,
                   color, thickness, cv.LINE_AA)
        y_pos += line_spacing
    
    add_stat_line("LEAF STATISTICS")
    add_stat_line(f"Average: {avg_area:.2f} sqcm")
    add_stat_line(f"Range: {min_area:.2f} - {max_area:.2f} sqcm")
    add_stat_line(f"Total: {len(components)} leaves")

# ----------------------------------------------------------------------------------------------------------------------------------
# 11) SEGMENTATION REVIEW FUNCTIONS
# ----------------------------------------------------------------------------------------------------------------------------------
def show_segmentation_debug(
    img,
    mask,
    components,
    Pixel_ratio,
    calibration_points=None,
    label_info=None,
    window="Segmentation review",
    save_path=None,
    debug: Optional[Debugger] = None
):
    """Segmentation review window showing image and mask side-by-side."""
    # ----- Sizing from UI config -----
    ui  = CONFIG.get("UI", {})
    viz = CONFIG.get("VISUALIZATION", {})
    H, W = img.shape[:2]

    # Use the shared reference max size so review windows align with
    # other reference-style windows on screen.
    base_w, base_h        = _get_reference_max_size()
    # Height cap for the composed review window; follow the same base
    # height as the reference image.
    debug_view_max        = (base_w, base_h)
    # Compute the width that the *single* reference image would use
    # when fitted into (base_w, base_h). The two review panels together
    # must NEVER exceed this width, so that the mask selection window
    # is not wider than the reference windows.
    ref_scale             = min(base_w / max(1, W), base_h / max(1, H), 1.0)
    single_ref_width      = int(max(1, W * ref_scale))

    panel_spacing         = int(ui.get("REVIEW_PANEL_SPACING", 10))          # gap between panels
    header_h              = int(ui.get("REVIEW_HEADER_HEIGHT", 40))          # header strip height

    # Total width of the two-panel window should match the single reference
    # width. We compute per-panel width so that:
    #   (per_panel * 2) + spacing = single_ref_width
    # This makes the mask review window the same total width as Image Label
    # and calibration reference windows.
    available_for_panels  = max(1, single_ref_width - panel_spacing)
    per_panel_max_w       = max(1, int(available_for_panels // 2))
    ref_pos               = tuple(ui.get("REFERENCE_WINDOW_POSITION", (0, 0)))
    panel_offset_x        = int(ui.get("PANEL_OFFSET_X", 15))

    # Determine per-panel target size with aspect ratio preserved
    target_width  = max(1, per_panel_max_w)
    target_height = int(target_width * H / max(1, W))

    # Respect max window height
    max_content_h = max(1, int(debug_view_max[1]) - header_h)
    if target_height > max_content_h:
        target_height = max_content_h
        target_width  = int(target_height * W / max(1, H))

    # Final composed canvas size
    total_width  = target_width * 2 + panel_spacing
    total_height = header_h + target_height

    # Window size caps (enforce single reference width as the absolute max)
    display_width  = min(single_ref_width, total_width)
    display_height = min(int(debug_view_max[1]), total_height)

    # ----- Build side-by-side panel -----
    mask_rgb     = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)
    img_resized  = cv.resize(img,  (target_width, target_height), interpolation=cv.INTER_AREA)
    mask_resized = cv.resize(mask_rgb, (target_width, target_height), interpolation=cv.INTER_NEAREST)

    combined_panel = np.zeros((total_height, total_width, 3), dtype=np.uint8)
    combined_panel[0:header_h, :] = (40, 40, 40)  # header background

    # Header labels with dedicated config key and black background boxes
    review_font_scale = float(viz.get("REVIEW_HEADER_FONT_SCALE", 0.4))
    review_thickness = int(viz.get("REVIEW_HEADER_THICKNESS", 1))
    text_color = viz.get("TEXT_COLOR", (255, 255, 255))
    
    # Original Image label with background
    orig_text = "Original Image"
    (tw1, th1), _ = cv.getTextSize(orig_text, cv.FONT_HERSHEY_SIMPLEX, review_font_scale, review_thickness)
    cv.rectangle(combined_panel, (19, min(28, header_h - 12) - th1 - 1), 
                 (21 + tw1, min(28, header_h - 12) + 1), (0, 0, 0), -1)
    cv.putText(combined_panel, orig_text, (20, min(28, header_h - 12)),
               cv.FONT_HERSHEY_SIMPLEX, review_font_scale, text_color, review_thickness, cv.LINE_AA)

    # Leaf Mask label with background
    mask_label_x = target_width + panel_spacing + 20
    mask_text = "Leaf Mask"
    (tw2, th2), _ = cv.getTextSize(mask_text, cv.FONT_HERSHEY_SIMPLEX, review_font_scale, review_thickness)
    cv.rectangle(combined_panel, (mask_label_x - 1, min(28, header_h - 12) - th2 - 1),
                 (mask_label_x + tw2 + 1, min(28, header_h - 12) + 1), (0, 0, 0), -1)
    cv.putText(combined_panel, mask_text, (mask_label_x, min(28, header_h - 12)),
               cv.FONT_HERSHEY_SIMPLEX, review_font_scale, text_color, review_thickness, cv.LINE_AA)

    # Separator line
    sep_x = target_width + panel_spacing // 2
    cv.line(combined_panel, (sep_x, header_h), (sep_x, total_height), (100, 100, 100), 2)

    # Place images
    combined_panel[header_h:header_h + target_height, 0:target_width] = img_resized
    mask_start_x = target_width + panel_spacing
    combined_panel[header_h:header_h + target_height, mask_start_x:mask_start_x + target_width] = mask_resized

    # Optional quick info overlay (count/area)
    if components and Pixel_ratio:
        total_area_cm2 = sum(c["area_px"] for c in components) / (Pixel_ratio ** 2)
        info_text = f"Leaves: {len(components)} | Area: {total_area_cm2:.2f} sqcm"
        info_y = header_h + target_height - 12
        cv.putText(combined_panel, info_text,
                   (10, info_y),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5,
                   (255, 255, 255), 1, cv.LINE_AA)

    # ----- Show window and review -----
    try:
        cv.namedWindow(window, cv.WINDOW_NORMAL)
        cv.imshow(window, combined_panel)
        cv.resizeWindow(window, display_width, display_height)
        cv.moveWindow(window, ref_pos[0], ref_pos[1])
        cv.waitKey(1)

        confirm_pos = (ref_pos[0] + display_width + panel_offset_x, ref_pos[1])
        decision = show_segmentation_confirmation("Segmentation Confirmation", confirm_pos)

        # ----- Save AFTER selection with structured naming -----
        if debug and (debug.save_dir or debug.auto_save or debug.mode >= 2):
            outputs_root = CONFIG["PATHS"].get("DIR_OUTPUTS", get_working_directory())
            file_formats = CONFIG.get("FILE_FORMATS", {})
            debug_subdir = CONFIG["PATHS"].get("DEBUG_SUBDIR", "Debug")

            # Extract date for naming; other tokens resolve via label_info
            date_str = None
            if isinstance(label_info, dict):
                dr = label_info.get("Date")
                if isinstance(dr, str):
                    date_str = dr.replace("-", "_")

            # Fallbacks
            if not date_str and '_persistent_date' in globals() and _persistent_date:
                date_str = _persistent_date.replace("-", "_")
            if not date_str and isinstance(label_info, dict):
                date_str = label_info.get("Original_Stem")
            if not date_str:
                date_str = "Output"
            # Legacy fields removed; rely solely on dynamic labels for tokens

            # Build folder path honoring dynamic naming schema
            experiment_folder = build_folder_name_from_pattern(date_str, label_info=label_info)
            flat_output = False
            if isinstance(label_info, dict):
                if not label_info.get("Date") and label_info.get("Original_Stem"):
                    flat_output = True
            if debug_subdir == CONFIG["PATHS"].get("DEBUG_SUBDIR", "Debug"):
                folder_path = os.path.join(outputs_root, debug_subdir) if flat_output else os.path.join(outputs_root, debug_subdir, experiment_folder)
            else:
                folder_path = os.path.join(outputs_root, debug_subdir) if flat_output else os.path.join(outputs_root, experiment_folder, debug_subdir)
            os.makedirs(folder_path, exist_ok=True)

            # Always use unified OUTPUT_FILE_NAME_PATTERN via dynamic label fields
            base_pattern = file_formats.get("OUTPUT_FILE_NAME_PATTERN", "{DATE}")
            base_with_ext = build_filename_generic(date_str, label_info or {}, base_pattern)
            base_no_ext = os.path.splitext(base_with_ext)[0]
            final_save_path = os.path.join(folder_path, base_no_ext + "_debug.jpg")

            cv.imwrite(final_save_path, combined_panel)
            if debug:
                debug.log(f"Saved debug comparison: {final_save_path}", level=2)

        cv.destroyWindow(window)
        return decision

    except Exception as e:
        print(f"Error showing segmentation debug: {e}")
        try:
            cv.destroyWindow(window)
        except:
            pass
        return "Finish"
    
def show_segmentation_confirmation(window_title="Segmentation Confirmation", window_pos=None):
    """
    Vertical confirmation dialog (single column), auto-sized to fit content.
    Order: Finish, Flag for Review, Retry Calibration, Retry Masks.
    Returns: "Finish", "Flag", "Retry Calib", "Retry Label".
    ESC or Q quits the app.
    """
    scale_factor = _get_hidpi_scale()
    panel_window = "Segmentation Confirmation Window"
    # Labels and return values
    buttons = [
        ("Finish",            "Finish"),
        ("Flag",              "Flag"),
        ("Retry Calibration", "Retry Calib"),
        ("Retry Masks",       "Retry Label"),
    ]
    # Compact sizing (unscaled)
    margin = 16
    title_sub_spacing = 8
    header_bottom_spacing = 24
    button_height = max(52, CONFIG["UI"]["BUTTON_HEIGHT"])
    # Compute dynamic width/height from content and then fit to screen
    # Estimate panel size: header + buttons + bottom margin
    approx_rows = len(buttons)
    grid_gap = int(CONFIG["UI"].get("GRID_SPACING", 10))
    header_h = margin * 2 + max(35, int(CONFIG["UI"].get("TITLE_SCALE", 0.7) * 30 * scale_factor)) + title_sub_spacing + header_bottom_spacing
    buttons_h = approx_rows * button_height + (approx_rows - 1) * grid_gap
    bottom_h = int(CONFIG["UI"].get("INSTRUCTION_BOTTOM_MARGIN", 20)) + grid_gap + 4
    panel_width, panel_height = _fit_panel_to_screen(400, header_h + buttons_h + bottom_h)
    button_spacing = max(10, CONFIG["UI"]["BUTTON_SPACING"])
    instr_bottom_margin = 10
    min_button_width = int(CONFIG["UI"].get("PANEL_MIN_WIDTH", 400))
    text_pad = int(CONFIG["UI"].get("CONFIRM_BUTTON_PADDING_PX", 38))

    def _scaled_to_unscaled(px):
        return int(round(px / max(scale_factor, 1)))

    # Header sizes
    ui = CONFIG.get("UI", {})
    _tw, title_h_s = calculate_text_size("Segmentation Review", float(ui.get("TITLE_FONT_SIZE", 0.85)), scale_factor)
    title_h = _scaled_to_unscaled(title_h_s)
    subtitle_h = 0
    _iw, instr_h_s = calculate_text_size("Click to Select / ENTER=Finish / ESC=Quit", float(ui.get("INSTRUCTION_FONT_SIZE", 0.45)), scale_factor)
    instr_h = _scaled_to_unscaled(instr_h_s)

    # Button width from text (use button text scale config) and centralized font face
    scaled_font = float(ui.get("BUTTON_FONT_SIZE", ui.get("BUTTON_SIZE", 0.6))) * scale_factor
    scaled_thick = max(1, int(int(ui.get("BUTTON_THICKNESS", 2)) * scale_factor))
    _font_face_cfg = _get_font_face()
    max_label_w_unscaled = 0
    for label, _ret in buttons:
        (tw_s, _th_s), _base = cv.getTextSize(label, _font_face_cfg, scaled_font, scaled_thick)
        max_label_w_unscaled = max(max_label_w_unscaled, _scaled_to_unscaled(tw_s))
    button_width = max(min_button_width, max_label_w_unscaled + text_pad)

    # Auto-size panel
    panel_width = button_width + margin * 2
    buttons_h = len(buttons) * button_height + (len(buttons) - 1) * button_spacing
    header_h = title_h + header_bottom_spacing
    panel_height = margin + header_h + buttons_h + instr_h + instr_bottom_margin + margin

    try:
        # HiDPI panel
        hires_panel, _ = create_high_dpi_panel(panel_width, panel_height, scale_factor)
        # Header
        _ = draw_hires_panel_header(hires_panel, "Segmentation Review", "",
                        scale_factor, int(ui.get("MARGIN", 18)))
        current_y = int(ui.get("MARGIN", 18)) + header_h
        # Draw buttons with colorblind palette
        click_zones = {}
        x_left = int(ui.get("MARGIN", 18))
        for i, (label, ret_val) in enumerate(buttons):
            rect = (x_left, current_y, x_left + button_width, current_y + button_height)
            draw_hires_styled_button(hires_panel, label, rect, palette_color(i), scale_factor,
                                     tuple(ui.get("TEXT_WHITE", (255, 255, 255))), float(ui.get("BUTTON_FONT_SIZE", ui.get("BUTTON_SIZE", 0.6))))
            click_zones[ret_val] = rect
            current_y += button_height + button_spacing
        # Bottom instructions
        instr_y = int((panel_height - instr_bottom_margin) * scale_factor)
        _add_instructions(hires_panel, "Click to Select / ENTER=Finish / ESC=Quit", instr_y, scale_factor)
        # Show panel and position it
        _ = show_hires_panel(panel_window, hires_panel, panel_width, panel_height)
        if window_pos:
            cv.moveWindow(panel_window, *window_pos)
        else:
            cv.moveWindow(panel_window, *CONFIG["UI"]["DEFAULT_WINDOW_POSITION"])
        cv.waitKey(1)
        # Robust, local event loop (mouse + keyboard)
        selected_item = None

        def mouse_callback(event, x, y, flags, param):
            nonlocal selected_item
            if event == cv.EVENT_LBUTTONDOWN:
                for ret_val, (x1, y1, x2, y2) in click_zones.items():
                    if x1 <= x <= x2 and y1 <= y <= y2:
                        selected_item = ret_val
                        return
        cv.setMouseCallback(panel_window, mouse_callback)
        # Flush stray keys
        start = dt.datetime.now()
        while (dt.datetime.now() - start).total_seconds() < 0.2:
            _ = cv.waitKey(1)
        # Wait for click/keypress
        while selected_item is None:
            key = cv.waitKey(30)
            if key == -1:
                continue
            if key in (27, ord('q'), ord('Q')):  # ESC/Q quits app
                try:
                    cv.destroyWindow(panel_window)
                    cv.destroyAllWindows()
                except Exception:
                    pass
                sys.exit(0)
            elif key in (10, 13):               # ENTER
                selected_item = "Finish"
            elif key in (ord('f'), ord('F')):   # Flag via keyboard
                selected_item = "Flag"
            elif key in (ord('c'), ord('C')):   # Retry calibration
                selected_item = "Retry Calib"
            elif key in (ord('l'), ord('L'), ord('m'), ord('M')):  # Retry masks
                selected_item = "Retry Label"
        cv.destroyWindow(panel_window)
        return selected_item
    except Exception as e:
        print(f"Error in segmentation confirmation: {e}")
        try:
            cv.destroyWindow(panel_window)
        except:
            pass
        return "Finish"
        
# ----------------------------------------------------------------------------------------------------------------------------------
# 12) PERSISTENT SETTINGS MANAGEMENT
# ----------------------------------------------------------------------------------------------------------------------------------
def set_persistent_settings(date_str=None, label_value=None):
    """Programmatically set persistent date.

    The optional second parameter is retained for backward compatibility but is not used.
    """
    global _persistent_date, _persistent_labels
    if date_str:
        _persistent_date = date_str
        # Also seed the primary DATE field in persistent labels if present
        df = next((f for f in CONFIG.get("SCHEMA_FIELDS", []) if f.get("type") == "DATE"), None)
        if df:
            _persistent_labels[df["name"]] = date_str
    print(f"Set persistent settings - date={_persistent_date}")

def get_current_persistent_settings():
    """Get current persistent settings (date plus dynamic label values)."""
    global _persistent_date, _persistent_labels
    return {
        "date": _persistent_date,
        "labels": dict(_persistent_labels),
    }

def reset_persistent_settings():
    """Reset persistent date and all dynamic label settings."""
    global _persistent_date, _persistent_labels
    _persistent_date = None
    _persistent_labels.clear()
    print("Persistent settings reset.")

# ----------------------------------------------------------------------------------------------------------------------------------
# 13) MAIN PROCESSING PIPELINE
# ----------------------------------------------------------------------------------------------------------------------------------
def process_image(path, debug: Optional[Debugger] = None):
    """Image processing with selective retry options at final review.
    Supports:
      - Finish: accept and finish (Flagged=False in CSV)
      - Flag for Review: accept and finish with review flag (Flagged=True in CSV)
      - Retry Calib: redo only calibration, then re-segment and re-review (also updates persistent calibration if enabled)
      - Retry Masks: redo only label/bag exclusion, then re-segment and re-review
    """
    global _persistent_date, _persistent_px_per_cm
    img = read_image_with_orientation(path)
    if img is None:
        raise FileNotFoundError(path)
    if debug:
        debug.log(f"\n=== Processing: {os.path.basename(path)} ===", level=2)

    # Step 1: Collect label information (filename/folder + UI where needed)
    label_info = manual_input_label_fields(path, debug=debug)
    if debug:
        debug.log(f"Label info: {label_info}", level=2)

    H, W = img.shape[:2]
    persistent_on = bool(CONFIG.get("CALIB", {}).get("PERSISTENT_CALIBRATION", False))

    # Step 2: Calibration (persistent if enabled and available; otherwise QR then manual)
    calibration_points = None
    qr_info = {"QR_detected": False, "QR_count": 0, "QR_marker_ids": [], "bbox": None, "marker_corners": None}
    if persistent_on and (_persistent_px_per_cm is not None):
        Pixel_ratio = float(_persistent_px_per_cm)
        calib_exclude_mask = np.zeros((H, W), np.uint8)
        qr_enabled = bool(CONFIG.get("CALIB", {}).get("QR_CODE_CALIBRATION", False))
        if qr_enabled:
            if debug:
                debug.log("Step: QR-based mask (using cached calibration)...", level=2)
            _qr_px, _qr_mask, qr_info = _qr_calibration(img, debug=debug)
            if _qr_mask is not None:
                calib_exclude_mask = _qr_mask
            elif qr_info.get("bbox"):
                calib_exclude_mask = _build_qr_exclude_mask(img, qr_info.get("bbox"), Pixel_ratio)
        if debug:
            debug.log(f"[Persistent] Using cached calibration: {Pixel_ratio:.2f} px/cm", level=2)
    else:
        Pixel_ratio = None
        calib_exclude_mask = None
        qr_enabled = bool(CONFIG.get("CALIB", {}).get("QR_CODE_CALIBRATION", False))
        if qr_enabled:
            if debug:
                debug.log("Step: QR-based calibration...", level=2)
            Pixel_ratio, calib_exclude_mask, qr_info = _qr_calibration(img, debug=debug)

        if Pixel_ratio is None:
            if debug:
                debug.log("Step: Manual calibration...", level=2)
            if not CONFIG["RUN"].get("INTERACTIVE_CALIB", True):
                if debug:
                    debug.log("Manual calibration disabled - returning empty results", level=2)
                return {
                    "image_path": path,
                    "Pixel_cm_ratio": None,
                    "Leaf_Num": 0,
                    "Leaf_Area": None,
                    "leaves": [],
                    "label": label_info,
                    "Flagged": False,
                    "Retry_num": 0,
                    "calibration_points": None,
                    "QR_detected": qr_info.get("QR_detected", False),
                    "QR_count": qr_info.get("QR_count", 0),
                    "qr_info": qr_info,
                }

            Pixel_ratio, calib_exclude_mask, calibration_points = manual_calibration(img, debug=debug)
            if qr_info.get("bbox"):
                qr_mask = _build_qr_exclude_mask(img, qr_info.get("bbox"), Pixel_ratio)
                if qr_mask is not None:
                    if calib_exclude_mask is None:
                        calib_exclude_mask = qr_mask
                    else:
                        try:
                            calib_exclude_mask = cv.bitwise_or(calib_exclude_mask, qr_mask)
                        except Exception:
                            calib_exclude_mask = qr_mask
            if Pixel_ratio is None:
                if debug:
                    debug.log("Calibration cancelled - returning empty results", level=2)
                return {
                    "image_path": path,
                    "Pixel_cm_ratio": None,
                    "Leaf_Num": 0,
                    "Leaf_Area": None,
                    "leaves": [],
                    "label": label_info,
                    "Flagged": False,
                    "Retry_num": 0,
                    "calibration_points": None,
                    "QR_detected": qr_info.get("QR_detected", False),
                    "QR_count": qr_info.get("QR_count", 0),
                    "qr_info": qr_info,
                }

        if persistent_on and Pixel_ratio is not None:
            _persistent_px_per_cm = float(Pixel_ratio)
            if debug:
                debug.log(f"[Persistent] Cached calibration set: {_persistent_px_per_cm:.2f} px/cm", level=2)

    # Step 3: Optional label/bag exclusion
    select_masks_enabled = bool(CONFIG.get("RUN", {}).get("SELECT_MASKS", True))
    if not select_masks_enabled:
        if debug:
            debug.log("Step: Mask selection skipped (SELECT_MASKS=False)", level=2)
        label_exclude_mask = None
    else:
        if debug:
            debug.log("Step: Paper bag/label area exclusion...", level=2)
        label_exclude_mask = manual_label_exclusion(img, debug=debug, calib_exclude_mask=calib_exclude_mask, qr_detected=qr_info.get("QR_detected", False))

    # Step 4: Review loop — supports Retry Calib and Retry Label
    review_flag = False
    while True:
        # Combine exclusion masks
        # Ensure masks are valid arrays before combining
        if calib_exclude_mask is None:
            calib_exclude_mask = np.zeros((H, W), np.uint8)
        combined_exclude_mask = calib_exclude_mask
        if label_exclude_mask is not None:
            try:
                combined_exclude_mask = cv.bitwise_or(calib_exclude_mask, label_exclude_mask)
            except Exception:
                combined_exclude_mask = calib_exclude_mask

        # Segment leaves
        if debug:
            debug.log("Step: Leaf segmentation and review...", level=2)
        leaf_mask, components = segment_leaves(img, Pixel_ratio, exclude_mask=combined_exclude_mask, debug=debug)

        # Area in cm²
        for c in components:
            c["area_cm2"] = c["area_px"] / (Pixel_ratio ** 2)

        # Per-leaf geometry (true contour L/W) then convert to cm
        annotate_components_with_length_width(leaf_mask, components, debug=debug)
        for c in components:
            c["length_cm"] = (c.get("length_px", 0.0) / Pixel_ratio) if Pixel_ratio else 0.0
            c["width_cm"]  = (c.get("width_px", 0.0)  / Pixel_ratio) if Pixel_ratio else 0.0

        # Review dialog
        review_enabled = CONFIG["RUN"].get("REVIEW_SEGMENTATION", True) and not CONFIG["RUN"].get("HEADLESS", False)
        if review_enabled:
            decision = show_segmentation_debug(
                img, leaf_mask, components, Pixel_ratio, calibration_points,
                label_info=label_info, debug=debug
            )
        else:
            decision = "Finish"

        # Handle decisions
        if decision == "Finish":
            review_flag = False
            if debug and debug.mode >= 2:
                debug.log("✓ Segmentation accepted (no flag)", level=2)
            break
        elif decision in ("Flag", "No"):
            review_flag = True
            if debug and debug.mode >= 2:
                debug.log("⚑ Segmentation flagged for review", level=2)
            break
        elif decision == "Retry Calib":
            if debug and debug.mode >= 2:
                debug.log("↻ Retrying CALIBRATION...", level=2)
            new_px, new_calib_mask, new_calib_pts = manual_calibration(img, debug=debug)
            if new_px is not None:
                Pixel_ratio = float(new_px)
                calib_exclude_mask = new_calib_mask
                calibration_points = new_calib_pts
                if persistent_on:
                    _persistent_px_per_cm = float(Pixel_ratio)
                    if debug:
                        debug.log(f"[Persistent] Cached calibration updated: {_persistent_px_per_cm:.2f} px/cm", level=2)
            # loop continues: re-segment and re-review
        elif decision == "Retry Label":
            if debug and debug.mode >= 2:
                debug.log("↻ Retrying MASKING (label/bag exclusion)...", level=2)
            label_exclude_mask = manual_label_exclusion(img, debug=debug, force_select=True, calib_exclude_mask=calib_exclude_mask, qr_detected=qr_info.get("QR_detected", False))
            # loop continues: re-segment and re-review
        else:
            # Unknown response: treat as finish
            review_flag = False
            break

    # Step 5: Save training annotations if enabled
    if CONFIG["RUN"].get("SAVE_ML_TRAINING_DATA", CONFIG["RUN"].get("SAVE_ML_DATA", False)):
        if debug:
            debug.log("Step: Saving ML training annotations...", level=2)
        training_dir = os.path.join(
            CONFIG["PATHS"]["DIR_OUTPUTS"],
            CONFIG["PATHS"].get("TRAINING_SUBDIR", "ML_Training_Data")
        )
        os.makedirs(training_dir, exist_ok=True)
        base_filename = os.path.splitext(os.path.basename(path))[0]
        try:
            save_ml_annotations(
                img, leaf_mask, components, Pixel_ratio, label_info,
                training_dir, base_filename, calib_exclude_mask, label_exclude_mask,
                qr_info=qr_info if 'qr_info' in locals() else None
            )
            if debug:
                debug.log(f"Saved ML training annotations to: {training_dir}", level=2)
        except Exception as e:
            if debug:
                debug.log(f"Error saving ML annotations: {e}", level=1)

    # Final results
    total_area = sum(c["area_cm2"] for c in components) if components else 0.0
    if debug:
        debug.log(f"Processing complete - {len(components) if components else 0} leaves, {total_area:.2f} cm² total", level=2)

    results = {
        "image_path": path,
        "Pixel_cm_ratio": Pixel_ratio if 'Pixel_ratio' in locals() else None,
        "Leaf_Num": len(components) if components else 0,
        "Leaf_Area": total_area,
        "leaves": components if components else [],
        "label": label_info,
        "Flagged": review_flag,
        "Retry_num": 0,
        "calibration_points": calibration_points if 'calibration_points' in locals() else None,
        "QR_detected": qr_info.get("QR_detected", False) if 'qr_info' in locals() else False,
        "QR_count": qr_info.get("QR_count", 0) if 'qr_info' in locals() else 0,
        "qr_info": qr_info if 'qr_info' in locals() else None
    }
    return results

# ----------------------------------------------------------------------------------------------------------------------------------
# 14) BATCH PROCESSING FUNCTIONS
# ----------------------------------------------------------------------------------------------------------------------------------
def process_folder(folder, out_csv=None, overlay_dir=None, debug: Optional[Debugger] = None):
    """Batch processing using CONFIG settings; resilient to unreadable files.
    - Skips non-image resource files (e.g., AppleDouble '._*', .DS_Store).
    - Continues on per-image errors without stopping the batch.
    - Writes main CSV + distribution + length + width CSVs incrementally.
    - Saves overlays if enabled.
    """
    rows = []
    results_data = []
    total_processed = 0
    accepted_count = 0
    rejected_count = 0
    retry_stats = {}

    outputs_root = CONFIG["PATHS"].get("DIR_OUTPUTS", get_working_directory())

    # Prepare overlay dir if requested
    if overlay_dir is None and CONFIG["RUN"].get("SAVE_OVERLAYS", False):
        overlay_dir = os.path.join(outputs_root, CONFIG["PATHS"].get("OVERLAY_SUBDIR", "Overlays"))
        os.makedirs(overlay_dir, exist_ok=True)

    # Ensure Debug dir is available for debug saves
    if debug:
        if not debug.save_dir:
            debug.save_dir = os.path.join(outputs_root, CONFIG["PATHS"].get("DEBUG_SUBDIR", "Debug"))
        try:
            os.makedirs(debug.save_dir, exist_ok=True)
        except Exception:
            pass

    # Rename-in-place only; duplicate image functionality removed
    rename_flag = CONFIG["RUN"].get("RENAME_EXISTING_FILE", False)

    # Prepare CSV paths
    if not out_csv:
        out_csv_name = CONFIG["PATHS"].get("LEAF_AREA_CSV_NAME", "Leaf_Area.csv")
        out_csv = os.path.join(outputs_root, out_csv_name)
    base_dir = os.path.dirname(out_csv) if out_csv else outputs_root
    os.makedirs(base_dir, exist_ok=True)

    # Distribution CSV
    leaf_dist_csv_path = os.path.join(base_dir, CONFIG["PATHS"].get("LEAF_DIST_CSV_NAME", "Leaf_Distribution.csv"))
    # Length/Width CSVs (if configured)
    leaf_length_csv_path = os.path.join(base_dir, CONFIG["PATHS"].get("LEAF_LENGTH_CSV_NAME", "Leaf_Length.csv"))
    leaf_width_csv_path  = os.path.join(base_dir, CONFIG["PATHS"].get("LEAF_WIDTH_CSV_NAME",  "Leaf_Width.csv"))
    if debug:
        debug.log(f"CSV outputs: {out_csv}", level=2)
        debug.log(f"CSV outputs: {leaf_dist_csv_path}", level=2)
        debug.log(f"CSV outputs: {leaf_length_csv_path}", level=2)
        debug.log(f"CSV outputs: {leaf_width_csv_path}", level=2)
    else:
        print(f"CSV outputs: {out_csv}")

    for File in sorted(os.listdir(folder)):
        # Skip non-images and common metadata/resource files
        if File.startswith("._") or File.lower() in (".ds_store",):
            if debug:
                debug.log(f"Skipping non-image resource: {File}", level=2)
            continue
        if not File.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff')):
            continue

        fpath = os.path.join(folder, File)

        # Content-type guard: skip mislabeled HEIF/HEIC masquerading as .jpg
        if not is_supported_image_type(fpath):
            if debug:
                debug.log(f"Skipping unsupported or mislabeled image type: {File}", level=2)
            continue

        # Resilient per-image processing: skip unreadable/bad files
        try:
            res = process_image(fpath, debug=debug)
        except FileNotFoundError as e:
            if debug:
                debug.log(str(e), level=1)
            continue
        except Exception as e:
            if debug:
                debug.log(f"Error processing {File}: {e}", level=1)
            continue

        results_data.append(res)
        total_processed += 1

        # Accepted/Rejected summary (based on Flagged flag: accepted = not flagged)
        if res.get("Flagged", False):
            rejected_count += 1
        else:
            accepted_count += 1

        Retry_num = res.get("Retry_num", 0)
        if Retry_num > 0:
            retry_stats[Retry_num] = retry_stats.get(Retry_num, 0) + 1

        # Save overlay into Outputs/<Configured Folder>/<Overlays>/... (per-image)
        if CONFIG["RUN"].get("SAVE_OVERLAYS", False):
            try:
                img = read_image_with_orientation(fpath)
                overlay = draw_results(
                    img,
                    res["leaves"],
                    res["Pixel_cm_ratio"],
                    res["label"],
                    res["calibration_points"],
                    qr_info=res.get("qr_info"),
                )

                # Gather naming pieces with safe fallbacks
                lbl = res.get("label", {}) or {}
                date_str = lbl.get("Date") or _persistent_date
                if not date_str:
                    date_str = lbl.get("Original_Stem") or os.path.splitext(File)[0]
                date_str = str(date_str).replace("-", "_")

                # Build overlay path using unified helpers
                outputs_root = CONFIG["PATHS"].get("DIR_OUTPUTS", get_working_directory())
                overlay_subdir = CONFIG["PATHS"].get("OVERLAY_SUBDIR", "Overlays")
                overlay_path = build_output_image_path(
                    date_str,
                    "_overlay.jpg",
                    overlay_subdir,
                    outputs_root,
                    label_info=lbl
                )

                cv.imwrite(overlay_path, overlay)
                if debug:
                    debug.log(f"Saved overlay: {overlay_path}", level=2)
            except Exception as e:
                if debug:
                    debug.log(f"Failed to save overlay for {File}: {e}", level=1)

        # Rename in-place or duplicate into Outputs/Renamed_Images (per-image)
        New_File = res["label"].get("New_File")
        if New_File:
            src_dirname = os.path.dirname(fpath)
            target_path_in_place = os.path.join(src_dirname, New_File)
            if rename_flag:
                try:
                    if os.path.abspath(fpath) != os.path.abspath(target_path_in_place):
                        os.rename(fpath, target_path_in_place)
                        if debug:
                            debug.log(f"Renamed original file to: {target_path_in_place}", level=2)
                    else:
                        if debug:
                            debug.log("Rename skipped (same name).", level=2)
                except Exception as e:
                    if debug:
                        debug.log(f"Failed to rename original file: {e}", level=1)
        else:
            if debug:
                debug.log(f"Skipping rename for {File} - missing New_File", level=2)

        # Build main CSV row (base metrics)
        try:
            label_dict = res.get("label") or {}
            # Date/time formats for CSV
            date_data_fmt = str(CONFIG.get("FILE_FORMATS", {}).get("DATE_DATA_FORMAT", "YYYY-MM-DD"))
            datetime_data_fmt = str(CONFIG.get("FILE_FORMATS", {}).get("DATETIME_DATA_FORMAT", "YYYY-MM-DD HH:MM:SS"))
            row = {
                "File": File,
                "New_File": label_dict.get("New_File", ""),
                "Leaf_Num": res["Leaf_Num"],
                "Leaf_Area": round(res["Leaf_Area"], 3) if res.get("Pixel_cm_ratio") else None,
                "Pixel_cm_ratio": round(res["Pixel_cm_ratio"], 4) if res.get("Pixel_cm_ratio") else None,
                "QR_detected": res.get("QR_detected", False),
                "QR_count": res.get("QR_count", 0),
                "Flagged": res.get("Flagged", False),
                "Date_Analyzed": dt.datetime.now().strftime(_tokens_to_strptime(datetime_data_fmt)),
            }
            # Add only configured input label fields from CONFIG['INPUTS']['LABELS']
            input_names = CONFIG.get("INPUTS", {}).get("LABELS", []) or CONFIG.get("INPUTS", {}).get("NAMES", []) or []
            for name in input_names:
                val = label_dict.get(name, "")
                row[name] = format_label_value_for_csv(name, val, date_data_fmt)
            # Do not add arbitrary label keys; only configured input names added above
            rows.append(row)
        except Exception as e:
            if debug:
                debug.log(f"Error building CSV row for {File}: {e}", level=1)
            else:
                print(f"Error building CSV row for {File}: {e}")
            continue

        # APPEND/UPDATE MAIN CSV AFTER EACH IMAGE (unified writer handles backups)
        if not out_csv:
            out_csv_name = CONFIG["PATHS"].get("LEAF_AREA_CSV_NAME", "Leaf_Area.csv")
            out_csv = os.path.join(outputs_root, out_csv_name)
        try:
            append_to_csv_with_deduplication([row], out_csv, debug)
        except Exception as e:
            if debug:
                debug.log(f"Error updating main CSV for {File}: {e}", level=1)

        # APPEND/UPDATE LEAF AREA DISTRIBUTION CSV AFTER EACH IMAGE
        try:
            append_to_leaf_distribution_csv_with_deduplication([res], leaf_dist_csv_path, debug)
        except Exception as e:
            if debug:
                debug.log(f"Error updating leaf distribution CSV for {File}: {e}", level=1)

        # APPEND/UPDATE LEAF LENGTH CSV AFTER EACH IMAGE
        try:
            append_to_leaf_metric_csv_with_deduplication([res], leaf_length_csv_path, "length_cm", debug)
        except Exception as e:
            if debug:
                debug.log(f"Error updating leaf length CSV for {File}: {e}", level=1)

        # APPEND/UPDATE LEAF WIDTH CSV AFTER EACH IMAGE
        try:
            append_to_leaf_metric_csv_with_deduplication([res], leaf_width_csv_path, "width_cm", debug)
        except Exception as e:
            if debug:
                debug.log(f"Error updating leaf width CSV for {File}: {e}", level=1)

    # Summary
    if debug and debug.mode >= 1 and total_processed > 0:
        print(f"\n=== SEGMENTATION REVIEW SUMMARY ===")
        print(f"Total images processed: {total_processed}")
        print(f"Segmentations accepted: {accepted_count} ({accepted_count/total_processed*100:.1f}%)")
        if rejected_count > 0:
            print(f"Segmentations flagged: {rejected_count} ({rejected_count/total_processed*100:.1f}%)")
        if retry_stats:
            print(f"\nRetry Usage:")
            for rnum, count in sorted(retry_stats.items()):
                print(f"  {count} image(s) required {rnum} retr{'y' if rnum == 1 else 'ies'}")
        else:
            print("No retries were requested")
        print(f"\nOutputs directory: {outputs_root}")
        print("CSVs updated incrementally after each image")

    return rows


# ----------------------------------------------------------------------------------------------------------------------------------
# 16) MAIN EXECUTION FUNCTIONS
# ----------------------------------------------------------------------------------------------------------------------------------
def process_with_preset_settings(date_str, label_value, input_path=None, output_csv_name=None):
    """Preset processing that writes CSVs, overlays, and debug panels directly into Outputs (no per-input subfolders)."""
    print("=== Processing with Preset Settings ===")
    set_persistent_settings(date_str, label_value)

    # Resolve source directory
    src_dir = input_path if input_path is not None else CONFIG["PATHS"].get("DIR_SAMPLE_IMAGES")

    # Base output root (Outputs)
    out_root = CONFIG["PATHS"].get("DIR_OUTPUTS", os.path.join(get_working_directory(), "Outputs"))
    os.makedirs(out_root, exist_ok=True)

    # Prepare overlay dir inside Outputs (if enabled)
    overlay_dir = os.path.join(out_root, CONFIG["PATHS"].get("OVERLAY_SUBDIR", "Overlays")) if CONFIG["RUN"].get("SAVE_OVERLAYS", False) else None
    if overlay_dir:
        os.makedirs(overlay_dir, exist_ok=True)

    # Always prepare Debug dir inside Outputs (regardless of SAVE_DEBUG_IMAGES)
    debug_dir = os.path.join(out_root, CONFIG["PATHS"].get("DEBUG_SUBDIR", "Debug"))
    try:
        os.makedirs(debug_dir, exist_ok=True)
    except Exception:
        pass

    # CSV filename lives directly in Outputs
    csv_filename = output_csv_name if output_csv_name else CONFIG["PATHS"].get("LEAF_AREA_CSV_NAME", "Leaf_Area.csv")
    out_csv = os.path.join(out_root, csv_filename) if csv_filename else None

    # Validate input path
    if not src_dir or not os.path.exists(src_dir):
        print(f"Error: Input directory does not exist: {src_dir}")
        return []

    # Debugger writing into Outputs/Debug (save_dir set even if SAVE_DEBUG_IMAGES is False)
    dbg = Debugger(
        mode=CONFIG["RUN"].get("TROUBLESHOOT_MODE", 3),
        headless=CONFIG["RUN"].get("HEADLESS", False),
        save_dir=debug_dir,
        auto_save=CONFIG["RUN"].get("SAVE_DEBUG_IMAGES", CONFIG["RUN"].get("SAVE_DEBUG", False))
    )

    # Process the folder
    rows = process_folder(src_dir, out_csv=out_csv, overlay_dir=overlay_dir, debug=dbg)

    # Summary / reporting
    if dbg.mode >= 1:
        print(f"\nProcessed {len(rows)} images with preset settings")
        print(f"Source directory: {src_dir}")
        print(f"Outputs directory: {out_root}")
        print(f"Output CSV: {out_csv}")
        settings = get_current_persistent_settings()
        print(f"\nUsed persistent settings:")
        print(f"  Date: {settings.get('date')}")
        print(f"  Labels: {settings.get('labels')}")
        print(f"  Input Path: {src_dir}")
        print(f"  Debug Panels: {debug_dir}")
    return rows


def run_gatorleaf(new_date, label_value, input_path, output_csv_name):
    """Enhanced convenience function using CONFIG validation for GatorLeaf."""
    print(f"Updating preset settings to: Date={new_date}, Label={label_value}, Path={input_path}, CSV={output_csv_name}")
    
    # Validate input path
    if not input_path:
        print("Error: Input path cannot be empty")
        return []
    
    if not os.path.exists(input_path):
        print(f"Error: Input path does not exist: {input_path}")
        return []
    
    if not os.path.isdir(input_path):
        print(f"Error: Input path is not a directory: {input_path}")
        return []
    
    # Validate output CSV filename
    if not output_csv_name:
        print("Error: Output CSV filename cannot be empty")
        return []
    
    if not output_csv_name.lower().endswith('.csv'):
        print(f"Warning: Output filename should end with .csv, got: {output_csv_name}")
        # Auto-add .csv extension if missing
        output_csv_name = output_csv_name + '.csv'
        print(f"Auto-corrected to: {output_csv_name}")
    
    # Validate date format (basic check)
    if new_date and len(str(new_date).split('_')) != 3:
        print(f"Warning: Date format should be YYYY_MM_DD, got: {new_date}")

    # Process with the provided input path and output CSV name
    return process_with_preset_settings(new_date, label_value, input_path, output_csv_name)

def append_to_csv_with_deduplication(rows, csv_path, debug=None):
    """Append rows to CSV file with deduplication and BACKUP protection.
    Handles Leaf_Area, Leaf_Distribution, Leaf_Length, Leaf_Width based on filename.
    """
    if not rows:
        if debug:
            debug.log("No rows to append to CSV", level=2)
        return

    # Unique key to match plants: prefer CONFIG['RUN']['DEDUP_KEYS'], else use configured input names
    dedup_cfg = CONFIG.get("RUN", {}).get("DEDUP_KEYS")
    if isinstance(dedup_cfg, (list, tuple)) and len(dedup_cfg) > 0:
        unique_key_columns = [str(k) for k in dedup_cfg]
    elif isinstance(dedup_cfg, str) and dedup_cfg.strip():
        unique_key_columns = [p.strip() for p in dedup_cfg.split(',') if p.strip()]
    else:
        unique_key_columns = list(CONFIG.get("INPUTS", {}).get("LABELS", []) or CONFIG.get("INPUTS", {}).get("NAMES", []) or [])
        if not unique_key_columns:
            unique_key_columns = ["File"]

    existing_rows = []
    existing_fieldnames = []
    # Compose fieldnames based on CSV type using configured input names only
    input_names = list(CONFIG.get("INPUTS", {}).get("LABELS", []) or CONFIG.get("INPUTS", {}).get("NAMES", []) or [])
    name_lower = os.path.basename(csv_path).lower()
    if "leaf_distribution" in name_lower or "leaf distribution" in name_lower:
        max_leaves = CONFIG.get("INPUTS", {}).get("LEAF_NUMBER_COLUMNS", 100)
        leaf_cols = [f"L{i}" for i in range(1, max_leaves + 1)]
        base_fieldnames = ["File", "New_File"] + input_names + leaf_cols + ["Date_Analyzed"]
    elif "leaf_length" in name_lower or "leaf length" in name_lower or "leaf_width" in name_lower or "leaf width" in name_lower:
        max_leaves = CONFIG.get("INPUTS", {}).get("LEAF_NUMBER_COLUMNS", 100)
        leaf_cols = [f"L{i}" for i in range(1, max_leaves + 1)]
        base_fieldnames = ["File", "New_File"] + input_names + leaf_cols + ["Date_Analyzed"]
    else:
        # Main Leaf_Area.csv schema
        base_fieldnames = ["File", "New_File"] + input_names + ["Leaf_Num", "Leaf_Area", "Pixel_cm_ratio", "QR_detected", "QR_count", "Flagged", "Date_Analyzed"]

    # Read existing CSV if it exists
    if os.path.exists(csv_path):
        try:
            with open(csv_path, "r", newline="") as f:
                reader = csv.DictReader(f)
                existing_fieldnames = reader.fieldnames or []
                existing_rows = [row for row in reader if any(row.values())]
                if debug:
                    debug.log(f"Read {len(existing_rows)} existing rows from {os.path.basename(csv_path)}", level=2)
        except Exception as e:
            if debug:
                debug.log(f"Error reading existing CSV {csv_path}: {e}", level=1)

    # Preserve legacy columns: union existing header + current schema + new row keys
    fieldnames = list(existing_fieldnames)
    for key in base_fieldnames:
        if key not in fieldnames:
            fieldnames.append(key)
    for new_row in rows:
        for key in new_row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    # Force Date_Analyzed to update on every write
    if "Date_Analyzed" in fieldnames:
        datetime_data_fmt = str(CONFIG.get("FILE_FORMATS", {}).get("DATETIME_DATA_FORMAT", "YYYY-MM-DD HH:MM:SS"))
        now_str = dt.datetime.now().strftime(_tokens_to_strptime(datetime_data_fmt))
        for new_row in rows:
            new_row["Date_Analyzed"] = now_str

    # Normalize rows to current fieldnames, preserving any legacy columns.
    def normalize_for_fieldnames(row):
        normalized = {}
        for k in fieldnames:
            normalized[k] = row.get(k, "")
        return normalized

    existing_rows = [normalize_for_fieldnames(r) for r in existing_rows]

    # Index existing rows by unique key
    def make_key(row):
        parts = []
        for k in unique_key_columns:
            val = row.get(k)
            # Normalize numeric to string; None -> ''
            if val is None:
                sval = ''
            else:
                try:
                    sval = str(val).strip()
                except Exception:
                    sval = ''
            parts.append(sval)
        # If all parts are empty, fallback to File
        if not any(parts):
            parts.append(str(row.get("File") or '').strip())
        return tuple(parts)

    existing_dict = {}
    for row in existing_rows:
        key = make_key(row)
        existing_dict[key] = row

    # Merge new rows
    updated_count = 0
    added_count = 0
    for new_row in rows:
        normalized_new_row = normalize_for_fieldnames(new_row)
        key = make_key(normalized_new_row)
        if key in existing_dict:
            for k, v in new_row.items():
                existing_dict[key][k] = v
            updated_count += 1
        else:
            existing_dict[key] = normalized_new_row
            added_count += 1

    # === BACKUP (single place for all CSV types) ===
    if os.path.exists(csv_path):
        csv_dir = os.path.dirname(csv_path)
        backup_dir = os.path.join(csv_dir, "Backup_Data")
        os.makedirs(backup_dir, exist_ok=True)

        csv_basename = os.path.basename(csv_path)
        name_lower = csv_basename.lower()

        # Decide which flag and label to use
        do_backup = False
        backup_label = None

        if "leaf_area" in name_lower or "leaf area" in name_lower:
            if CONFIG["RUN"].get("SAVE_BACKUP_LEAF_AREA", True):
                do_backup = True
                backup_label = "Leaf_Area"

        elif "leaf_distribution" in name_lower or "leaf distribution" in name_lower:
            if CONFIG["RUN"].get("SAVE_BACKUP_LEAF_DISTRIBUTION", True):
                do_backup = True
                backup_label = "Leaf_Distribution"

        elif "leaf_length" in name_lower or "leaf length" in name_lower:
            if CONFIG["RUN"].get("SAVE_BACKUP_LEAF_LENGTH", True):
                do_backup = True
                backup_label = "Leaf_Length"

        elif "leaf_width" in name_lower or "leaf width" in name_lower:
            if CONFIG["RUN"].get("SAVE_BACKUP_LEAF_WIDTH", True):
                do_backup = True
                backup_label = "Leaf_Width"

        if do_backup and backup_label:
            # BACKUP_FREQUENCY supports minute frequencies (e.g. "5", "5min", "10min", "1min").
            # Numeric values are interpreted as minutes. If not defined, fall back to
            freq_val = CONFIG.get("RUN", {}).get("BACKUP_FREQUENCY", None)

            def parse_duration_to_seconds(v):
                """Parse BACKUP_FREQUENCY as minutes-only and return seconds.

                Accepted formats:
                  - integer or numeric type: interpreted as minutes (e.g. 5 -> 5 minutes)
                  - string with optional minute suffix: '5', '5min', '10m', '1min'

                Returns:
                  - integer seconds (minutes * 60) or None on parse failure / None input
                """
                try:
                    if v is None:
                        return None
                    # Numeric types: interpret as minutes
                    if isinstance(v, (int, float)):
                        return int(v) * 60
                    s = str(v).strip().lower()
                    # numeric-only string -> minutes
                    if s.isdigit():
                        return int(s) * 60
                    # match digits with optional minute unit
                    m = re.match(r"^(\d+)\s*(m|min|mins|minute|minutes)?$", s)
                    if m:
                        return int(m.group(1)) * 60
                except Exception:
                    pass
                return None

            freq_seconds = parse_duration_to_seconds(freq_val)

            # Decide whether to create a new backup now.
            create_backup = True
            if freq_seconds and freq_seconds > 0:
                # Find the most recent backup file for this label (if any)
                candidate_files = [f for f in os.listdir(backup_dir) if f.startswith(f"{backup_label}_") and f.lower().endswith('.csv')]
                if candidate_files:
                    full_paths = [os.path.join(backup_dir, f) for f in candidate_files]
                    latest = max(full_paths, key=os.path.getmtime)
                    try:
                        last_mtime = os.path.getmtime(latest)
                        now_ts = time.time()
                        if (now_ts - last_mtime) < freq_seconds:
                            create_backup = False
                            if debug:
                                debug.log(f"Skipping backup for {csv_basename}; last backup {int((now_ts-last_mtime)/60)} minutes ago (<{freq_seconds/60} min)", level=2)
                    except Exception:
                        # On error, fall back to creating the backup
                        create_backup = True

            if create_backup:
                # Timestamp with seconds for uniqueness
                current_timestamp = dt.datetime.now().strftime("%y%m%d_%H%M%S")
                backup_filename = f"{backup_label}_{current_timestamp}.csv"
                backup_path = os.path.join(backup_dir, backup_filename)
                try:
                    shutil.copy2(csv_path, backup_path)
                    if debug:
                        debug.log(f"Created backup: {backup_path}", level=2)
                except Exception as e:
                    if debug:
                        debug.log(f"Warning: Could not create backup for {csv_basename}: {e}", level=1)

    # === Write updated CSV (atomic) ===
    temp_path = csv_path + ".tmp"
    try:
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        with open(temp_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(existing_dict.values())

        if os.path.exists(csv_path):
            os.replace(temp_path, csv_path)
        else:
            os.rename(temp_path, csv_path)

        try:
            os.utime(csv_path, None)
        except Exception:
            pass

        if debug:
            debug.log(
                f"{os.path.basename(csv_path)} updated: {added_count} new, {updated_count} updated",
                level=1
            )
    except Exception as e:
        if debug:
            debug.log(f"Error writing CSV {csv_path}: {e}", level=1)
        else:
            print(f"Error writing CSV {csv_path}: {e}")
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass
        raise


def append_to_leaf_distribution_csv_with_deduplication(results_list, csv_path, debug=None):
    """
    Build leaf distribution rows (per-sample L1..LN areas) and delegate write+backup
    to append_to_csv_with_deduplication.
    """
    if not results_list:
        if debug:
            debug.log("No results to append to leaf distribution CSV", level=2)
        return

    max_leaves = CONFIG.get("INPUTS", {}).get("LEAF_NUMBER_COLUMNS", 100)
    new_rows = []

    for result in results_list:
        lbl = result.get("label", {}) or {}
        datetime_data_fmt = str(CONFIG.get("FILE_FORMATS", {}).get("DATETIME_DATA_FORMAT", "YYYY-MM-DD HH:MM:SS"))
        row = {
            "File": os.path.basename(result.get("image_path", "")),
            "New_File": lbl.get("New_File", ""),
            "Date_Analyzed": dt.datetime.now().strftime(_tokens_to_strptime(datetime_data_fmt)),
        }
        # Add only configured input label fields, formatting DATE per CSV config
        date_data_fmt = str(CONFIG.get("FILE_FORMATS", {}).get("DATE_DATA_FORMAT", "YYYY-MM-DD"))
        input_names = CONFIG.get("INPUTS", {}).get("LABELS", []) or CONFIG.get("INPUTS", {}).get("NAMES", []) or []
        for name in input_names:
            val = lbl.get(name, "")
            row[name] = format_label_value_for_csv(name, val, date_data_fmt)

        # Initialize L1..LN as NA
        for i in range(1, max_leaves + 1):
            row[f"L{i}"] = "NA"

        leaves = result.get("leaves", [])
        pix_ratio = result.get("Pixel_cm_ratio")
        if leaves and pix_ratio:
            sorted_leaves = sorted(leaves, key=lambda x: x["area_px"], reverse=True)
            for idx, leaf in enumerate(sorted_leaves):
                if idx < max_leaves:
                    area_cm2 = leaf["area_px"] / (pix_ratio ** 2)
                    row[f"L{idx + 1}"] = round(area_cm2, 4)

        new_rows.append(row)

    if not new_rows:
        if debug:
            debug.log("No new distribution rows built", level=2)
        return

    # Use the unified writer (handles backups based on filename)
    append_to_csv_with_deduplication(new_rows, csv_path, debug)

def append_to_leaf_metric_csv_with_deduplication(results_list, csv_path, metric_key, debug=None):
    """
    Build per-leaf metric CSV rows (length or width) and delegate write+backup
    to append_to_csv_with_deduplication.

    metric_key: 'length_cm' or 'width_cm'
    """
    if not results_list:
        if debug:
            debug.log(f"No results to append to {metric_key} CSV", level=2)
        return

    max_leaves = CONFIG.get("INPUTS", {}).get("LEAF_NUMBER_COLUMNS", 100)

    new_rows = []
    for result in results_list:
        lbl = result.get("label", {}) or {}
        datetime_data_fmt = str(CONFIG.get("FILE_FORMATS", {}).get("DATETIME_DATA_FORMAT", "YYYY-MM-DD HH:MM:SS"))
        row = {
            "File": os.path.basename(result.get("image_path", "")),
            "New_File": lbl.get("New_File", ""),
            "Date_Analyzed": dt.datetime.now().strftime(_tokens_to_strptime(datetime_data_fmt)),
        }
        # Add only configured input label fields, formatting DATE per CSV config
        date_data_fmt = str(CONFIG.get("FILE_FORMATS", {}).get("DATE_DATA_FORMAT", "YYYY-MM-DD"))
        input_names = CONFIG.get("INPUTS", {}).get("LABELS", []) or CONFIG.get("INPUTS", {}).get("NAMES", []) or []
        for name in input_names:
            val = lbl.get(name, "")
            row[name] = format_label_value_for_csv(name, val, date_data_fmt)

        # Initialize L1..LN as NA
        for i in range(1, max_leaves + 1):
            row[f"L{i}"] = "NA"

        leaves = result.get("leaves", [])
        if leaves and result.get("Pixel_cm_ratio"):
            enriched = [leaf for leaf in leaves if metric_key in leaf]
            sorted_leaves = sorted(enriched, key=lambda x: x.get("area_px", 0), reverse=True)
            for idx, leaf in enumerate(sorted_leaves):
                if idx < max_leaves:
                    row[f"L{idx + 1}"] = round(leaf[metric_key], 4)

        new_rows.append(row)

    if not new_rows:
        if debug:
            debug.log(f"No new {metric_key} rows built", level=2)
        return

    # Use the unified writer (handles backups based on filename)
    append_to_csv_with_deduplication(new_rows, csv_path, debug)

def save_leaf_distribution_csv(results_list, csv_path, debug=None):
    """Legacy function - now calls the append function with deduplication."""
    append_to_leaf_distribution_csv_with_deduplication(results_list, csv_path, debug)


# ----------------------------------------------------------------------------------------------------------------------------------
# 17) ML TRAINING ANNOTATION FUNCTIONS
# ----------------------------------------------------------------------------------------------------------------------------------
def save_ml_annotations(img, leaf_mask, components, Pixel_ratio, label_info, output_dir, filename_base,
                              calib_exclude_mask=None, label_exclude_mask=None, qr_info=None):
    
    # Global override: if SAVE_ML_TRAINING_DATA or SAVE_ML_DATA is not True, do nothing
    if not CONFIG["RUN"].get("SAVE_ML_TRAINING_DATA", CONFIG["RUN"].get("SAVE_ML_DATA", False)):
        return
    """Save ML/ML training artifacts selectively based on CONFIG['ML_TRAINING_OUTPUTS'] flags.
    Ensures the top-level training directory exists, and uses label_info['New_File'] as base name if present.
    """
    # Ensure the top-level training directory exists
    try:
        os.makedirs(output_dir, exist_ok=True)
    except Exception:
        pass

    # Prefer the user-defined file name base (without extension) if available
    try:
        if label_info and label_info.get("New_File"):
            filename_base = os.path.splitext(label_info["New_File"])[0]
    except Exception:
        pass

    # Only save if we actually detected leaves and have a good mask
    if len(components) == 0 or leaf_mask is None:
        print(f"⚠️  Skipping ML annotations for {filename_base} - no leaves detected")
        return

    # Check if mask is mostly empty
    mask_coverage = np.sum(leaf_mask > 0) / leaf_mask.size
    if mask_coverage < 0.01:  # Less than 1% coverage
        print(f"⚠️  Skipping ML annotations for {filename_base} - mask too sparse ({mask_coverage*100:.1f}% coverage)")
        return

    # Read flags
    outs = CONFIG.get("ML_TRAINING_OUTPUTS", {})
    want_masks = outs.get("Segmentation_Masks", True)
    want_contours = outs.get("Leaf_Contours", True)
    want_objects = outs.get("Object_Annotations", True)
    want_coco = outs.get("COCO", True)
    want_yolo = outs.get("YOLO", True)

    # 1) Binary segmentation mask
    if want_masks:
        save_quality_leaf_mask(img, leaf_mask, output_dir, filename_base)

    # 2) Individual leaf contours
    if want_contours:
        save_leaf_contours(img, leaf_mask, components, output_dir, filename_base)

    # 3) Calibration and label object annotations (visualization + JSON)
    if want_objects:
        save_calibration_label_annotations(img, calib_exclude_mask, label_exclude_mask, output_dir, filename_base, qr_info=qr_info)

    # 4) COCO format JSON (in dedicated COCO subfolder)
    if want_coco:
        save_comprehensive_coco_annotations(img, leaf_mask, components, Pixel_ratio, label_info,
                                            calib_exclude_mask, label_exclude_mask, output_dir, filename_base, qr_info=qr_info)

    # 5) YOLO txt (in dedicated YOLO subfolder)
    if want_yolo:
        save_comprehensive_yolo_annotations(img, leaf_mask, components, calib_exclude_mask,
                                            label_exclude_mask, output_dir, filename_base, qr_info=qr_info)

    print(f"✅ Saved ML annotations for {filename_base} ({len(components)} leaves, {mask_coverage*100:.1f}% coverage)")


def save_quality_leaf_mask(img, leaf_mask, output_dir, filename_base):
    """Save the actual segmentation mask, not empty rectangles."""
    masks_dir = os.path.join(output_dir, CONFIG["PATHS"]["MASKS_SUBDIR"])
    os.makedirs(masks_dir, exist_ok=True)
    
    # Save the binary mask
    mask_path = os.path.join(masks_dir, f"{filename_base}_mask.png")
    cv.imwrite(mask_path, leaf_mask)
    
    # Create colored overlay for visualization
    overlay = img.copy()
    green_mask = np.zeros_like(img)
    green_mask[:, :, 1] = leaf_mask  # Green channel
    overlay = cv.addWeighted(overlay, 0.7, green_mask, 0.3, 0)
    
def save_leaf_contours(img, leaf_mask, components, output_dir, filename_base):
    """Extract and save individual leaf contours."""
    contours_dir = os.path.join(output_dir, CONFIG["PATHS"]["CONTOURS_SUBDIR"])
    os.makedirs(contours_dir, exist_ok=True)
    
    # Find contours in the leaf mask
    contours, _ = cv.findContours(leaf_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    # Create visualization
    contour_img = img.copy()
    colors = [(255,0,0), (0,255,0), (0,0,255), (255,255,0), (255,0,255), (0,255,255)]
    
    contour_data = []
    for i, contour in enumerate(contours):
        if len(contour) >= 3:  # Valid contour
            color = colors[i % len(colors)]
            cv.drawContours(contour_img, [contour], -1, color, 2)
            points = contour.reshape(-1, 2).tolist()
            contour_data.append({
                "leaf_id": i + 1,
                "points": points,
                "area": cv.contourArea(contour)
            })
    
    # Save contour visualization and data
    contour_path = os.path.join(contours_dir, f"{filename_base}_contours.jpg")
    cv.imwrite(contour_path, contour_img)
    if contour_data:
        json_path = os.path.join(contours_dir, f"{filename_base}_contours.json")
        with open(json_path, 'w') as f:
            json.dump(contour_data, f, indent=2)

def save_comprehensive_coco_annotations(img, leaf_mask, components, Pixel_ratio, label_info,
                                       calib_exclude_mask, label_exclude_mask, output_dir, filename_base, qr_info=None):
    """Save COCO annotations including leaves, calibration cards, and labels to a dedicated COCO subfolder."""
    H, W = img.shape[:2]
    coco_dir = os.path.join(output_dir, CONFIG["PATHS"].get("COCO_SUBDIR"))
    os.makedirs(coco_dir, exist_ok=True)

    coco_data = {
        "images": [{
            "id": 1,
            "file_name": f"{filename_base}.jpg",
            "width": int(W),
            "height": int(H)
        }],
        "categories": [
            {"id": 1, "name": "leaf", "supercategory": "sample"},
            {"id": 2, "name": "calibration_card", "supercategory": "equipment"},
            {"id": 3, "name": "label", "supercategory": "equipment"},
            {"id": 4, "name": "qr_square", "supercategory": "equipment"}
        ],
        "annotations": []
    }
    annotation_id = 1

    # Add leaf annotations
    leaf_contours, _ = cv.findContours(leaf_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    for contour in leaf_contours:
        if len(contour) >= 3 and cv.contourArea(contour) > 50:
            x, y, w, h = cv.boundingRect(contour)
            area_px = cv.contourArea(contour)
            segmentation = contour.reshape(-1, 2).flatten().tolist()
            coco_data["annotations"].append({
                "id": annotation_id,
                "image_id": 1,
                "category_id": 1,  # leaf
                "bbox": [int(x), int(y), int(w), int(h)],
                "area": float(area_px),
                "area_cm2": float(area_px / (Pixel_ratio ** 2)) if Pixel_ratio > 0 else 0,
                "iscrowd": 0,
                "segmentation": [segmentation]
            })
            annotation_id += 1

    # Add calibration card annotations
    if calib_exclude_mask is not None:
        calib_contours, _ = cv.findContours(calib_exclude_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        for contour in calib_contours:
            if len(contour) >= 3 and cv.contourArea(contour) > 100:
                x, y, w, h = cv.boundingRect(contour)
                area_px = cv.contourArea(contour)
                segmentation = contour.reshape(-1, 2).flatten().tolist()
                coco_data["annotations"].append({
                    "id": annotation_id,
                    "image_id": 1,
                    "category_id": 2,  # calibration_card
                    "bbox": [int(x), int(y), int(w), int(h)],
                    "area": float(area_px),
                    "iscrowd": 0,
                    "segmentation": [segmentation]
                })
                annotation_id += 1

    # Add label annotations
    if label_exclude_mask is not None:
        label_contours, _ = cv.findContours(label_exclude_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        for contour in label_contours:
            if len(contour) >= 3 and cv.contourArea(contour) > 100:
                x, y, w, h = cv.boundingRect(contour)
                area_px = cv.contourArea(contour)
                segmentation = contour.reshape(-1, 2).flatten().tolist()
                coco_data["annotations"].append({
                    "id": annotation_id,
                    "image_id": 1,
                    "category_id": 3,  # label
                    "bbox": [int(x), int(y), int(w), int(h)],
                    "area": float(area_px),
                    "iscrowd": 0,
                    "segmentation": [segmentation]
                })
                annotation_id += 1

    # Add QR square annotation
    if qr_info and qr_info.get("QR_detected") and qr_info.get("bbox"):
        x, y, w, h = qr_info["bbox"]
        coco_data["annotations"].append({
            "id": annotation_id,
            "image_id": 1,
            "category_id": 4,  # qr_square
            "bbox": [int(x), int(y), int(w), int(h)],
            "area": float(int(w) * int(h)),
            "iscrowd": 0,
            "segmentation": []
        })
        annotation_id += 1

    # Add comprehensive metadata
    coco_data["metadata"] = {
        "pixels_per_cm": float(Pixel_ratio) if Pixel_ratio is not None else None,
        "labels": label_info if isinstance(label_info, dict) else None,
        "total_leaves": len([a for a in coco_data["annotations"] if a["category_id"] == 1]),
        "total_calibration_cards": len([a for a in coco_data["annotations"] if a["category_id"] == 2]),
        "total_labels": len([a for a in coco_data["annotations"] if a["category_id"] == 3]),
        "QR_detected": bool(qr_info.get("QR_detected")) if qr_info else False,
        "QR_count": int(qr_info.get("QR_count", 0)) if qr_info else 0,
        "QR_marker_ids": qr_info.get("QR_marker_ids", []) if qr_info else []
    }

    # Save comprehensive COCO JSON to COCO subfolder
    json_path = os.path.join(coco_dir, f"{filename_base}_comprehensive_coco.json")
    with open(json_path, 'w') as f:
        json.dump(coco_data, f, indent=2)

def save_comprehensive_yolo_annotations(img, leaf_mask, components, calib_exclude_mask, label_exclude_mask, output_dir, filename_base, qr_info=None):
    """Save YOLO format with all object classes: 0=leaf, 1=calibration_card, 2=label, 3=qr_square."""
    H, W = img.shape[:2]
    yolo_lines = []
    
    # Class 0: Leaves
    leaf_contours, _ = cv.findContours(leaf_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    for contour in leaf_contours:
        if cv.contourArea(contour) > 50:
            x, y, w, h = cv.boundingRect(contour)
            center_x, center_y = (x + w/2) / W, (y + h/2) / H
            norm_w, norm_h = w / W, h / H
            yolo_lines.append(f"0 {center_x:.6f} {center_y:.6f} {norm_w:.6f} {norm_h:.6f}")
    
    # Class 1: Calibration cards
    if calib_exclude_mask is not None and label_exclude_mask is not None:
        combined_exclude_mask = cv.bitwise_or(calib_exclude_mask, label_exclude_mask)
        calib_contours, _ = cv.findContours(calib_exclude_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        for contour in calib_contours:
            if cv.contourArea(contour) > 100:
                x, y, w, h = cv.boundingRect(contour)
                center_x, center_y = (x + w/2) / W, (y + h/2) / H
                norm_w, norm_h = w / W, h / H
                yolo_lines.append(f"1 {center_x:.6f} {center_y:.6f} {norm_w:.6f} {norm_h:.6f}")
    
    # Class 2: Labels
    if label_exclude_mask is not None:
        label_contours, _ = cv.findContours(label_exclude_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        for contour in label_contours:
            if cv.contourArea(contour) > 100:
                x, y, w, h = cv.boundingRect(contour)
                center_x, center_y = (x + w/2) / W, (y + h/2) / H
                norm_w, norm_h = w / W, h / H
                yolo_lines.append(f"2 {center_x:.6f} {center_y:.6f} {norm_w:.6f} {norm_h:.6f}")

        # Class 3: QR square
        if qr_info and qr_info.get("QR_detected") and qr_info.get("bbox"):
            x, y, w, h = qr_info["bbox"]
            center_x, center_y = (x + w/2) / W, (y + h/2) / H
            norm_w, norm_h = w / W, h / H
            yolo_lines.append(f"3 {center_x:.6f} {center_y:.6f} {norm_w:.6f} {norm_h:.6f}")
    
    # New: Place YOLO files into a dedicated subdirectory
    yolo_dir = os.path.join(output_dir, CONFIG["PATHS"]["YOLO_SUBDIR"])
    os.makedirs(yolo_dir, exist_ok=True)

    txt_path = os.path.join(yolo_dir, f"{filename_base}_comprehensive.txt")
    with open(txt_path, 'w') as f:
        f.write('\n'.join(yolo_lines))

    # Save the class names file in the YOLO directory as well
    classes_path = os.path.join(yolo_dir, "classes.txt")
    with open(classes_path, 'w') as f:
        f.write("leaf\ncalibration_card\nlabel\nqr_square\n")

def save_calibration_label_annotations(img, calib_exclude_mask, label_exclude_mask, output_dir, filename_base, qr_info=None):
    """Save annotations for calibration cards and labels for AI training."""
    objects_dir = os.path.join(output_dir, CONFIG["PATHS"]["OBJECTS_SUBDIR"]) 
    os.makedirs(objects_dir, exist_ok=True)
    
    # Create visualization image
    annotated_img = img.copy()
    
    object_data = {
        "calibration_cards": [],
        "labels": [],
        "qr_squares": []
    }
    
    # Process calibration card mask
    if calib_exclude_mask is not None and label_exclude_mask is not None:
        combined_exclude_mask = cv.bitwise_or(calib_exclude_mask, label_exclude_mask)
        calib_contours, _ = cv.findContours(calib_exclude_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        for i, contour in enumerate(calib_contours):
            if cv.contourArea(contour) > 100:  # Filter tiny artifacts
                # Get bounding box
                x, y, w, h = cv.boundingRect(contour)
                # Draw on the visualization image
                cv.drawContours(annotated_img, [contour], -1, (255, 0, 0), 3) # Blue contour
                cv.rectangle(annotated_img, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv.putText(annotated_img, f"Calib {i+1}", (x, y-10), 
                          cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                
                # Store data for JSON
                object_data["calibration_cards"].append({
                    "id": i + 1,
                    "bbox": [int(x), int(y), int(w), int(h)],
                    "contour": contour.reshape(-1, 2).tolist(),
                    "area": float(cv.contourArea(contour))
                })
    
    # Process label/bag mask
    if label_exclude_mask is not None:
        label_contours, _ = cv.findContours(label_exclude_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        for i, contour in enumerate(label_contours):
            if cv.contourArea(contour) > 100:  # Filter tiny artifacts
                # Get bounding box
                x, y, w, h = cv.boundingRect(contour)
                # Draw on the visualization image
                cv.drawContours(annotated_img, [contour], -1, (0, 0, 255), 3) # Red contour
                cv.rectangle(annotated_img, (x, y), (x+w, y+h), (0, 0, 255), 2)
                cv.putText(annotated_img, f"Label {i+1}", (x, y-10), 
                          cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Store data for JSON
                object_data["labels"].append({
                    "id": i + 1,
                    "bbox": [int(x), int(y), int(w), int(h)],
                    "contour": contour.reshape(-1, 2).tolist(),
                    "area": float(cv.contourArea(contour))
                })

    # Process QR square (if detected)
    if qr_info and qr_info.get("QR_detected") and qr_info.get("bbox"):
        x, y, w, h = qr_info["bbox"]
        cv.rectangle(annotated_img, (x, y), (x + w, y + h), (255, 0, 255), 2)
        ids_text = "QR" if not qr_info.get("QR_marker_ids") else "QR: " + ",".join([str(i) for i in qr_info.get("QR_marker_ids", [])])
        cv.putText(annotated_img, ids_text, (x, max(0, y - 10)),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        object_data["qr_squares"].append({
            "id": 1,
            "bbox": [int(x), int(y), int(w), int(h)],
            "QR_marker_ids": qr_info.get("QR_marker_ids", []),
            "QR_count": qr_info.get("QR_count", 0)
        })
    
    # Save visualization image
    vis_path = os.path.join(objects_dir, f"{filename_base}_objects.jpg")
    cv.imwrite(vis_path, annotated_img)
    
    # Save object data as JSON
    json_path = os.path.join(objects_dir, f"{filename_base}_objects.json")
    with open(json_path, 'w') as f:
        json.dump(object_data, f, indent=2)


# ------------------------------------------------------------------------------------------
# FINAL SETUP
# ------------------------------------------------------------------------------------------

def main():
    """Main function to run the analysis on all experiment folders."""
    show_startup_window()
    base_input_dir = CONFIG["PATHS"]["DIR_SAMPLE_IMAGES"]
    if not os.path.exists(base_input_dir):
        print(f"❌ Input directory not found: {base_input_dir}")
        print(f"Please create it and place your experiment folders inside.")
        return
    experiment_folders = [f for f in os.listdir(base_input_dir) if os.path.isdir(os.path.join(base_input_dir, f))]
    if not experiment_folders:
        print(f"⚠️ No experiment folders found in {base_input_dir}")
        return
    print(f"Found {len(experiment_folders)} experiments: {', '.join(experiment_folders)}")
    for experiment_name in experiment_folders:
        bar = "=" * 20
        print(f"\n\n{bar} PROCESSING EXPERIMENT: {experiment_name} {bar}")
        input_path = os.path.join(base_input_dir, experiment_name)
        output_csv_name = CONFIG["PATHS"]["LEAF_AREA_CSV_NAME"]
        reset_persistent_settings()
        # Legacy auto-detection of experiment/date removed. Run directly; UI collects labels.
        results = run_gatorleaf(None, None, input_path, output_csv_name)
        if not results:
            print(f"No images processed for experiment {experiment_name}.")
    move_outputs_to_current_directory()
    print(f"\n\n🎉 ALL EXPERIMENTS COMPLETE! 🎉")
    if is_frozen():
        results_dir = os.path.join(os.getcwd(), "Outputs")
        print(f"📂 Your results are saved in: {results_dir}")
    else:
        print(f"📂 Your results are saved in: {CONFIG['PATHS']['DIR_OUTPUTS']}")


if __name__ == '__main__':
    main()
