# GatorLeaf — config.JSON Instruction Manual

This file is a complete reference for every setting available in `config.JSON`.
Use it to understand allowed values, default behaviors, and how each setting
impacts the app.

---

## File Location

`config.JSON` lives in the same folder as `GatorLeaf.app` (your working folder).
The app reads it automatically at startup. If no `config.JSON` is present,
GatorLeaf uses its built-in default settings.

```
Your Working Folder/
|-- GatorLeaf.app
|-- config.JSON        <- place it here
|-- Inputs/
|-- Outputs/
```

---

## Top-Level Sections

| Section | Description |
|---|---|
| `INPUTS` | Defines label fields, input modes, and UI choices |
| `FILE_FORMATS` | Defines filename patterns and date/time formatting |
| `CALIB` | Calibration length and persistence settings |
| `UI` | UI window sizing options |
| `COLORS` | UI button colors for label selection panels |
| `PATHS` | Output filenames and subdirectory names |
| `RUN` | Runtime switches (labeling, outputs, backups, etc.) |
| `ML_TRAINING_OUTPUTS` | Output switches for ML training artifacts |
| `SEG` | Segmentation tuning parameters |

---

## 1) INPUTS

Defines label fields shown in the UI and the options or ranges for each field.

### LABELS

- **Type:** array of strings
- The ordered list of label field names stored in CSV data and shown in UI
  panels. Labels are referenced in order as L1 to Ln.

**Example:**
```json
"LABELS": ["Date", "Exp_ID", "Exp_Num", "Cultivar", "Sample_Num"]
```
```
L1 = Date
L2 = Exp_ID
L3 = Exp_Num
L4 = Cultivar
L5 = Sample_Num
```

---

### Label Field Definitions (L1, L2, L3, ...)

Each `Lx` entry defines the input mode and values for the label at the same
position in `LABELS`.

**Format:** `[TYPE, ...VALUES]`

**Supported types:**

| Type | Description |
|---|---|
| `DATE` | Uses date selection panels (year/month/day). Ranges drawn from `YEAR`, `MONTH`, `DAY` keys. |
| `DISCRETE` | Shows a button-selection panel using the provided options. |
| `NUMERIC` | Shows a numeric entry panel with min and max values. |
| `TIME` | Uses time selection panels (hour/minute/second). Ranges drawn from `HOUR`, `MINUTE`, `SECOND` keys. |

**Examples:**
```json
"L1": ["DATE",     "MONTH", "DAY", "YEAR"]              // Date
"L2": ["DISCRETE", "GH", "TNFT"]                        // Exp_ID
"L3": ["NUMERIC",  1, 34]                               // Exp_Num
"L4": ["DISCRETE", "FLB", "CAB", "SGC", "..."]         // Cultivar
"L5": ["NUMERIC",  1, 120]                              // Sample_Num
```

---

### Date / Time Ranges

| Key | Description |
|---|---|
| `YEAR` | List of year values shown in the Year selection panel |
| `MONTH` | List of month values shown in the Month selection panel |
| `DAY` | List of day values shown in the Day selection panel |
| `HOUR` | List of hour values shown in the Hour selection panel |
| `MINUTE` | List of minute values shown in the Minute selection panel |
| `SECOND` | List of second values shown in the Second selection panel |

---

### Distribution / Metrics Columns

- **`LEAF_NUMBER_COLUMNS`** *(integer)*
  Maximum number of L-columns written to Leaf_Distribution, Leaf_Length,
  and Leaf_Width CSVs.

---

### CSV Headers Based on Current Config

**Headers:**
```
File, New_File, Date, Exp_ID, Exp_Num, Cultivar, Sample_Num,
Leaf_Num, Leaf_Area, Pixel_cm_ratio, QR_detected, QR_count,
Flagged, Date_Analyzed
```

**Data types:**
```
str, str, date, str, num, str, num,
num, num, num, bool, num,
bool, datetime
```

---

## 2) FILE_FORMATS

Controls date/time formatting and input/output filename parsing.

### Data Formatting

- **`DATE_DATA_FORMAT`** *(string)*
  CSV date output format using tokens `YYYY`, `MM`, `DD`.

- **`DATETIME_DATA_FORMAT`** *(string)*
  CSV datetime output format.

### Input Filename Parsing

- **`INPUT_DATE_FORMAT`** *(string or array)*
  Accepted date formats when parsing dates from filenames.

- **`INPUT_FILENAME_PATTERN`** *(string)*
  Token pattern used to parse label values from input filenames.
  Uses `{L1}`, `{L2}`, ... tokens matching the `LABELS` order.

### Output Naming

- **`OUTPUT_DATE_FORMAT`** *(string or array)*
  Date format used when building output filenames and folder names.

- **`OUTPUT_FOLDER_PATTERN`** *(string)*
  Output subfolder name pattern using `{L1}`, `{L2}`, etc.

- **`OUTPUT_FILE_NAME_PATTERN`** *(string)*
  Output filename pattern using `{L1}`, `{L2}`, etc.

---

## 3) CALIB

Controls calibration length and persistence behavior.

- **`MANUAL_CALIBRATION_CM`** *(number)*
  The real-world length of the ruler or calibration card used during
  manual calibration (cm).

- **`QR_CODE_CALIBRATION`** *(boolean)*
  If true, attempts QR-based auto-calibration before falling back to
  manual calibration.

- **`PERSISTENT_CALIBRATION`** *(boolean)*
  If true, reuses the last successful px/cm ratio across images in the
  same session until manually changed.

---

## 4) UI

Controls base UI window sizing.

- **`REFERENCE_IMAGE_SIZE`** *(array [width, height])*
  Maximum pixel dimensions for reference image windows. Reduce if windows
  appear too large for your screen.

---

## 5) COLORS

Controls button colors for label selection panels in the UI. Colors are
defined as hex strings (e.g., `"#3c78b4"`).

- **`Labels`**
  Array of hex colors used for the Update buttons in the Image Label panel.
  One color per label field in `LABELS` order.

- **`L2`, `L4`, `L5`, `L6`, ...**
  Arrays of hex colors for each option in that label's `DISCRETE` list.
  Provide one color per option.

> **Note:** The number of colors must match the number of options in the
> corresponding `Lx` DISCRETE list.

---

## 6) PATHS

Sets output CSV filenames and output subdirectory names. All paths are
relative to the Outputs folder.

### CSV Filenames

| Key | Description |
|---|---|
| `LEAF_AREA_CSV_NAME` | Total leaf area results |
| `LEAF_DIST_CSV_NAME` | Individual leaf area distribution |
| `LEAF_LENGTH_CSV_NAME` | Individual leaf lengths |
| `LEAF_WIDTH_CSV_NAME` | Individual leaf widths |

### Output Subdirectories

| Key | Description |
|---|---|
| `OVERLAY_SUBDIR` | Annotated overlay images |
| `DEBUG_SUBDIR` | Side-by-side debug panels |

### ML Training Subdirectories

| Key | Description |
|---|---|
| `TRAINING_SUBDIR` | Root folder for all training artifacts |
| `OBJECTS_SUBDIR` | Calibration and label annotations |
| `MASKS_SUBDIR` | Binary segmentation masks |
| `CONTOURS_SUBDIR` | Individual leaf contours |
| `YOLO_SUBDIR` | YOLO format label files |
| `COCO_SUBDIR` | COCO format annotation files |

---

## 7) RUN

Feature toggles for labeling, calibration, outputs, and backups.

### Core Behavior

- **`LABEL_IMAGES`** *(boolean)*
  If true, shows the label input UI panel for each image. If false,
  attempts to parse labels from filenames only.

- **`SELECT_MASKS`** *(boolean)*
  If true, prompts the user to manually draw exclusion regions
  (calibration card, labels, bags) before segmentation.

- **`BACKUP_FREQUENCY`** *(string)*
  Minimum time between automatic CSV backups. Accepts formats like
  `"10min"`, `"5"`, `"1min"`.

### Output Controls

- **`SAVE_DEBUG`** *(boolean)*
  If true, saves side-by-side debug panels (original image and leaf mask)
  to the Debug subfolder.

- **`SAVE_OVERLAYS`** *(boolean)*
  If true, saves annotated overlay images showing detected leaves to the
  Overlays subfolder.

- **`SAVE_ML_DATA`** *(boolean)*
  If true, saves ML training outputs as controlled by the
  `ML_TRAINING_OUTPUTS` flags.

- **`RENAME_EXISTING_FILE`** *(boolean)*
  If true, renames source images in-place to the filename generated from
  `OUTPUT_FILE_NAME_PATTERN`.

### CSV Backups

| Key | Description |
|---|---|
| `BACKUP_LEAF_AREA` | Enable backups for Leaf_Area.csv |
| `BACKUP_LEAF_DISTRIBUTION` | Enable backups for Leaf_Distribution.csv |
| `BACKUP_LEAF_LENGTH` | Enable backups for Leaf_Length.csv |
| `BACKUP_LEAF_WIDTH` | Enable backups for Leaf_Width.csv |

When true, a timestamped backup copy of that CSV is saved to
`Outputs/Backup_Data/` at the interval set by `BACKUP_FREQUENCY`.

---

## 8) ML_TRAINING_OUTPUTS

Fine-grained control of which ML training artifacts are saved when
`RUN.SAVE_ML_DATA` is true.

| Key | Description |
|---|---|
| `Leaf_Contours` | Saves leaf contour outlines as JSON and visualization images |
| `Object_Annotations` | Saves bounding box annotations for calibration cards and labels |
| `Segmentation_Masks` | Saves binary leaf masks as PNG images |
| `YOLO` | Saves YOLO-format bounding box label files (.txt) |
| `COCO` | Saves COCO-format annotation JSON files |

---

## 9) SEG

Controls leaf segmentation parameters. These are the most impactful settings
for detection accuracy and should be tuned to match your imaging setup.

### Size Thresholds

- **`NOISE_CM2`** *(number)*
  Minimum component area to keep before merging (cm2). Components smaller
  than this are removed as noise.

- **`MIN_LEAF_CM2`** *(number)*
  Minimum final leaf area (cm2). Leaves smaller than this are excluded
  from results.

- **`MIN_LEAF_LENGTH_CM`** *(number)*
  Minimum leaf length (cm). Leaves shorter than this are excluded.

- **`MIN_LEAF_WIDTH_CM`** *(number)*
  Minimum leaf width (cm). Leaves narrower than this are excluded.

### Background

- **`BACKGROUND_COLOR`** *(string)*
  Tells the segmentation pipeline what kind of background to expect.
  Options: `"Black"` `"White"` `"Auto"`
  `"Auto"` detects background type from image corner brightness.

### HSV Thresholds

- **`HSV_S_MIN` / `HSV_S_MAX`** — Saturation range (0–255)
- **`HSV_V_MIN` / `HSV_V_MAX`** — Brightness range (0–255)

### Hue Bands

OpenCV hue range: 0–180
`Red = 0/180` `Yellow = 30` `Green = 60` `Cyan = 90` `Blue = 120` `Magenta = 150`

| Band | Keys |
|---|---|
| GREEN | `GREEN_H_MIN` / `GREEN_H_MAX`, `GREEN_V_MIN` / `GREEN_V_MAX`, `GREEN_S_MIN` / `GREEN_S_MAX` |
| YELLOW | `YELLOW_H_MIN` / `YELLOW_H_MAX` |
| PURPLE | `PURPLE_H_MIN` / `PURPLE_H_MAX`, `PURPLE_S_MIN`, `PURPLE_V_MIN` |

### Exclusion Ranges

| Range | Keys |
|---|---|
| BLACK | `BLACK_V_MIN` / `BLACK_V_MAX` — brightness range treated as black |
| WHITE | `WHITE_S_MIN` / `WHITE_S_MAX` — saturation range treated as white |
| WHITE | `WHITE_V_MIN` / `WHITE_V_MAX` — brightness range treated as white |

### Lab Color Space Gating

Provides additional color filtering in L\*a\*b\* space on top of HSV.
Values use centered a\* and b\* ranges (−128 to 127).

- **`LAB_L_MIN` / `LAB_L_MAX`** — Lightness range
- **`LAB_A_MIN` / `LAB_A_MAX`** — a\* range
- **`LAB_B_MIN` / `LAB_B_MAX`** — b\* range

---

## Examples

**1) Disable QR calibration**
Set `CALIB.QR_CODE_CALIBRATION` to `false`.

**2) Change label fields and options**
Edit `INPUTS.LABELS` to add or rename fields.
Add or update the matching `L1..Ln` entries with the new type and values.
Update `COLORS.Lx` arrays to match the new option count.

**3) Limit leaf columns in distribution CSV**
Reduce `INPUTS.LEAF_NUMBER_COLUMNS` to the maximum expected leaf count.

**4) Speed up processing by skipping ML outputs**
Set `RUN.SAVE_ML_DATA` to `false`.

**5) Adjust for white background imaging**
Set `SEG.BACKGROUND_COLOR` to `"White"` or `"Auto"`.

---

## Notes

- Label fields are entirely driven by `INPUTS.LABELS` and matching `L1..Ln`
  definitions. Adding a new label requires both a new entry in `LABELS` and
  a corresponding `Lx` definition.
- Filename parsing and output naming rely on `FILE_FORMATS` patterns using
  `{L1}`, `{L2}`, etc. tokens. The token count must match `LABELS` length.
- QR calibration stores `QR_detected` and `QR_count` in output CSVs
  regardless of whether QR calibration succeeded.
- SEG parameters are the most experiment-specific settings. If leaves are
  being missed or background is being detected as leaves, start by tuning
  `BACKGROUND_COLOR`, the GREEN hue band, and the HSV brightness thresholds.