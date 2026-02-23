GatorLeaf — config.JSON Instruction Manual

This README is a complete reference for every setting available in `config.JSON`. Use it to understand allowed values, default behaviors, and how each setting impacts the app.

File Location
- `config.JSON` lives at the project root. The app reads it at startup.

Top-Level Sections
- `INPUTS`: Defines label fields, their input modes, and UI choices.
- `FILE_FORMATS`: Defines input/output filename patterns and date/time formatting.
- `CALIB`: Calibration length and persistence settings.
- `UI`: UI sizing options.
- `ANNOTATIONS`: Non-leaf object classes for object annotation outputs.
- `COLORS`: UI colors for label selection buttons.
- `PATHS`: Output file names and output subdirectories.
- `RUN`: Runtime switches (labeling, calibration mode, backups, overlays, etc.).
- `ML_TRAINING_OUTPUTS`: Output switches for ML training artifacts.

----------------------------------------
1) INPUTS
----------------------------------------
Purpose: Controls label fields shown in the UI and the options/ranges for each field.

Keys
- `LABELS` (array of strings)
  - The ordered list of label field names stored in csv data and used in UI panels.
  - Example: `["Date", "Exp_ID", "Exp_Num", "Genotype", "Sample", "Location", "Replicate", "Harvest_Time"]`
  - The above labels will be referenced in order as L1 - Ln where you can include n number of labels
  - L1 = Date, L2 = Exp_ID, L3 = Exp_Num, L4 = Genotype, L5 = Sample, L6  = Location, L7 = Replicate, L8 = Harvest_Time

Label field definitions
- `L1`, `L2`, `L3`, ...
  - Each `Lx` defines the input mode and values for the label at the same index in `LABELS`.
  - Format: `[TYPE, ...VALUES]`
  - Supported types:
    - `DATE`: Uses date selection panels (year/month/day) based on `YEAR/MONTH/DAY` ranges (date)
    - `SELECT`: Shows a selection panel using the provided options (string).
    - `RANGE`: Shows a numeric entry panel with min/max (numeric).
    - `TIME`: Uses time selection panels (hour/minute/second) based on `HOUR:MINUTE:SECOND` ranges (time)
  - Examples from current config:
    - `"L1": ["DATE", "MONTH", "DAY", "YEAR"]` (uses the date ranges below)       (L1 = Date)
    - `"L2": ["SELECT", "Control", "Treatment"]`                                  (L2 = Exp_ID)
    - `"L3": ["RANGE", 1, 34]`                                                    (L3 = Exp_Num)
    - `"L4": ["SELECT", "B. napus", "B. juncea", "B. olaraceae", "B. rapa"]`      (L4 = Genotype)
    - `"L5": ["RANGE", 1, 10]`                                                    (L5 = Sample)
    - `"L6": ["SELECT", "Control", "Field", "Hoophouse", "Growth Chamber"]`       (L6 = Location)
    - `"L7": ["RANGE", 1, 5]`                                                     (L7 = Replicate)
    - `"L8": ["TIME", "HOUR", "MINUTE", "SECOND"]`                                (L8 = Harvest_Time)


Example of Leaf_Area.csv headers and data types based on the `LABELS` and `TYPE` define above
- `File,New_File,Date,Exp_ID,Exp_Num,Cultivar,Sample,Location,Replicate,Harvest_Time,Leaf_Num,Leaf_Area,Pixel_cm_ratio,QR_detected,QR_count,Flagged,Date_Analyzed`
- `str,str,date,str,num,str,num,str,num,time,num,num,num,bool,num,bool,datetime`

Date/time ranges
- `YEAR`: Selection of year values shown in the Year panel.
- `MONTH`: Selection month values shown in the Month picker.
- `DAY`: Selection day values shown in the Day picker.
- `HOUR`: Selection hour values shown in the Hour picker.
- `MINUTE`: Selection minute values shown in the Minue picker.
- `SECOND`: Selection second values shown in the Second picker.

Distribution/metrics columns
- `LEAF_NUMBER_COLUMNS` (int)
  - Maximum number of L-columns written to Leaf_Distribution/Length/Width CSVs.

----------------------------------------
2) FILE_FORMATS
----------------------------------------
Purpose: Controls date/time formatting and input/output filename parsing.

Data formatting
- `DATE_DATA_FORMAT` (string)
  - CSV date output format, using tokens like `YYYY`, `MM`, `DD`.
- `DATETIME_DATA_FORMAT` (string)
  - CSV datetime output format, using tokens like `YYYY-MM-DD HH:MM:SS`.

Input filename parsing
- `INPUT_DATE_FORMAT` (string or array)
  - Accepted date formats when parsing dates from filenames.
- `INPUT_FILENAME_PATTERN` (string)
  - Token pattern for filename parsing, using `{L1}`, `{L2}`, ...
  - Example: `{L1}_{L2}{L3}_{L4}_{L5}_{L6}_{L7}.jpg`

Output naming
- `OUTPUT_DATE_FORMAT` (string or array)
  - Output date format used in filename generation.
- `OUTPUT_FOLDER_PATTERN` (string)
  - Output subfolder name pattern using `{L1}`, `{L2}`, etc.
- `OUTPUT_FILE_NAME_PATTERN` (string)
  - Output filename pattern using `{L1}`, `{L2}`, etc.

----------------------------------------
3) CALIB
----------------------------------------
Purpose: Controls calibration length and persistence behavior.

Keys
- `CALIBRATION_LENGTH_CM` (number)
  - Real-world length used during calibration (cm).
- `PERSISTENT_CALIBRATION` (boolean)
  - When true, reuses last successful px/cm across images until changed.

----------------------------------------
4) UI
----------------------------------------
Purpose: Controls base UI sizing.

Keys
- `REFERENCE_IMAGE_SIZE` (array [width, height])
  - Maximum size for reference image windows.

----------------------------------------
5) ANNOTATIONS
----------------------------------------
Purpose: Defines object classes for object annotation outputs.

Keys
- `OBJECTS` (array of strings)
  - Labels displayed for object annotation categories.
- `Obj1`, `Obj2`, `Obj3`, ... (arrays)
  - Per-object mode configuration (e.g., `MASK`, `LABEL`).
  - Example: `"Obj1": ["MASK"]`, `"Obj2": ["LABEL"]`.

----------------------------------------
6) COLORS
----------------------------------------
Purpose: Controls button colors for UI panels.

Keys
- `Update_Buttons` (array of hex strings)
  - Button colors for update options in the Image Label panel.
- `L2`, `L4`, `L5`, `L6` (arrays of hex strings)
  - Button colors corresponding to each option for those labels.
  - Use one color per option in the label’s `Lx` list.

----------------------------------------
7) PATHS
----------------------------------------
Purpose: Sets CSV filenames and output subdirectories.

CSV filenames
- `LEAF_AREA_CSV_NAME`
- `LEAF_DIST_CSV_NAME`
- `LEAF_LENGTH_CSV_NAME`
- `LEAF_WIDTH_CSV_NAME`

Output subdirectories
- `OVERLAY_SUBDIR`
- `DEBUG_SUBDIR`

ML training subdirectories
- `TRAINING_SUBDIR`
- `OBJECTS_SUBDIR`
- `MASKS_SUBDIR`
- `CONTOURS_SUBDIR`
- `YOLO_SUBDIR`
- `COCO_SUBDIR`

----------------------------------------
8) RUN
----------------------------------------
Purpose: Feature toggles for labeling, calibration, outputs, and backups.

Core behavior
- `LABEL_IMAGES` (boolean)
  - If true, shows label input UI for each image.
- `QR_CODE_CALIBRATION` (boolean)
  - If true, attempts QR-based calibration before manual calibration.
- `BACKUP_FREQUENCY` (string)
  - Minimum time between CSV backups (e.g., `"10min"`).

Output controls
- `SAVE_DEBUG` (boolean)
  - Save debug panels.
- `SAVE_OVERLAYS` (boolean)
  - Save overlays.
- `SAVE_ML_DATA` (boolean)
  - Save ML training outputs (gated by `ML_TRAINING_OUTPUTS`).
- `RENAME_EXISTING_FILE` (boolean)
  - Rename source images in-place to the generated `New_File`.

Backups
- `BACKUP_LEAF_AREA` (boolean)
- `BACKUP_LEAF_DISTRIBUTION` (boolean)
- `BACKUP_LEAF_LENGTH` (boolean)
- `BACKUP_LEAF_WIDTH` (boolean)

----------------------------------------
9) ML_TRAINING_OUTPUTS
----------------------------------------
Purpose: Fine-grained control of ML outputs saved when `SAVE_ML_DATA` is true.

Keys
- `Leaf_Contours` (boolean)
- `Object_Annotations` (boolean)
- `Segmentation_Masks` (boolean)
- `YOLO` (boolean)
- `COCO` (boolean)

----------------------------------------
Examples
----------------------------------------
1) Disable QR calibration
   - Set `RUN.QR_CODE_CALIBRATION` to `false`.

2) Change label options
   - Edit `INPUTS.LABELS` and corresponding `INPUTS.Lx` entries.
   - Update `COLORS.Lx` arrays to match option count.

3) Limit leaf columns in distribution CSV
   - Reduce `INPUTS.LEAF_NUMBER_COLUMNS`.

----------------------------------------
Notes
----------------------------------------
- Label fields are entirely driven by `INPUTS.LABELS` and matching `L1..Ln` specs.
- Filename parsing and output naming rely on `FILE_FORMATS` patterns using `{L1}`, `{L2}`, etc.
- QR calibration stores `QR_detected` and `QR_count` in output CSVs.

