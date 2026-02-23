# GatorLeaf - Application Architecture

This document provides visual diagrams of the GatorLeaf application architecture and workflows using Mermaid diagrams. This reflects the current version with config-driven labels, QR calibration, and persistent calibration support.

## Application Architecture Overview

```mermaid
graph TB
    subgraph "User Interface Layer"
        UI[OpenCV Interactive Windows]
        ImageLabel["Image Label Panel<br/>Config-Driven Labels"]
        InputDialogs["Selection Dialogs<br/>Date/Select/Range/Time"]
        Display[Image Display and Review]
    end
    
    subgraph "Core Application"
        Main[GatorLeaf.py Main Entry]
        Config[Configuration Manager<br/>config.JSON]
        Debug[Debugger Class]
        Persistent["Persistent State<br/>Date + Labels + Calibration"]
    end
    
    subgraph "Image Processing Pipeline"
        CalibrationMgmt["Calibration Management<br/>QR or Manual or Persistent"]
        LabelExclude["Label Exclusion Areas<br/>Multi-ROI Selection"]
        Seg["Leaf Segmentation<br/>ExG plus HSV plus Lab"]
        Analysis["Leaf Analysis<br/>Area/Length/Width"]
        Metrics["Metric Calculation<br/>Per-Leaf Stats"]
    end
    
    subgraph "Data Management"
        CSVMain[Main CSV Export<br/>Leaf_Area.csv]
        CSVDist[Distribution CSV<br/>Leaf_Distribution.csv]
        CSVMetrics[Metrics CSV<br/>Leaf_Length.csv<br/>Leaf_Width.csv]
        Overlay[Overlay Generation]
        Rename[Optional Rename In-Place]
        MLExport[ML Training Export<br/>Masks/Contours/YOLO/COCO/Objects]
    end
    
    subgraph "External Dependencies"
        CV2[OpenCV cv2]
        NP[NumPy]
        JSON[JSON Config]
    end
    
    subgraph "File System"
        Input_Dir[Inputs/]
        Config_File[config.JSON]
        Output_Dir[Outputs/]
        Overlays_Dir[Overlays/]
        Debug_Dir[Debug_Panels/]
        ML_Dir[ML_Training_Data/]
        Renamed_Dir[Renamed_Images/]
    end
    
    UI --> Main
    ImageLabel --> Main
    InputDialogs --> Main
    Display --> Main
    
    Main --> Config
    Main --> Debug
    Main --> Persistent
    Main --> CalibrationMgmt
    Main --> LabelExclude
    Main --> Seg
    Main --> Analysis
    
    CalibrationMgmt --> CV2
    Seg --> CV2
    Seg --> NP
    Analysis --> NP
    Metrics --> NP
    
    Analysis --> CSVMain
    Analysis --> CSVDist
    Metrics --> CSVMetrics
    Analysis --> Overlay
    Analysis --> Rename
    Seg --> MLExport
    
    Config --> Config_File
    Main --> Input_Dir
    CSVMain --> Output_Dir
    CSVDist --> Output_Dir
    CSVMetrics --> Output_Dir
    Overlay --> Overlays_Dir
    Debug --> Debug_Dir
    MLExport --> ML_Dir
    Rename --> Renamed_Dir
```

## End-to-End Processing Flow (Current)

```mermaid
flowchart TD
    A([Launch GatorLeaf]) --> B[Show startup window]
    B --> C{Inputs directory exists?}
    C -- No --> C1[Print error and exit]
    C -- Yes --> D[List experiment folders]
    D --> E{Any experiments?}
    E -- No --> E1[Warn and exit]
    E -- Yes --> F[For each experiment]

    F --> G[Reset persistent settings]
    G --> H[Run batch on experiment folder]

    subgraph Batch[Process each image]
        H --> I{Next image?}
        I -- No --> Z[Batch summary]
        I -- Yes --> J[Read image]
        J --> K["Collect label info - config-driven UI"]
        K --> L{Persistent calibration enabled and cached?}
        L -- Yes --> L1[Use cached px/cm]
        L -- No --> M{QR calibration enabled?}
        M -- Yes --> M1[Detect QR + compute px/cm]
        M1 --> M2{QR detected & valid?}
        M2 -- Yes --> N[Set px/cm + calib exclusion]
        M2 -- No --> O[Manual calibration]
        M -- No --> O[Manual calibration]
        O --> O1{Manual calibration canceled?}
        O1 -- Yes --> O2[Return empty results]
        O1 -- No --> N
        L1 --> N

        N --> P[Manual label/bag exclusion]
        P --> Q[Segment leaves]
        Q --> R[Review segmentation]
        R --> S{Decision}
        S -- Finish --> T[Accept Flagged=false]
        S -- Flag --> U[Accept Flagged=true]
        S -- Retry Calib --> O
        S -- Retry Masks --> P

        T --> V[Save overlays/ML outputs if enabled]
        U --> V
        V --> W[Append CSV rows: Leaf_Area, Distribution, Length, Width]
        W --> I
    end

    Z --> F
    F --> Y[Move outputs to current directory]
    Y --> X([Done])

    %% Notes
    M1 -.-> M3[Store QR_detected + QR_count]
    N -.-> N1[Cache px/cm if persistent calibration enabled]
```

## User Interface Flow - Selection Dialogs

```mermaid
flowchart TD
    Launch[Launch GatorLeaf] --> InitUI[Initialize OpenCV<br/>Window Handlers]
    InitUI --> LoadFirstImage[Load First Image<br/>Show in Reference Window]
    
    LoadFirstImage --> ShowImageLabel["Show Image Label Panel<br/>Display Saved Labels (config-driven)<br/>Allow Updates or Done"]
    
    ShowImageLabel --> ImageLabelAction{User Selection?}
    ImageLabelAction -->|Done| CalibStage
    ImageLabelAction -->|Update Label| LabelDialog
    
    LabelDialog["Config-Driven Inputs<br/>Date/Select/Range/Time"] --> ShowImageLabel
    
    CalibStage{Calibration Path}
    CalibStage -->|Persistent cached| UsePersist["Use saved px/cm"]
    CalibStage -->|QR enabled| QRCalib["Auto QR calibration"]
    CalibStage -->|Manual| ManualCalib["Manual calibration"]
    QRCalib --> QRCheck{QR detected & valid?}
    QRCheck -->|Yes| CalibDone
    QRCheck -->|No| ManualCalib
    UsePersist --> CalibDone
    ManualCalib --> CalibDone
    
    CalibDone --> LabelCheck{Label Exclusion<br/>Needed?}
    LabelCheck -->|Yes| LabelUI["Multi-ROI Selection"]
    LabelCheck -->|No| SegmentationProcess["Segmentation Processing"]
    LabelUI --> SegmentationProcess
    
    SegmentationProcess --> ShowResults["Display Segmentation Results"]
    
    ShowResults --> ConfirmSegment{Review and<br/>Confirm?}
    ConfirmSegment -->|Flag| FlagReview["Flag for Review"]
    ConfirmSegment -->|Retry Calib| ManualCalib
    ConfirmSegment -->|Retry Masks| LabelUI
    ConfirmSegment -->|Approve| ProcessNext["Save CSVs/Overlays/ML"]
    
    FlagReview --> ProcessNext
    ProcessNext --> NextImageCheck{More<br/>Images?}
    NextImageCheck -->|Yes| LoadFirstImage
    NextImageCheck -->|No| ShowComplete["Completion Message<br/>Output Locations"]
    ShowComplete --> Exit[Exit Application<br/>Reset Persistent Settings]
    
    style Launch fill:#90EE90
    style Exit fill:#FFB6C1
    style ShowImageLabel fill:#FFE4B5
    style CalibDone fill:#FFE4B5
    style SegmentationProcess fill:#87CEEB
    style ProcessNext fill:#90EE90
```

## Configuration System

```mermaid
graph LR
    subgraph "Configuration Sources"
        Default["Default CONFIG Dict<br/>Hard-coded in Python"]
        JSON["config.JSON<br/>User Customizations<br/>Optional Override"]
    end
    
    subgraph "Configuration Categories"
        Paths["PATHS SECTION<br/>Inputs/Outputs/Subdirs/Names"]
        Inputs["INPUTS SECTION<br/>Label Names + Options"]
        Schema["SCHEMA_FIELDS<br/>Type + Range + Date"]
        Calib["CALIB SECTION<br/>Calibration + Persistence"]
        Seg["SEG SECTION<br/>Thresholds + Morphology"]
        Run["RUN SECTION<br/>Save Flags + Modes"]
        Formats["FILE_FORMATS SECTION<br/>Filename + Date Patterns"]
        UI_Config["UI SECTION<br/>Window Sizing + Layouts"]
        Viz["VISUALIZATION SECTION<br/>Overlay Styling"]
        ML["ML_TRAINING_OUTPUTS SECTION<br/>COCO/YOLO/Masks/Objects"]
    end
    
    subgraph "Configuration Loading"
        LoadJSON["Load JSON Config<br/>if exists"]
        DeepMerge["Deep Merge<br/>JSON overrides defaults"]
        Normalize["Normalize Parameters<br/>Expand ranges"]
        Validate["Validate Configuration<br/>Check types and ranges"]
        ApplyConfig["Apply Final Config<br/>Initialize System"]
    end
    
    Default --> DeepMerge
    JSON --> LoadJSON
    LoadJSON --> DeepMerge
    DeepMerge --> Normalize
    Normalize --> Validate
    Validate --> ApplyConfig
    
    ApplyConfig --> Paths
    ApplyConfig --> Inputs
    ApplyConfig --> Schema
    ApplyConfig --> Calib
    ApplyConfig --> Seg
    ApplyConfig --> Run
    ApplyConfig --> Formats
    ApplyConfig --> UI_Config
    ApplyConfig --> Viz
    ApplyConfig --> ML
    
    style Default fill:#E6E6FA
    style JSON fill:#98FB98
    style ApplyConfig fill:#F0E68C
```

## Data Flow and CSV Outputs

```mermaid
flowchart LR
    subgraph "Input Data"
        Images["Input Images<br/>jpg, png, tif<br/>from Inputs/ folder"]
        Config["config.JSON<br/>Processing parameters"]
        UserInputs["User Selections<br/>Config-driven labels"]
        Calibration["Calibration Result<br/>Pixel/cm Ratio"]
        QR["QR Detection<br/>QR_detected + QR_count"]
    end
    
    subgraph "Processing Steps"
        LoadImg["1. Load Image<br/>with Orientation"]
        CalibStep["2. QR or Manual or Persistent<br/>Calibration"]
        Exclusion["3. Exclusion Areas<br/>Labels and Cards"]
        Segmentation["4. Leaf Segmentation<br/>ExG plus HSV plus Lab"]
        AreaCalc["5. Area Calculation<br/>cm squared per Leaf"]
        MetricCalc["6. Metric Calculation<br/>Length and Width per Leaf"]
    end
    
    subgraph "CSV Outputs"
        MainCSV["Leaf_Area.csv<br/>Summary per Image<br/>Pixel_cm_ratio + QR fields"]
        DistCSV["Leaf_Distribution.csv<br/>Individual Leaf Areas"]
        LengthCSV["Leaf_Length.csv<br/>Per-Leaf Length"]
        WidthCSV["Leaf_Width.csv<br/>Per-Leaf Width"]
    end
    
    subgraph "Visual Outputs"
        Overlays["Overlay Images<br/>Leaf Outlines plus Text"]
        Debug["Debug Images<br/>Original and Mask Pair"]
        Renamed["Renamed Images<br/>Optional"]
    end
    
    subgraph "ML Training Outputs"
        Masks["Segmentation Masks<br/>Binary PNG Files"]
        Contours["Leaf Contours<br/>JSON Coordinates"]
        YOLO["YOLO Labels<br/>Normalized Coordinates"]
        COCO["COCO Annotations<br/>JSON Metadata"]
        Objects["Object Annotations<br/>Calibration/Label/QR"]
    end
    
    Images --> LoadImg
    Config --> Segmentation
    UserInputs --> AreaCalc
    Calibration --> AreaCalc
    QR --> MainCSV
    
    LoadImg --> CalibStep
    CalibStep --> Exclusion
    Exclusion --> Segmentation
    Segmentation --> AreaCalc
    AreaCalc --> MetricCalc
    
    MetricCalc --> MainCSV
    MetricCalc --> DistCSV
    MetricCalc --> LengthCSV
    MetricCalc --> WidthCSV
    
    MetricCalc --> Overlays
    Segmentation --> Debug
    LoadImg --> Renamed
    
    Segmentation --> Masks
    Segmentation --> Contours
    Segmentation --> YOLO
    Segmentation --> COCO
    Exclusion --> Objects
    
    style Images fill:#FFE4E1
    style MainCSV fill:#90EE90
    style DistCSV fill:#90EE90
    style LengthCSV fill:#90EE90
    style WidthCSV fill:#90EE90
    style Overlays fill:#87CEEB
    style Masks fill:#DDA0DD
```

## Build and Distribution Process

```mermaid
flowchart TD
    subgraph "Development Environment"
        Source["Source Code<br/>GatorLeaf.py"]
        Deps["Dependencies<br/>requirements.txt"]
        Spec["Build Configuration<br/>GatorLeaf.spec"]
        ConfigFile["Runtime Config<br/>config.JSON"]
        Icon["Application Icon<br/>icon.icns"]
    end
    
    subgraph "Build Process"
        VEnv["1. Virtual Environment<br/>python -m venv venv"]
        Install["2. Install Dependencies<br/>pip install -r requirements.txt"]
        Build["3. PyInstaller Build<br/>pyinstaller GatorLeaf.spec"]
        Sign["4. Code Sign on macOS<br/>codesign --force --deep<br/>-s - GatorLeaf.app"]
    end
    
    subgraph "Build Artifacts"
        StandaloneExe["Standalone Executable<br/>with Dependencies"]
        AppBundle["macOS Application Bundle<br/>Frameworks and Resources"]
        BuildDir["Intermediate Build Files"]
    end
    
    subgraph "Distribution Package"
        DistFolder["dist/ Folder"]
        DistExe["GatorLeaf executable<br/>or GatorLeaf.app"]
        DistInputs["Inputs/ folder"]
        DistConfig["config.JSON"]
        DistReadme["README files"]
    end

    Source --> VEnv
    Deps --> Install
    Icon --> Build
    VEnv --> Install
    Install --> Build
    Spec --> Build
    Build --> StandaloneExe
    Build --> AppBundle
    Build --> BuildDir
    StandaloneExe --> DistExe
    AppBundle --> DistExe
    ConfigFile --> DistConfig
    DistExe --> DistFolder
    DistInputs --> DistFolder
    DistConfig --> DistFolder
    DistReadme --> DistFolder
```

## Key Functions and Modules

```mermaid
graph TD
    Main["main()"] --> Run["run_gatorleaf()"]
    Run --> Batch["process_folder()"]
    Batch --> Image["process_image()"]
    Image --> Labels["manual_input_label_fields()"]
    Image --> Calib["manual_calibration() / _qr_calibration()"]
    Image --> Excl["manual_label_exclusion()"]
    Image --> Seg["segment_leaves()"]
    Image --> Review["show_segmentation_debug()"]
    Image --> Draw["draw_results()"]
    Batch --> CSV["append_to_csv_with_deduplication()"]
    Batch --> Dist["append_to_leaf_distribution_csv_with_deduplication()"]
    Batch --> Metric["append_to_leaf_metric_csv_with_deduplication()"]
    Batch --> ML["save_ml_annotations()"]
    Config["load_config_from_json() + normalize_dynamic_schema()"] --> Main
```

## Global Variables for Persistent Settings

GatorLeaf maintains global values to preserve settings across images in a batch:

```
_persistent_date          : Last selected date (YYYY-MM-DD or partial format)
_persistent_px_per_cm     : Last calculated calibration ratio
_persistent_labels        : Dict of current label values (config-driven names)
```

These values allow users to process multiple images with the same metadata without re-entering information for each image, while still providing the ability to update any field at any time via the Image Label panel buttons.
