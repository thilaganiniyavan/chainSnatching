# 📹 CCTV Dataset Analysis Report

This report provides a detailed structure and quality sweep of the **Snatch 1.0** dataset for the forensic search FYP.

## 📂 Folder Structure & Organization

The dataset is organized hierarchically as follows:
```text
Snatch 1.0/
└── Chain Snatching Videos/
    ├── Normal/         (Standard surveillance footage files)
    └── Snatch Theft/   (Target criminal snatching videos)
```

### Folder Distribution:
- **Normal**: 7 videos
- **Snatch Theft**: 35 videos

## 📊 Summary Statistics

- **Total Video Files**: 42
- **Total File Size**: 1.76 GB (1801.74 MB)
- **Total Duration**: 3839.75 seconds (64.0 minutes)
- **Average Video Duration**: 91.42 seconds
- **Resolutions Found**: 1006x720, 1056x780, 1152x720, 1280x720, 1920x1080, 198x360, 294x240, 320x240, 352x288, 364x360, 384x288, 400x224, 400x328, 480x360, 640x352, 640x360, 658x480, 800x480
- **FPS Configurations**: 23.98, 24.0, 25.0, 29.97, 30.0 FPS
- **Codecs**: FMP4, h264

## ⚠️ Dataset Health & Integrity

### Corrupted or Unreadable Videos:
✅ **No corrupted or unreadable videos detected!** All files opened and read successfully.

### Duplicate Videos:
✅ **No duplicate videos found.** All video files represent unique physical files.

## 🏷️ Observable Characteristics & Categorization

A preliminary classification of lighting and camera motion was executed using luminance and frame-difference heuristics:

- **Lighting conditions**: 39 Day feeds, 3 Night feeds
- **Camera movement**: 39 Static cameras, 3 Panning/Moving cameras

> [!NOTE]
> Detailed manual parameters (Indoor/Outdoor, Pedestrian/Vehicle density, Traffic flow) cannot be estimated with 100% accuracy using simple heuristics. A templates-filled CSV has been generated at [outputs/video_metadata.csv](file:///c:/Users/tejes/Downloads/fyp/outputs/video_metadata.csv) for you to manually complete these specific categories.
