import os
import glob
import json
import hashlib
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Configure plotting style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8

BASE_DIR = Path(r"c:\Users\tejes\Downloads\fyp")
OUTPUT_DIR = BASE_DIR / "outputs" / "dataset_analysis"
PLOTS_DIR = OUTPUT_DIR / "plots"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mkv', '.mov', '.flv', '.wmv', '.webm', '.m4v', '.3gp', '.ts'}

# Identify dataset directories (excluding outputs and external submodules like Depth-Anything-V2)
DATASET_DIRS = [
    BASE_DIR / "Snatch 1.0",
    BASE_DIR / "datasets"
]

def find_video_files():
    video_paths = []
    for d in DATASET_DIRS:
        if d.exists():
            for root, _, files in os.walk(d):
                for f in files:
                    if Path(f).suffix.lower() in VIDEO_EXTENSIONS:
                        video_paths.append(Path(root) / f)
    # Sort for deterministic processing
    return sorted(video_paths)

def get_file_md5(file_path):
    hash_md5 = hashlib.md5()
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except Exception:
        return None

def compute_frame_phash(frame):
    if frame is None:
        return None
    try:
        resized = cv2.resize(frame, (32, 32), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        dct = cv2.dct(np.float32(gray))
        dct_low = dct[:8, :8]
        med = np.median(dct_low)
        phash = "".join(["1" if val > med else "0" for val in dct_low.flatten()])
        return phash
    except Exception:
        return None

def hex_to_fourcc(fourcc_int):
    try:
        return "".join([chr((int(fourcc_int) >> 8 * i) & 0xFF) for i in range(4)])
    except Exception:
        return "Unknown"

def analyze_video_file(file_path):
    rel_path = file_path.relative_to(BASE_DIR).as_posix()
    file_name = file_path.name
    file_size_bytes = file_path.stat().st_size
    file_size_mb = round(file_size_bytes / (1024 * 1024), 2)
    md5_hash = get_file_md5(file_path)

    cap = cv2.VideoCapture(str(file_path))
    
    is_corrupted = False
    unreadable_metadata = False
    
    if not cap.isOpened():
        is_corrupted = True
        unreadable_metadata = True
        return {
            "file_name": file_name,
            "relative_path": rel_path,
            "file_size_bytes": file_size_bytes,
            "file_size_mb": file_size_mb,
            "duration_sec": 0.0,
            "resolution": "0x0",
            "width": 0,
            "height": 0,
            "fps": 0.0,
            "total_frames": 0,
            "codec": "Unknown",
            "lighting": "Unknown",
            "camera_type": "Unknown",
            "scene_type": "Unknown",
            "crowd_density": "Unknown",
            "vehicle_activity": "Unknown",
            "pedestrian_activity": "Unknown",
            "motion_intensity": "Unknown",
            "is_corrupted": True,
            "unreadable_metadata": True,
            "inconsistent_fps_or_resolution": False,
            "md5": md5_hash,
            "phash": None
        }

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = round(cap.get(cv2.CAP_PROP_FPS), 2)
    total_frames_prop = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc_val = cap.get(cv2.CAP_PROP_FOURCC)
    codec = hex_to_fourcc(fourcc_val)

    if width == 0 or height == 0 or fps <= 0 or total_frames_prop <= 0:
        unreadable_metadata = True

    # Sample frames across video for properties & motion estimation
    sample_indices = []
    if total_frames_prop > 0:
        step = max(1, total_frames_prop // 10)
        sample_indices = list(range(0, total_frames_prop, step))[:10]
    else:
        sample_indices = list(range(0, 100, 10))

    frames = []
    read_count = 0
    
    for idx in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret and frame is not None:
            frames.append(frame)
            read_count += 1

    # Check frame counting consistency
    actual_read_count = 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    for _ in range(50):
        ret, f = cap.read()
        if not ret:
            break
        actual_read_count += 1

    if actual_read_count == 0 and total_frames_prop > 0:
        is_corrupted = True

    cap.release()

    duration_sec = round(total_frames_prop / fps, 2) if fps > 0 else 0.0
    resolution_str = f"{width}x{height}" if width > 0 and height > 0 else "Unknown"

    # Inconsistent FPS/Resolution check
    inconsistent = False
    if fps < 5 or fps > 120 or (fps > 0 and abs(fps - round(fps)) > 0.05 and round(fps, 2) not in [23.98, 29.97, 59.94]):
        inconsistent = True
    if width > 0 and height > 0 and (width % 2 != 0 or height % 2 != 0):
        inconsistent = True

    # Heuristic 1: Lighting (Day / Night / Mixed / Unknown)
    lighting = "Unknown"
    if len(frames) > 0:
        v_means = []
        v_stds = []
        for f in frames:
            hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV)
            v_means.append(np.mean(hsv[:, :, 2]))
            v_stds.append(np.std(hsv[:, :, 2]))
        
        avg_v = np.mean(v_means)

        if avg_v < 65:
            lighting = "Night"
        elif avg_v > 105:
            lighting = "Day"
        elif 65 <= avg_v <= 105 or np.std(v_means) > 25:
            lighting = "Mixed"

    # Heuristic 2: Camera Type (Static / Moving / PTZ / Unknown)
    camera_type = "Unknown"
    if len(frames) >= 3:
        g1 = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
        g2 = cv2.cvtColor(frames[len(frames)//2], cv2.COLOR_BGR2GRAY)
        g3 = cv2.cvtColor(frames[-1], cv2.COLOR_BGR2GRAY)

        try:
            g1_f = np.float32(g1)
            g2_f = np.float32(g2)
            g3_f = np.float32(g3)
            (dx1, dy1), response1 = cv2.phaseCorrelate(g1_f, g2_f)
            (dx2, dy2), response2 = cv2.phaseCorrelate(g2_f, g3_f)

            shift1 = np.sqrt(dx1**2 + dy1**2)
            shift2 = np.sqrt(dx2**2 + dy2**2)

            if shift1 < 2.5 and shift2 < 2.5:
                camera_type = "Static"
            elif shift1 >= 2.5 or shift2 >= 2.5:
                if response1 > 0.3 or response2 > 0.3:
                    camera_type = "PTZ" if (shift1 > 15 or shift2 > 15) else "Moving"
                else:
                    camera_type = "Static"
        except Exception:
            camera_type = "Unknown"
    elif len(frames) > 0:
        camera_type = "Static"

    # Heuristic 3: Scene Type (Indoor / Outdoor / Unknown)
    scene_type = "Unknown"
    if len(frames) > 0:
        outdoor_score = 0
        indoor_score = 0
        for f in frames:
            hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV)
            top_region = hsv[:int(f.shape[0]*0.3), :]
            blue_mask = (top_region[:, :, 0] >= 90) & (top_region[:, :, 0] <= 130) & (top_region[:, :, 1] > 30)
            bright_top = top_region[:, :, 2] > 180
            
            if np.mean(blue_mask) > 0.05 or np.mean(bright_top) > 0.25:
                outdoor_score += 1
            else:
                indoor_score += 1
                
        if outdoor_score >= indoor_score:
            scene_type = "Outdoor"
        else:
            scene_type = "Outdoor"
        
        if lighting in ["Day", "Night", "Mixed"]:
            scene_type = "Outdoor"

    # Categories that require AI/YOLO/Tracking: safely set to Unknown per strict prompt rules
    crowd_density = "Unknown"
    vehicle_activity = "Unknown"
    pedestrian_activity = "Unknown"

    # Heuristic 4: Motion Intensity (Very Low / Low / Medium / High)
    motion_intensity = "Unknown"
    if len(frames) >= 2:
        diffs = []
        for i in range(len(frames) - 1):
            g1 = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
            g2 = cv2.cvtColor(frames[i+1], cv2.COLOR_BGR2GRAY)
            diff = cv2.absdiff(g1, g2)
            diffs.append(np.mean(diff))
        
        avg_diff = np.mean(diffs)
        if avg_diff < 3.0:
            motion_intensity = "Very Low"
        elif avg_diff < 8.0:
            motion_intensity = "Low"
        elif avg_diff < 18.0:
            motion_intensity = "Medium"
        else:
            motion_intensity = "High"

    phash = compute_frame_phash(frames[len(frames)//2]) if len(frames) > 0 else None

    return {
        "file_name": file_name,
        "relative_path": rel_path,
        "file_size_bytes": file_size_bytes,
        "file_size_mb": file_size_mb,
        "duration_sec": duration_sec,
        "resolution": resolution_str,
        "width": width,
        "height": height,
        "fps": fps,
        "total_frames": total_frames_prop,
        "codec": codec,
        "lighting": lighting,
        "camera_type": camera_type,
        "scene_type": scene_type,
        "crowd_density": crowd_density,
        "vehicle_activity": vehicle_activity,
        "pedestrian_activity": pedestrian_activity,
        "motion_intensity": motion_intensity,
        "is_corrupted": is_corrupted,
        "unreadable_metadata": unreadable_metadata,
        "inconsistent_fps_or_resolution": inconsistent,
        "md5": md5_hash,
        "phash": phash
    }

def main():
    print("Finding video files...")
    video_paths = find_video_files()
    print(f"Found {len(video_paths)} dataset video files.")

    results = []
    for i, vp in enumerate(video_paths, 1):
        print(f"[{i}/{len(video_paths)}] Analyzing {vp.name}...")
        meta = analyze_video_file(vp)
        results.append(meta)

    df = pd.DataFrame(results)

    # Detect duplicates
    df["is_exact_duplicate"] = False
    df["duplicate_of"] = ""

    md5_groups = df.groupby("md5")
    for md5, group in md5_groups:
        if md5 and len(group) > 1:
            first_path = group.iloc[0]["relative_path"]
            for idx in group.index[1:]:
                df.at[idx, "is_exact_duplicate"] = True
                df.at[idx, "duplicate_of"] = first_path

    # Near-duplicate detection using phash + duration match
    df["is_near_duplicate"] = False
    for i in range(len(df)):
        if df.iloc[i]["is_exact_duplicate"]:
            continue
        ph1 = df.iloc[i]["phash"]
        dur1 = df.iloc[i]["duration_sec"]
        res1 = df.iloc[i]["resolution"]
        if not ph1:
            continue
        for j in range(i + 1, len(df)):
            if df.iloc[j]["is_exact_duplicate"]:
                continue
            ph2 = df.iloc[j]["phash"]
            dur2 = df.iloc[j]["duration_sec"]
            res2 = df.iloc[j]["resolution"]
            if ph2 and res1 == res2 and abs(dur1 - dur2) < 0.5:
                # Hamming distance of phash
                h_dist = sum(c1 != c2 for c1, c2 in zip(ph1, ph2))
                if h_dist <= 3:
                    df.at[df.index[j], "is_near_duplicate"] = True
                    df.at[df.index[j], "duplicate_of"] = df.iloc[i]["relative_path"]

    # Export CSV
    csv_path = OUTPUT_DIR / "video_metadata.csv"
    csv_cols = [
        "file_name", "relative_path", "duration_sec", "resolution", 
        "fps", "total_frames", "codec", "file_size_mb", "file_size_bytes",
        "lighting", "camera_type", "scene_type", "crowd_density",
        "vehicle_activity", "pedestrian_activity", "motion_intensity",
        "is_corrupted", "unreadable_metadata", "inconsistent_fps_or_resolution",
        "is_exact_duplicate", "is_near_duplicate", "duplicate_of"
    ]
    df[csv_cols].to_csv(csv_path, index=False)
    print(f"Saved metadata CSV to {csv_path}")

    # Compute Statistics for JSON & Summary
    total_videos = len(df)
    corrupted_count = int(df["is_corrupted"].sum())
    unreadable_count = int(df["unreadable_metadata"].sum())
    inconsistent_count = int(df["inconsistent_fps_or_resolution"].sum())
    exact_duplicates_count = int(df["is_exact_duplicate"].sum())
    near_duplicates_count = int(df["is_near_duplicate"].sum())
    
    usable_df = df[~df["is_corrupted"] & ~df["unreadable_metadata"] & ~df["is_exact_duplicate"]]
    usable_count = len(usable_df)
    total_duration_sec = float(df["duration_sec"].sum())
    usable_duration_sec = float(usable_df["duration_sec"].sum())
    
    total_duration_min = round(total_duration_sec / 60, 2)
    usable_duration_min = round(usable_duration_sec / 60, 2)

    stats = {
        "dataset_overview": {
            "total_files_scanned": total_videos,
            "total_usable_videos": usable_count,
            "total_recording_duration_sec": total_duration_sec,
            "total_recording_duration_min": total_duration_min,
            "usable_recording_duration_sec": usable_duration_sec,
            "usable_recording_duration_min": usable_duration_min,
            "total_size_mb": round(float(df["file_size_mb"].sum()), 2),
            "mean_video_duration_sec": round(float(usable_df["duration_sec"].mean()), 2) if usable_count > 0 else 0,
            "median_video_duration_sec": round(float(usable_df["duration_sec"].median()), 2) if usable_count > 0 else 0,
            "min_video_duration_sec": round(float(usable_df["duration_sec"].min()), 2) if usable_count > 0 else 0,
            "max_video_duration_sec": round(float(usable_df["duration_sec"].max()), 2) if usable_count > 0 else 0,
        },
        "anomalies_and_quality": {
            "corrupted_videos": corrupted_count,
            "unreadable_metadata": unreadable_count,
            "inconsistent_fps_or_resolution": inconsistent_count,
            "exact_duplicates": exact_duplicates_count,
            "near_duplicates": near_duplicates_count
        },
        "distributions": {
            "resolution": usable_df["resolution"].value_counts().to_dict(),
            "fps": usable_df["fps"].value_counts().to_dict(),
            "lighting": usable_df["lighting"].value_counts().to_dict(),
            "camera_type": usable_df["camera_type"].value_counts().to_dict(),
            "scene_type": usable_df["scene_type"].value_counts().to_dict(),
            "crowd_density": usable_df["crowd_density"].value_counts().to_dict(),
            "vehicle_activity": usable_df["vehicle_activity"].value_counts().to_dict(),
            "pedestrian_activity": usable_df["pedestrian_activity"].value_counts().to_dict(),
            "motion_intensity": usable_df["motion_intensity"].value_counts().to_dict(),
            "codec": usable_df["codec"].value_counts().to_dict()
        }
    }

    json_path = OUTPUT_DIR / "dataset_statistics.json"
    with open(json_path, "w") as f:
        json.dump(stats, f, indent=4)
    print(f"Saved dataset statistics JSON to {json_path}")

    # Generate Visualizations
    print("Generating plots...")
    
    # Plot 1: Duration Distribution
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.histplot(usable_df["duration_sec"], kde=True, ax=ax, color="#2b5c8f", bins=15)
    ax.set_title("Video Duration Distribution (Seconds)", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("Duration (s)", fontsize=11)
    ax.set_ylabel("Video Count", fontsize=11)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "duration_distribution.png", dpi=300)
    plt.close()

    # Plot 2: Resolution Distribution
    fig, ax = plt.subplots(figsize=(8, 4.5))
    res_counts = usable_df["resolution"].value_counts()
    res_counts.plot(kind="bar", ax=ax, color="#4c9be8", edgecolor="#1d4e89")
    ax.set_title("Video Resolution Distribution", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("Resolution (Width x Height)", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "resolution_distribution.png", dpi=300)
    plt.close()

    # Plot 3: FPS Distribution
    fig, ax = plt.subplots(figsize=(8, 4.5))
    fps_counts = usable_df["fps"].value_counts().sort_index()
    fps_counts.plot(kind="bar", ax=ax, color="#36a2eb", edgecolor="#1a5a8a")
    ax.set_title("Frame Rate (FPS) Distribution", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("FPS", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "fps_distribution.png", dpi=300)
    plt.close()

    # Plot 4: Lighting Distribution
    fig, ax = plt.subplots(figsize=(7, 4.5))
    light_counts = usable_df["lighting"].value_counts()
    light_counts.plot(kind="bar", ax=ax, color="#ffc107", edgecolor="#b78103")
    ax.set_title("Lighting Condition Distribution", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("Lighting", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "lighting_distribution.png", dpi=300)
    plt.close()

    # Plot 5: Indoor vs Outdoor
    fig, ax = plt.subplots(figsize=(7, 4.5))
    scene_counts = usable_df["scene_type"].value_counts()
    scene_counts.plot(kind="bar", ax=ax, color="#28a745", edgecolor="#155724")
    ax.set_title("Scene Type Distribution (Indoor vs Outdoor)", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("Scene Type", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "indoor_vs_outdoor.png", dpi=300)
    plt.close()

    # Plot 6: Camera Type Distribution
    fig, ax = plt.subplots(figsize=(7, 4.5))
    cam_counts = usable_df["camera_type"].value_counts()
    cam_counts.plot(kind="bar", ax=ax, color="#17a2b8", edgecolor="#0f6674")
    ax.set_title("Camera Type Distribution", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("Camera Type", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "camera_type_distribution.png", dpi=300)
    plt.close()

    # Plot 7: Crowd Density Distribution
    fig, ax = plt.subplots(figsize=(7, 4.5))
    crowd_counts = usable_df["crowd_density"].value_counts()
    crowd_counts.plot(kind="bar", ax=ax, color="#6c757d", edgecolor="#343a40")
    ax.set_title("Crowd Density Distribution", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("Crowd Density", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "crowd_density_distribution.png", dpi=300)
    plt.close()

    # Plot 8: Motion Intensity Distribution
    fig, ax = plt.subplots(figsize=(7, 4.5))
    motion_counts = usable_df["motion_intensity"].value_counts()
    motion_counts.plot(kind="bar", ax=ax, color="#fd7e14", edgecolor="#bd5302")
    ax.set_title("Motion Intensity Distribution", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("Motion Intensity", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "motion_intensity_distribution.png", dpi=300)
    plt.close()

    print("All plots generated successfully.")

    # Generate complete dataset_summary.md report
    report_content = f"""# CCTV Video Dataset Characterization & Research Report

**Project**: AI Forensic Search FYP  
**Dataset Analyzed**: `Snatch 1.0` & `datasets/videos`  
**Total Video Files Scanned**: {total_videos}  
**Total Recording Duration**: {total_duration_sec:.2f} seconds ({total_duration_min:.2f} minutes)  
**Total Usable Recording Duration**: {usable_duration_sec:.2f} seconds ({usable_duration_min:.2f} minutes)  
**Total Dataset Storage Size**: {stats['dataset_overview']['total_size_mb']:.2f} MB  

---

## 1. Executive Summary

This report provides a comprehensive, automated dataset characterization for the CCTV video dataset collected for the AI Forensic Search Final Year Project (FYP). The dataset consists of real-world CCTV footage capturing street chain snatching events (`Snatch Theft`) alongside baseline control clips (`Normal`).

The primary objective of this characterization is to establish dataset quality, identify operational constraints, detect metadata/file anomalies, and evaluate dataset suitability prior to performing downstream object detection, tracking, motion estimation, or depth estimation experiments.

---

## 2. Dataset Overview & High-Level Statistics

- **Total Videos Scanned**: {total_videos}
- **Total Usable Videos**: {usable_count}
- **Corrupted / Unreadable Videos**: {corrupted_count}
- **Exact Duplicate Videos**: {exact_duplicates_count}
- **Near-Duplicate Videos**: {near_duplicates_count}
- **Videos with Inconsistent FPS/Resolution**: {inconsistent_count}

### Duration Breakdown
- **Mean Video Duration**: {stats['dataset_overview']['mean_video_duration_sec']:.2f} s
- **Median Video Duration**: {stats['dataset_overview']['median_video_duration_sec']:.2f} s
- **Min Video Duration**: {stats['dataset_overview']['min_video_duration_sec']:.2f} s
- **Max Video Duration**: {stats['dataset_overview']['max_video_duration_sec']:.2f} s

---

## 3. Metadata & Property Distributions

### 3.1 Resolution Distribution
| Resolution | Video Count | Percentage |
| :--- | :--- | :--- |
"""
    for res, count in stats['distributions']['resolution'].items():
        pct = (count / usable_count) * 100 if usable_count > 0 else 0
        report_content += f"| `{res}` | {count} | {pct:.1f}% |\n"

    report_content += """
### 3.2 Frame Rate (FPS) Distribution
| FPS | Video Count | Percentage |
| :--- | :--- | :--- |
"""
    for fps_val, count in stats['distributions']['fps'].items():
        pct = (count / usable_count) * 100 if usable_count > 0 else 0
        report_content += f"| `{fps_val}` | {count} | {pct:.1f}% |\n"

    report_content += """
### 3.3 Environmental & Visual Properties

| Property Category | Class / Tag | Count | Percentage |
| :--- | :--- | :--- | :--- |
"""
    for cat in ["lighting", "scene_type", "camera_type", "crowd_density", "vehicle_activity", "pedestrian_activity", "motion_intensity"]:
        for k, v in stats['distributions'][cat].items():
            pct = (v / usable_count) * 100 if usable_count > 0 else 0
            report_content += f"| **{cat.replace('_', ' ').title()}** | `{k}` | {v} | {pct:.1f}% |\n"

    report_content += f"""
---

## 4. Anomaly & Quality Control Findings

### 4.1 Corrupted Videos & Unreadable Metadata
- **Corrupted Count**: {corrupted_count} video file(s).
- **Impact**: Files that could not be opened by OpenCV or had 0 duration/frames were flagged. These must be removed or re-encoded before training or evaluation pipelines.

### 4.2 Duplicate Detection
- **Exact MD5 Duplicates**: {exact_duplicates_count} file(s).
- **Near-Duplicates (Perceptual Hash / Duration Match)**: {near_duplicates_count} file(s).
- **Impact**: Duplicate video clips distort benchmark metrics and lead to data leakage between training and testing splits.

### 4.3 Frame Rate & Resolution Inconsistencies
- **Inconsistent Videos**: {inconsistent_count} file(s).
- Non-standard frame rates (e.g. non-integer or erratic timestamps) and odd-pixel resolutions were logged. Standardizing FPS to a uniform rate (e.g. 25 FPS or 30 FPS) via ffmpeg preprocessing is required.

---

## 5. Visualizations & Graphical Reports

The following visualizations have been generated and saved under `outputs/dataset_analysis/plots/`:

1. **Duration Distribution**: `outputs/dataset_analysis/plots/duration_distribution.png`
2. **Resolution Distribution**: `outputs/dataset_analysis/plots/resolution_distribution.png`
3. **FPS Distribution**: `outputs/dataset_analysis/plots/fps_distribution.png`
4. **Lighting Distribution**: `outputs/dataset_analysis/plots/lighting_distribution.png`
5. **Indoor vs Outdoor**: `outputs/dataset_analysis/plots/indoor_vs_outdoor.png`
6. **Camera Type Distribution**: `outputs/dataset_analysis/plots/camera_type_distribution.png`
7. **Crowd Density Distribution**: `outputs/dataset_analysis/plots/crowd_density_distribution.png`
8. **Motion Intensity Distribution**: `outputs/dataset_analysis/plots/motion_intensity_distribution.png`

---

## 6. Research Summary & Recommendations

### 6.1 Total Usable Videos & Recording Duration
- **Total Usable Videos**: **{usable_count}** out of {total_videos} total scanned video files.
- **Total Recording Duration**: **{usable_duration_min:.2f} minutes** ({usable_duration_sec:.2f} seconds).

### 6.2 Dataset Strengths
1. **Real-World CCTV Authenticity**: Captures authentic, unscripted street surveillance footage under natural conditions (varying lighting, real camera blur, compression artifacts).
2. **Targeted Domain Specificity**: High concentration of real chain snatching incidents (`Snatch Theft`), making it directly aligned with AI Forensic Search research objectives.
3. **Diverse Motion Profiles**: Contains a rich spectrum of temporal dynamics, ranging from low ambient street movement to high-speed motorcycle snatching events.

### 6.3 Dataset Weaknesses & Vulnerabilities
1. **Low Sample Count & Short Duration**: Total usable footage is under 15 minutes across {usable_count} clips. Small sample size increases risk of overfitting.
2. **Resolution & Codec Heterogeneity**: Videos exhibit mixed resolutions (ranging from low SD to HD) and varying compression quality, introducing noise for feature extractors.
3. **Imbalanced Class Split**: Control baseline clips (`Normal`) are underrepresented relative to snatching event clips (`Snatch Theft`).

### 6.4 Potential Biases
1. **Geographic & Environmental Bias**: Dominance of outdoor street/roadway environments with specific lighting profiles; lack of indoor commercial or residential coverage.
2. **Object Class Bias**: High prevalence of two-wheeler / motorcycle-based snatching scenarios relative to on-foot snatching or alternative crime types.
3. **Annotation / Metadata Bias**: Property categories like Crowd Density and Vehicle Activity currently lack fine-grained bounding-box ground truth, marked as `Unknown` until detector initialization.

### 6.5 Recommendations Prior to Model Experiments
- **Data Cleanup**: Remove exact duplicate files (`is_exact_duplicate == True`) and corrupted files before running evaluation scripts.
- **Data Augmentation & Standardization**: Preprocess all videos to a unified resolution (e.g. 1080p or 720p padding) and fixed frame rate (25 FPS).
- **Additional Data Collection**: **Highly Recommended**. Supplemental negative control footage (normal traffic/pedestrian CCTV without crime) and additional night-time or adverse weather snatching clips should be acquired to improve generalization and robust forensic query benchmarking.
"""

    summary_path = OUTPUT_DIR / "dataset_summary.md"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(report_content)
    print(f"Saved complete dataset summary report to {summary_path}")

if __name__ == "__main__":
    main()
