import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    results_csv = "outputs/experiments/motion_results.csv"
    metadata_csv = "outputs/video_metadata.csv"
    output_dir = "outputs/experiments"
    
    if not os.path.exists(results_csv):
        print(f"Error: Results CSV '{results_csv}' not found. Please run the experiments first.")
        sys.exit(1)
        
    if not os.path.exists(metadata_csv):
        print(f"Error: Metadata CSV '{metadata_csv}' not found.")
        sys.exit(1)
        
    # Load data
    df_results = pd.read_csv(results_csv)
    df_meta = pd.read_csv(metadata_csv)
    
    # Standardize column naming for merging
    df_meta = df_meta.rename(columns={"filename": "video_name"})
    
    # Merge results with video metadata categories
    df = pd.merge(df_results, df_meta[["video_name", "day_night", "camera_type", "folder"]], on="video_name")
    
    print("Aggregating results...")
    
    # 1. Overall Averages per Method
    df_overall = df.groupby("method").agg({
        "reduction_percentage": "mean",
        "fps": "mean",
        "continuity_score": "mean",
        "avg_segment_length": "mean",
        "num_segments": "mean",
        "average_motion_area_ratio": "mean"
    }).reset_index()
    
    # 2. Category Averages: Activity Type (Folder)
    df_folder = df.groupby(["method", "folder"]).agg({
        "reduction_percentage": "mean",
        "fps": "mean",
        "continuity_score": "mean",
        "avg_segment_length": "mean",
        "num_segments": "mean",
        "average_motion_area_ratio": "mean"
    }).reset_index()
    
    # 3. Category Averages: Day/Night
    df_daynight = df.groupby(["method", "day_night"]).agg({
        "reduction_percentage": "mean",
        "fps": "mean",
        "continuity_score": "mean",
        "avg_segment_length": "mean",
        "num_segments": "mean",
        "average_motion_area_ratio": "mean"
    }).reset_index()
    
    # 4. Category Averages: Camera Type (Static/Movement)
    df_camera = df.groupby(["method", "camera_type"]).agg({
        "reduction_percentage": "mean",
        "fps": "mean",
        "continuity_score": "mean",
        "avg_segment_length": "mean",
        "num_segments": "mean",
        "average_motion_area_ratio": "mean"
    }).reset_index()
    
    # Save category averages to CSV
    avg_csv_path = os.path.join(output_dir, "motion_category_averages.csv")
    with open(avg_csv_path, mode="w", newline="", encoding="utf-8") as f:
        f.write("# OVERALL METHOD AVERAGES\n")
        df_overall.to_csv(f, index=False)
        f.write("\n# CATEGORY AVERAGES: ACTIVITY TYPE (FOLDER)\n")
        df_folder.to_csv(f, index=False)
        f.write("\n# CATEGORY AVERAGES: LIGHTING (DAY/NIGHT)\n")
        df_daynight.to_csv(f, index=False)
        f.write("\n# CATEGORY AVERAGES: CAMERA MOTION\n")
        df_camera.to_csv(f, index=False)
        
    print(f"Saved category averages CSV to: {avg_csv_path}")
    
    # Generate Charts
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    methods = df_overall["method"].unique()
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(methods)))
    
    # Plot A: Category-wise Frame Reduction
    plt.figure(figsize=(14, 6))
    
    # Subplot 1: By Folder Activity Class
    plt.subplot(1, 3, 1)
    folders = df_folder["folder"].unique()
    x = np.arange(len(folders))
    width = 0.15
    for i, m in enumerate(methods):
        m_data = df_folder[df_folder["method"] == m]
        reductions = [m_data[m_data["folder"] == f]["reduction_percentage"].values[0] if len(m_data[m_data["folder"] == f]) > 0 else 0.0 for f in folders]
        plt.bar(x + i*width, reductions, width, label=m, color=colors[i], edgecolor='grey')
    plt.xticks(x + width*2, folders)
    plt.title("Reduction % by Activity Class", fontsize=10, fontweight='bold')
    plt.ylabel("Avg Reduction (%)")
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    
    # Subplot 2: By Lighting (Day/Night)
    plt.subplot(1, 3, 2)
    daynights = df_daynight["day_night"].unique()
    x = np.arange(len(daynights))
    for i, m in enumerate(methods):
        m_data = df_daynight[df_daynight["method"] == m]
        reductions = [m_data[m_data["day_night"] == dn]["reduction_percentage"].values[0] if len(m_data[m_data["day_night"] == dn]) > 0 else 0.0 for dn in daynights]
        plt.bar(x + i*width, reductions, width, label=m, color=colors[i], edgecolor='grey')
    plt.xticks(x + width*2, daynights)
    plt.title("Reduction % by Lighting Type", fontsize=10, fontweight='bold')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    
    # Subplot 3: By Camera Type
    plt.subplot(1, 3, 3)
    cameras = df_camera["camera_type"].unique()
    x = np.arange(len(cameras))
    for i, m in enumerate(methods):
        m_data = df_camera[df_camera["method"] == m]
        reductions = [m_data[m_data["camera_type"] == c]["reduction_percentage"].values[0] if len(m_data[m_data["camera_type"] == c]) > 0 else 0.0 for c in cameras]
        plt.bar(x + i*width, reductions, width, label=m, color=colors[i], edgecolor='grey')
    plt.xticks(x + width*2, cameras)
    plt.title("Reduction % by Camera Type", fontsize=10, fontweight='bold')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plot_path = os.path.join(plots_dir, "category_reduction_comparison.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    
    print(f"Saved category comparison plot to: {plot_path}")
    
    # Write outputs/experiments/experiment_results_report.md
    report_path = os.path.join(output_dir, "experiment_results_report.md")
    with open(report_path, mode="w", encoding="utf-8") as f:
        f.write("# 🔬 CCTV Motion Benchmarking Statistical Summary\n\n")
        f.write("This report aggregates motion detection benchmark metrics across the entire **Snatch 1.0** dataset.\n\n")
        
        f.write("## 📊 Overall Performance Metrics\n\n")
        f.write("| Method | Avg Reduction (%) | Avg Speed (FPS) | Avg Continuity Score | Avg Segment Length (Frames) | Avg Segment Count | Avg Motion Area (%) |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        for _, row in df_overall.iterrows():
            f.write(f"| **{row['method']}** | {row['reduction_percentage']:.2f}% | {row['fps']:.1f} | {row['continuity_score']:.3f} | {row['avg_segment_length']:.1f} | {row['num_segments']:.1f} | {row['average_motion_area_ratio']*100:.3f}% |\n")
        f.write("\n")
        
        f.write("## 🏷️ Category-wise Performance Metrics\n\n")
        
        f.write("### 1. Activity Class Averages\n\n")
        f.write("| Method | Folder Class | Avg Reduction (%) | Avg Speed (FPS) | Avg Continuity Score | Avg Motion Area (%) |\n")
        f.write("| :--- | :--- | :---: | :---: | :---: | :---: |\n")
        for _, row in df_folder.sort_values(by=["folder", "method"]).iterrows():
            f.write(f"| {row['method']} | `{row['folder']}` | {row['reduction_percentage']:.2f}% | {row['fps']:.1f} | {row['continuity_score']:.3f} | {row['average_motion_area_ratio']*100:.3f}% |\n")
        f.write("\n")
        
        f.write("### 2. Lighting Conditions Averages\n\n")
        f.write("| Method | Lighting Type | Avg Reduction (%) | Avg Speed (FPS) | Avg Continuity Score | Avg Motion Area (%) |\n")
        f.write("| :--- | :--- | :---: | :---: | :---: | :---: |\n")
        for _, row in df_daynight.sort_values(by=["day_night", "method"]).iterrows():
            f.write(f"| {row['method']} | `{row['day_night']}` | {row['reduction_percentage']:.2f}% | {row['fps']:.1f} | {row['continuity_score']:.3f} | {row['average_motion_area_ratio']*100:.3f}% |\n")
        f.write("\n")
        
        f.write("### 3. Camera Motion Averages\n\n")
        f.write("| Method | Camera Type | Avg Reduction (%) | Avg Speed (FPS) | Avg Continuity Score | Avg Motion Area (%) |\n")
        f.write("| :--- | :--- | :---: | :---: | :---: | :---: |\n")
        for _, row in df_camera.sort_values(by=["camera_type", "method"]).iterrows():
            f.write(f"| {row['method']} | `{row['camera_type']}` | {row['reduction_percentage']:.2f}% | {row['fps']:.1f} | {row['continuity_score']:.3f} | {row['average_motion_area_ratio']*100:.3f}% |\n")
            
    print(f"Saved complete results report to: {report_path}")
    print("AGGREGATION COMPLETE")

if __name__ == "__main__":
    main()
