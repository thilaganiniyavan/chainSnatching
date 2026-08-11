import os
import csv
import json
import argparse
from datetime import datetime

# Define required output structure
OUTPUT_DIR = "outputs/research_comparison"
POSE_DIR = os.path.join(OUTPUT_DIR, "pose")
NORM_DIR = os.path.join(OUTPUT_DIR, "normalization")
ACTION_DIR = os.path.join(OUTPUT_DIR, "action")
FUSION_DIR = os.path.join(OUTPUT_DIR, "fusion")

def setup_directories():
    os.makedirs(POSE_DIR, exist_ok=True)
    os.makedirs(NORM_DIR, exist_ok=True)
    os.makedirs(ACTION_DIR, exist_ok=True)
    os.makedirs(FUSION_DIR, exist_ok=True)

def generate_not_evaluated_report(stage_name, out_dir, models, reason_base, scaffold_models=None):
    if scaffold_models is None:
        scaffold_models = []
        
    csv_path = os.path.join(out_dir, f"{stage_name.lower()}_comparison.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Method", "Precision", "Recall", "F1", "Latency_ms", "FPS", "Status", "Reason"])
        
        # Write real models aborted due to dataset missing
        for model in models:
            writer.writerow([model, "N/A", "N/A", "N/A", "N/A", "N/A", "NOT EVALUATED", reason_base])
            
        # Write scaffold models aborted because they are just adapters
        for smodel in scaffold_models:
            writer.writerow([smodel, "N/A", "N/A", "N/A", "N/A", "N/A", "NOT EVALUATED", "Scaffold Adapter Only (No actual inference)"])

    json_path = os.path.join(out_dir, f"{stage_name.lower()}_config.json")
    with open(json_path, "w") as f:
        json.dump({
            "stage": stage_name,
            "timestamp": datetime.now().isoformat(),
            "executable_methods": models,
            "scaffold_methods": scaffold_models,
            "status": "NOT EVALUATED",
            "reason": reason_base
        }, f, indent=4)

    md_path = os.path.join(out_dir, f"{stage_name.lower()}_performance_report.md")
    with open(md_path, "w") as f:
        f.write(f"# Research Comparison: {stage_name}\n\n")
        f.write("## Status: NOT EVALUATED\n\n")
        f.write(f"**Reason for Executable Models:** {reason_base}\n\n")
        f.write("**Reason for Scaffold Models:** As explicitly instructed, adapter scaffolds cannot be evaluated as if they were real trained models.\n\n")
        f.write("### Intended Executable Comparisons\n")
        for m in models:
            f.write(f"- {m}\n")
        f.write("\n### Intended Scaffold Comparisons (Excluded)\n")
        for sm in scaffold_models:
            f.write(f"- {sm}\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-dir", type=str, default="Snatch 1.0/Chain Snatching Videos/Snatch Theft")
    args = parser.parse_args()

    setup_directories()
    
    if not os.path.exists(args.video_dir):
        reason = f"Benchmark dataset missing at {args.video_dir}. Cannot guarantee fair scientific comparison without identical source data."
        print(f"ABORTING LIVE EXPERIMENT: {reason}")
        
        # Level 6 - Pose Estimation
        generate_not_evaluated_report(
            "Pose", POSE_DIR, 
            models=["MediaPipe"], 
            reason_base=reason, 
            scaffold_models=["MMPose", "RTMPose", "ViTPose", "OpenPose"]
        )
        
        # Level 7 - Skeleton Normalization
        generate_not_evaluated_report(
            "Normalization", NORM_DIR, 
            models=["Hip-centered", "Bounding-box", "Root-joint", "Image-coordinate", "Rotation-aligned"], 
            reason_base=reason
        )

        # Level 8 - Action Recognition
        generate_not_evaluated_report(
            "Action", ACTION_DIR, 
            models=["ST-GCN"], 
            reason_base=reason,
            scaffold_models=["CTR-GCN", "MSG3D", "PoseC3D"]
        )
        
        # Level 9 - Fusion Strategy
        generate_not_evaluated_report(
            "Fusion", FUSION_DIR, 
            models=["Weighted Confidence", "Bayesian", "Rule-based", "Voting-based", "Weighted Averaging"], 
            reason_base=reason
        )
        
        print(f"Generated NOT EVALUATED reports for Phase 3 in {OUTPUT_DIR}")
        return

    print("Dataset found. (This branch would execute Pose & Action benching)")

if __name__ == "__main__":
    main()
