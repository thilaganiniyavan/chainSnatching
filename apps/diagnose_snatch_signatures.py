"""Snatch Signature Diagnostic & Evidence Breakdown Tool.

Runs Config D across sample videos (or a specified video) and prints
a detailed forensic breakdown:
- Overall Decision Score & Ground Truth
- Individual Snatch Signatures evaluated
- Matched Evidence (with component weights)
- Missing Evidence (with component weights)
- Pose / ST-GCN Action Predictions
- Behaviour Graph Pattern nodes
- Kinematic Speeds and Spatial Distances
"""

import argparse
import os
import sys

# Ensure root directory is in python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
import time
from src.pipeline.pipeline import Pipeline
from src.core.models.frame_context import FrameContext
from src.evaluation.research_live_evaluator import (
    LiveResearchEvaluator,
    full_spec,
    discover_snatch_videos,
    is_incident_video,
    person_vehicle_proximity_score,
    _finalize_stage,
    MOTION_PIXEL_THRESHOLD,
)


def main():
    parser = argparse.ArgumentParser(description="Snatch Signature Evidence Diagnostic")
    parser.add_argument("--snatch-videos", type=int, default=None, help="Number of snatch incident videos to evaluate (default: 4)")
    parser.add_argument("--normal-videos", type=int, default=None, help="Number of normal control videos to evaluate (default: 4)")
    parser.add_argument("--max-videos", type=int, default=8, help="Total videos to sample (e.g. 8 = 4 snatch + 4 normal)")
    parser.add_argument("--max-frames", type=int, default=1000, help="Max frames evaluated per video (default: 1000)")
    parser.add_argument("--video", type=str, default=None, help="Specific video path to evaluate")
    args = parser.parse_args()

    evaluator = LiveResearchEvaluator(max_frames=args.max_frames)
    spec = full_spec("Config D (Proposed Framework)")

    if args.video:
        if not os.path.exists(args.video):
            print(f"Error: Video file not found: {args.video}")
            return
        test_videos = [os.path.abspath(args.video)]
    else:
        all_videos = discover_snatch_videos()
        pos = [v for v in all_videos if is_incident_video(v)]
        neg = [v for v in all_videos if not is_incident_video(v)]
        
        # Prioritize clear daylight normal videos over dark night CCTV videos
        day_normal = [v for v in neg if any(d in os.path.basename(v) for d in ("s_15", "s_18", "s_20", "s_23", "s_24"))]
        night_normal = [v for v in neg if v not in day_normal]
        neg_ordered = day_normal + night_normal

        if args.snatch_videos is not None or args.normal_videos is not None:
            n_pos = args.snatch_videos if args.snatch_videos is not None else 4
            n_neg = args.normal_videos if args.normal_videos is not None else 4
        else:
            n_neg = max(1, args.max_videos // 2) if neg else 0
            n_pos = max(1, args.max_videos - n_neg) if pos else 0

        test_videos = pos[:n_pos] + neg_ordered[:n_neg]

    print("\n" + "=" * 80)
    print(" FORENSIC SNATCH SIGNATURE EVIDENCE BREAKDOWN TOOL")
    print(f" Evaluating {len(test_videos)} video(s) | Frame limit: {args.max_frames} frames/video")
    print("=" * 80)

    summary_rows = []

    for idx, vpath in enumerate(test_videos, start=1):
        vname = os.path.basename(vpath)
        gt_is_snatch = is_incident_video(vpath)
        gt_label = "SNATCH THEFT" if gt_is_snatch else "NORMAL"

        print(f"\n[{idx}/{len(test_videos)}] Video: {vname} (Ground Truth: {gt_label})")
        print("-" * 80)

        cap = cv2.VideoCapture(vpath)
        if not cap.isOpened():
            print(f"  ! Could not open {vpath}")
            continue

        fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
        if fps <= 0 or fps != fps:
            fps = 20.0

        detector = evaluator._ensure_detector()
        stages, handles = evaluator._build_stages(spec, fps=fps, video_name=vname)
        pipeline = Pipeline(stages=stages)
        mog2 = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)

        frame_number = 0
        processed_cnt = 0
        max_proximity = 0.0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_number += 1
                if args.max_frames and frame_number > args.max_frames:
                    break

                if frame_number % 100 == 0 or frame_number == args.max_frames:
                    print(f"\r  ⏳ Processing frame {frame_number}/{args.max_frames} (active: {processed_cnt})...", end="", flush=True)

                if mog2 is not None:
                    mask = mog2.apply(frame)
                    if cv2.countNonZero(mask) <= MOTION_PIXEL_THRESHOLD:
                        continue

                detections = detector.detect(frame)
                max_proximity = max(max_proximity, person_vehicle_proximity_score(detections))

                if not detections:
                    continue

                processed_cnt += 1
                context = FrameContext(
                    frame=frame,
                    frame_number=frame_number,
                    timestamp=frame_number / fps,
                    detections=detections,
                )
                pipeline.run(context)
            print()  # Newline after progress
        finally:
            cap.release()
            for stage in stages:
                _finalize_stage(stage)

        # Retrieve forensic signatures
        sig_stage = handles.get("signature")
        all_results = sig_stage.engine.get_all_results() if sig_stage else []
        sorted_results = sorted(all_results, key=lambda r: r.signature_score, reverse=True)
        best_result = sorted_results[0] if sorted_results else None

        score, evidence_ratio = evaluator._decision_score(spec, handles, max_proximity)
        predicted = int(score >= evaluator.decision_threshold)
        gt = int(gt_is_snatch)
        correct = (predicted == gt)
        pred_label = "POSITIVE (Snatch)" if predicted == 1 else "NEGATIVE (Normal)"
        status = "CORRECT" if correct else "MISCLASSIFIED"

        print(f"  • Overall Decision Score : {score:.4f} (Threshold = {evaluator.decision_threshold:.2f})")
        print(f"  • Final Classification   : {pred_label}  -->  [{status}]")
        print(f"  • Evaluated Signatures   : {len(all_results)} signature candidate(s)")

        if best_result is not None:
            print(f"\n  EVIDENCE COMPONENTS BREAKDOWN (Top Candidate: {best_result.interaction_id} | Score: {best_result.signature_score:.4f}):")
            print(f"  {'-'*76}")
            print(f"  {'Evidence Component':<25} | {'Weight':<6} | {'Status':<9} | {'Score Contrib':<13} | Details")
            print(f"  {'-'*76}")

            template_weights = {}
            if sig_stage and hasattr(sig_stage, "engine"):
                matcher = getattr(sig_stage.engine, "matcher", None)
                if matcher and hasattr(matcher, "template"):
                    template_weights = getattr(matcher.template, "evidence_weights", {})

            weights = template_weights if template_weights else {
                "closing_approach_vector": 0.20,
                "directed_grab": 0.35,
                "victim_reactive_jerk": 0.25,
                "rapid_getaway_escape": 0.20,
            }

            matched_map = {m["component"]: m for m in best_result.matched_evidence}
            missing_map = {m["component"]: m for m in best_result.missing_evidence}

            total_contrib = 0.0
            for comp, w in weights.items():
                if comp in matched_map:
                    stat = "MATCHED ✓"
                    contrib = f"+{w:.2f}"
                    total_contrib += w
                    desc = matched_map[comp]["description"]
                else:
                    stat = "MISSING ✗"
                    contrib = " 0.00"
                    desc = missing_map.get(comp, {}).get("description", "Not detected")

                print(f"  {comp:<25} | {w:<6.2f} | {stat:<9} | {contrib:<13} | {desc}")

            print(f"  {'-'*76}")
            print(f"  {'TOTAL SIGNATURE SCORE':<25} | 1.00   | {'--':<9} | {best_result.signature_score:<13.4f} | Decision: {best_result.decision}")
            print(f"  • Pose Actions Detected : {best_result.action_evidence if best_result.action_evidence else ['None / Unknown']}")
            print(f"  • Graph Patterns Detected: {list(best_result.behaviour_evidence) if best_result.behaviour_evidence else ['None']}")
        else:
            print(f"\n  ! No completed snatch signatures formed for this video (Fallback proximity: {max_proximity:.4f})")

        summary_rows.append({
            "video": vname,
            "gt": gt_label,
            "pred": pred_label,
            "score": score,
            "status": status,
        })

    print("\n" + "=" * 80)
    print(" CONSOLIDATED SUMMARY TABLE")
    print("=" * 80)
    print(f"{'Video':<18} | {'Ground Truth':<14} | {'Prediction':<18} | {'Score':<7} | {'Status'}")
    print("-" * 80)
    tp = fp = tn = fn = 0
    for r in summary_rows:
        print(f"{r['video']:<18} | {r['gt']:<14} | {r['pred']:<18} | {r['score']:<7.4f} | {r['status']}")
        if r['gt'] == "SNATCH THEFT" and "POSITIVE" in r['pred']:
            tp += 1
        elif r['gt'] == "SNATCH THEFT" and "NEGATIVE" in r['pred']:
            fn += 1
        elif r['gt'] == "NORMAL" and "POSITIVE" in r['pred']:
            fp += 1
        elif r['gt'] == "NORMAL" and "NEGATIVE" in r['pred']:
            tn += 1

    total = len(summary_rows)
    acc = (tp + tn) / total if total > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec_rate = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

    print("=" * 80)
    print(" OVERALL PERFORMANCE METRICS")
    print("=" * 80)
    print(f" Total Videos Evaluated : {total} (Snatch: {tp + fn}, Normal: {tn + fp})")
    print(f" True Positives  (TP)   : {tp}")
    print(f" False Positives (FP)   : {fp}")
    print(f" True Negatives  (TN)   : {tn}")
    print(f" False Negatives (FN)   : {fn}")
    print("-" * 80)
    print(f" Overall Accuracy       : {acc * 100:.2f}%")
    print(f" Precision              : {prec * 100:.2f}%")
    print(f" Recall / Sensitivity   : {rec * 100:.2f}%")
    print(f" Specificity            : {spec_rate * 100:.2f}%")
    print(f" F1-Score               : {f1:.4f}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
