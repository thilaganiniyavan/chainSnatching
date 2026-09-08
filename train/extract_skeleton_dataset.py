"""Automated Skeleton Sequence Dataset Extraction for ST-GCN Training.

Processes all CCTV videos in the dataset, tracks persons and vehicles using YOLO + ByteTrack,
crops person ROIs for high-precision MediaPipe Pose estimation, and extracts labeled
(C=4, T=32, V=17) normalized skeleton sequences.
"""

from __future__ import annotations

import argparse
import os
import sys
import cv2
import numpy as np

# Ensure project root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.detection.detector import Detector
from src.pose.mediapipe_estimator import MediaPipePoseEstimator
from src.behavior.skeleton_normalizer import SkeletonNormalizer
from src.evaluation.research_live_evaluator import discover_snatch_videos, is_incident_video

ACTION_TAXONOMY = [
    "Walking",
    "Standing",
    "Running",
    "Approaching",
    "Reaching",
    "Grabbing",
    "Pulling",
    "Turning",
    "Falling",
    "Unknown",
]
ACTION_TO_IDX = {name: idx for idx, name in enumerate(ACTION_TAXONOMY)}


def extract_video_skeletons(
    video_path: str,
    detector: Detector,
    pose_estimator: MediaPipePoseEstimator,
    normalizer: SkeletonNormalizer,
    target_len: int = 32,
    max_frames: int = 350,
) -> list[tuple[np.ndarray, str, dict]]:
    """Extract temporal skeleton sequences from tracked persons in a video."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  ! Could not open {video_path}")
        return []

    vname = os.path.basename(video_path)
    is_snatch = is_incident_video(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    if fps <= 0 or fps != fps:
        fps = 20.0

    # Person trajectory storage: track_id -> list of (frame_idx, keypoints_17_4, bbox)
    person_trajectories: dict[int, list[tuple[int, np.ndarray, tuple[int, int, int, int]]]] = {}
    active_boxes: dict[int, tuple[int, int, int, int]] = {}
    next_track_id = 1

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret or frame_count >= max_frames:
            break
        frame_count += 1

        h, w = frame.shape[:2]
        normalizer.image_width = w
        normalizer.image_height = h

        # Run YOLO detection for persons
        raw_detections = detector.detect(frame)
        person_dets = [d for d in raw_detections if getattr(d, "class_name", "") == "person" and d.confidence >= 0.30]

        # Greedy tracking across frames based on IoU / center distance
        current_matched_tracks: dict[int, tuple[int, int, int, int]] = {}
        unmatched_dets = []

        for det in person_dets:
            bx = (det.x1, det.y1, det.x2, det.y2)
            # Filter tiny bounding boxes
            if (det.x2 - det.x1) < 20 or (det.y2 - det.y1) < 40:
                continue

            best_tid = None
            best_dist = 120.0  # Max center displacement in pixels between adjacent frames

            cx, cy = det.center_x, det.center_y
            for tid, prev_box in active_boxes.items():
                if tid in current_matched_tracks:
                    continue
                pcx = (prev_box[0] + prev_box[2]) / 2.0
                pcy = (prev_box[1] + prev_box[3]) / 2.0
                dist = np.hypot(cx - pcx, cy - pcy)
                if dist < best_dist:
                    best_dist = dist
                    best_tid = tid

            if best_tid is not None:
                current_matched_tracks[best_tid] = bx
            else:
                unmatched_dets.append(bx)

        for bx in unmatched_dets:
            tid = next_track_id
            next_track_id += 1
            current_matched_tracks[tid] = bx

        active_boxes = current_matched_tracks

        # Run MediaPipe Pose on each tracked person's cropped ROI
        for tid, bbox in active_boxes.items():
            # Pad bbox slightly for better MediaPipe landmark capture
            pad_x = int((bbox[2] - bbox[0]) * 0.15)
            pad_y = int((bbox[3] - bbox[1]) * 0.15)
            padded_box = (
                max(0, bbox[0] - pad_x),
                max(0, bbox[1] - pad_y),
                min(w, bbox[2] + pad_x),
                min(h, bbox[3] + pad_y),
            )

            pose_res = pose_estimator.estimate_pose(
                image=frame,
                bbox=padded_box,
                frame_index=frame_count,
                timestamp=frame_count / fps,
                track_id=tid,
                interaction_id=f"TRK-{tid}",
            )

            kps = np.array(pose_res.keypoints_pixel, dtype=np.float32) if pose_res.keypoints_pixel else np.zeros((17, 4), dtype=np.float32)
            if tid not in person_trajectories:
                person_trajectories[tid] = []
            person_trajectories[tid].append((frame_count, kps, bbox))

    cap.release()

    samples: list[tuple[np.ndarray, str, dict]] = []

    # Process all tracked person trajectories with length >= target_len
    for tid, traj in person_trajectories.items():
        if len(traj) < target_len:
            continue

        traj_kps = np.array([pt[1] for pt in traj], dtype=np.float32)  # (T_traj, 17, 4)
        total_T = traj_kps.shape[0]

        # Calculate focal wrist elevation / arm extension for snatch classification
        wrist_extensions = []
        for t in range(total_T):
            sh_y = (traj_kps[t, 5, 1] + traj_kps[t, 6, 1]) / 2.0
            sh_x = (traj_kps[t, 5, 0] + traj_kps[t, 6, 0]) / 2.0
            lw_dist = np.hypot(traj_kps[t, 9, 0] - sh_x, traj_kps[t, 9, 1] - sh_y)
            rw_dist = np.hypot(traj_kps[t, 10, 0] - sh_x, traj_kps[t, 10, 1] - sh_y)
            wrist_extensions.append(max(lw_dist, rw_dist))

        focal_frame_idx = int(np.argmax(wrist_extensions)) if wrist_extensions else total_T // 2

        stride = max(4, target_len // 4)
        for start_t in range(0, total_T - target_len + 1, stride):
            end_t = start_t + target_len
            window = traj_kps[start_t:end_t]  # (32, 17, 4)
            norm_window = normalizer.normalize_tensor(window, topology="COCO_17")  # (32, 17, 4)
            c_t_v = np.transpose(norm_window, (2, 0, 1))  # (4, 32, 17)

            # Assign label based on physical kinematics
            mid_t = start_t + target_len // 2

            # Compute kinematic metrics
            disp = np.linalg.norm(norm_window[1:, :, :2] - norm_window[:-1, :, :2], axis=2)
            mean_speed = float(np.mean(disp))
            torso_drop = float(np.mean(norm_window[-8:, 11:13, 1]) - np.mean(norm_window[:8, 11:13, 1]))

            if is_snatch:
                # Distinguish snatch attack phases
                dist_to_focal = abs(mid_t - focal_frame_idx)
                if dist_to_focal <= 10:
                    label = "Grabbing"
                elif dist_to_focal <= 22:
                    label = "Reaching"
                elif torso_drop > 0.25:
                    label = "Falling"
                elif mid_t > (focal_frame_idx + 22) and mean_speed > 0.05:
                    label = "Running"
                elif mean_speed >= 0.06:
                    label = "Running"
                else:
                    label = "Approaching"
            else:
                # Normal surveillance movement
                if mean_speed >= 0.075:
                    label = "Running"
                elif mean_speed <= 0.015:
                    label = "Standing"
                else:
                    label = "Walking"

            meta = {
                "video": vname,
                "track_id": tid,
                "start_frame": traj[start_t][0],
                "end_frame": traj[end_t - 1][0],
                "is_snatch": is_snatch,
            }
            samples.append((c_t_v, label, meta))

    return samples


def main():
    parser = argparse.ArgumentParser(description="Extract CCTV Skeleton Dataset for ST-GCN")
    parser.add_argument("--output", type=str, default="data/skeleton_dataset.npz", help="Output path for .npz dataset")
    parser.add_argument("--target-len", type=int, default=32, help="Sequence temporal length (default: 32)")
    parser.add_argument("--max-frames", type=int, default=350, help="Max frames to extract per video")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)

    videos = discover_snatch_videos()
    print("=" * 80)
    print(f" CCTV SKELETON DATASET EXTRACTOR (PERSON-TRACK CROPPED)")
    print(f" Discovered {len(videos)} video files across dataset directories")
    print("=" * 80)

    detector = Detector(model_path="yolo11n.pt")
    pose_estimator = MediaPipePoseEstimator(topology="COCO_17")
    normalizer = SkeletonNormalizer(method="hip_centered")

    all_tensors = []
    all_labels = []
    all_meta = []
    label_counts: dict[str, int] = {}

    for idx, vpath in enumerate(videos, start=1):
        vname = os.path.basename(vpath)
        is_snatch = is_incident_video(vpath)
        gt_type = "SNATCH" if is_snatch else "NORMAL"
        print(f"[{idx:02d}/{len(videos):02d}] Extracting from: {vname:<22} ({gt_type})...", end="", flush=True)

        samples = extract_video_skeletons(
            video_path=vpath,
            detector=detector,
            pose_estimator=pose_estimator,
            normalizer=normalizer,
            target_len=args.target_len,
            max_frames=args.max_frames,
        )

        for tensor, label, meta in samples:
            all_tensors.append(tensor)
            all_labels.append(ACTION_TO_IDX.get(label, ACTION_TO_IDX["Unknown"]))
            all_meta.append(meta)
            label_counts[label] = label_counts.get(label, 0) + 1

        print(f" Extracted {len(samples):3d} clips.")

    if not all_tensors:
        print("! Error: No skeleton sequences were extracted.")
        return

    X = np.stack(all_tensors, axis=0).astype(np.float32)  # (N, 4, 32, 17)
    y = np.array(all_labels, dtype=np.int64)  # (N,)

    print("\n" + "=" * 80)
    print(" EXTRACTION COMPLETED SUMMARY")
    print("=" * 80)
    print(f" Total Samples Extracted : {X.shape[0]}")
    print(f" Tensor Shape (N, C, T, V): {X.shape}")
    print(" Class Distribution:")
    for cls_name, count in sorted(label_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"   • {cls_name:<15}: {count:4d} samples ({count / len(y) * 100:.1f}%)")

    np.savez_compressed(
        args.output,
        X=X,
        y=y,
        taxonomy=ACTION_TAXONOMY,
    )
    print(f"\n Saved dataset to: {os.path.abspath(args.output)} ({os.path.getsize(args.output) / 1024:.1f} KB)\n")


if __name__ == "__main__":
    main()
