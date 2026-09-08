"""Diagnostic script to inspect ST-GCN kinematics and action classifications across Snatch and Normal videos."""

from __future__ import annotations

import os
import sys
import glob
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

import cv2
import numpy as np

from src.detection.yolo_detector import YOLODetector
from src.tracking.deepsort_tracker import DeepSORTTracker
from src.pose.mediapipe_estimator import MediaPipeEstimator
from src.behavior.skeleton_sequence_builder import SkeletonSequenceBuilder
from src.action.stgcn_recognizer import STGCNRecognizer
from src.snatch.signature_matcher import SignatureMatcher
from src.snatch.signature_config import StandardMotorcycleSnatchSignature


def diagnose_video(video_path: str, max_frames: int = 150):
    print(f"\n{'='*70}\nAnalyzing: {Path(video_path).name} ({'SNATCH' if 'Snatch Theft' in video_path else 'NORMAL'})\n{'='*70}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open {video_path}")
        return

    # Initialize components
    detector = YOLODetector(model_name="yolov8n.pt", conf_threshold=0.25)
    tracker = DeepSORTTracker(max_age=30, n_init=3)
    pose_estimator = MediaPipeEstimator()
    seq_builder = SkeletonSequenceBuilder(window_size=16, stride=4, min_valid_ratio=0.3)
    action_recognizer = STGCNRecognizer()
    matcher = SignatureMatcher(template=StandardMotorcycleSnatchSignature())

    frame_idx = 0
    detected_actions = []
    kinematics_log = []

    while cap.isOpened() and frame_idx < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        # 1. Detection
        detections = detector.detect(frame, frame_idx=frame_idx)
        # 2. Tracking
        tracks = tracker.update(detections, frame)

        # 3. Pose
        person_tracks = [t for t in tracks if t.class_name == "person"]
        for p in person_tracks:
            # Crop bbox
            bbox = None
            if p.detections:
                bbox = p.detections[-1].bbox
            if bbox:
                x1, y1, x2, y2 = [int(v) for v in bbox]
                h_img, w_img = frame.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w_img, x2), min(h_img, y2)
                if x2 > x1 and y2 > y1:
                    crop = frame[y1:y2, x1:x2]
                    pose = pose_estimator.estimate_keypoints(crop, bbox=bbox, frame_index=frame_idx, track_id=p.tracking_id)
                    seq_builder.add_pose(pose, p.tracking_id, frame_idx)

        # 4. Check ready sequences
        for p in person_tracks:
            if seq_builder.is_ready(p.tracking_id):
                seq = seq_builder.get_sequence(p.tracking_id)
                if seq:
                    # Inspect tensor kinematics
                    tensor = seq.get_tensor(num_joints=17) # [T, V, C]
                    T = tensor.shape[0]
                    head_idx = 0
                    l_shoulder, r_shoulder = 5, 6
                    l_wrist, r_wrist = 9, 10

                    horiz_exts = []
                    arm_elevs = []
                    wrist_vels = []
                    head_dists = []

                    for t in range(T):
                        dx_l = abs(float(tensor[t, l_wrist, 0] - tensor[t, l_shoulder, 0])) if l_wrist < tensor.shape[1] else 0.0
                        dx_r = abs(float(tensor[t, r_wrist, 0] - tensor[t, r_shoulder, 0])) if r_wrist < tensor.shape[1] else 0.0
                        dy_l = abs(float(tensor[t, l_wrist, 1] - tensor[t, l_shoulder, 1])) if l_wrist < tensor.shape[1] else 1.0
                        dy_r = abs(float(tensor[t, r_wrist, 1] - tensor[t, r_shoulder, 1])) if r_wrist < tensor.shape[1] else 1.0
                        dh_l = float(np.linalg.norm(tensor[t, l_wrist, :2] - tensor[t, head_idx, :2])) if l_wrist < tensor.shape[1] else 0.0
                        dh_r = float(np.linalg.norm(tensor[t, r_wrist, :2] - tensor[t, head_idx, :2])) if r_wrist < tensor.shape[1] else 0.0
                        horiz_exts.append(max(dx_l, dx_r))
                        arm_elevs.append(min(dy_l, dy_r))
                        head_dists.append(max(dh_l, dh_r))

                    for t in range(1, T):
                        vl = float(np.linalg.norm(tensor[t, l_wrist, :2] - tensor[t-1, l_wrist, :2])) if l_wrist < tensor.shape[1] else 0.0
                        vr = float(np.linalg.norm(tensor[t, r_wrist, :2] - tensor[t-1, r_wrist, :2])) if r_wrist < tensor.shape[1] else 0.0
                        wrist_vels.append(max(vl, vr))

                    max_h_ext = max(horiz_exts, default=0.0)
                    min_elev = min(arm_elevs, default=1.0)
                    max_w_vel = max(wrist_vels, default=0.0)
                    max_h_dist = max(head_dists, default=0.0)
                    h_delta = horiz_exts[-1] - horiz_exts[0] if horiz_exts else 0.0
                    peak_idx = int(np.argmax(horiz_exts))
                    retract = (horiz_exts[peak_idx] - horiz_exts[-1]) if 0 < peak_idx < len(horiz_exts) - 1 else 0.0
                    head_retract = (head_dists[peak_idx] - head_dists[-1]) if 0 < peak_idx < len(head_dists) - 1 else 0.0

                    act = action_recognizer.classify(seq)
                    detected_actions.append(act.predicted_action)
                    kinematics_log.append({
                        "frame": frame_idx,
                        "track_id": p.tracking_id,
                        "action": act.predicted_action,
                        "conf": round(act.action_confidence, 2),
                        "max_h_ext": round(max_h_ext, 3),
                        "min_elev": round(min_elev, 3),
                        "max_w_vel": round(max_w_vel, 3),
                        "max_h_dist": round(max_h_dist, 3),
                        "retract": round(retract, 3),
                        "head_retract": round(head_retract, 3),
                        "h_delta": round(h_delta, 3),
                    })

    cap.release()
    print(f"Total action windows evaluated: {len(kinematics_log)}")
    print(f"Detected actions summary: {set(detected_actions)}")
    target_matches = [k for k in kinematics_log if k["action"] in matcher.template.target_actions]
    print(f"Target actions matched: {len(target_matches)} / {len(kinematics_log)}")
    for k in kinematics_log[:8]:
        print(f"  Frame {k['frame']} | Track {k['track_id']} | Action: {k['action']} (conf={k['conf']}) | max_h_ext={k['max_h_ext']} | min_elev={k['min_elev']} | max_w_vel={k['max_w_vel']} | max_h_dist={k['max_h_dist']} | retract={k['retract']}")


def main():
    snatch_videos = sorted(glob.glob(str(root_dir / "Snatch 1.0" / "Chain Snatching Videos" / "Snatch Theft" / "*.mp4")))[:3]
    normal_videos = sorted(glob.glob(str(root_dir / "Snatch 1.0" / "Chain Snatching Videos" / "Normal" / "*.avi")))[:2]

    print("================ RUNNING SNATCH THEFT VIDEOS ================")
    for v in snatch_videos:
        diagnose_video(v)

    print("\n================ RUNNING NORMAL VIDEOS ================")
    for v in normal_videos:
        diagnose_video(v)


if __name__ == "__main__":
    main()
