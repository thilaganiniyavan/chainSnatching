"""Pose Estimation Abstraction Layer for the forensic surveillance framework.

Provides model-agnostic pose estimator interface, MediaPipe implementation,
scaffolded adapters for RTMPose, ViTPose, MMPose, OpenPose, PoseEstimatorFactory,
post-processing, visualization, and structured dataset export.
"""

from src.pose.base_estimator import AbstractPoseEstimator
from src.pose.mediapipe_estimator import MediaPipePoseEstimator
from src.pose.adapters.rtmpose_adapter import RTMPoseAdapter
from src.pose.adapters.vitpose_adapter import ViTPoseAdapter
from src.pose.adapters.mmpose_adapter import MMPoseAdapter
from src.pose.adapters.openpose_adapter import OpenPoseAdapter
from src.pose.factory import PoseEstimatorFactory
from src.pose.pose_post_processor import PosePostProcessor
from src.pose.pose_visualizer import PoseOverlayVisualizer, PosePreviewExporter
from src.pose.pose_logger import PoseLogger
