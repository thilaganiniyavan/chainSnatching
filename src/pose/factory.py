"""Pose Estimator Factory.

Instantiates concrete AbstractPoseEstimator implementations and adapters
by backend name ("mediapipe", "rtmpose", "vitpose", "mmpose", "openpose").
"""

from __future__ import annotations

from src.pose.base_estimator import AbstractPoseEstimator
from src.pose.mediapipe_estimator import MediaPipePoseEstimator
from src.pose.adapters.rtmpose_adapter import RTMPoseAdapter
from src.pose.adapters.vitpose_adapter import ViTPoseAdapter
from src.pose.adapters.mmpose_adapter import MMPoseAdapter
from src.pose.adapters.openpose_adapter import OpenPoseAdapter


class PoseEstimatorFactory:
    """Factory for instantiating pose estimation model backends."""

    _BACKENDS = {
        "mediapipe": MediaPipePoseEstimator,
        "rtmpose": RTMPoseAdapter,
        "vitpose": ViTPoseAdapter,
        "mmpose": MMPoseAdapter,
        "openpose": OpenPoseAdapter,
    }

    @classmethod
    def create(
        cls,
        backend_name: str = "mediapipe",
        **kwargs,
    ) -> AbstractPoseEstimator:
        """Create and return a pose estimator instance for *backend_name*.

        Args:
            backend_name: Name of backend ("mediapipe", "rtmpose", "vitpose", "mmpose", "openpose").
            **kwargs: Additional arguments passed to backend constructor.

        Returns:
            An instance of :class:`AbstractPoseEstimator`.

        Raises:
            ValueError: If *backend_name* is not supported.
        """
        key = backend_name.lower().strip()
        if key not in cls._BACKENDS:
            valid_keys = ", ".join(cls._BACKENDS.keys())
            raise ValueError(
                f"Unsupported pose backend '{backend_name}'. Supported backends: {valid_keys}"
            )

        estimator_cls = cls._BACKENDS[key]
        return estimator_cls(**kwargs)

    @classmethod
    def register_backend(cls, name: str, estimator_cls: type[AbstractPoseEstimator]) -> None:
        """Register a new custom pose estimator class."""
        cls._BACKENDS[name.lower().strip()] = estimator_cls

    @classmethod
    def get_supported_backends(cls) -> list[str]:
        """Return list of supported backend names."""
        return list(cls._BACKENDS.keys())
