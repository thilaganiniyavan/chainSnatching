"""Action Recognizer Factory.

Instantiates concrete AbstractActionRecognizer implementations and adapters
by backend name ("stgcn", "ctrgcn", "msg3d", "posec3d").
"""

from __future__ import annotations

from src.action.base_recognizer import AbstractActionRecognizer
from src.action.stgcn_recognizer import STGCNRecognizer
from src.action.adapters.ctrgcn_adapter import CTRGCNRecognizer
from src.action.adapters.msg3d_adapter import MSG3DRecognizer
from src.action.adapters.posec3d_adapter import PoseC3DRecognizer


class ActionRecognizerFactory:
    """Factory for instantiating human action recognition model backends."""

    _BACKENDS = {
        "stgcn": STGCNRecognizer,
        "ctrgcn": CTRGCNRecognizer,
        "msg3d": MSG3DRecognizer,
        "posec3d": PoseC3DRecognizer,
    }

    @classmethod
    def create(
        cls,
        backend_name: str = "stgcn",
        **kwargs,
    ) -> AbstractActionRecognizer:
        """Create and return an action recognizer instance for *backend_name*.

        Args:
            backend_name: Name of backend ("stgcn", "ctrgcn", "msg3d", "posec3d").
            **kwargs: Additional arguments passed to backend constructor.

        Returns:
            An instance of :class:`AbstractActionRecognizer`.

        Raises:
            ValueError: If *backend_name* is not supported.
        """
        key = backend_name.lower().strip()
        if key not in cls._BACKENDS:
            valid_keys = ", ".join(cls._BACKENDS.keys())
            raise ValueError(
                f"Unsupported action recognizer backend '{backend_name}'. Supported backends: {valid_keys}"
            )

        recognizer_cls = cls._BACKENDS[key]
        return recognizer_cls(**kwargs)

    @classmethod
    def register_backend(cls, name: str, recognizer_cls: type[AbstractActionRecognizer]) -> None:
        """Register a new custom action recognizer class."""
        cls._BACKENDS[name.lower().strip()] = recognizer_cls

    @classmethod
    def get_supported_backends(cls) -> list[str]:
        """Return list of supported backend names."""
        return list(cls._BACKENDS.keys())
