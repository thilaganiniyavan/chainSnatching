"""Abstract Action Recognizer Interface.

Defines the contract for all human action recognition backends and framework adapters.
Every implementation accepts SkeletonSequence objects containing (T, V, C) tensors
and produces standardized ActionResult classification outputs.

Does NOT perform forensic decision-making or chain-snatching classification.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from src.core.models.action_result import ActionResult
from src.core.models.skeleton_sequence import SkeletonSequence


DEFAULT_ACTION_TAXONOMY = [
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


class AbstractActionRecognizer(ABC):
    """Abstract base class for all skeleton action recognition models and adapters.

    Args:
        backend_name: Label of backend model architecture.
        action_taxonomy: List of action class names.
    """

    def __init__(
        self,
        backend_name: str = "Abstract",
        action_taxonomy: list[str] | None = None,
    ) -> None:
        self.backend_name = backend_name
        self.action_taxonomy = (
            list(action_taxonomy)
            if action_taxonomy is not None
            else list(DEFAULT_ACTION_TAXONOMY)
        )

    @abstractmethod
    def predict_action(self, sequence: SkeletonSequence) -> ActionResult:
        """Classify human action for a single SkeletonSequence object.

        Args:
            sequence: Input :class:`SkeletonSequence` instance carrying (T, V, C) tensor.

        Returns:
            An :class:`ActionResult` object containing prediction label and confidence.
        """
        pass

    @abstractmethod
    def predict_batch(self, sequences: list[SkeletonSequence]) -> list[ActionResult]:
        """Classify human actions for a batch of SkeletonSequence objects.

        Args:
            sequences: List of :class:`SkeletonSequence` instances.

        Returns:
            List of :class:`ActionResult` objects.
        """
        pass
