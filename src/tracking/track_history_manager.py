from typing import Optional
from src.core.models.track_history import TrackHistory
from src.core.models.track import Track

class TrackHistoryManager:
    """Manages the temporal history for every tracked object."""

    def __init__(self):
        self.history: dict[int, TrackHistory] = {}

    def update(self, track: Track, timestamp: float) -> None:
        """
        Update the history for the given track.
        
        If it's a new tracking ID, a new TrackHistory is created.
        Otherwise, the track's center position and timestamp are appended.
        """
        # If no tracking ID is provided, we can't reliably track history
        if track.tracking_id < 0:
            return

        # Ensure TrackHistory exists for this ID
        if track.tracking_id not in self.history:
            self.history[track.tracking_id] = TrackHistory(tracking_id=track.tracking_id)

        history = self.history[track.tracking_id]

        # Ensure center is calculated if lacking (as per requirements)
        # Note: center is expected to be assigned by TrackingStage.
        # If not, we skip appending position.
        if track.center is not None:
            history.positions.append(track.center)
            
        history.timestamps.append(timestamp)

    def get(self, tracking_id: int) -> Optional[TrackHistory]:
        """Return the TrackHistory for a given tracking_id."""
        return self.history.get(tracking_id)

    def remove(self, tracking_id: int) -> None:
        """Delete a lost track's history."""
        if tracking_id in self.history:
            del self.history[tracking_id]

    def clear(self) -> None:
        """Clear all histories."""
        self.history.clear()
