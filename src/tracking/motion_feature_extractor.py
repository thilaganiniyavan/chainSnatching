import math
from src.core.models.track import Track

class MotionFeatureExtractor:
    """Computes basic motion features for a tracked object based on its history."""

    def compute(self, track: Track) -> None:
        """
        Calculates distance travelled, average speed, instantaneous speed,
        and direction, then stores them back in the track object.
        """
        if not track.history or len(track.history.positions) <= 1:
            track.instantaneous_speed = 0.0
            track.average_speed = 0.0
            track.direction = None
            track.distance_travelled = 0.0
            return

        positions = track.history.positions

        # 1. Distance travelled (Euclidean distance between consecutive points)
        total_distance = 0.0
        for i in range(1, len(positions)):
            x1, y1 = positions[i - 1]
            x2, y2 = positions[i]
            dist = math.hypot(x2 - x1, y2 - y1)
            total_distance += dist

        track.distance_travelled = total_distance

        # 2. Instantaneous speed (Distance between last two positions)
        x1, y1 = positions[-2]
        x2, y2 = positions[-1]
        inst_speed = math.hypot(x2 - x1, y2 - y1)
        track.instantaneous_speed = inst_speed

        # 3. Average speed (Total distance / number of intervals)
        intervals = len(positions) - 1
        track.average_speed = total_distance / intervals if intervals > 0 else 0.0

        # 4. Direction of movement (Angle between last two positions)
        # Using math.atan2(y2 - y1, x2 - x1)
        direction_rad = math.atan2(y2 - y1, x2 - x1)
        direction_deg = math.degrees(direction_rad)
        
        # Normalize to 0-360 degrees
        direction_deg = (direction_deg + 360) % 360
        track.direction = direction_deg
