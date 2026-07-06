import math
from typing import Any
from src.core.models.track import Track
from src.core.models.relationship import Relationship

class RelationshipEngine:
    """Engine to compute simple spatial relationships between tracked objects."""

    def __init__(self, distance_threshold: float = 150.0):
        self.distance_threshold = distance_threshold
        self.vehicle_classes = {"bicycle", "motorcycle", "car", "bus", "truck"}
        self.person_class = "person"

    def compute(self, tracks: list[Track], timestamp: float) -> list[Relationship]:
        """Computes spatial relationships between persons and vehicles."""
        relationships = []

        persons = [t for t in tracks if t.class_name == self.person_class and t.center is not None]
        vehicles = [t for t in tracks if t.class_name in self.vehicle_classes and t.center is not None]

        for person in persons:
            nearest_vehicle = None
            min_distance = float('inf')

            for vehicle in vehicles:
                dist = self._euclidean_distance(person.center, vehicle.center)
                if dist < min_distance:
                    min_distance = dist
                    nearest_vehicle = vehicle

            if nearest_vehicle is not None and min_distance < self.distance_threshold:
                rel = Relationship(
                    subject_id=person.tracking_id,
                    subject_class=person.class_name,
                    object_id=nearest_vehicle.tracking_id,
                    object_class=nearest_vehicle.class_name,
                    relationship_type="near",
                    distance=min_distance,
                    timestamp=timestamp,
                    metadata={}
                )
                relationships.append(rel)

        return relationships

    @staticmethod
    def _euclidean_distance(p1: tuple[int, int], p2: tuple[int, int]) -> float:
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
