import math
from typing import List
from src.core.models.track import Track
from src.core.models.relationship import Relationship

class RelationshipEngine:
    """Computes spatial relationships between tracked objects."""

    def __init__(self, distance_threshold: float = 150.0):
        self.distance_threshold = distance_threshold
        self.vehicle_classes = {"bicycle", "motorcycle", "car", "bus", "truck"}

    def compute(self, tracks: List[Track], timestamp: float) -> List[Relationship]:
        """
        Calculates spatial relationships between tracks.
        Currently focuses on finding the nearest vehicle for each person.
        """
        relationships = []
        
        # Separate people and vehicles
        people = [t for t in tracks if t.class_name == "person"]
        vehicles = [t for t in tracks if t.class_name in self.vehicle_classes]

        # Ensure both lists are not empty
        if not people or not vehicles:
            return relationships

        for person in people:
            # We need valid centers to compute distance
            if not person.center:
                continue
                
            nearest_vehicle = None
            min_distance = float('inf')

            # Find nearest vehicle
            for vehicle in vehicles:
                if not vehicle.center:
                    continue
                    
                px, py = person.center
                vx, vy = vehicle.center
                
                dist = math.hypot(vx - px, vy - py)
                
                if dist < min_distance:
                    min_distance = dist
                    nearest_vehicle = vehicle

            # If the nearest vehicle is within the threshold, create a relationship
            if nearest_vehicle and min_distance < self.distance_threshold:
                rel = Relationship(
                    subject_id=person.tracking_id,
                    subject_class=person.class_name,
                    object_id=nearest_vehicle.tracking_id,
                    object_class=nearest_vehicle.class_name,
                    relationship_type="near",
                    distance=min_distance,
                    timestamp=timestamp
                )
                relationships.append(rel)

        return relationships
