import cv2
from src.core.interfaces import Stage
from src.core.models import FrameContext
from src.behavior.relationship_engine import RelationshipEngine

class RelationshipStage(Stage):
    """Pipeline stage that computes spatial relationships between tracked objects."""

    def __init__(self, distance_threshold: float = 150.0):
        self.engine = RelationshipEngine(distance_threshold=distance_threshold)

    def process(self, context: FrameContext) -> FrameContext:
        """Computes relationships and visualizes them on a new frame."""
        
        # Compute relationships using the RelationshipEngine
        relationships = self.engine.compute(context.tracks, context.timestamp)
        
        # Store in metadata
        context.metadata["relationships"] = relationships
        
        # Visualization
        # Use trajectory_frame as the base so we can layer information, but copy it
        # so we do not overwrite the original trajectory_frame as per instructions.
        base_frame = context.metadata.get("trajectory_frame", context.frame)
        viz_frame = base_frame.copy()
        
        # Fast lookup for track centers
        track_centers = {t.tracking_id: t.center for t in context.tracks if t.center}

        for rel in relationships:
            p1 = track_centers.get(rel.subject_id)
            p2 = track_centers.get(rel.object_id)
            
            if p1 and p2:
                # Draw thin line between objects
                cv2.line(viz_frame, p1, p2, (0, 0, 255), 1)
                
                # Display distance at the midpoint
                midpoint = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
                cv2.putText(
                    viz_frame, 
                    f"{rel.distance:.1f}px", 
                    (midpoint[0] + 5, midpoint[1] - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, 
                    (0, 0, 255), 
                    1
                )
                
        # Store annotated frame separately
        context.metadata["relationship_frame"] = viz_frame
        
        return context
