"""
Event preservation evaluation metrics for CCTV Forensic Search.
These metrics quantify whether motion filtering preserves critical event frame ranges.
"""

def calculate_event_recall(event_ranges: list[tuple[int, int]], motion_decisions: list[bool]) -> float:
    """
    Calculate the recall of event frames under motion filtering.
    
    Args:
        event_ranges: List of tuples specifying start and end frame ranges (inclusive).
                      Ranges are assumed to be 0-indexed.
        motion_decisions: List of booleans indicating whether motion was detected 
                           for each frame index.
                           
    Returns:
        The fraction of event frames that were correctly preserved (detected as motion).
        Returns 1.0 if there are no event frames defined.
    """
    event_frames = set()
    for start, end in event_ranges:
        for f in range(start, end + 1):
            event_frames.add(f)
            
    if not event_frames:
        return 1.0
        
    preserved_count = 0
    for f in event_frames:
        if f < len(motion_decisions) and motion_decisions[f]:
            preserved_count += 1
            
    return preserved_count / len(event_frames)

def calculate_frame_preservation(event_ranges: list[tuple[int, int]], motion_decisions: list[bool]) -> dict:
    """
    Generate detailed statistics about frame preservation.
    
    Args:
        event_ranges: List of tuples specifying start and end frame ranges (inclusive).
        motion_decisions: List of booleans indicating motion detection for each frame.
        
    Returns:
        Dictionary containing detailed preservation stats.
    """
    event_frames = set()
    for start, end in event_ranges:
        for f in range(start, end + 1):
            event_frames.add(f)
            
    total_event_frames = len(event_frames)
    
    if total_event_frames == 0:
        return {
            "total_event_frames": 0,
            "preserved_event_frames": 0,
            "missed_event_frames": 0,
            "preservation_rate": 1.0
        }
        
    preserved_event_frames = 0
    for f in event_frames:
        if f < len(motion_decisions) and motion_decisions[f]:
            preserved_event_frames += 1
            
    missed_event_frames = total_event_frames - preserved_event_frames
    preservation_rate = preserved_event_frames / total_event_frames
    
    return {
        "total_event_frames": total_event_frames,
        "preserved_event_frames": preserved_event_frames,
        "missed_event_frames": missed_event_frames,
        "preservation_rate": preservation_rate
    }
