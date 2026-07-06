# 👁️ AI Surveillance Framework

Welcome to the central repository for our research-grade AI Surveillance Framework. This project is aimed at building a highly modular, intelligent video analytics pipeline. The ultimate long-term goal of this system is to identify complex, suspicious human-vehicle interactions (such as chain-snatching incidents) in real-time.

---

## 🚀 What We've Built So Far

Instead of building one massive script, we've taken a highly modular, software-engineering-first approach. We've built an orchestration pipeline where independent "stages" operate on a shared `FrameContext`. This makes the system incredibly easy to extend.

Here's the current state of our fully functioning, end-to-end pipeline:

1. **Motion Detection (Pre-filter)**
   - We implemented a background subtraction stage (`MOG2`) to filter out completely static frames. If there's no movement in the camera view, we don't waste precious GPU cycles running heavy neural networks.

2. **Object Detection (YOLO)**
   - We integrated a YOLO-based detector that is configured to specifically look for classes relevant to our domain: `person`, `bicycle`, `motorcycle`, `car`, `bus`, and `truck`.

3. **Persistent Object Tracking**
   - We implemented a robust tracking module that assigns unique IDs to objects and follows them across frames, ensuring we know that "Person 7" in frame 1 is the same "Person 7" in frame 100.

4. **Trajectory & Track History**
   - We built a `TrackHistoryManager` and a `TrajectoryVisualizer` that remembers where objects have been. It draws visual paths on the screen so we can see the exact historical movement of every tracked object.

5. **Motion Feature Extraction**
   - Our `MotionFeatureExtractor` analyzes the track history mathematically to compute real-time physics properties for each object:
     - Instantaneous Speed
     - Average Speed
     - Direction of Movement
     - Total Distance Travelled

6. **Relationship Engine (Version 1)**
   - We recently introduced our first behavior analysis module. The `RelationshipEngine` evaluates the spatial dynamics between objects. Currently, it continuously calculates the Euclidean distance between every person and their nearest vehicle. If they come within a specific threshold (e.g., 150 pixels), it creates a formal "Near" relationship and draws a visual tether on the screen along with the distance.

7. **The Orchestrator (`surveillance_demo.py`)**
   - We tied everything together into a master application. The demo seamlessly streams video (or a live webcam), pushes each frame through the above stages, and renders a unified output video displaying bounding boxes, active tracking IDs, trajectories, live speed/direction stats, and proximity tethers.

---

## 📍 Current Stage

We have successfully completed the foundational tracking, feature extraction, and simple spatial relationship layers. The system is now capable of "understanding" the physical scene, knowing who is moving, how fast they are going, where they came from, and who/what they are standing near.

### Next Steps:
Now that the physics and basic relationships are established, the next phase will focus on **Complex Event Detection**. We will start building modules that analyze these motion features and relationships over time to detect specific behaviors (e.g., a motorcycle approaching a person at high speed, stopping briefly, and then accelerating away rapidly). 

---

## 💻 How to Run the Demo

To see the current pipeline in action on a video file:
```bash
python apps/surveillance_demo.py \
    --input videos/test.mp4 \
    --output outputs/surveillance_output.mp4 \
    --show \
    --save
```

To run it live on your webcam:
```bash
python apps/surveillance_demo.py \
    --input 0 \
    --output outputs/webcam_output.mp4 \
    --show \
    --save
```
