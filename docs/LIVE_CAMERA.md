# Live camera detection

Install the central, edge and learned dependencies and the detector and pose models described in the main setup guide. Put a trusted, compatible SnatchGuard verifier `.pt` checkpoint in `models/verifier/` on the machine running the central server. These checkpoints are created by the project's verifier training tools; generic YOLO or unrelated PyTorch models cannot be used as verifier checkpoints. Model weights are not included in Git.

Start the central server, open **Footage & demo**, select the verifier, enter a camera name and an RTSP/HTTP stream URL or a webcam index such as `0`, then select **Start live detection**. Webcam indices refer to the server machine. Only one footage or camera job can run at a time. Use **Stop processing** to end it. The overview shows camera connection and verifier status, snapshots, and alerts. A running job means the process started; check the camera and verifier indicators to confirm they are online. RTSP credentials are retained in the job configuration under the central data directory; protect that directory.

The current detector is a research prototype. It can miss incidents and produce false alarms; review alerts and validate performance on the intended cameras before depending on it. A compatible checkpoint and a reachable camera are needed to test actual live detection.
