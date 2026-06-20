# 🦾 Spatial Event Detector — Active Blueprint

Real-time arm curl repetition counter using YOLO11-Pose and a flexion/extension state machine. Tracks the left arm (shoulder → elbow → wrist) via webcam and counts completed movement cycles on-device.

## Architecture

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Capture** | OpenCV (V4L2 / RTSP) | Webcam or IP camera input |
| **Pose** | YOLO11-Pose (nano) | Left arm keypoints: shoulder, elbow, wrist |
| **Kinematics** | Rolling window + angle calc | Elbow interior angle from 3 keypoints |
| **State** | EXTENSION/FLEXION FSM | Detect completed curl cycles |

## Quick Start

```bash
git clone https://github.com/aragit/spatial-event-detector.git
cd spatial-event-detector
python3 -m venv venv
source venv/bin/activate
pip install ultralytics opencv-python numpy pyyaml
```
Model weights (yolo11n-pose.pt) download automatically on first run.
```bash
python main.py
```

A window opens showing your webcam feed. Perform left arm curls — the display shows elbow angle and repetition count.

```text
Angle: 45.2 Deg | State: FLEXION | Events: 5
```
- EXTENSION → arm straightens (angle > 160°)
- FLEXION → arm bends (angle < 90°)
- Each completed EXTENSION → FLEXION → EXTENSION cycle increments Events
  
**Configuration**
Edit configs/engine_config.yaml:
```text
camera:
  source: 0                         # 0=webcam, or "rtsp://..."
  resolution: [640, 480]
  fps_target: 30

model:
  weights: "yolo11n-pose.pt"        # Nano model for edge
  confidence_threshold: 0.5
  device: "cpu"                     # "cuda" or "mps" if available

kinematics:
  temporal_window_size: 15          # Frames for velocity tracking
  angle_smoothing_alpha: 0.3        # EMA smoothing factor
```
**How It Works**
1. Each frame runs through YOLO11-Pose to extract 17 COCO keypoints
2. The left arm chain (shoulder-5 → elbow-7 → wrist-9) is isolated
3. KinematicTelemetryEngine.calculate_joint_angle() computes the interior elbow angle
4. SpatialEventStateMachine transitions between EXTENSION and FLEXION at 90°/160° thresholds
5. Each full cycle increments the event counter; display updates in real time

**Use Cases**

- Rehabilitation — Track range-of-motion progress for post-stroke or post-surgery patients
- Fitness — Count curls / triceps extensions with no wearable
- Ergonomics — Monitor repetitive strain patterns

**Repository Structure**
```text
spatial-event-detector/
├── main.py                          # Entry point + display loop
├── configs/
│   └── engine_config.yaml           # Camera, model, kinematics settings
├── src/
│   ├── vision/
│   │   └── stream.py                # DecoupledVideoStream wrapper
│   ├── kinematics/
│   │   └── telemetry.py             # Joint angle calculation + velocity
│   └── state/
│       └── machine.py               # EXTENSION/FLEXION state machine
└── .gitignore
```

**Roadmap**
 - [] Right arm tracking (mirror keypoint indices)
 - []Bilateral tracking (both arms simultaneously)
 - []Configurable threshold angles
 - []Rep log export (CSV / JSON)
 - []TensorRT backend for Jetson deployment

  *Built for rehabilitation monitoring. Designed for edge deployment.*


