import cv2
import yaml
import os
import numpy as np
os.environ["YOLO_VERBOSE"] = "False"
from ultralytics import YOLO
from src.vision.stream import DecoupledVideoStream
from src.kinematics.telemetry import KinematicTelemetryEngine
from src.state.machine import SpatialEventStateMachine

def main():
    with open("configs/engine_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    stream = DecoupledVideoStream(source=config["camera"]["source"], 
                                  resolution=tuple(config["camera"]["resolution"])).start()
    model = YOLO(config["model"]["weights"])
    telemetry = KinematicTelemetryEngine(window_size=config["kinematics"]["temporal_window_size"])
    sm = SpatialEventStateMachine()

    window_name = "Spatial Telemetry Active"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    # COCO Keypoint Map: 5=L_Shoulder, 7=L_Elbow, 9=L_Wrist
    L_SHOULDER, L_ELBOW, L_WRIST = 5, 7, 9

    while True:
        frame = stream.read()
        if frame is None: continue

        results = model(frame, verbose=False, device=config["model"]["device"])[0]
        
        status_text = "Status: Locating Left Arm..."
        color = (0, 0, 255) # Red

        if results.keypoints is not None and results.keypoints.data.numel() > 0:
            # kpts shape: [1, 17, 3] -> [x, y, confidence]
            kpts = results.keypoints.data[0].cpu().numpy()
            
            # Check confidence for the joints we need
            if kpts[L_SHOULDER][2] > 0.5 and kpts[L_ELBOW][2] > 0.5 and kpts[L_WRIST][2] > 0.5:
                p_shoulder = kpts[L_SHOULDER][:2]
                p_elbow = kpts[L_ELBOW][:2]
                p_wrist = kpts[L_WRIST][:2]
                
                angle = telemetry.calculate_joint_angle(p_shoulder, p_elbow, p_wrist)
                telemetry.update_history({L_ELBOW: p_elbow})
                metrics = sm.process_angle(angle)
                
                status_text = f"Angle: {angle:.1f} Deg | State: {metrics['state']} | Events: {metrics['count']}"
                color = (0, 255, 0) # Green

        # Draw status
        cv2.putText(frame, status_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        cv2.imshow(window_name, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    stream.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
