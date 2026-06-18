import numpy as np

class KinematicTelemetryEngine:
    def __init__(self, window_size: int = 15):
        self.window_size = window_size
        self.history = []

    def calculate_joint_angle(self, p_start: np.ndarray, p_mid: np.ndarray, p_end: np.ndarray) -> float:
        """Computes the interior angle (in degrees) formed by three 2D/3D keypoints."""
        if p_start is None or p_mid is None or p_end is None:
            return 0.0
            
        u = p_start - p_mid
        v = p_end - p_mid
        
        cosine_angle = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-6)
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        
        return float(np.degrees(angle))

    def update_history(self, metrics: dict):
        """Appends metrics to a rolling temporal window."""
        self.history.append(metrics)
        if len(self.history) > self.window_size:
            self.history.pop(0)

    def compute_velocity(self, keypoint_id: int) -> float:
        """Calculates frame-over-frame pixel velocity of a tracking keypoint."""
        if len(self.history) < 2:
            return 0.0
        pos_t1 = self.history[-1].get(keypoint_id)
        pos_t2 = self.history[-2].get(keypoint_id)
        
        if pos_t1 is None or pos_t2 is None:
            return 0.0
            
        return float(np.linalg.norm(pos_t1 - pos_t2))
