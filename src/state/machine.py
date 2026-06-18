class SpatialEventStateMachine:
    def __init__(self, threshold_low: float = 90.0, threshold_high: float = 160.0):
        self.threshold_low = threshold_low
        self.threshold_high = threshold_high
        self.current_state = "EXTENSION"  # EXTENSION vs FLEXION
        self.event_count = 0

    def process_angle(self, current_angle: float) -> dict:
        """Evaluates thresholds to determine structural state mutations."""
        event_triggered = False
        
        if self.current_state == "EXTENSION" and current_angle < self.threshold_low:
            self.current_state = "FLEXION"
        elif self.current_state == "FLEXION" and current_angle > self.threshold_high:
            self.current_state = "EXTENSION"
            self.event_count += 1
            event_triggered = True
            
        return {
            "state": self.current_state,
            "count": self.event_count,
            "event_triggered": event_triggered
        }
