import cv2

class DecoupledVideoStream:
    def __init__(self, source=0, resolution=(640, 480)):
        self.stream = cv2.VideoCapture(source)
        if isinstance(source, int):
            self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
            self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
        
    def start(self):
        # Kept for interface compatibility with our orchestrator
        return self

    def read(self):
        grabbed, frame = self.stream.read()
        return frame if grabbed else None

    def stop(self):
        self.stream.release()
