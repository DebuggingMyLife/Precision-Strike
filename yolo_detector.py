"""
Wraps a trained YOLO nano model to match drone_soccer_env.py's detector
interface: detector(rgb) -> list of (x1, y1, x2, y2, conf, cls).
"""

from ultralytics import YOLO

DEFAULT_WEIGHTS = "runs/detect/defender_yolo/weights/best.pt"


class YoloDetector:
    # device="cpu": with N_ENVS parallel SubprocVecEnv workers each loading
    # their own YoloDetector, letting ultralytics auto-select CUDA means all
    # of them queue for the one GPU instead of doing independent work — CPU
    # keeps each worker's inference genuinely parallel across cores.
    def __init__(self, weights_path=DEFAULT_WEIGHTS, conf_threshold=0.25, device="cpu"):
        self.model = YOLO(weights_path)
        self.conf_threshold = conf_threshold
        self.device = device

    def __call__(self, rgb):
        result = self.model.predict(
            rgb, conf=self.conf_threshold, device=self.device, verbose=False
        )[0]
        boxes = result.boxes
        return [
            (*boxes.xyxy[i].tolist(), float(boxes.conf[i]), int(boxes.cls[i]))
            for i in range(len(boxes))
        ]
