from ultralytics import YOLO

DEFAULT_WEIGHTS = "runs/detect/defender_yolo/weights/best.pt"


class YoloDetector:
    # CPU so parallel env workers don't queue on one GPU
    def __init__(self, weights_path=DEFAULT_WEIGHTS, conf_threshold=0.25, device="cpu"):
        self.model = YOLO(weights_path)
        self.conf_threshold = conf_threshold
        self.device = device

    def __call__(self, rgb):
        boxes = self.model.predict(
            rgb, conf=self.conf_threshold, device=self.device, verbose=False
        )[0].boxes
        return [
            (*boxes.xyxy[i].tolist(), float(boxes.conf[i]), int(boxes.cls[i]))
            for i in range(len(boxes))
        ]
