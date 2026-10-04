"""
Fine-tunes a YOLO nano model on detector_dataset/ (see
generate_detector_dataset.py) to localize the defender drone. Saves to
runs/detect/defender_yolo/weights/best.pt, which yolo_detector.py loads.

Run:
    python train_detector.py
"""

from ultralytics import YOLO

DATASET_YAML = "detector_dataset/dataset.yaml"
BASE_WEIGHTS = "yolo11n.pt"  # pretrained nano checkpoint, downloaded on first use
IMG_SIZE = 160  # must match DroneSoccerEnv's img_size
EPOCHS = 50

if __name__ == "__main__":
    model = YOLO(BASE_WEIGHTS)
    model.train(
        data=DATASET_YAML,
        imgsz=IMG_SIZE,
        epochs=EPOCHS,
        batch=64,
        device=0,  # GPU: no multi-process contention here, unlike in-loop RL inference
        name="defender_yolo",  # project omitted: ultralytics' own default is runs/detect
        patience=15,  # early stop if val loss stalls
    )
