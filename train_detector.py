from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("yolo11n.pt")
    model.train(
        data="detector_dataset/dataset.yaml",
        imgsz=160,
        epochs=50,
        batch=64,
        device=0,
        name="defender_yolo",
        patience=15,
    )
