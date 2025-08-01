from ultralytics import YOLO
import os

if __name__ == '__main__':
    directory_path = "/path/to/train/script"
    os.chdir(directory_path)  # switch to the directory

    print(f"✅ Current Dictionary Path: {os.getcwd()}")

    model = YOLO("yolo11n.yaml")

    # Load a model
    model = YOLO("path/to/best.pt")  # load a custom model

    # Validate the model
    metrics = model.val()  # no arguments needed, dataset and settings remembered
    metrics.box.map  # map50-95
    metrics.box.map50  # map50
    metrics.box.map75  # map75
    metrics.box.maps  # a list contains map50-95 of each category


