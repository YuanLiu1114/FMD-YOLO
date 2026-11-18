from ultralytics import YOLO
import torch
import os

# 加载 YOLOv8-L 模型
model = YOLO("./ultralytics/cfg/models/v8/yolov8l-decoupledhead_with_C2fFM.yaml")

model.train(
    data = "/path/to/full_training_dataset.yaml",
    lr0 = 0.01,
    device = 0,
    epochs = 300,
    patience = 100,
    batch = 16,
    workers = 8,
    imgsz = 1280,
    rect = False,
    verbose = True,
    optimizer = "SGD",

    hsv_h = 0.015,
    hsv_s = 0.7,
    hsv_v = 0.4,
    mosaic = 1.0,
    scale = 0.5,
    seed = 0
)
