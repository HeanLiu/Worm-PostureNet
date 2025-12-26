from ultralytics import YOLO
import os
import torch
import cv2

def train():
 
    model = YOLO("yolo11s-pose.pt") 

    print(f"Loss function type: {type(model.loss)}")
    print(f"Keypoint shape: {model.model[-1].kpt_shape}")


    results = model.train(
        data="data.yaml",  
        pretrained=True,  
        epochs=300, 
        imgsz=640,  
        batch=16,  
        lr0=0.01,  
        device=0  
    )

    print("Training results:", results)