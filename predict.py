from ultralytics import YOLO
import os
import torch
import cv2

def predict():
   
    model_path = 'runs/weights/best.pt'  
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found!")
        return
    print(f"Model file '{model_path}' found. Loading model...")

    model = YOLO(model_path)

    image_path = 'imgs.jpg'  
    if not os.path.exists(image_path):
        print(f"Error: Image file '{image_path}' not found!")
        return
    print(f"Image file '{image_path}' found. Predicting...")

    results = model(image_path, conf=0.25)  

    image = cv2.imread(image_path)

    keypoint_colors = [
        (0, 0, 255),  
        (0, 255, 0),  
        (255, 0, 0),   
        (255, 255, 0), 
        (128, 0, 128),
       
    ]

    for result in results:
      
        xy = result.keypoints.xy  
        xyn = result.keypoints.xyn  
        kpts = result.keypoints.data 

        print("Keypoints (xy):", xy)
        print("Keypoints (xyn):", xyn)
        print("Keypoints (data):", kpts)

     
        for target_kpts in xy: 
            for i, kpt in enumerate(target_kpts):  
                x, y = int(kpt[0]), int(kpt[1]) 
                color = keypoint_colors[i % len(keypoint_colors)] 
                cv2.circle(image, (x, y), 3, color, -1) 

 
    output_path = 'outputs/_manual.jpg' 
    cv2.imwrite(output_path, image)
    print(f"Manual prediction saved to '{output_path}'")