from ultralytics import YOLO
import os
import torch
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.interpolate import interp1d


def predict_video_with_datawrite_wh():

    model_path = 'runs/weights/best.pt'
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found!")
        return
    print(f"Model file '{model_path}' found. Loading model...")

    model = YOLO(model_path)

 
    video_path = 'imgs/darkfield.avi'
    if not os.path.exists(video_path):
        print(f"Error: Video file '{video_path}' not found!")
        return
    print(f"Video file '{video_path}' found. Predicting...")

     cap = cv2.VideoCapture(video_path)

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    output_path = 'outputs/output_video_tracked.avi'
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    data_file = open('outputs/worm_tracking_data_wh.txt', 'w')
  
    data_file.write("frame\tworm_id\tx1\ty1\tx2\ty2\tx3\ty3\tx4\ty4\tx5\ty5\tw\th\n")

    keypoint_colors = [
        (0, 0, 255),  
        (0, 255, 0),  
        (255, 0, 0),  
        (255, 255, 0),  
        (128, 0, 128),  
    ]


    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

  
        results = model.track(frame, conf=0.25, persist=True)


        for result in results:
       
            track_ids = result.boxes.id.int().tolist() if result.boxes.id is not None else []

         
            boxes = result.boxes.xywh 

           
            xy = result.keypoints.xy  

            for i, (target_kpts, track_id, box) in enumerate(zip(xy, track_ids, boxes)):
               
                if len(target_kpts) != 5:
                    continue

                
                w, h = int(box[2]), int(box[3])

              
                kpt_coords = []
                for kpt in target_kpts:
                    x, y = int(kpt[0]), int(kpt[1])
                    kpt_coords.extend([x, y])

               
                data_line = f"{frame_count}\t{track_id}"
                for coord in kpt_coords:
                    data_line += f"\t{coord}"
                data_line += f"\t{w}\t{h}\n"  
                data_file.write(data_line)

             
                x1, y1 = int(target_kpts[0][0]), int(target_kpts[0][1])
                cv2.putText(frame, f"ID:{track_id}", (x1 - 10, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

              
                for kpt_idx, kpt in enumerate(target_kpts):
                    x, y = int(kpt[0]), int(kpt[1])
                    color = keypoint_colors[kpt_idx % len(keypoint_colors)]
                    cv2.circle(frame, (x, y), 3, color, -1)

  
        cv2.imshow('Video Prediction with Tracking', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    cap.release()
    out.release()
    data_file.close()
    cv2.destroyAllWindows()
    print(f"Video with tracking saved to '{output_path}'")
    print(f"Tracking data saved to 'outputs/worm_tracking_data_wh.txt'")

if __name__ == '__main__':
    predict_video_with_datawrite_wh()