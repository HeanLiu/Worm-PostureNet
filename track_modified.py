import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
# Load the YOLO11 model
model = YOLO("runs/weights/best.pt")


video_path = "imgs/darkfield.avi"
result_path = "result1_track_Box.mp4"

track_history = defaultdict(lambda: [])

if __name__ == "__main__":
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        print("Error opening video stream or file")
        exit()
    # print(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    # print(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # print(capture.get(cv2.CAP_PROP_FPS))
    fps = capture.get(cv2.CAP_PROP_FPS)
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    videoWriter = None
    fps_count = 0;
    while True:
        success, frame = capture.read()
        if not success:
            print("fail")
            break
        fps_count = fps_count + 1 
        print("fps_count:", fps_count)
        results = model.track([frame], persist=True, show=True, show_labels=True, boxes=True,line_width=1)

        a_frame = results[0].plot(line_width=1, font_size=10, labels=True, boxes=True)  
       
        boxes = results[0].boxes.xywh.cpu()
        # print("boxes:", boxes)
  
        track_ids = results[0].boxes.id.int().cpu().tolist()
        # print("track_ids: ", track_ids)
        for box, track_id in zip(boxes, track_ids):
            x, y, w, h = box
            # print(x)
            track = track_history[track_id]
            track.append((float(x), float(y)))
            # print("track.append: ", track)
            if len(track) > 250:   
                track.pop(0)
       
            points = np.hstack(track).astype(np.int32).reshape(-1,1,2)
            cv2.putText(a_frame, str(fps_count), org=(420,350),fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.5, color=(0, 255, 0),thickness=1)
            cv2.polylines(a_frame, [points], isClosed=False, color=(0, 255, 0), thickness=1)

        if videoWriter is None:
            fourcc = cv2.VideoWriter.fourcc("m","p","4","v")
            videoWriter = cv2.VideoWriter(result_path,fourcc,fps,(int(frame_width),int(frame_height)))


        # videoWriter.write( a_frame)
        cv2.imshow("yolo_track", a_frame)
        cv2.waitKey(1)

    capture.release()  
    # videoWriter.release()
    cv2.destroyAllWindows() 