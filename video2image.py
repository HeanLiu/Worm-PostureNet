import os
import cv2


def video_to_frames(video_path, output_dir, save_interval=1):

    os.makedirs(output_dir, exist_ok=True)


    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"error: {video_path}")
        return
  
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
  
        if frame_count % save_interval == 0:
           
            frame_filename = os.path.join(output_dir, f"frame_{frame_count:06d}.jpg")
            cv2.imwrite(frame_filename, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            saved_count += 1
      
        if frame_count % 100 == 0:
          

    cap.release()

if __name__ == "__main__":
  
    video_path = "outputs/output_video_tracked-with-wh.avi" 
    output_dir = "outputs/sort"  
    save_interval = 1  

   
    video_to_frames(video_path, output_dir, save_interval)