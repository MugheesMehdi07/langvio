import os
import cv2
from ultralytics.solutions import ObjectCounter

# --- CONFIGURATION ---
MODEL_PATH = "yolo11n.pt"  # Make sure the model file exists
CONFIDENCE = 0.3
VIDEO_PATH = "data/traffic_video.mp4"  # Replace with your actual path

def count_objects_in_region(video_path, model_path):
    """Count objects in a specific region within a video."""
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), "Error reading video file"
    w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

    region_points = [(0, 0), (w, 0), (w, h), (0, h)]
    counter = ObjectCounter(show=True, region=region_points, model=model_path)

    while cap.isOpened():
        success, im0 = cap.read()
        if not success:
            print("Video frame is empty or processing is complete.")
            break
        results = counter(im0)
        # video_writer.write(results.plot_im)

    cap.release()
    # video_writer.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    count_objects_in_region(VIDEO_PATH,MODEL_PATH)
