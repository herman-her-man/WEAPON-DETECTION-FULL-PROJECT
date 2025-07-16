import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

# Load the trained YOLO model
model_choice = input("Enter model to use (best.pt or last.pt): ").strip()
model_path = model_choice if os.path.exists(model_choice) else f"runs/detect/weapon-detection-debug/weights/{model_choice}"

if not os.path.exists(model_path):
    print(f"Error: Model file '{model_path}' does not exist.")
    exit()

model = YOLO(model_path)
print(f"Using model: {model_path}")

# Function to process a video for weapon detection and classification
def process_video(video_path, output_path="output_video.mp4"):
    print(f"Processing video: {video_path}")
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("End of video stream.")
            break

        # Ensure the frame is a valid NumPy array
        if not isinstance(frame, np.ndarray):
            print("Invalid frame read from video. Skipping...")
            continue

        # Predict on the current frame
        results = model(frame)

        for result in results:
            for box, cls in zip(result.boxes.xyxy, result.boxes.cls):
                x_min, y_min, x_max, y_max = map(int, box)
                weapon_class = model.names[int(cls)]  # Get the class name
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                cv2.putText(frame, f"{weapon_class}", (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Write the processed frame to the output video
        out.write(frame)

    cap.release()
    out.release()
    print(f"Processed video saved at: {output_path}")

# Function to process an image for weapon detection and classification
def process_image(image_path):
    print(f"Processing image: {image_path}")
    results = model.predict(source=image_path, save=True, show=True)
    for result in results:
        for box, cls in zip(result.boxes.xyxy, result.boxes.cls):
            x_min, y_min, x_max, y_max = map(int, box)
            weapon_class = model.names[int(cls)]
            print(f"Detected {weapon_class} at [{x_min}, {y_min}, {x_max}, {y_max}]")

# Main script
if __name__ == "__main__":
    mode = input("Select mode (image/video/live): ").strip().lower()

    if mode == "image":
        image_path = input("Enter the path to the image: ").strip()
        process_image(image_path)

    elif mode == "video":
        video_path = input("Enter the path to the video: ").strip()
        output_path = input("Enter the path to save the output video (default: output_video.mp4): ").strip()
        if not output_path:
            output_path = "output_video.mp4"
        process_video(video_path, output_path)

    elif mode == "live":
        print("Live detection mode not currently supported for classification.")
    else:
        print("Invalid mode selected. Please choose 'image', 'video', or 'live'.")
