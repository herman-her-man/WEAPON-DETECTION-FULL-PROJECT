import os
import cv2
import numpy as np
from ultralytics import YOLO

# Paths
dataset_dir = r"C:\Users\cvr\Desktop\weapon-detection\weapon_detection"
train_images_dir = os.path.join(dataset_dir, "train", "images")
train_labels_dir = os.path.join(dataset_dir, "train", "labels")
val_images_dir = os.path.join(dataset_dir, "val", "images")
val_labels_dir = os.path.join(dataset_dir, "val", "labels")
data_yaml_path = os.path.join(dataset_dir, "data.yaml")

# Define classes and their names
label_queries = ["pistol", "knife"]  # Classes to detect

def validate_image(image_path):
    """Validate if the image can be read properly."""
    try:
        img = cv2.imread(image_path)
        if img is None or img.size == 0:
            print(f"WARNING: Corrupt image skipped: {image_path}")
            return False
        return True
    except Exception as e:
        print(f"ERROR: Unable to read image {image_path}. Exception: {e}")
        return False

def is_label_empty_or_invalid(label_path):
    """Check if the label file is empty or contains invalid entries."""
    try:
        with open(label_path, 'r') as f:
            lines = f.readlines()
            if not lines:
                return True
            for line in lines:
                parts = line.strip().split()
                if len(parts) != 5:
                    print(f"WARNING: Malformed label skipped: {label_path}")
                    return True
        return False
    except Exception as e:
        print(f"ERROR: Unable to read label file {label_path}. Exception: {e}")
        return True

def create_data_yaml(data_yaml_path, train_images_dir, val_images_dir, label_queries):
    """Create a YOLOv8-compatible data.yaml file."""
    try:
        with open(data_yaml_path, 'w') as f:
            f.write(f"train: {train_images_dir}\n")
            f.write(f"val: {val_images_dir}\n")
            f.write(f"nc: {len(label_queries)}\n")
            f.write(f"names: {label_queries}\n")
        print(f"data.yaml created at {data_yaml_path}.")
    except Exception as e:
        print(f"ERROR: Failed to create data.yaml. Exception: {e}")

def train_yolo_model(data_yaml_path, epochs=30):
    """Train YOLOv8 model."""
    print("Starting YOLOv8 training...")
    try:
        model = YOLO('yolov8n.pt')
        model.train(
            data=data_yaml_path,
            epochs=epochs,
            imgsz=640,
            batch=16,
            workers=4,
            name="weapon-detection-model",
            pretrained=True,
            device="cpu"  # Force CPU training to avoid CUDA issues
        )
        print("Training complete.")
    except Exception as e:
        print(f"ERROR: Training failed. Exception: {e}")

if __name__ == "__main__":
    try:
        # Validate images and labels for training
        print("Validating training images and labels...")
        valid_images = [f for f in os.listdir(train_images_dir) if validate_image(os.path.join(train_images_dir, f))]
        valid_labels = [f for f in os.listdir(train_labels_dir) if not is_label_empty_or_invalid(os.path.join(train_labels_dir, f))]

        print(f"Valid images: {len(valid_images)}, Valid labels: {len(valid_labels)}")

        # Validate images and labels for validation
        print("Validating validation images and labels...")
        valid_images = [f for f in os.listdir(val_images_dir) if validate_image(os.path.join(val_images_dir, f))]
        valid_labels = [f for f in os.listdir(val_labels_dir) if not is_label_empty_or_invalid(os.path.join(val_labels_dir, f))]

        print(f"Valid images: {len(valid_images)}, Valid labels: {len(valid_labels)}")

        # Create data.yaml
        create_data_yaml(data_yaml_path, train_images_dir, val_images_dir, label_queries)

        # Train YOLOv8
        train_yolo_model(data_yaml_path, epochs=30)

    except Exception as e:
        print(f"ERROR: An unexpected error occurred. Exception: {e}")
