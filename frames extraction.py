!pip install transformers datasets accelerate

# Step 2: Import Dependencies
import os
import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import (
    VideoMAEImageProcessor,
    VideoMAEForVideoClassification,
    TrainingArguments,
    Trainer,
    VideoMAEModel
)
import torch.nn as nn

# Step 3: Define Labels
LABELS = [
    "Abuse", "Arrest", "Arson", "Assault", "Burglary", "Explosion",
    "Fighting", "Normal", "Road Accidents", "Robbery",
    "Shooting", "Shoplifting", "Stealing", "Vandalism"
]
label2id = {label: i for i, label in enumerate(LABELS)}
id2label = {i: label for label, i in label2id.items()}

# Step 4: Frame Extraction Function
def extract_16_frames(video_path, output_folder, desired_frames=16, resize=(224, 224)):
    os.makedirs(output_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        print(f"⚠️ Skipping: {video_path} (0 frames)")
        return False

    frame_indices = []
    if total_frames < desired_frames:
        frame_indices = list(range(total_frames)) + [total_frames - 1] * (desired_frames - total_frames)
    else:
        frame_indices = list(map(int, np.linspace(0, total_frames - 1, desired_frames)))

    saved = 0
    for idx in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        if idx == frame_indices[saved]:
            if resize:
                frame = cv2.resize(frame, resize)
            save_path = os.path.join(output_folder, f"frame_{saved:04d}.jpg")
            cv2.imwrite(save_path, frame)
            saved += 1
            if saved == desired_frames:
                break
    cap.release()
    return True

# Step 5: Process UCF-Crime Dataset
def process_ucf_crime_videos(source_root, output_root):
    for category in os.listdir(source_root):
        category_path = os.path.join(source_root, category)
        if not os.path.isdir(category_path):
            continue
        print(f"\n📂 Category: {category}")
        for video_file in tqdm(os.listdir(category_path), desc=f"Processing {category}"):
            if not video_file.lower().endswith(('.mp4', '.avi')):
                continue
            video_path = os.path.join(category_path, video_file)
            output_folder = os.path.join(output_root, category, os.path.splitext(video_file)[0])
            extract_16_frames(video_path, output_folder)