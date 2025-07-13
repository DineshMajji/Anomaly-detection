!pip install gradio numpy torch transformers

import gradio as gr
import numpy as np
import torch
from transformers import VideoMAEImageProcessor
from torch import nn
class VideoAnomalyDetector(nn.Module):
    def __init__(self, num_classes=14):
        super().__init__()
        self.backbone = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base")
        self.temporal_attn = nn.MultiheadAttention(embed_dim=768, num_heads=4)
        self.classifier = nn.Linear(768, num_classes)
    
    def forward(self, inputs):
        features = self.backbone(**inputs).last_hidden_state
        attn_out, _ = self.temporal_attn(features, features, features)
        return self.classifier(attn_out.mean(dim=1))
model = VideoAnomalyDetector()
model.load_state_dict(torch.load('videomae-ucfcrime-temporal.h5', map_location='cpu'))  
model.eval()
processor = VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-base")


def predict(video):
   
    cap = cv2.VideoCapture(video)
    frames = []
    for _ in range(16):
        ret, frame = cap.read()
        if not ret: break
        frames.append(Image.fromarray(frame[..., ::-1])) 
    
  
    inputs = processor(frames, return_tensors="pt")
    
   
    with torch.no_grad():
        logits = model(inputs)
    
    
    probs = torch.softmax(logits, dim=-1).squeeze()
    
    
    return {config.LABELS[i]: float(probs[i]) for i in range(len(config.LABELS))}

iface = gr.Interface(
    fn=process_video,
    inputs=gr.Video(),
    outputs="text",
    title="🚨  Anomaly Detection",
    description="Upload a video for anomaly detection."
)
