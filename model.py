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
class VideoMAECustomDataset(Dataset):
    def __init__(self, root_dir, label2id, processor, samples=None, num_frames=16):
        self.root_dir = root_dir
        self.label2id = label2id
        self.processor = processor
        self.num_frames = num_frames
        self.samples = samples if samples else self._load_samples()

    def _load_samples(self):
        samples = []
        for label in os.listdir(self.root_dir):
            label_path = os.path.join(self.root_dir, label)
            if not os.path.isdir(label_path): continue
            for vid in os.listdir(label_path):
                vid_path = os.path.join(label_path, vid)
                if os.path.isdir(vid_path):
                    samples.append((vid_path, self.label2id[label]))
        return samples

    def _load_frames(self, folder):
        frames = sorted([os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".jpg")])
        total = len(frames)
        if total < self.num_frames:
            frames += [frames[-1]] * (self.num_frames - total)
        indices = torch.linspace(0, len(frames) - 1, steps=self.num_frames).long()
        return [Image.open(frames[i]).convert("RGB") for i in indices]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, label = self.samples[idx]
        frames = self._load_frames(video_path)
        inputs = self.processor(frames, return_tensors="pt")
        return {
            "pixel_values": inputs["pixel_values"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long)
        }

class VideoMAEWithTemporalAttention(nn.Module):
    def __init__(self, base_model_name, num_labels):
        super().__init__()
        self.videomae = VideoMAEModel.from_pretrained(base_model_name)
        self.temporal_attn = nn.MultiheadAttention(embed_dim=768, num_heads=4, batch_first=True)
        self.classifier = nn.Linear(768, num_labels)

    def forward(self, pixel_values, labels=None):
        outputs = self.videomae(pixel_values=pixel_values)
        sequence = outputs.last_hidden_state 
        attn_out, _ = self.temporal_attn(sequence, sequence, sequence)  
        pooled = attn_out.mean(dim=1) 
        logits = self.classifier(pooled)

        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)

        return {"loss": loss, "logits": logits}

processor = VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-base")
model = VideoMAEWithTemporalAttention("MCG-NJU/videomae-base", num_labels=len(LABELS))

root_dir = "/content/processed_frames" 
full_dataset = VideoMAECustomDataset(root_dir, label2id, processor)
train_samples, val_samples = train_test_split(full_dataset.samples, test_size=0.2, random_state=42)
train_dataset = VideoMAECustomDataset(root_dir, label2id, processor, samples=train_samples)
val_dataset = VideoMAECustomDataset(root_dir, label2id, processor, samples=val_samples)


training_args = TrainingArguments(
    output_dir="./videomae-ucfcrime",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    logging_dir="./logs",
    learning_rate=5e-5,
    remove_unused_columns=False,
    logging_steps=5,
    report_to="none",
    fp16=True,
    save_total_limit=2
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

trainer.train()


model_path = "videomae-ucfcrime-temporal.h5"
os.makedirs(model_path, exist_ok=True)
torch.save(model.state_dict(), os.path.join(model_path, "pytorch_model.bin"))
processor.save_pretrained(model_path)

print("\n✅ Training complete. Model saved with temporal attention.")
