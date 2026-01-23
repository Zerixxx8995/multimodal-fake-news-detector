
# 1. Imports

import os
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import pandas as pd
from torchvision import models, transforms
from transformers import DistilBertTokenizerFast, DistilBertModel

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support



# 2. Paths

DATA_PATH = "data/processed/real_multimodal_dataset.csv"
IMAGE_DIR = "data/raw/flickr8k/images"



# 3. Load Dataset

df = pd.read_csv(DATA_PATH)

texts = df["headline"].tolist()
images = df["image"].tolist()
labels = df["label"].values



# 4. Tokenizer & Transforms

tokenizer = DistilBertTokenizerFast.from_pretrained(
    "distilbert-base-uncased"
)

TEXT_MAX_LEN = 32

image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])



# 5. Multimodal Dataset

class MultimodalDataset(Dataset):
    def __init__(self, texts, images, labels, tokenizer, max_len):
        self.texts = texts
        self.images = images
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Load real image
        image_path = os.path.join(IMAGE_DIR, self.images[idx])
        image = Image.open(image_path).convert("RGB")
        image = image_transform(image)

        # Tokenize text
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )

        return {
            "image": image,
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.float)
        }



# 6. Train / Validation Split

train_idx, val_idx = train_test_split(
    range(len(labels)),
    test_size=0.2,
    random_state=42,
    stratify=labels
)

train_dataset = MultimodalDataset(
    [texts[i] for i in train_idx],
    [images[i] for i in train_idx],
    labels[train_idx],
    tokenizer,
    TEXT_MAX_LEN
)

val_dataset = MultimodalDataset(
    [texts[i] for i in val_idx],
    [images[i] for i in val_idx],
    labels[val_idx],
    tokenizer,
    TEXT_MAX_LEN
)

train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=16,
    num_workers=0
)



# 7. Multimodal Fusion Model

class MultimodalFusionModel(nn.Module):
    def __init__(self):
        super().__init__()

        # Text encoder
        self.text_encoder = DistilBertModel.from_pretrained(
            "distilbert-base-uncased"
        )

        # Image encoder
        self.image_encoder = models.resnet50(pretrained=True)
        for name, param in self.image_encoder.named_parameters():
          if "layer4" in name:
            param.requires_grad = True
          else:
            param.requires_grad = False

        self.image_encoder.fc = nn.Identity()

        # Fusion head
        self.fusion = nn.Sequential(
            nn.Linear(768 + 2048, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 1)
        )

    def forward(self, input_ids, attention_mask, images):
        text_features = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        ).last_hidden_state[:, 0]

        image_features = self.image_encoder(images)

        fused = torch.cat([text_features, image_features], dim=1)
        logits = self.fusion(fused)

        return logits.squeeze(-1)



# 8. Training Setup

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

model = MultimodalFusionModel().to(device)

criterion = nn.BCEWithLogitsLoss()

optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-4
)




# 9. Training Loop

def train_epoch(model, loader):
    model.train()
    total_loss = 0

    for batch in loader:
        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        logits = model(input_ids, attention_mask, images)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)



# 10. Evaluation

def evaluate(model, loader):
    model.eval()
    preds, true = [], []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            logits = model(input_ids, attention_mask, images)
            probs = torch.sigmoid(logits)

            preds.extend((probs > 0.5).cpu().numpy())
            true.extend(labels.cpu().numpy())

    p, r, f, _ = precision_recall_fscore_support(
        true, preds, average="binary", zero_division=0
    )
    return p, r, f



# 11. Train Multimodal Model

EPOCHS = 5

for epoch in range(EPOCHS):
    loss = train_epoch(model, train_loader)
    p, r, f = evaluate(model, val_loader)

    print(f"Epoch {epoch + 1}")
    print(f"Loss: {loss:.4f}")
    print(f"Precision: {p:.3f}")
    print(f"Recall: {r:.3f}")
    print(f"F1-score: {f:.3f}")
    print("-" * 30)