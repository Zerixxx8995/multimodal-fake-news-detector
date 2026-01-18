#1. imports
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torchvision.datasets import FakeData

import pandas as pd
from transformers import DistilBertTokenizerFast, DistilBertModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support


#2. Load Dataset (Same CSV as Phase 3 & 4)
DATA_PATH = "data/processed/multimodal_dataset.csv"

df = pd.read_csv(DATA_PATH)
labels = df["label"].values
texts = df["headline"].tolist()

#3. Tokenizer & Image Transforms
tokenizer = DistilBertTokenizerFast.from_pretrained(
    "distilbert-base-uncased"
)

text_max_len = 16

image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


#4. Image Dataset (Synthetic, Same as Phase 4)

image_dataset = FakeData(
    size=len(labels),
    image_size=(3, 224, 224),
    num_classes=2,
    transform=image_transform
)


#5. Multimodal Dataset (IMPORTANT)
class MultimodalDataset(Dataset):
    def __init__(self, texts, labels, image_dataset, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.image_dataset = image_dataset
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image, _ = self.image_dataset[idx]

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

#6. Train / Validation Split
indices = list(range(len(labels)))

train_idx, val_idx = train_test_split(
    indices,
    test_size=0.2,
    random_state=42,
    stratify=labels
)

train_dataset = MultimodalDataset(
    [texts[i] for i in train_idx],
    labels[train_idx],
    torch.utils.data.Subset(image_dataset, train_idx),
    tokenizer,
    text_max_len
)

val_dataset = MultimodalDataset(
    [texts[i] for i in val_idx],
    labels[val_idx],
    torch.utils.data.Subset(image_dataset, val_idx),
    tokenizer,
    text_max_len
)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)


#7.Multimodal Fusion Model (CORE)
class MultimodalFusionModel(nn.Module):
    def __init__(self):
        super().__init__()

        # Text encoder
        self.text_encoder = DistilBertModel.from_pretrained(
            "distilbert-base-uncased"
        )

        # Image encoder
        self.image_encoder = models.resnet50(pretrained=True)
        for param in self.image_encoder.parameters():
            param.requires_grad = False
        self.image_encoder.fc = nn.Identity()

        # Fusion head
        self.fusion = nn.Sequential(
            nn.Linear(768 + 2048, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 1)
        )

    def forward(self, input_ids, attention_mask, image):
        text_out = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        ).last_hidden_state[:, 0]

        image_out = self.image_encoder(image)

        fused = torch.cat([text_out, image_out], dim=1)
        logits = self.fusion(fused)

        return logits.squeeze(-1)


#8. Training Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = MultimodalFusionModel().to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(
    model.fusion.parameters(), lr=2e-4
)

#9. Training Loop
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


#10.Evaluation
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


#11. Train Multimodal Model
for epoch in range(2):
    loss = train_epoch(model, train_loader)
    p, r, f = evaluate(model, val_loader)

    print(f"Epoch {epoch+1}")
    print(f"Loss: {loss:.4f}")
    print(f"Precision: {p:.3f}")
    print(f"Recall: {r:.3f}")
    print(f"F1-score: {f:.3f}")
    print("-" * 30)

