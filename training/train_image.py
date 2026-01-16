#1. imports
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torchvision.datasets import FakeData

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support

#2. Load Labels (Reuse Phase 2 CSV)
DATA_PATH = "data/processed/multimodal_dataset.csv"

df = pd.read_csv(DATA_PATH)
labels = df["label"].values


#3. Image Transforms (VERY IMPORTANT)
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

#4. Image Dataset (Synthetic but Correct)
image_dataset = FakeData(
    size=len(labels),
    image_size=(3, 224, 224),
    num_classes=2,
    transform=image_transform
)

#5. Custom Dataset Wrapper (IMPORTANT)
class ImageOnlyDataset(Dataset):
    def __init__(self, image_dataset, labels):
        self.image_dataset = image_dataset
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image, _ = self.image_dataset[idx]
        label = torch.tensor(self.labels[idx], dtype=torch.float)
        return image, label

#6. Train / Validation Split
indices = list(range(len(labels)))

train_idx, val_idx = train_test_split(
    indices,
    test_size=0.2,
    random_state=42,
    stratify=labels
)

train_dataset = torch.utils.data.Subset(
    ImageOnlyDataset(image_dataset, labels),
    train_idx
)

val_dataset = torch.utils.data.Subset(
    ImageOnlyDataset(image_dataset, labels),
    val_idx
)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)


#7. Model — ResNet-50 Classifier
class ResNetImageClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.resnet50(pretrained=True)

        for param in self.backbone.parameters():
            param.requires_grad = False

        self.backbone.fc = nn.Linear(2048, 1)

    def forward(self, x):
        return self.backbone(x).squeeze(-1)


#8. Training setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ResNetImageClassifier().to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.backbone.fc.parameters(), lr=1e-3)

#9. Training Loop
def train_epoch(model, loader):
    model.train()
    total_loss = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


#10. Evaluation
def evaluate(model, loader):
    model.eval()
    preds, true = [], []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            probs = torch.sigmoid(logits)

            preds.extend((probs > 0.5).cpu().numpy())
            true.extend(labels.cpu().numpy())

    p, r, f, _ = precision_recall_fscore_support(
        true, preds, average="binary", zero_division=0
    )
    return p, r, f

#11. Training Baseline Run
for epoch in range(2):
    loss = train_epoch(model, train_loader)
    p, r, f = evaluate(model, val_loader)

    print(f"Epoch {epoch+1}")
    print(f"Loss: {loss:.4f}")
    print(f"Precision: {p:.3f}")
    print(f"Recall: {r:.3f}")
    print(f"F1-score: {f:.3f}")
    print("-" * 30)

