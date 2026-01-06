"""
Phase 3: Text-only baseline using DistilBERT

This script trains a text-only classifier to predict
image–text consistency using headline text alone.

Expected behavior on synthetic data:
- Near-random performance
- Possible collapse to majority class
"""

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizerFast, DistilBertModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support


DATA_PATH = "data/processed/multimodal_dataset.csv"
MAX_LEN = 16
BATCH_SIZE = 16
EPOCHS = 2
LR = 2e-5


class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.float),
        }


class DistilBERTClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained(
            "distilbert-base-uncased"
        )
        self.classifier = nn.Linear(768, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        cls_output = outputs.last_hidden_state[:, 0]
        logits = self.classifier(cls_output)
        return logits.squeeze(-1)


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    for batch in loader:
        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate(model, loader, device):
    model.eval()
    preds, true = [], []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            logits = model(input_ids, attention_mask)
            probs = torch.sigmoid(logits)

            preds.extend((probs > 0.5).cpu().numpy())
            true.extend(labels.cpu().numpy())

    p, r, f, _ = precision_recall_fscore_support(
        true, preds, average="binary", zero_division=0
    )
    return p, r, f


def main():
    df = pd.read_csv(DATA_PATH)
    text_df = df[["headline", "label"]]

    train_df, val_df = train_test_split(
        text_df,
        test_size=0.2,
        random_state=42,
        stratify=text_df["label"],
    )

    tokenizer = DistilBertTokenizerFast.from_pretrained(
        "distilbert-base-uncased"
    )

    train_dataset = TextDataset(
        train_df["headline"].tolist(),
        train_df["label"].tolist(),
        tokenizer,
        MAX_LEN,
    )

    val_dataset = TextDataset(
        val_df["headline"].tolist(),
        val_df["label"].tolist(),
        tokenizer,
        MAX_LEN,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = DistilBERTClassifier().to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        loss = train_epoch(
            model, train_loader, optimizer, criterion, device
        )
        p, r, f = evaluate(model, val_loader, device)

        print(f"Epoch {epoch+1}")
        print(f"Loss: {loss:.4f}")
        print(f"Precision: {p:.3f}")
        print(f"Recall: {r:.3f}")
        print(f"F1-score: {f:.3f}")
        print("-" * 30)


if __name__ == "__main__":
    main()
