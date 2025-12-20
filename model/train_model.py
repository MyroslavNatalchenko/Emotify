import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
import pandas as pd
import numpy as np
from tqdm import tqdm
import os

from dataset import MultimodalDataset
from model import MultimodalAudioModel

CSV_FILE = '../datasets/MTG/moodtheme_low_npy.csv'
MERT_EMB_DIR = '/Volumes/T7 Shield/Emotify/MTG_dataset/mertspecs'
MODEL_SAVE_PATH = "model/test.pth"

BATCH_SIZE = 16
LEARNING_RATE = 1e-3
EPOCHS = 10
TARGET_FRAMES = 1366
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def train_model():
    print(f"Device: {DEVICE}")

    df = pd.read_csv(CSV_FILE)

    meta_cols = ['TRACK_ID', 'PATH', 'DURATION']
    num_classes = len([c for c in df.columns if c not in meta_cols])

    train_df, val_df = train_test_split(df, test_size=0.15, random_state=42)

    print("Preparing Train dataset...")
    train_ds = MultimodalDataset(train_df, MERT_EMB_DIR, target_length=TARGET_FRAMES)

    print("Preparing Val dataset...")
    val_ds = MultimodalDataset(val_df, MERT_EMB_DIR, target_length=TARGET_FRAMES)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    model = MultimodalAudioModel(num_classes=num_classes).to(DEVICE)

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    criterion = nn.BCEWithLogitsLoss()

    best_pr_auc = 0.0

    # Training Loop
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}")

        for specs, merts, labels in loop:
            specs = specs.to(DEVICE)
            merts = merts.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()

            outputs = model(specs, merts)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        # Validation
        model.eval()
        val_loss = 0
        all_targets = []
        all_probs = []

        with torch.no_grad():
            for specs, merts, labels in val_loader:
                specs = specs.to(DEVICE)
                merts = merts.to(DEVICE)
                labels = labels.to(DEVICE)

                logits = model(specs, merts)
                loss = criterion(logits, labels)
                val_loss += loss.item()

                probs = torch.sigmoid(logits)

                all_targets.append(labels.cpu().numpy())
                all_probs.append(probs.cpu().numpy())

        # Metrics
        avg_val_loss = val_loss / len(val_loader)
        all_targets = np.concatenate(all_targets)
        all_probs = np.concatenate(all_probs)

        try:
            roc_auc = roc_auc_score(all_targets, all_probs, average='macro')
            pr_auc = average_precision_score(all_targets, all_probs, average='macro')

            predicted = (all_probs > 0.4).astype(int)
            f1 = f1_score(all_targets, predicted, average='macro')
        except:
            roc_auc, pr_auc, f1 = 0, 0, 0

        print(f"\nVal Loss: {avg_val_loss:.4f} | ROC-AUC: {roc_auc:.4f} | PR-AUC: {pr_auc:.4f} | F1: {f1:.4f}")

        scheduler.step(pr_auc)

        if pr_auc > best_pr_auc:
            best_pr_auc = pr_auc
            if not os.path.exists(os.path.dirname(MODEL_SAVE_PATH)):
                os.makedirs(os.path.dirname(MODEL_SAVE_PATH))
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"--> Model Saved! New Best PR-AUC: {best_pr_auc:.4f}\n")


if __name__ == "__main__":
    train_model()