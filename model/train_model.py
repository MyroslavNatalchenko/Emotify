import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
import pandas as pd
import numpy as np
from tqdm import tqdm

from model import AudioCNN
from dataset import AudioMoodDataset

CSV_FILE = '../datasets/MTG/moodtheme_low_npy.csv'
MODEL_SAVE_PATH = "trained_model/audio_rec_model_stratify.pth"
BATCH_SIZE = 16
LEARNING_RATE = 0.001
EPOCHS = 15
TARGET_FRAMES = 1366
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

def train_model():
    print(f"Device: {DEVICE}")

    df = pd.read_csv(CSV_FILE)
    num_classes = len(df.columns) - 3

    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    # TODO: stratify data

    train_ds = AudioMoodDataset(train_df, target_length=TARGET_FRAMES)
    val_ds = AudioMoodDataset(val_df, target_length=TARGET_FRAMES)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = AudioCNN(num_classes=num_classes).to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # TODO: try another learning rates
    criterion = nn.BCEWithLogitsLoss()

    best_pr_auc = 0.0

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}")

        for specs, labels in loop:
            specs, labels = specs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(specs)  # Forward pass
            loss = criterion(outputs, labels)  # BCEWithLogitsLoss
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        model.eval()
        val_loss = 0
        all_targets = []
        all_probs = []

        with torch.no_grad():
            for specs, labels in val_loader:
                specs, labels = specs.to(DEVICE), labels.to(DEVICE)

                logits = model(specs)
                loss = criterion(logits, labels)
                val_loss += loss.item()

                probs = torch.sigmoid(logits)

                all_targets.append(labels.cpu().numpy())
                all_probs.append(probs.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        all_targets = np.concatenate(all_targets)
        all_probs = np.concatenate(all_probs)

        # Metrics
        try:
            roc_auc = roc_auc_score(all_targets, all_probs, average='macro')
            pr_auc = average_precision_score(all_targets, all_probs, average='macro')

            threshold = 0.3 # TODO: test -> need to try other thresholds
            predicted_labels = (all_probs > threshold).astype(int)
            f1 = f1_score(all_targets, predicted_labels, average='macro')
        except ValueError:
            roc_auc, pr_auc, f1 = 0, 0, 0

        print(f"\nVal Loss: {avg_val_loss:.4f} | ROC-AUC: {roc_auc:.4f} | PR-AUC: {pr_auc:.4f} | F1: {f1:.4f}\n")

        if pr_auc > best_pr_auc:
            best_pr_auc = pr_auc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"Model saved! New Best PR-AUC: {best_pr_auc:.4f}")


if __name__ == "__main__":
    train_model()