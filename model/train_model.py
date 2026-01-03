import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from skmultilearn.model_selection import iterative_train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import matplotlib.pyplot as plt

from dataset import MultimodalDataset
from model import MultimodalAudioModel

CSV_FILE = '../datasets/MTG/moodtheme_low_npy.csv'
MERT_EMB_DIR = '/Volumes/T7 Shield/Emotify/MTG_dataset/mertspecs'

OUTPUT_DIR = "training_output"
MODEL_SAVE_PATH = os.path.join(OUTPUT_DIR, "multimodal_finetuned.pth")
PLOT_SAVE_PATH = os.path.join(OUTPUT_DIR, "training_history_finetuned.png")

BATCH_SIZE = 64
LEARNING_RATE = 1e-3
EPOCHS = 15
TARGET_FRAMES = 1366
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        """
        Focal Loss for multi-label classification.
        gamma: Focusing parameter. Higher values focus more on hard examples.
        alpha: Balancing parameter.
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def plot_history(history):
    epochs_range = range(1, len(history['train_loss']) + 1)
    plt.figure(figsize=(15, 6))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, history['train_loss'], label='Train Loss', marker='o')
    plt.plot(epochs_range, history['val_loss'], label='Val Loss', marker='o', linestyle='--')
    plt.title('Loss: Train vs Validation')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, history['roc_auc'], label='Val ROC-AUC', color='green')
    plt.plot(epochs_range, history['pr_auc'], label='Val PR-AUC', color='blue', linewidth=2)
    plt.plot(epochs_range, history['f1'], label='Val F1', color='orange', linestyle='--')
    plt.title('Validation Metrics')
    plt.xlabel('Epochs')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(PLOT_SAVE_PATH)
    print(f"\n[Info] Training plots saved to: {PLOT_SAVE_PATH}")
    plt.close()


def find_optimal_thresholds(targets, probs, num_classes):
    print("\n[Info] Tuning thresholds for each class...")
    best_thresholds = []

    for i in range(num_classes):
        y_true = targets[:, i]
        y_prob = probs[:, i]

        best_f1 = 0.0
        best_t = 0.2

        for t in np.arange(0.05, 0.8, 0.05):
            y_pred = (y_prob >= t).astype(int)
            score = f1_score(y_true, y_pred, average='binary', zero_division=0)
            if score > best_f1:
                best_f1 = score
                best_t = t

        best_thresholds.append(best_t)
    return best_thresholds


def train_model():
    print(f"Device: {DEVICE}")

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # --- 1. DATA ---
    print("[Data] Loading CSV...")
    df = pd.read_csv(CSV_FILE)

    meta_cols = ['TRACK_ID', 'PATH', 'DURATION']
    label_cols = [c for c in df.columns if c not in meta_cols]
    num_classes = len(label_cols)
    print(f"[Data] Detected {num_classes} classes.")

    X_indices = df.index.values.reshape(-1, 1)
    y_labels = df[label_cols].values

    print("[Data] Performing Iterative Stratified Split...")
    X_train_idx, _, X_val_idx, _ = iterative_train_test_split(X_indices, y_labels, test_size=0.15)

    train_df = df.loc[X_train_idx[:, 0]]
    val_df = df.loc[X_val_idx[:, 0]]

    print(f"[Data] Train size: {len(train_df)} | Val size: {len(val_df)}")

    print("[Data] Initializing Datasets...")
    train_ds = MultimodalDataset(train_df, MERT_EMB_DIR, target_length=TARGET_FRAMES)
    val_ds = MultimodalDataset(val_df, MERT_EMB_DIR, target_length=TARGET_FRAMES)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # --- 2. MODEL & CONFIGURATION ---
    model = MultimodalAudioModel(num_classes=num_classes).to(DEVICE)

    criterion = FocalLoss(alpha=1, gamma=2)
    print("[Setup] Using Focal Loss (gamma=2).")

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-2)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    print("[Setup] Using CosineAnnealingLR scheduler.")

    # --- 3. TRAINING LOOP ---
    best_pr_auc = 0.0
    history = {'train_loss': [], 'val_loss': [], 'roc_auc': [], 'pr_auc': [], 'f1': []}

    for epoch in range(EPOCHS):
        model.train()
        running_train_loss = 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}")

        for specs, merts, labels in loop:
            specs, merts, labels = specs.to(DEVICE), merts.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(specs, merts)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_train_loss = running_train_loss / len(train_loader)

        model.eval()
        running_val_loss = 0
        all_targets = []
        all_probs = []

        with torch.no_grad():
            for specs, merts, labels in val_loader:
                specs, merts, labels = specs.to(DEVICE), merts.to(DEVICE), labels.to(DEVICE)
                logits = model(specs, merts)
                loss = criterion(logits, labels)
                running_val_loss += loss.item()
                probs = torch.sigmoid(logits)
                all_targets.append(labels.cpu().numpy())
                all_probs.append(probs.cpu().numpy())

        avg_val_loss = running_val_loss / len(val_loader)
        all_targets = np.concatenate(all_targets)
        all_probs = np.concatenate(all_probs)

        try:
            roc_auc = roc_auc_score(all_targets, all_probs, average='macro')
            pr_auc = average_precision_score(all_targets, all_probs, average='macro')
            predicted = (all_probs > 0.3).astype(int)
            f1 = f1_score(all_targets, predicted, average='macro')
        except ValueError:
            roc_auc, pr_auc, f1 = 0, 0, 0

        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['roc_auc'].append(roc_auc)
        history['pr_auc'].append(pr_auc)
        history['f1'].append(f1)

        print(f"\nVal Loss: {avg_val_loss:.4f} | ROC-AUC: {roc_auc:.4f} | PR-AUC: {pr_auc:.4f} | F1: {f1:.4f}")

        # CHANGE 2 (Update): CosineAnnealingLR does not require arguments
        scheduler.step()

        if pr_auc > best_pr_auc:
            best_pr_auc = pr_auc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"--> Best Model Weights Saved! PR-AUC: {best_pr_auc:.4f}\n")

    # --- 4. POST-TRAINING: THRESHOLD OPTIMIZATION ---
    print("\n[Post-Training] Loading best model to tune thresholds...")
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    model.eval()

    all_targets = []
    all_probs = []

    with torch.no_grad():
        for specs, merts, labels in val_loader:
            specs, merts, labels = specs.to(DEVICE), merts.to(DEVICE), labels.to(DEVICE)
            logits = model(specs, merts)
            probs = torch.sigmoid(logits)
            all_targets.append(labels.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    all_targets = np.concatenate(all_targets)
    all_probs = np.concatenate(all_probs)

    best_thresholds = find_optimal_thresholds(all_targets, all_probs, num_classes)

    final_checkpoint = {
        'model_state_dict': model.state_dict(),
        'thresholds': best_thresholds,
        'pr_auc': best_pr_auc,
        'labels': label_cols
    }
    torch.save(final_checkpoint, MODEL_SAVE_PATH)
    print(f"\n[Success] Final model with optimized thresholds saved to: {MODEL_SAVE_PATH}")

    plot_history(history)


if __name__ == "__main__":
    train_model()