import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
import pandas as pd
import numpy as np
from tqdm import tqdm

# CONFIG
CSV_FILE = '../../datasets/MTG/moodtheme_mp3.csv'
EMB_DIR = 'F:/Emotify/MTG_dataset/mertspecs'
MODEL_SAVE_PATH = "trained_model/mert_model_best.pth"

BATCH_SIZE = 64
LEARNING_RATE = 1e-3
EPOCHS = 50
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# MODEL
class MERTClassifier(nn.Module):
    def __init__(self, input_size=768, num_classes=56):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.network(x)


# DATASET
class MERTDataset(Dataset):
    def __init__(self, df, embeddings_dir):
        self.df = df
        self.embeddings_dir = embeddings_dir

        self.meta_cols = ['TRACK_ID', 'PATH', 'DURATION']
        self.label_cols = [col for col in self.df.columns if col not in self.meta_cols]

        print(f"Loading {len(df)} vectors into RAM")

        self.X_data = []
        self.Y_data = []

        labels_matrix = self.df[self.label_cols].values.astype('float32')
        paths = self.df['PATH'].values

        found_count = 0
        missing_count = 0

        for idx in tqdm(range(len(self.df)), desc="Caching"):
            original_path = paths[idx]

            folder = os.path.basename(os.path.dirname(original_path))
            filename = os.path.basename(original_path)

            track_id = filename.split('.')[0]

            final_path = None
            candidates = [f"{track_id}.npy", f"{track_id}.low.npy"]

            for cand in candidates:
                p = os.path.join(self.embeddings_dir, folder, cand)
                if os.path.exists(p):
                    final_path = p
                    break

            if final_path:
                try:
                    embedding = np.load(final_path)
                    if embedding.ndim > 1:
                        embedding = embedding.squeeze()
                    if embedding.ndim > 1:
                        embedding = embedding.mean(axis=0)

                    self.X_data.append(embedding)
                    found_count += 1
                except:
                    missing_count += 1
            else:
                missing_count += 1

            self.Y_data.append(labels_matrix[idx])

        # Convert list to tensor
        self.X_data = torch.tensor(np.array(self.X_data), dtype=torch.float32)
        self.Y_data = torch.tensor(np.array(self.Y_data), dtype=torch.float32)

        print(f"Caching complete: Found {found_count}, Missing/Error {missing_count}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return self.X_data[idx], self.Y_data[idx]


# TRAINING
def train_model():
    print(f"Device: {DEVICE}")

    if not os.path.exists(CSV_FILE):
        print(f"Error: CSV file not found at {CSV_FILE}")
        return

    df = pd.read_csv(CSV_FILE)

    meta_cols = ['TRACK_ID', 'PATH', 'DURATION']
    num_classes = len([c for c in df.columns if c not in meta_cols])
    print(f"Number of classes: {num_classes}")

    # Split
    train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)

    print("Preparing Train Dataset")
    train_ds = MERTDataset(train_df, EMB_DIR)
    print("Preparing Val Dataset")
    val_ds = MERTDataset(val_df, EMB_DIR)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = MERTClassifier(num_classes=num_classes).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

    best_pr_auc = 0.0

    print("Start Training")
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}")

        for vecs, labels in loop:
            vecs, labels = vecs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(vecs)
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
            for vecs, labels in val_loader:
                vecs, labels = vecs.to(DEVICE), labels.to(DEVICE)

                logits = model(vecs)
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

            predicted_labels = (all_probs > 0.4).astype(int)
            f1 = f1_score(all_targets, predicted_labels, average='macro')
        except ValueError:
            roc_auc, pr_auc, f1 = 0, 0, 0

        # Step & Save
        scheduler.step(pr_auc)

        print(f"Val Loss: {avg_val_loss:.4f} | ROC-AUC: {roc_auc:.4f} | PR-AUC: {pr_auc:.4f} | F1: {f1:.4f}")

        if pr_auc > best_pr_auc:
            best_pr_auc = pr_auc
            if not os.path.exists("trained_model"):
                os.makedirs("trained_model")
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"Saved Best Model! PR-AUC: {best_pr_auc:.4f}")
        print("-" * 30)


if __name__ == "__main__":
    train_model()