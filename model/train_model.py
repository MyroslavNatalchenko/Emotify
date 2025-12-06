import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd
from tqdm import tqdm
import sys

from model import AudioCNN
from dataset import AudioMoodDataset

sys.path.append('../performing_data')

CSV_FILE = '../datasets/MTG/dataset_metadata.csv'
MODEL_SAVE_PATH = "trained_best_model/best_mood_model.pth"
BATCH_SIZE = 16
LR = 0.001
EPOCHS = 15
TARGET_FRAMES = 1366
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def train_model():
    print(f"Device: {DEVICE}")

    df = pd.read_csv(CSV_FILE)
    num_classes = len(df.columns) - 3  # minus ID, PATH, DURATION

    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

    train_ds = AudioMoodDataset(train_df, target_length=TARGET_FRAMES)
    val_ds = AudioMoodDataset(val_df, target_length=TARGET_FRAMES)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = AudioCNN(num_classes).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCEWithLogitsLoss()  # for multi-label classification

    best_loss = float('inf')

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}")

        for specs, labels in loop:
            specs, labels = specs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(specs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        # Валидация
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for specs, labels in val_loader:
                specs, labels = specs.to(DEVICE), labels.to(DEVICE)
                outputs = model(specs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print("Model saved!")


if __name__ == "__main__":
    train_model()