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

# --- НАСТРОЙКИ ---
# Проверьте эти пути перед запуском!
CSV_FILE = '../../datasets/MTG/moodtheme_mp3.csv'  # Ваш CSV файл
EMB_DIR = 'F:/Emotify/MTG_dataset/mertspecs'  # Папка с MERT векторами (.npy)
MODEL_SAVE_PATH = "trained_model/mert_model_best.pth"  # Куда сохранять модель

BATCH_SIZE = 64  # Батч
LEARNING_RATE = 1e-3  # Скорость обучения
EPOCHS = 50  # Количество эпох
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- 1. КЛАСС МОДЕЛИ (MLP) ---
class MERTClassifier(nn.Module):
    def __init__(self, input_size=768, num_classes=56):
        super().__init__()
        self.network = nn.Sequential(
            # Вход: 768
            nn.Linear(input_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),

            # Скрытый слой
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),

            # Выход: 56 классов
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.network(x)


# --- 2. ДАТАСЕТ (С ЗАГРУЗКОЙ В ПАМЯТЬ) ---
class MERTDataset(Dataset):
    def __init__(self, df, embeddings_dir):
        self.df = df
        self.embeddings_dir = embeddings_dir

        self.meta_cols = ['TRACK_ID', 'PATH', 'DURATION']
        self.label_cols = [col for col in self.df.columns if col not in self.meta_cols]

        # --- ЗАГРУЗКА ДАННЫХ В RAM ---
        print(f"Загружаем {len(df)} векторов в память... Подождите минуту.")

        self.X_data = []
        self.Y_data = []

        # Конвертируем лейблы сразу в массив
        labels_matrix = self.df[self.label_cols].values.astype('float32')
        paths = self.df['PATH'].values

        found_count = 0
        missing_count = 0

        for idx in tqdm(range(len(self.df)), desc="Caching"):
            original_path = paths[idx]

            # Разбираем путь
            # В CSV: .../audio_low/41/1145441.low.mp3
            # Нам нужно найти .npy файл в папке EMB_DIR

            filename = os.path.basename(original_path)  # 1145441.low.mp3
            folder = os.path.basename(os.path.dirname(original_path))  # 41

            # Меняем расширение на .npy
            if filename.endswith('.mp3'):
                filename_npy = filename.replace('.mp3', '.npy')
            else:
                filename_npy = filename

            # Полный путь к ожидаемому файлу
            emb_path = os.path.join(self.embeddings_dir, folder, filename_npy)

            # --- ЛОГИКА ПОИСКА ФАЙЛА ---
            final_path = None

            if os.path.exists(emb_path):
                final_path = emb_path
            else:
                # Пробуем варианты имени (с .low и без)
                # 1. Если искали .low.npy, попробуем просто .npy
                alt_name_1 = filename_npy.replace('.low.npy', '.npy')
                alt_path_1 = os.path.join(self.embeddings_dir, folder, alt_name_1)

                # 2. Если искали просто .npy, попробуем .low.npy
                alt_name_2 = filename_npy.replace('.npy', '.low.npy')
                alt_path_2 = os.path.join(self.embeddings_dir, folder, alt_name_2)

                if os.path.exists(alt_path_1):
                    final_path = alt_path_1
                elif os.path.exists(alt_path_2):
                    final_path = alt_path_2

            # --- ЗАГРУЗКА ---
            if final_path:
                try:
                    embedding = np.load(final_path)
                    # Убираем лишние размерности (1, 768) -> (768,)
                    if embedding.ndim > 1:
                        embedding = embedding.squeeze()
                    # Если вдруг (Time, 768), усредняем
                    if embedding.ndim > 1:
                        embedding = embedding.mean(axis=0)

                    self.X_data.append(embedding)
                    found_count += 1
                except:
                    # Был файл, но битый
                    self.X_data.append(np.zeros(768))
                    missing_count += 1
            else:
                # Файл вообще не найден
                self.X_data.append(np.zeros(768))
                missing_count += 1

            self.Y_data.append(labels_matrix[idx])

        # Превращаем в Tensor (весь датасет сразу)
        self.X_data = torch.tensor(np.array(self.X_data), dtype=torch.float32)
        self.Y_data = torch.tensor(np.array(self.Y_data), dtype=torch.float32)

        print(f"Кеширование завершено: Найдено {found_count}, Пустых/Ошибок {missing_count}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Просто берем из памяти
        return self.X_data[idx], self.Y_data[idx]


# --- 3. ФУНКЦИЯ ОБУЧЕНИЯ ---
def train_model():
    print(f"Device: {DEVICE}")

    # Загрузка CSV
    if not os.path.exists(CSV_FILE):
        print(f"ОШИБКА: Не найден CSV файл: {CSV_FILE}")
        return

    df = pd.read_csv(CSV_FILE)

    # Считаем классы
    meta_cols = ['TRACK_ID', 'PATH', 'DURATION']
    num_classes = len([c for c in df.columns if c not in meta_cols])
    print(f"Количество классов: {num_classes}")

    # Разделение Train / Val
    train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)

    # Инициализация датасетов (тут будет пауза на загрузку в RAM)
    print("\n--- Подготовка Train Dataset ---")
    train_ds = MERTDataset(train_df, EMB_DIR)
    print("\n--- Подготовка Val Dataset ---")
    val_ds = MERTDataset(val_df, EMB_DIR)

    # Dataloaders
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Модель
    model = MERTClassifier(num_classes=num_classes).to(DEVICE)

    # Оптимизатор
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

    best_pr_auc = 0.0

    print("\n--- Start Training ---")
    for epoch in range(EPOCHS):
        # 1. Train
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

        # 2. Validation
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

        # 3. Metrics
        try:
            roc_auc = roc_auc_score(all_targets, all_probs, average='macro')
            pr_auc = average_precision_score(all_targets, all_probs, average='macro')

            threshold = 0.4
            predicted_labels = (all_probs > threshold).astype(int)
            f1 = f1_score(all_targets, predicted_labels, average='macro')
        except ValueError:
            roc_auc, pr_auc, f1 = 0, 0, 0

        # 4. Step & Save
        scheduler.step(pr_auc)

        print(f"Val Loss: {avg_val_loss:.4f} | ROC-AUC: {roc_auc:.4f} | PR-AUC: {pr_auc:.4f} | F1: {f1:.4f}")

        if pr_auc > best_pr_auc:
            best_pr_auc = pr_auc
            if not os.path.exists("trained_model"):
                os.makedirs("trained_model")
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"--> Saved Best Model! PR-AUC: {best_pr_auc:.4f}")
        print("-" * 30)


if __name__ == "__main__":
    train_model()