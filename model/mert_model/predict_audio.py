import os
import torch
import torch.nn as nn
import librosa  # <--- Используем Librosa
import pandas as pd
import numpy as np
from transformers import Wav2Vec2FeatureExtractor, AutoModel

# --- НАСТРОЙКИ ---
CSV_FILE = '../../datasets/MTG/moodtheme_mp3.csv'
MODEL_PATH = "trained_model/mert_model_best.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MERT_HF_ID = "m-a-p/MERT-v1-95M"


# --- 1. КЛАСС ВАШЕЙ МОДЕЛИ ---
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


# --- 2. КЛАСС ПРЕДСКАЗАНИЯ ---
class AudioPredictor:
    def __init__(self):
        print(f"Загрузка MERT ({MERT_HF_ID})...")
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(MERT_HF_ID, trust_remote_code=True)
        self.mert_model = AutoModel.from_pretrained(MERT_HF_ID, trust_remote_code=True).to(DEVICE)
        self.mert_model.eval()

        # Получаем названия классов
        if os.path.exists(CSV_FILE):
            df = pd.read_csv(CSV_FILE, nrows=1)
            meta_cols = ['TRACK_ID', 'PATH', 'DURATION']
            self.labels = [col for col in df.columns if col not in meta_cols]
        else:
            self.labels = []  # Если файла нет, будет работать, но без имен тегов

        # Загружаем вашу модель
        self.classifier = MERTClassifier(num_classes=len(self.labels)).to(DEVICE)
        try:
            self.classifier.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
            self.classifier.eval()
            print("Ваша модель готова к работе.")
        except Exception as e:
            print(f"ОШИБКА весов: {e}")

    def process_audio(self, audio_path):
        """Версия с Librosa"""
        try:
            # 1. Загрузка + Ресемплинг + Моно (все в одной строчке!)
            # sr=24000 — требование модели MERT
            # mono=True — смешиваем каналы
            audio_array, _ = librosa.load(audio_path, sr=24000, mono=True)

            # 2. Обрезка (берем первые 30 сек, чтобы не переполнить память GPU)
            max_samples = 24000 * 30
            if len(audio_array) > max_samples:
                audio_array = audio_array[:max_samples]

            # 3. Подготовка для модели
            # Librosa возвращает numpy, processor его отлично понимает
            input_values = self.processor(audio_array,
                                          sampling_rate=24000,
                                          return_tensors="pt").input_values.to(DEVICE)

            # 4. Прогон через MERT
            with torch.no_grad():
                outputs = self.mert_model(input_values)
                # Усредняем по времени
                embedding = outputs.last_hidden_state.mean(dim=1)

            return embedding

        except Exception as e:
            print(f"Librosa error: {e}")
            return None

    def predict(self, audio_path):
        if not os.path.exists(audio_path):
            print("❌ Файл не найден")
            return

        print(f"Processing: {os.path.basename(audio_path)} ...")
        embedding = self.process_audio(audio_path)

        if embedding is None: return

        # Предсказание вашей модели
        with torch.no_grad():
            logits = self.classifier(embedding)
            probs = torch.sigmoid(logits)[0].cpu().numpy()

        # Вывод
        top_indices = probs.argsort()[::-1]
        print(f"\n🎧 Результат:")
        for i in range(5):
            idx = top_indices[i]
            if probs[idx] > 0.05:  # Показываем только то, что хоть немного вероятно
                print(f"{probs[idx]:.1%} | {self.labels[idx]}")
        print("-" * 30)


if __name__ == "__main__":
    predictor = AudioPredictor()

    while True:
        path = input("\n>> Путь к mp3/wav (или 'exit'): ").strip().strip('"')
        if path.lower() in ['exit', 'quit']: break
        predictor.predict(path)