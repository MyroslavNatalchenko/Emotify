import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np

# --- НАСТРОЙКИ ---
# Путь к CSV нужен ТОЛЬКО для того, чтобы узнать названия тегов (happy, sad и т.д.)
CSV_FILE = '../../datasets/MTG/moodtheme_mp3.csv'
MODEL_PATH = "trained_model/mert_model_best.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- 1. КЛАСС МОДЕЛИ (Тот же, что при обучении) ---
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


def load_labels_and_model():
    print("--- Инициализация ---")

    # 1. Получаем список тегов из заголовка CSV
    if os.path.exists(CSV_FILE):
        df = pd.read_csv(CSV_FILE, nrows=1)  # Читаем только первую строку
        meta_cols = ['TRACK_ID', 'PATH', 'DURATION']
        label_names = [col for col in df.columns if col not in meta_cols]
        print(f"Загружено {len(label_names)} классов из CSV.")
    else:
        print("ОШИБКА: Не найден CSV файл для получения названий тегов.")
        return None, None

    # 2. Загружаем модель
    model = MERTClassifier(num_classes=len(label_names)).to(DEVICE)
    try:
        state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(state_dict)
        model.eval()  # Режим предсказания (выключает Dropout)
        print("Веса модели успешно загружены.")
    except Exception as e:
        print(f"ОШИБКА загрузки модели: {e}")
        return None, None

    return model, label_names


def predict_interactive():
    model, label_names = load_labels_and_model()
    if model is None: return

    print("\n" + "=" * 50)
    print("ГОТОВ К РАБОТЕ!")
    print("Введите путь к файлу .npy (или 'exit' для выхода)")
    print("=" * 50)

    while True:
        # Ввод пути пользователем
        user_path = input("\n>> Путь к файлу: ").strip().strip('"').strip("'")  # Убираем лишние кавычки

        if user_path.lower() in ['exit', 'quit', 'выход']:
            break

        if not os.path.exists(user_path):
            print("❌ Файл не найден. Проверьте путь.")
            continue

        if not user_path.endswith('.npy'):
            print("⚠️ Внимание: модель ожидает файл .npy (вектор).")
            # Если это mp3, можно вывести предупреждение
            if user_path.endswith('.mp3') or user_path.endswith('.wav'):
                print("   Вы ввели аудиофайл. Сначала его нужно конвертировать в вектор MERT.")
                continue

        # Обработка и предсказание
        try:
            # 1. Загрузка вектора
            embedding = np.load(user_path)

            # 2. Препроцессинг (как в обучении)
            # Если размерность (Time, 768) -> усредняем до (768,)
            if embedding.ndim > 1:
                embedding = embedding.mean(axis=0)

            # Добавляем Batch dimension: (768,) -> (1, 768)
            tensor_input = torch.tensor(embedding, dtype=torch.float32).unsqueeze(0).to(DEVICE)

            # 3. Инференс
            with torch.no_grad():
                logits = model(tensor_input)
                probs = torch.sigmoid(logits)[0]  # Берем нулевой элемент батча

            # 4. Вывод результатов
            probs_np = probs.cpu().numpy()

            # Сортируем от большего к меньшему
            top_indices = probs_np.argsort()[::-1]

            print(f"\n📊 Результаты для {os.path.basename(user_path)}:")
            print("-" * 30)

            # Выводим Топ-5 тегов
            for i in range(5):
                idx = top_indices[i]
                score = probs_np[idx]
                tag = label_names[idx]

                # Рисуем красивый прогресс-бар
                bar_len = int(score * 20)
                bar = "█" * bar_len + "░" * (20 - bar_len)

                print(f"{bar} {score:.1%} | {tag}")

        except Exception as e:
            print(f"❌ Ошибка при обработке файла: {e}")


if __name__ == "__main__":
    predict_interactive()