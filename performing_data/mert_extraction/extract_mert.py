import os
import argparse
import torch
import pandas as pd
import numpy as np
from transformers import Wav2Vec2FeatureExtractor, AutoModel
from tqdm import tqdm
import librosa

# Настройки модели
MODEL_NAME = "m-a-p/MERT-v1-95M"
TARGET_SR = 24000  # MERT работает только с 24kHz


def load_model(device):
    print(f"Loading MERT model: {MODEL_NAME}...")
    processor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model.to(device)
    model.eval()
    return processor, model


def process_audio(filepath, processor, model, device):
    try:
        wav_numpy, sr = librosa.load(filepath, sr=TARGET_SR, mono=True)

        # 2. Конвертируем в Tensor (MERT ждет тензор)
        # wav_numpy имеет форму (Time,), делаем (1, Time)
        wav = torch.from_numpy(wav_numpy).unsqueeze(0)

        # 3. Обрезка (30 сек)
        # 30 сек * 24000 = 720,000 сэмплов
        max_samples = 30 * TARGET_SR
        if wav.shape[1] > max_samples:
            start = (wav.shape[1] - max_samples) // 2
            wav = wav[:, start: start + max_samples]

        # 4. Инференс
        # processor сам разберется с паддингом если трек короче
        inputs = processor(wav.squeeze(), sampling_rate=TARGET_SR, return_tensors="pt")
        input_values = inputs.input_values.to(device)

        with torch.no_grad():
            outputs = model(input_values)

        # 5. Агрегация
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()

        return embedding

    except Exception as e:
        print(f"\nError processing {filepath}: {e}")
        return None


def main(args):
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Загружаем CSV
    df = pd.read_csv(args.csv_path)
    print(f"Loaded CSV with {len(df)} tracks.")

    # 2. Грузим модель
    processor, model = load_model(device)

    # 3. Создаем папку для вывода
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    print(f"Saving embeddings to: {args.output_dir}")
    print("Starting extraction... (This may take time)")

    count = 0
    # Проходим по всем трекам
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        # --- ЛОГИКА ПОСТРОЕНИЯ ПУТИ ---
        # В CSV путь типа: .../melspecs/48/948.npy
        # Нам нужно:       .../audio_low/48/948.mp3

        # Берем имя файла (948.npy) и имя папки (48)
        old_path = row['PATH']
        filename_npy = os.path.basename(old_path)  # 948.npy
        folder_name = os.path.basename(os.path.dirname(old_path))  # 48

        filename_mp3 = filename_npy.replace('.npy', '.mp3')

        # Собираем реальный путь к MP3
        # args.audio_root это "/Volumes/T7 Shield/.../audio_low"
        mp3_path = os.path.join(args.audio_root, folder_name, filename_mp3)

        # Путь куда сохраним вектор
        save_folder = os.path.join(args.output_dir, folder_name)
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        save_path = os.path.join(save_folder, filename_npy)  # Сохраняем как .npy

        # Пропускаем, если уже есть
        if os.path.exists(save_path):
            continue

        if not os.path.exists(mp3_path):
            # Иногда в пути к файлу в CSV бывают ошибки или файл отсутствует
            # print(f"File not found: {mp3_path}")
            continue

        # Обработка
        emb = process_audio(mp3_path, processor, model, device)

        if emb is not None:
            np.save(save_path, emb)
            count += 1

    print(f"\nDone! Processed {count} tracks.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, required=True, help="Путь к dataset_metadata.csv")
    parser.add_argument("--audio_root", type=str, required=True, help="Папка, где лежат папки 00, 01 с mp3 файлами")
    parser.add_argument("--output_dir", type=str, required=True, help="Куда сохранять новые npy файлы")
    args = parser.parse_args()
    main(args)