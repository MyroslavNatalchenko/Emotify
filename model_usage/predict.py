import sys
import torch
import pandas as pd
import numpy as np
import os

sys.path.append('../performing_data')
sys.path.append('../model')

try:
    from melspectograms import load_audio, melspectrogram
    from model import AudioCNN
except ImportError as e:
    print("Error imports. Run from 'model_usage' directory.")
    sys.exit(1)

MODEL_PATH = '../model/trained_model/audio_rec_model.pth'
CSV_PATH = '../datasets/MTG/moodtheme_low_npy.csv'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
TARGET_FRAMES = 1366

def predict_mp3(mp3_path):
    if not os.path.exists(mp3_path):
        print(f"File not found: {mp3_path}")
        return

    try:
        df = pd.read_csv(CSV_PATH, nrows=1)
        classes = df.columns[3:].tolist()
    except FileNotFoundError:
        print(f"CSV not found: {CSV_PATH}")
        return

    model = AudioCNN(num_classes=len(classes)).to(DEVICE)

    if not os.path.exists(MODEL_PATH):
        print(f"Weights not found: {MODEL_PATH}")
        return

    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    print(f"Analyzing: {mp3_path}")
    try:
        audio = load_audio(mp3_path, segment_duration=29.1)
        spec = melspectrogram(audio)

        if spec.shape[1] > TARGET_FRAMES:
            spec = spec[:, :TARGET_FRAMES]
        elif spec.shape[1] < TARGET_FRAMES:
            pad = TARGET_FRAMES - spec.shape[1]
            spec = np.pad(spec, ((0, 0), (0, pad)), mode='constant')

        inp = torch.from_numpy(spec).float().unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            logits = model(inp)
            probs = torch.sigmoid(logits).cpu().numpy()[0]

        results = sorted(zip(classes, probs), key=lambda x: x[1], reverse=True)

        print("\n--- Results ---")
        found = False
        for tag, prob in results:
            if prob > 0.15:
                print(f"{tag}: {prob * 100:.1f}%")
                found = True
            elif found and prob < 0.1:
                break
        if not found:
            print("No clear moods found.")

    except Exception as e:
        print(f"Processing error: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        predict_mp3(sys.argv[1])
    else:
        print("Usage: python predict.py song.mp3")