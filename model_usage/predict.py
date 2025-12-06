import sys
import torch
import pandas as pd
import os

sys.path.append('../performing_data')
sys.path.append('../model')

try:
    from melspectograms import load_audio, melspectrogram
    from model import AudioCNN
except ImportError as e:
    print("Error: Run this script from the 'model_usage' directory.")
    sys.exit(1)

MODEL_PATH = '../model/trained_best_model/best_mood_model.pth'
CSV_PATH = '../datasets/MTG/dataset_metadata.csv'

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

    model = AudioCNN(len(classes)).to(DEVICE)

    if not os.path.exists(MODEL_PATH):
        print(f"Weights not found: {MODEL_PATH}")
        print("Run train_model.py first!")
        return

    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    print(f"Analyzing: {mp3_path}")
    try:
        audio = load_audio(mp3_path, segment_duration=29.1)
        spec = melspectrogram(audio)

        inp = torch.from_numpy(spec).float().unsqueeze(0).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            probs = torch.sigmoid(model(inp)).cpu().numpy()[0]

        results = sorted(zip(classes, probs), key=lambda x: x[1], reverse=True)

        print("\n--- Results ---")
        found = False
        for tag, prob in results:
            if prob > 0.1:
                print(f"{tag}: {prob * 100:.1f}%")
                found = True
            elif found and prob < 0.1:
                break
        if not found:
            print("No clear moods found (all < 10%)")

    except Exception as e:
        print(f"Processing error: {e}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        predict_mp3(sys.argv[1])
    else:
        print("Usage: python predict.py song.mp3")