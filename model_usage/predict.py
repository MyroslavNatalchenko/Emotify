import sys
import torch
import numpy as np
import os


try:
    from performing_data.melspectograms import load_audio, melspectrogram
    from model.model import AudioCNN
    from datasets.mood_themes import MOOD_THEMES
except ImportError as e:
    sys.path.append("../")
    from performing_data.melspectograms import load_audio, melspectrogram
    from model.model import AudioCNN
    from datasets.mood_themes import MOOD_THEMES

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
TARGET_FRAMES = 1366

class EmotionPredictor:
    def __init__(self, model_path):
        self.device = DEVICE
        self.model = None

        if not os.path.exists(model_path):
            print(f"Weights not found: {model_path}")
            return

        try:
            self.model = AudioCNN(num_classes=len(MOOD_THEMES)).to(self.device)
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None

    def predict(self, mp3_path):
        if self.model is None:
            return []

        if not os.path.exists(mp3_path):
            print(f"File not found: {mp3_path}")
            return []

        try:
            audio = load_audio(mp3_path, segment_duration=29.1)
            spec = melspectrogram(audio)

            if spec.shape[1] > TARGET_FRAMES:
                spec = spec[:, :TARGET_FRAMES]
            elif spec.shape[1] < TARGET_FRAMES:
                pad = TARGET_FRAMES - spec.shape[1]
                spec = np.pad(spec, ((0, 0), (0, pad)), mode='constant')

            inp = torch.from_numpy(spec).float().unsqueeze(0).to(self.device)

            with torch.no_grad():
                logits = self.model(inp)
                probs = torch.sigmoid(logits).cpu().numpy()[0]

            results = []
            for tag, prob in enumerate(probs):
                if tag < len(MOOD_THEMES):
                    if prob > 0.15:
                        results.append({
                            "tag": MOOD_THEMES[tag],
                            "confidence": float(prob)
                        })
            results.sort(key=lambda x: x["confidence"], reverse=True)
            return results

        except Exception as e:
            print(f"Processing error: {e}")
            return []

if __name__ == "__main__":
    if len(sys.argv) > 1:
        MODEL_PATH = '../model/trained_model/audio_rec_model.pth'

        predictor = EmotionPredictor(MODEL_PATH)
        print(f"Analyzing: {sys.argv[1]}")

        predictions = predictor.predict(sys.argv[1])

        print("\n--- Results ---")
        found = False
        for item in predictions:
            print(f"{item['tag']}: {item['confidence'] * 100:.1f}%")
            found = True

        if not found:
            print("No clear moods found.")
    else:
        print("Usage: python predict.py song.mp3")