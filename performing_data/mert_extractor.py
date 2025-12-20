import os
import sys
import argparse
import numpy as np
import torch
import warnings

# PATH SETUP
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_root)

try:
    from model.model import MultimodalAudioModel
    from performing_data.mert_extractor import load_model as load_mert_lib, get_file_embedding
    from performing_data.melspectograms import load_audio as load_audio_mel, melspectrogram
except ImportError as e:
    print(f"Import Error: {e}")
    sys.exit(1)

warnings.filterwarnings("ignore")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
MODEL_PATH = os.path.join(project_root, "model/mert/model/multimodal_best.pth")
TARGET_FRAMES = 1366
LABELS = [
    'action', 'adventure', 'advertising', 'ambiental', 'background', 'ballad', 'calm',
    'children', 'christmas', 'commercial', 'cool', 'corporate', 'dark', 'deep',
    'documentary', 'drama', 'dramatic', 'dream', 'emotional', 'energetic', 'epic',
    'fast', 'film', 'fun', 'funny', 'game', 'groovy', 'happy', 'heavy', 'holiday',
    'hopeful', 'horror', 'inspiring', 'love', 'meditative', 'melancholic', 'mellow',
    'melodic', 'motivational', 'movie', 'nature', 'party', 'positive', 'powerful',
    'relaxing', 'retro', 'romantic', 'sad', 'sexy', 'slow', 'soft', 'soundscape',
    'space', 'sport', 'summer', 'trailer', 'travel', 'upbeat', 'uplifting'
]


class MoodPredictor:
    def __init__(self):
        print(f"Initializing on device: {DEVICE}")

        self.mert_processor, self.mert_extractor = load_mert_lib(DEVICE)

        print(f"Loading classifier weights from: {MODEL_PATH}")
        self.model = MultimodalAudioModel(num_classes=len(LABELS)).to(DEVICE)

        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")

        state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        print("System ready!\n")

    def _get_melspec_features(self, filepath):
        """
        Wrapper to use imported melspectrogram logic and ensure correct shape.
        """
        try:
            audio = load_audio_mel(filepath, segment_duration=29.1)

            spec = melspectrogram(audio)

            current_width = spec.shape[1]
            if current_width < TARGET_FRAMES:
                pad_amount = TARGET_FRAMES - current_width
                spec = np.pad(spec, ((0, 0), (0, pad_amount)), 'constant')
            elif current_width > TARGET_FRAMES:
                spec = spec[:, :TARGET_FRAMES]

            return spec.astype(np.float32)

        except Exception as e:
            print(f"Error extracting MelSpectrogram: {e}")
            return None

    def predict(self, audio_path):
        if not os.path.exists(audio_path):
            print(f"File not found: {audio_path}")
            return

        print(f"--- Analyzing: {os.path.basename(audio_path)} ---")

        spec_np = self._get_melspec_features(audio_path)

        mert_np = get_file_embedding(audio_path, self.mert_processor, self.mert_extractor, DEVICE)

        if spec_np is None or mert_np is None:
            print("Failed to extract features (MERT or MelSpec failed). Aborting.")
            return

        spec_tensor = torch.from_numpy(spec_np).unsqueeze(0).to(DEVICE)
        mert_tensor = torch.from_numpy(mert_np).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            logits = self.model(spec_tensor, mert_tensor)
            probs = torch.sigmoid(logits).squeeze().cpu().numpy()

        results = []
        for i, prob in enumerate(probs):
            if prob >= 0.1:
                results.append((LABELS[i], prob))

        results.sort(key=lambda x: x[1], reverse=True)

        if not results:
            print("No genres with confidence > 10%.")
        else:
            print(f"{'Genre/Mood':<20} | {'Confidence':<10}")
            print("-" * 35)
            for label, prob in results:
                print(f"{label:<20} | {prob:.1%}")
        print("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multimodal Genre/Mood prediction for MP3 files.")
    parser.add_argument("file_path", type=str, help="Path to the audio file (.mp3, .wav)")

    args = parser.parse_args()

    predictor = MoodPredictor()
    predictor.predict(args.file_path)