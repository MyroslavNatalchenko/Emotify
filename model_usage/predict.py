import os
import sys
import argparse
import numpy as np
import torch
import librosa
from transformers import Wav2Vec2FeatureExtractor, AutoModel
from essentia.standard import MonoLoader, Windowing, Spectrum, MelBands, UnaryOperator, FrameGenerator
import essentia
import warnings

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_root)

try:
    from model.model import (MultimodalAudioModel)
except ImportError:
    print(
        "Import Error! Ensure you are running the script from the correct directory or the project structure is correct.")
    sys.exit(1)

warnings.filterwarnings("ignore")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
MODEL_PATH = os.path.join(project_root, "model/model/multimodal_best.pth")
MERT_MODEL_NAME = "m-a-p/MERT-v1-95M"

# Label list (MUST match the CSV column order starting from the 4th column)
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


def load_mert_model():
    print(f"Loading MERT: {MERT_MODEL_NAME}...")
    processor = Wav2Vec2FeatureExtractor.from_pretrained(MERT_MODEL_NAME, trust_remote_code=True)
    model = AutoModel.from_pretrained(MERT_MODEL_NAME, trust_remote_code=True)
    model.to(DEVICE)
    model.eval()
    return processor, model


def get_mert_embedding(filepath, processor, model):
    target_sr = 24000
    duration = 30

    try:
        wav, _ = librosa.load(filepath, sr=target_sr, mono=True)
    except Exception as e:
        print(f"Error loading audio for MERT: {e}")
        return None

    target_samples = int(duration * target_sr)
    if len(wav) > target_samples:
        start = (len(wav) - target_samples) // 2
        wav = wav[start:start + target_samples]

    # 2. Inference
    inputs = processor(wav, sampling_rate=target_sr, return_tensors="pt", padding=True)
    input_values = inputs.input_values.to(DEVICE)

    with torch.no_grad():
        outputs = model(input_values)

    # (Time, 768) -> (768,)
    embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
    return embedding


def get_melspectrogram(filename):
    sampleRate = 12000
    frameSize = 512
    hopSize = 256
    numberBands = 96
    segment_duration = 29.1  # ~1366 frames

    try:
        loader = MonoLoader(filename=filename, sampleRate=sampleRate, resampleQuality=4)
        audio = loader()
    except Exception as e:
        print(f"Error loading audio for MelSpec: {e}")
        return None

    target_samples = int(round(segment_duration * sampleRate))
    if len(audio) >= target_samples:
        start = (len(audio) - target_samples) // 2
        audio = audio[start:start + target_samples]
    else:
        pad_needed = target_samples - len(audio)
        pad_l = pad_needed // 2
        pad_r = pad_needed - pad_l
        audio = np.pad(audio, (pad_l, pad_r), 'constant')

    windowing = Windowing(type='hann', normalized=False, zeroPadding=0)
    spectrum = Spectrum()
    melbands = MelBands(numberBands=numberBands, sampleRate=sampleRate,
                        lowFrequencyBound=0, highFrequencyBound=sampleRate / 2,
                        inputSize=(frameSize // 2) + 1, weighting='linear',
                        warpingFormula='slaneyMel', type='power', normalize='unit_tri')
    amp2db = UnaryOperator(type='lin2db', scale=2)

    pool = essentia.Pool()

    for frame in FrameGenerator(audio, frameSize=frameSize, hopSize=hopSize, startFromZero=True):
        f_win = windowing(frame)
        f_spec = spectrum(f_win)
        f_mel = melbands(f_spec)
        f_db = amp2db(f_mel)
        pool.add('mel', f_db)

    return pool['mel'].T.astype(np.float32)


class MoodPredictor:
    def __init__(self):
        print(f"Initializing on device: {DEVICE}")

        self.mert_processor, self.mert_extractor = load_mert_model()

        print(f"Loading model weights from: {MODEL_PATH}")
        self.model = MultimodalAudioModel(num_classes=len(LABELS)).to(DEVICE)

        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

        state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        print("System ready!\n")

    def predict(self, audio_path):
        if not os.path.exists(audio_path):
            print(f"File not found: {audio_path}")
            return

        print(f"--- Analyzing: {os.path.basename(audio_path)} ---")

        # MelSpec: (96, 1366)
        spec_np = get_melspectrogram(audio_path)
        # MERT: (768,)
        mert_np = get_mert_embedding(audio_path, self.mert_processor, self.mert_extractor)

        if spec_np is None or mert_np is None:
            print("Failed to extract features.")
            return


        spec_tensor = torch.from_numpy(spec_np).unsqueeze(0).to(DEVICE)
        mert_tensor = torch.from_numpy(mert_np).unsqueeze(0).to(DEVICE)

        # 3. Forward pass
        with torch.no_grad():
            logits = self.model(spec_tensor, mert_tensor)
            probs = torch.sigmoid(logits).squeeze().cpu().numpy()

        results = []
        for i, prob in enumerate(probs):
            if prob >= 0.10:  # Threshold 10%
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
    parser = argparse.ArgumentParser(description="Genre/Mood prediction for MP3 files.")
    parser.add_argument("file_path", type=str, help="Path to audio file (.mp3, .wav)")

    args = parser.parse_args()

    # Run
    predictor = MoodPredictor()
    predictor.predict(args.file_path)