from argparse import ArgumentParser
import numpy as np
from essentia.standard import *
import essentia


def load_audio(filename: str, sampleRate: int = 12000, segment_duration: float = None) -> np.ndarray:
    """
    Loads audio and extracts a center segment. 
    If audio is shorter than segment_duration, it pads with zeros (silence).
    """
    # Load audio
    loader = MonoLoader(filename=filename, sampleRate=sampleRate, resampleQuality=4)
    audio = loader()

    if segment_duration:
        target_samples = int(round(segment_duration * sampleRate))
        audio_length = len(audio)

        # Case 1: Audio is longer than target (Center Crop)
        if audio_length >= target_samples:
            segment_start = (audio_length - target_samples) // 2
            segment_end = segment_start + target_samples
            return audio[segment_start:segment_end]

        # Case 2: Audio is shorter than target (Zero Pad)
        else:
            padding_needed = target_samples - audio_length
            pad_left = padding_needed // 2
            pad_right = padding_needed - pad_left
            # Pad with zeros
            return np.pad(audio, (pad_left, pad_right), 'constant')

    return audio


def melspectrogram(audio: np.ndarray,
                   sampleRate: int = 12000,
                   frameSize: int = 512,
                   hopSize: int = 256,
                   window: str = 'hann',
                   zeroPadding: int = 0,
                   center: bool = True,
                   numberBands: int = 96,
                   lowFrequencyBound: int = 0,
                   highFrequencyBound: int = None,
                   weighting: str = 'linear',
                   warpingFormula: str = 'slaneyMel',
                   normalize: str = 'unit_tri') -> np.ndarray:
    if highFrequencyBound is None:
        highFrequencyBound = sampleRate / 2

    # Initialize Algorithms
    windowing = Windowing(type=window, normalized=False, zeroPadding=zeroPadding)
    spectrum = Spectrum()
    melbands = MelBands(numberBands=numberBands,
                        sampleRate=sampleRate,
                        lowFrequencyBound=lowFrequencyBound,
                        highFrequencyBound=highFrequencyBound,
                        inputSize=(frameSize + zeroPadding) // 2 + 1,
                        weighting=weighting,
                        normalize=normalize,
                        warpingFormula=warpingFormula,
                        type='power')

    # Scale=2 converts Power to dB (10 * log10(x^2) is equivalent to 20 * log10(x))
    amp2db = UnaryOperator(type='lin2db', scale=2)

    pool = essentia.Pool()

    # Processing Loop
    # Note: startFromZero=True aligns the first frame at t=0. 
    # If center=True, standard libs usually pad input. 
    # Here we stick to your logic: startFromZero=False mimics skipping half a window.
    for frame in FrameGenerator(audio,
                                frameSize=frameSize,
                                hopSize=hopSize,
                                startFromZero=not center):
        # Chain: Window -> FFT -> MelFilterbank -> LogScale
        frame_windowed = windowing(frame)
        frame_spectrum = spectrum(frame_windowed)
        frame_mel = melbands(frame_spectrum)
        frame_db = amp2db(frame_mel)

        pool.add('mel', frame_db)

    # Return shape: (n_bands, n_frames)
    return pool['mel'].T


def analyze(audio_file: str, npy_file: str, full_audio: bool):
    # Choi et al. / MusicTagger architecture specific duration
    CHOI_DURATION = 29.1

    segment_duration = None if full_audio else CHOI_DURATION

    try:
        audio = load_audio(audio_file, segment_duration=segment_duration)
        mel = melspectrogram(audio)

        # Save compressed to save disk space, or standard binary
        np.save(npy_file, mel, allow_pickle=False)
        print(f"Successfully saved {mel.shape} to {npy_file}")

    except Exception as e:
        print(f"Error processing {audio_file}: {e}")


if __name__ == '__main__':
    parser = ArgumentParser(description="Computes a mel-spectrogram for Choi et al. Music Tagger.")

    parser.add_argument('audio_file', help='input audio file')
    parser.add_argument('npy_file', help='output NPY file')
    parser.add_argument('--full', dest='full_audio', help='analyze full audio instead of 29.1s segment',
                        action='store_true')
    args = parser.parse_args()

    analyze(args.audio_file, args.npy_file, args.full_audio)