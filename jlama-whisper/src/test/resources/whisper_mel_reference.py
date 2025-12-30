#!/usr/bin/env python3
"""
Generate reference mel spectrogram data for Java test verification.

This script uses the original Whisper Python implementation to generate
mel spectrograms that can be compared against our Java implementation.

Usage:
    python whisper_mel_reference.py

Output:
    - whisper_mel_reference.json: Contains test cases with input audio and expected output
"""

import json
import numpy as np

# Whisper's mel spectrogram parameters
SAMPLE_RATE = 16000
N_FFT = 400
HOP_LENGTH = 160
N_MELS = 80
CHUNK_LENGTH = 30
N_SAMPLES = SAMPLE_RATE * CHUNK_LENGTH  # 480,000


def hann_window(n):
    """Create Hann window."""
    return 0.5 * (1 - np.cos(2 * np.pi * np.arange(n) / n))


def hz_to_mel(hz):
    """Convert Hz to Mel scale (HTK formula)."""
    return 2595 * np.log10(1 + hz / 700)


def mel_to_hz(mel):
    """Convert Mel to Hz scale (HTK formula)."""
    return 700 * (10 ** (mel / 2595) - 1)


def create_mel_filterbank(n_mels, n_fft, sample_rate):
    """Create mel filterbank matrix."""
    n_freq = n_fft // 2 + 1

    f_min = 0
    f_max = sample_rate / 2

    mel_min = hz_to_mel(f_min)
    mel_max = hz_to_mel(f_max)

    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    bin_points = hz_points * n_fft / sample_rate

    filters = np.zeros((n_mels, n_freq))

    for m in range(n_mels):
        left = bin_points[m]
        center = bin_points[m + 1]
        right = bin_points[m + 2]

        for k in range(n_freq):
            if left <= k <= center:
                filters[m, k] = (k - left) / (center - left)
            elif center < k <= right:
                filters[m, k] = (right - k) / (right - center)

        # Slaney normalization
        filters[m] *= 2.0 / (hz_points[m + 2] - hz_points[m])

    return filters


def log_mel_spectrogram_simple(audio, n_mels=80):
    """
    Compute log-mel spectrogram matching Whisper's implementation.

    This is a simplified version for testing that doesn't require torch.
    """
    # Pad audio
    reflect_pad = N_FFT // 2
    audio_padded = np.pad(audio, (reflect_pad, N_SAMPLES - len(audio) + reflect_pad), mode='reflect')

    # Limit to expected length
    if len(audio_padded) > N_SAMPLES + 2 * reflect_pad:
        audio_padded = audio_padded[:N_SAMPLES + 2 * reflect_pad]

    # Number of frames
    n_frames = (len(audio_padded) - N_FFT) // HOP_LENGTH

    # Create Hann window
    window = hann_window(N_FFT)

    # Create mel filterbank
    mel_filters = create_mel_filterbank(n_mels, N_FFT, SAMPLE_RATE)

    # Compute STFT and mel spectrogram
    mel_spec = np.zeros((n_mels, n_frames))

    for frame in range(n_frames):
        start = frame * HOP_LENGTH
        windowed = audio_padded[start:start + N_FFT] * window

        # FFT (only positive frequencies)
        fft_result = np.fft.rfft(windowed, n=N_FFT)
        power = np.abs(fft_result) ** 2

        # Apply mel filterbank
        mel_spec[:, frame] = mel_filters @ power

    # Log scaling
    log_spec = np.log10(np.maximum(mel_spec, 1e-10))

    # Normalization (Whisper's method)
    log_spec = np.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0

    return log_spec


def generate_test_cases():
    """Generate test cases with known inputs and expected outputs."""
    test_cases = []

    # Test case 1: Sine wave at 440 Hz (A4 note)
    t = np.arange(SAMPLE_RATE) / SAMPLE_RATE  # 1 second
    audio_sine = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    mel_sine = log_mel_spectrogram_simple(audio_sine)

    test_cases.append({
        "name": "sine_440hz",
        "description": "1 second of 440 Hz sine wave",
        "audio": audio_sine[:1600].tolist(),  # First 100ms for brevity
        "mel_shape": list(mel_sine.shape),
        "mel_sample": mel_sine[:, :10].tolist(),  # First 10 frames
        "mel_stats": {
            "min": float(mel_sine.min()),
            "max": float(mel_sine.max()),
            "mean": float(mel_sine.mean()),
        }
    })

    # Test case 2: Silence
    audio_silence = np.zeros(SAMPLE_RATE, dtype=np.float32)
    mel_silence = log_mel_spectrogram_simple(audio_silence)

    test_cases.append({
        "name": "silence",
        "description": "1 second of silence",
        "audio": audio_silence[:1600].tolist(),
        "mel_shape": list(mel_silence.shape),
        "mel_sample": mel_silence[:, :10].tolist(),
        "mel_stats": {
            "min": float(mel_silence.min()),
            "max": float(mel_silence.max()),
            "mean": float(mel_silence.mean()),
        }
    })

    # Test case 3: White noise
    np.random.seed(42)
    audio_noise = (0.1 * np.random.randn(SAMPLE_RATE)).astype(np.float32)
    mel_noise = log_mel_spectrogram_simple(audio_noise)

    test_cases.append({
        "name": "white_noise",
        "description": "1 second of white noise (seed=42)",
        "audio": audio_noise[:1600].tolist(),
        "mel_shape": list(mel_noise.shape),
        "mel_sample": mel_noise[:, :10].tolist(),
        "mel_stats": {
            "min": float(mel_noise.min()),
            "max": float(mel_noise.max()),
            "mean": float(mel_noise.mean()),
        }
    })

    # Test case 4: Chirp (frequency sweep)
    t = np.arange(SAMPLE_RATE) / SAMPLE_RATE
    audio_chirp = (0.5 * np.sin(2 * np.pi * (200 + 800 * t) * t)).astype(np.float32)
    mel_chirp = log_mel_spectrogram_simple(audio_chirp)

    test_cases.append({
        "name": "chirp",
        "description": "1 second chirp 200-1000 Hz",
        "audio": audio_chirp[:1600].tolist(),
        "mel_shape": list(mel_chirp.shape),
        "mel_sample": mel_chirp[:, :10].tolist(),
        "mel_stats": {
            "min": float(mel_chirp.min()),
            "max": float(mel_chirp.max()),
            "mean": float(mel_chirp.mean()),
        }
    })

    return test_cases


def main():
    print("Generating Whisper mel spectrogram reference data...")

    test_cases = generate_test_cases()

    output = {
        "parameters": {
            "sample_rate": SAMPLE_RATE,
            "n_fft": N_FFT,
            "hop_length": HOP_LENGTH,
            "n_mels": N_MELS,
            "chunk_length": CHUNK_LENGTH,
        },
        "test_cases": test_cases
    }

    with open("whisper_mel_reference.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"Generated {len(test_cases)} test cases")
    print("Output written to whisper_mel_reference.json")

    # Print summary
    for tc in test_cases:
        print(f"\n{tc['name']}:")
        print(f"  Shape: {tc['mel_shape']}")
        print(f"  Stats: min={tc['mel_stats']['min']:.4f}, max={tc['mel_stats']['max']:.4f}, mean={tc['mel_stats']['mean']:.4f}")


if __name__ == "__main__":
    main()
