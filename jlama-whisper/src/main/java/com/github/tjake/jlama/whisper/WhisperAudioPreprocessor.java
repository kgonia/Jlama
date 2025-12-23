/*
 * Copyright 2024 T Jake Luciani
 *
 * The Jlama Project licenses this file to you under the Apache License,
 * version 2.0 (the "License"); you may not use this file except in compliance
 * with the License. You may obtain a copy of the License at:
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 */
package com.github.tjake.jlama.whisper;

import com.jlibrosa.audio.JLibrosa;

import javax.sound.sampled.UnsupportedAudioFileException;
import java.io.IOException;

/**
 * Audio preprocessor for Whisper models.
 * Converts audio files to mel spectrograms.
 */
public class WhisperAudioPreprocessor {

    // Whisper expects 16kHz audio
    private static final int SAMPLE_RATE = 16000;

    // Whisper uses n_fft=400, but JLibrosa requires power of 2
    // Using 512 as a compatible approximation
    private static final int N_FFT = 512;

    // Hop length for 10ms steps at 16kHz
    private static final int HOP_LENGTH = 160;

    // Number of mel filterbanks
    private static final int N_MELS = 80;

    // Maximum audio length in samples (30 seconds)
    private static final int MAX_AUDIO_LENGTH = SAMPLE_RATE * 30;

    /**
     * Load audio file and convert to mel spectrogram.
     *
     * @param filePath Path to audio file (WAV format)
     * @return Mel spectrogram of shape [n_mels, time_frames]
     */
    public static float[][] fromFile(String filePath) throws UnsupportedAudioFileException, IOException,
            com.jlibrosa.audio.wavFile.WavFileException,
            com.jlibrosa.audio.exception.FileFormatNotSupportedException {

        JLibrosa jLibrosa = new JLibrosa();

        // Load audio and resample to 16kHz
        float[] audioFloats = jLibrosa.loadAndRead(filePath, SAMPLE_RATE, -1);

        // Pad or truncate to 30 seconds (Whisper's expected input)
        audioFloats = padOrTruncate(audioFloats, MAX_AUDIO_LENGTH);

        // Generate mel spectrogram
        float[][] melSpectrogram = jLibrosa.generateMelSpectroGram(audioFloats, SAMPLE_RATE, N_FFT, N_MELS, HOP_LENGTH);

        // Apply log mel transformation (Whisper uses log mel)
        for (int i = 0; i < melSpectrogram.length; i++) {
            for (int j = 0; j < melSpectrogram[i].length; j++) {
                // Clamp to minimum value and apply log
                float val = Math.max(melSpectrogram[i][j], 1e-10f);
                melSpectrogram[i][j] = (float) Math.log10(val);
            }
        }

        return melSpectrogram;
    }

    /**
     * Pad audio to target length with zeros, or truncate if too long.
     */
    private static float[] padOrTruncate(float[] audio, int targetLength) {
        if (audio.length >= targetLength) {
            // Truncate
            float[] result = new float[targetLength];
            System.arraycopy(audio, 0, result, 0, targetLength);
            return result;
        } else {
            // Pad with zeros
            float[] result = new float[targetLength];
            System.arraycopy(audio, 0, result, 0, audio.length);
            // Rest is already 0.0f
            return result;
        }
    }
}
