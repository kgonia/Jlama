package com.github.tjake.jlama.whisper;

import com.jlibrosa.audio.JLibrosa;

import javax.sound.sampled.UnsupportedAudioFileException;
import java.io.IOException;

/**
 * Audio preprocessor for Whisper models.
 * Converts audio files to mel spectrograms.
 *
 * Uses JLibrosa for audio loading/resampling and custom Vector API
 * implementation for mel spectrogram generation matching Whisper's specs.
 */
public class WhisperAudioPreprocessor {

    // Whisper expects 16kHz mono audio
    private static final int SAMPLE_RATE = 16000;

    // Default mel bins (80 for most models, 128 for large-v3)
    private static final int DEFAULT_N_MELS = 80;

    // Cached mel spectrogram generator
    private static volatile WhisperMelSpectrogram melGenerator80;
    private static volatile WhisperMelSpectrogram melGenerator128;

    /**
     * Load audio file and convert to mel spectrogram using 80 mel bins.
     *
     * @param filePath Path to audio file (WAV format)
     * @return Mel spectrogram of shape [n_mels, time_frames]
     */
    public static float[][] fromFile(String filePath) throws UnsupportedAudioFileException, IOException,
            com.jlibrosa.audio.wavFile.WavFileException,
            com.jlibrosa.audio.exception.FileFormatNotSupportedException {
        return fromFile(filePath, DEFAULT_N_MELS);
    }

    /**
     * Load audio file and convert to mel spectrogram.
     *
     * @param filePath Path to audio file (WAV format)
     * @param nMels Number of mel bins (80 or 128)
     * @return Mel spectrogram of shape [n_mels, time_frames]
     */
    public static float[][] fromFile(String filePath, int nMels) throws UnsupportedAudioFileException, IOException,
            com.jlibrosa.audio.wavFile.WavFileException,
            com.jlibrosa.audio.exception.FileFormatNotSupportedException {

        // Load and resample audio to 16kHz using JLibrosa
        JLibrosa jLibrosa = new JLibrosa();
        float[] audio = jLibrosa.loadAndRead(filePath, SAMPLE_RATE, -1);

        return fromSamples(audio, nMels);
    }

    /**
     * Convert raw audio samples to mel spectrogram.
     *
     * @param audio Audio samples at 16kHz
     * @param nMels Number of mel bins (80 or 128)
     * @return Mel spectrogram of shape [n_mels, time_frames]
     */
    public static float[][] fromSamples(float[] audio, int nMels) {
        WhisperMelSpectrogram generator = getMelGenerator(nMels);
        return generator.compute(audio);
    }

    /**
     * Convert raw audio samples to mel spectrogram using 80 mel bins.
     *
     * @param audio Audio samples at 16kHz
     * @return Mel spectrogram of shape [80, time_frames]
     */
    public static float[][] fromSamples(float[] audio) {
        return fromSamples(audio, DEFAULT_N_MELS);
    }

    /**
     * Get or create mel spectrogram generator for given mel bins.
     * Uses double-checked locking for thread-safe lazy initialization.
     */
    private static WhisperMelSpectrogram getMelGenerator(int nMels) {
        if (nMels == 80) {
            if (melGenerator80 == null) {
                synchronized (WhisperAudioPreprocessor.class) {
                    if (melGenerator80 == null) {
                        melGenerator80 = new WhisperMelSpectrogram(80);
                    }
                }
            }
            return melGenerator80;
        } else if (nMels == 128) {
            if (melGenerator128 == null) {
                synchronized (WhisperAudioPreprocessor.class) {
                    if (melGenerator128 == null) {
                        melGenerator128 = new WhisperMelSpectrogram(128);
                    }
                }
            }
            return melGenerator128;
        } else {
            // For non-standard mel counts, create new instance
            return new WhisperMelSpectrogram(nMels);
        }
    }

    /**
     * Get sample rate expected by Whisper.
     */
    public static int getSampleRate() {
        return SAMPLE_RATE;
    }
}
