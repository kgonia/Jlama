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

import com.github.tjake.jlama.model.WhisperDecoder;
import com.github.tjake.jlama.safetensors.Config;
import com.github.tjake.jlama.safetensors.DType;
import com.github.tjake.jlama.safetensors.WeightLoader;
import com.github.tjake.jlama.safetensors.tokenizer.Tokenizer;
import com.github.tjake.jlama.tensor.AbstractTensor;
import com.github.tjake.jlama.tensor.FloatBufferTensor;
import com.github.tjake.jlama.tensor.TensorShape;

import java.io.File;
import java.nio.FloatBuffer;
import java.util.Optional;
import java.util.function.BiConsumer;

/**
 * High-level facade for Whisper speech-to-text transcription.
 */
public class WhisperModel {

    private final WhisperDecoder decoder;

    public WhisperModel(Config config, WeightLoader weightLoader, Tokenizer tokenizer) {
        this(config, weightLoader, tokenizer, weightLoader.getModelDType(), weightLoader.getModelDType(), Optional.empty());
    }

    public WhisperModel(Config config, WeightLoader weightLoader, Tokenizer tokenizer,
                        DType workingDType, DType workingQType, Optional<DType> modelQType) {
        this.decoder = new WhisperDecoder(config, weightLoader, tokenizer, workingDType, workingQType, modelQType);
    }

    /**
     * Transcribe audio from a WAV file.
     *
     * @param audioFile Path to the WAV file
     * @return Transcribed text
     */
    public String transcribe(String audioFile) throws Exception {
        return transcribe(audioFile, 256, 0.0f, null);
    }

    /**
     * Transcribe audio from a WAV file with options.
     *
     * @param audioFile Path to the WAV file
     * @param maxTokens Maximum number of tokens to generate
     * @param temperature Sampling temperature (0 for greedy decoding)
     * @param onToken Callback for each generated token
     * @return Transcribed text
     */
    public String transcribe(String audioFile, int maxTokens, float temperature,
                             BiConsumer<String, Float> onToken) throws Exception {
        // Load and preprocess audio
        float[][] melSpectrogram = WhisperAudioPreprocessor.fromFile(audioFile);

        // Convert to tensor [n_mels, time_frames]
        int nMels = melSpectrogram.length;
        int timeFrames = melSpectrogram[0].length;

        float[] flat = new float[nMels * timeFrames];
        for (int m = 0; m < nMels; m++) {
            System.arraycopy(melSpectrogram[m], 0, flat, m * timeFrames, timeFrames);
        }

        try (AbstractTensor melTensor = new FloatBufferTensor(
                FloatBuffer.wrap(flat),
                TensorShape.of(nMels, timeFrames),
                true)) {

            return decoder.transcribe(melTensor, maxTokens, temperature, onToken);
        }
    }

    /**
     * Transcribe from a pre-computed mel spectrogram tensor.
     *
     * @param melSpectrogram Tensor of shape [n_mels, time_frames]
     * @param maxTokens Maximum number of tokens to generate
     * @param temperature Sampling temperature (0 for greedy decoding)
     * @param onToken Callback for each generated token
     * @return Transcribed text
     */
    public String transcribe(AbstractTensor melSpectrogram, int maxTokens, float temperature,
                             BiConsumer<String, Float> onToken) {
        return decoder.transcribe(melSpectrogram, maxTokens, temperature, onToken);
    }

    /**
     * Get the underlying decoder model.
     */
    public WhisperDecoder getDecoder() {
        return decoder;
    }
}
