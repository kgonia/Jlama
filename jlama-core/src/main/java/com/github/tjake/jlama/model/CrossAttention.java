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
package com.github.tjake.jlama.model;

import com.github.tjake.jlama.math.VectorMath;
import com.github.tjake.jlama.safetensors.Config;
import com.github.tjake.jlama.tensor.AbstractTensor;
import com.github.tjake.jlama.tensor.operations.TensorOperationsProvider;

import java.util.Optional;

/**
 * Cross-attention layer for encoder-decoder models (like Whisper decoder).
 * Query comes from the decoder hidden states, Key and Value come from encoder output.
 * K and V are cached since encoder output doesn't change during generation.
 */
public class CrossAttention {

    private final AbstractModel model;
    private final Config config;
    private final int numberOfHeads;
    private final int headSize;
    private final float attentionScale;

    private final AbstractTensor queryWeights;
    private final AbstractTensor keyWeights;
    private final AbstractTensor valueWeights;
    private final AbstractTensor outputWeights;

    private final Optional<AbstractTensor> queryBias;
    private final Optional<AbstractTensor> keyBias;
    private final Optional<AbstractTensor> valueBias;
    private final Optional<AbstractTensor> outputBias;

    // Cached K and V projections from encoder (computed once)
    private AbstractTensor cachedK;
    private AbstractTensor cachedV;
    private int cachedEncoderSeqLen;

    public CrossAttention(
            AbstractModel model,
            AbstractTensor queryWeights,
            AbstractTensor keyWeights,
            AbstractTensor valueWeights,
            AbstractTensor outputWeights) {
        this(model, queryWeights, keyWeights, valueWeights, outputWeights,
             Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty());
    }

    public CrossAttention(
            AbstractModel model,
            AbstractTensor queryWeights,
            AbstractTensor keyWeights,
            AbstractTensor valueWeights,
            AbstractTensor outputWeights,
            Optional<AbstractTensor> queryBias,
            Optional<AbstractTensor> keyBias,
            Optional<AbstractTensor> valueBias,
            Optional<AbstractTensor> outputBias) {
        this.model = model;
        this.config = model.c;
        this.numberOfHeads = config.numberOfHeads;
        this.headSize = config.headSize;
        this.attentionScale = (float) (1.0 / Math.sqrt(headSize));

        this.queryWeights = queryWeights;
        this.keyWeights = keyWeights;
        this.valueWeights = valueWeights;
        this.outputWeights = outputWeights;

        this.queryBias = queryBias;
        this.keyBias = keyBias;
        this.valueBias = valueBias;
        this.outputBias = outputBias;

        this.cachedK = null;
        this.cachedV = null;
        this.cachedEncoderSeqLen = 0;
    }

    /**
     * Reset cached K and V when processing new audio.
     */
    public void resetCache() {
        if (cachedK != null) {
            cachedK.close();
            cachedK = null;
        }
        if (cachedV != null) {
            cachedV.close();
            cachedV = null;
        }
        cachedEncoderSeqLen = 0;
    }

    /**
     * Forward pass for cross-attention.
     *
     * @param decoderInput Tensor of shape [batchSize, embeddingLength] - decoder hidden states
     * @param encoderOutput Tensor of shape [encoderSeqLen, embeddingLength] - encoder output
     * @return Output tensor of shape [batchSize, embeddingLength]
     */
    public AbstractTensor forward(AbstractTensor decoderInput, AbstractTensor encoderOutput) {
        int decoderBatchSize = decoderInput.shape().first();
        int encoderSeqLen = encoderOutput.shape().first();
        int embeddingLength = config.embeddingLength;
        int attentionLength = numberOfHeads * headSize;

        // Cache K and V from encoder if not already cached or if encoder output changed
        if (cachedK == null || cachedEncoderSeqLen != encoderSeqLen) {
            if (cachedK != null) cachedK.close();
            if (cachedV != null) cachedV.close();

            cachedK = model.makeDenseTensor(encoderSeqLen, attentionLength);
            cachedV = model.makeDenseTensor(encoderSeqLen, attentionLength);
            cachedEncoderSeqLen = encoderSeqLen;

            // Project encoder output to K and V
            VectorMath.pchunk(0, attentionLength, (chunkStart, chunkLength) -> {
                TensorOperationsProvider.get()
                    .dotProductChunk(cachedK, encoderOutput, keyWeights, 0, embeddingLength, chunkStart, chunkLength);
                TensorOperationsProvider.get()
                    .dotProductChunk(cachedV, encoderOutput, valueWeights, 0, embeddingLength, chunkStart, chunkLength);
            });

            // Add biases if present
            keyBias.ifPresent(bias -> {
                for (int i = 0; i < encoderSeqLen; i++) {
                    TensorOperationsProvider.get().accumulate(cachedK.slice(i), bias, 0, attentionLength);
                }
            });
            valueBias.ifPresent(bias -> {
                for (int i = 0; i < encoderSeqLen; i++) {
                    TensorOperationsProvider.get().accumulate(cachedV.slice(i), bias, 0, attentionLength);
                }
            });
        }

        // Compute Q from decoder input
        AbstractTensor Q = model.makeDenseTensor(decoderBatchSize, attentionLength);
        VectorMath.pchunk(0, attentionLength, (chunkStart, chunkLength) -> {
            TensorOperationsProvider.get()
                .dotProductChunk(Q, decoderInput, queryWeights, 0, embeddingLength, chunkStart, chunkLength);
        });

        queryBias.ifPresent(bias -> {
            for (int i = 0; i < decoderBatchSize; i++) {
                TensorOperationsProvider.get().accumulate(Q.slice(i), bias, 0, attentionLength);
            }
        });

        // Output tensor for attention results
        AbstractTensor attnOutput = model.makeDenseTensor(decoderBatchSize, attentionLength);

        // Process each head independently
        VectorMath.pfor(0, numberOfHeads, h -> {
            int headOffset = h * headSize;

            // For each decoder position (query)
            for (int qPos = 0; qPos < decoderBatchSize; qPos++) {
                // Compute attention scores against all encoder positions (keys)
                float[] scores = new float[encoderSeqLen];

                for (int kPos = 0; kPos < encoderSeqLen; kPos++) {
                    float score = 0.0f;
                    for (int d = 0; d < headSize; d++) {
                        score += Q.get(qPos, headOffset + d) * cachedK.get(kPos, headOffset + d);
                    }
                    scores[kPos] = score * attentionScale;
                }

                // Softmax over all encoder positions
                float maxScore = Float.NEGATIVE_INFINITY;
                for (int i = 0; i < encoderSeqLen; i++) {
                    maxScore = Math.max(maxScore, scores[i]);
                }
                float sum = 0.0f;
                for (int i = 0; i < encoderSeqLen; i++) {
                    scores[i] = (float) Math.exp(scores[i] - maxScore);
                    sum += scores[i];
                }
                for (int i = 0; i < encoderSeqLen; i++) {
                    scores[i] /= sum;
                }

                // Weighted sum of encoder values
                for (int d = 0; d < headSize; d++) {
                    float value = 0.0f;
                    for (int vPos = 0; vPos < encoderSeqLen; vPos++) {
                        value += scores[vPos] * cachedV.get(vPos, headOffset + d);
                    }
                    attnOutput.set(value, qPos, headOffset + d);
                }
            }
        });

        Q.close();

        // Output projection
        AbstractTensor result = model.makeDenseTensor(decoderBatchSize, embeddingLength);
        VectorMath.pchunk(0, embeddingLength, (chunkStart, chunkLength) -> {
            TensorOperationsProvider.get()
                .dotProductChunk(result, attnOutput, outputWeights, 0, attentionLength, chunkStart, chunkLength);
        });

        attnOutput.close();

        // Add output bias if present
        outputBias.ifPresent(bias -> {
            for (int i = 0; i < decoderBatchSize; i++) {
                TensorOperationsProvider.get().accumulate(result.slice(i), bias, 0, embeddingLength);
            }
        });

        return result;
    }
}
