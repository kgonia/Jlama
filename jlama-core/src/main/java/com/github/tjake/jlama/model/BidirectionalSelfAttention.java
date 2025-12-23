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
 * Bidirectional self-attention for encoder models (like Whisper encoder).
 * Unlike CausalSelfAttention, this attends to the full sequence without masking.
 * No KV cache is used since encoders process the full sequence at once.
 */
public class BidirectionalSelfAttention {

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

    public BidirectionalSelfAttention(
            AbstractModel model,
            AbstractTensor queryWeights,
            AbstractTensor keyWeights,
            AbstractTensor valueWeights,
            AbstractTensor outputWeights) {
        this(model, queryWeights, keyWeights, valueWeights, outputWeights,
             Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty());
    }

    public BidirectionalSelfAttention(
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
    }

    /**
     * Forward pass for bidirectional attention.
     *
     * @param input Tensor of shape [seqLen, embeddingLength]
     * @return Output tensor of shape [seqLen, embeddingLength]
     */
    public AbstractTensor forward(AbstractTensor input) {
        int seqLen = input.shape().first();
        int embeddingLength = config.embeddingLength;
        int attentionLength = numberOfHeads * headSize;

        // Compute Q, K, V projections for full sequence
        // Shape: [seqLen, attentionLength]
        AbstractTensor Q = model.makeDenseTensor(seqLen, attentionLength);
        AbstractTensor K = model.makeDenseTensor(seqLen, attentionLength);
        AbstractTensor V = model.makeDenseTensor(seqLen, attentionLength);

        // Project input to Q, K, V
        VectorMath.pchunk(0, attentionLength, (chunkStart, chunkLength) -> {
            TensorOperationsProvider.get()
                .dotProductChunk(Q, input, queryWeights, 0, embeddingLength, chunkStart, chunkLength);
            TensorOperationsProvider.get()
                .dotProductChunk(K, input, keyWeights, 0, embeddingLength, chunkStart, chunkLength);
            TensorOperationsProvider.get()
                .dotProductChunk(V, input, valueWeights, 0, embeddingLength, chunkStart, chunkLength);
        });

        // Add biases if present
        queryBias.ifPresent(bias -> {
            for (int i = 0; i < seqLen; i++) {
                TensorOperationsProvider.get().accumulate(Q.slice(i), bias, 0, attentionLength);
            }
        });
        keyBias.ifPresent(bias -> {
            for (int i = 0; i < seqLen; i++) {
                TensorOperationsProvider.get().accumulate(K.slice(i), bias, 0, attentionLength);
            }
        });
        valueBias.ifPresent(bias -> {
            for (int i = 0; i < seqLen; i++) {
                TensorOperationsProvider.get().accumulate(V.slice(i), bias, 0, attentionLength);
            }
        });

        // Output tensor for attention results
        AbstractTensor output = model.makeDenseTensor(seqLen, attentionLength);

        // Process each head independently
        VectorMath.pfor(0, numberOfHeads, h -> {
            int headOffset = h * headSize;

            // For each query position
            for (int qPos = 0; qPos < seqLen; qPos++) {
                // Compute attention scores for this query against all keys
                float[] scores = new float[seqLen];

                for (int kPos = 0; kPos < seqLen; kPos++) {
                    float score = 0.0f;
                    for (int d = 0; d < headSize; d++) {
                        score += Q.get(qPos, headOffset + d) * K.get(kPos, headOffset + d);
                    }
                    scores[kPos] = score * attentionScale;
                }

                // Softmax over all positions (full attention, no masking)
                float maxScore = Float.NEGATIVE_INFINITY;
                for (int i = 0; i < seqLen; i++) {
                    maxScore = Math.max(maxScore, scores[i]);
                }
                float sum = 0.0f;
                for (int i = 0; i < seqLen; i++) {
                    scores[i] = (float) Math.exp(scores[i] - maxScore);
                    sum += scores[i];
                }
                for (int i = 0; i < seqLen; i++) {
                    scores[i] /= sum;
                }

                // Weighted sum of values
                for (int d = 0; d < headSize; d++) {
                    float value = 0.0f;
                    for (int vPos = 0; vPos < seqLen; vPos++) {
                        value += scores[vPos] * V.get(vPos, headOffset + d);
                    }
                    output.set(value, qPos, headOffset + d);
                }
            }
        });

        // Clean up Q, K, V
        Q.close();
        K.close();
        V.close();

        // Output projection
        AbstractTensor result = model.makeDenseTensor(seqLen, embeddingLength);
        VectorMath.pchunk(0, embeddingLength, (chunkStart, chunkLength) -> {
            TensorOperationsProvider.get()
                .dotProductChunk(result, output, outputWeights, 0, attentionLength, chunkStart, chunkLength);
        });

        output.close();

        // Add output bias if present
        outputBias.ifPresent(bias -> {
            for (int i = 0; i < seqLen; i++) {
                TensorOperationsProvider.get().accumulate(result.slice(i), bias, 0, embeddingLength);
            }
        });

        return result;
    }
}
