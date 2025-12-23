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

import com.github.tjake.jlama.model.functions.FeedForward;
import com.github.tjake.jlama.tensor.AbstractTensor;
import com.github.tjake.jlama.tensor.KvBufferCache;
import com.github.tjake.jlama.tensor.operations.TensorOperationsProvider;

import java.util.List;
import java.util.Optional;
import java.util.function.Consumer;
import java.util.function.Supplier;

/**
 * Whisper decoder block with self-attention, cross-attention, and feed-forward.
 * The cross-attention layer attends to the encoder output.
 */
public class WhisperDecoderBlock extends TransformerBlock {

    private final CrossAttention crossAttention;
    private final LayerNorm crossAttentionNorm;
    private final Supplier<AbstractTensor> encoderOutputSupplier;

    public WhisperDecoderBlock(
            AbstractModel model,
            int layerIndex,
            LayerNorm selfAttentionNorm,
            CausalSelfAttention selfAttention,
            LayerNorm crossAttentionNorm,
            CrossAttention crossAttention,
            LayerNorm ffNorm,
            FeedForward ffBlock,
            Supplier<AbstractTensor> encoderOutputSupplier) {
        super(model, layerIndex,
              Optional.of(selfAttentionNorm),
              selfAttention,
              Optional.empty(),
              Optional.of(ffNorm),
              ffBlock,
              Optional.empty(),
              Optional.empty());

        this.crossAttention = crossAttention;
        this.crossAttentionNorm = crossAttentionNorm;
        this.encoderOutputSupplier = encoderOutputSupplier;
    }

    @Override
    public AbstractTensor forward(
            AbstractTensor embedding,
            int position,
            KvBufferCache.KvBuffer kvBuffer,
            Optional<Consumer<List<AbstractTensor>>> tensorReducer) {

        // 1. Self-attention with residual
        AbstractTensor afterSelfAttn = applyAttention(embedding, position, kvBuffer, tensorReducer);

        // 2. Cross-attention with residual
        AbstractTensor afterCrossAttn = applyCrossAttention(afterSelfAttn);

        // 3. Feed-forward with residual
        return applyFeedForward(afterCrossAttn, tensorReducer);
    }

    private AbstractTensor applyCrossAttention(AbstractTensor input) {
        int batchSize = input.shape().first();
        int embeddingLength = input.shape().last();

        // Apply layer norm before cross-attention
        AbstractTensor normed = crossAttentionNorm.forward(input);

        // Get encoder output from supplier
        AbstractTensor encoderOutput = encoderOutputSupplier.get();

        // Apply cross-attention
        AbstractTensor crossAttnOut = crossAttention.forward(normed, encoderOutput);
        normed.close();

        // Residual connection: input + cross_attn_output
        for (int b = 0; b < batchSize; b++) {
            for (int e = 0; e < embeddingLength; e++) {
                float val = input.get(b, e) + crossAttnOut.get(b, e);
                crossAttnOut.set(val, b, e);
            }
        }

        input.close();
        return crossAttnOut;
    }

    /**
     * Reset the cross-attention cache when processing new audio.
     */
    public void resetCrossAttentionCache() {
        crossAttention.resetCache();
    }
}
