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

import com.github.tjake.jlama.math.ActivationFunction;
import com.github.tjake.jlama.math.VectorMath;
import com.github.tjake.jlama.safetensors.Config;
import com.github.tjake.jlama.safetensors.WeightLoader;
import com.github.tjake.jlama.tensor.AbstractTensor;
import com.github.tjake.jlama.tensor.operations.TensorOperationsProvider;

import java.util.Optional;

/**
 * Whisper encoder that processes mel spectrogram features into hidden states.
 * Uses bidirectional self-attention (no causal masking).
 */
public class WhisperEncoder {
    private static final int CONV1_KERNEL_SIZE = 3;
    private static final int CONV2_KERNEL_SIZE = 3;
    private static final int CONV2_STRIDE = 2;

    private final AbstractModel model;
    private final Config config;
    private final int nMels;
    private final int embeddingLength;
    private final int numberOfLayers;

    // Conv1D weights: [out_channels, in_channels, kernel_size]
    // In Whisper: conv1 is [d_model, n_mels, 3], conv2 is [d_model, d_model, 3]
    private final AbstractTensor conv1Weight;
    private final AbstractTensor conv1Bias;
    private final AbstractTensor conv2Weight;
    private final AbstractTensor conv2Bias;

    // Positional embeddings: [max_source_positions, d_model]
    private final AbstractTensor positionalEmbedding;

    // Encoder layers
    private final EncoderBlock[] layers;

    // Final layer norm
    private final LayerNorm layerNorm;

    public WhisperEncoder(AbstractModel model, Config config, WeightLoader weights, int numberOfEncoderLayers) {
        this.model = model;
        this.config = config;
        this.nMels = config.n_mels > 0 ? config.n_mels : 80;
        this.embeddingLength = config.embeddingLength;
        this.numberOfLayers = numberOfEncoderLayers;

        // Load convolutional weights
        this.conv1Weight = weights.load("model.encoder.conv1.weight");
        this.conv1Bias = weights.load("model.encoder.conv1.bias");
        this.conv2Weight = weights.load("model.encoder.conv2.weight");
        this.conv2Bias = weights.load("model.encoder.conv2.bias");

        // Load positional embeddings
        this.positionalEmbedding = weights.load("model.encoder.embed_positions.weight");

        // Load encoder layers
        this.layers = new EncoderBlock[numberOfLayers];
        for (int i = 0; i < numberOfLayers; i++) {
            layers[i] = new EncoderBlock(model, config, weights, i);
        }

        // Final layer norm
        this.layerNorm = new LayerNorm(
            model,
            weights.load("model.encoder.layer_norm.weight"),
            weights.load("model.encoder.layer_norm.bias")
        );
    }

    /**
     * Encode mel spectrogram features.
     *
     * @param melSpectrogram Tensor of shape [n_mels, time_frames] - mel spectrogram
     * @return Encoded tensor of shape [seq_len, embeddingLength]
     */
    public AbstractTensor encode(AbstractTensor melSpectrogram) {
        int timeFrames = melSpectrogram.shape().dim(1);

        // Apply Conv1D layers
        // conv1: [n_mels, time] -> [d_model, time] with kernel=3, padding=1
        AbstractTensor afterConv1 = applyConv1D(melSpectrogram, conv1Weight, conv1Bias, nMels, embeddingLength, 1, 1);
        applyGelu(afterConv1);

        // conv2: [d_model, time] -> [d_model, time/2] with kernel=3, stride=2, padding=1
        int conv2OutLen = (afterConv1.shape().dim(1) + 2 * 1 - CONV2_KERNEL_SIZE) / CONV2_STRIDE + 1;
        AbstractTensor afterConv2 = applyConv1D(afterConv1, conv2Weight, conv2Bias, embeddingLength, embeddingLength, CONV2_STRIDE, 1);
        afterConv1.close();
        applyGelu(afterConv2);

        // Transpose to [seq_len, d_model] for transformer
        int seqLen = afterConv2.shape().dim(1);
        AbstractTensor x = model.makeDenseTensor(seqLen, embeddingLength);
        for (int t = 0; t < seqLen; t++) {
            for (int d = 0; d < embeddingLength; d++) {
                x.set(afterConv2.get(d, t), t, d);
            }
        }
        afterConv2.close();

        // Add positional embeddings (truncated or padded to seq_len)
        int posLen = Math.min(seqLen, positionalEmbedding.shape().first());
        for (int t = 0; t < posLen; t++) {
            for (int d = 0; d < embeddingLength; d++) {
                float val = x.get(t, d) + positionalEmbedding.get(t, d);
                x.set(val, t, d);
            }
        }

        // Run through encoder layers
        for (EncoderBlock layer : layers) {
            AbstractTensor newX = layer.forward(x);
            x.close();
            x = newX;
        }

        // Final layer norm
        AbstractTensor output = layerNorm.forward(x);
        x.close();

        return output;
    }

    /**
     * Apply 1D convolution with padding.
     */
    private AbstractTensor applyConv1D(AbstractTensor input, AbstractTensor weight, AbstractTensor bias,
                                        int inChannels, int outChannels, int stride, int padding) {
        int inputLen = input.shape().dim(1);
        int kernelSize = weight.shape().dim(2);
        int outputLen = (inputLen + 2 * padding - kernelSize) / stride + 1;

        AbstractTensor output = model.makeDenseTensor(outChannels, outputLen);

        // Initialize with bias
        for (int oc = 0; oc < outChannels; oc++) {
            float biasVal = bias.get(0, oc);
            for (int t = 0; t < outputLen; t++) {
                output.set(biasVal, oc, t);
            }
        }

        // Convolution
        for (int oc = 0; oc < outChannels; oc++) {
            for (int t = 0; t < outputLen; t++) {
                int inputStart = t * stride - padding;
                float sum = output.get(oc, t); // Start with bias

                for (int k = 0; k < kernelSize; k++) {
                    int inputIdx = inputStart + k;
                    if (inputIdx >= 0 && inputIdx < inputLen) {
                        for (int ic = 0; ic < inChannels; ic++) {
                            sum += input.get(ic, inputIdx) * weight.get(oc, ic, k);
                        }
                    }
                }
                output.set(sum, oc, t);
            }
        }

        return output;
    }

    /**
     * Apply GELU activation in-place.
     */
    private void applyGelu(AbstractTensor tensor) {
        int dim0 = tensor.shape().first();
        int dim1 = tensor.shape().dim(1);
        for (int i = 0; i < dim0; i++) {
            for (int j = 0; j < dim1; j++) {
                float val = tensor.get(i, j);
                tensor.set(ActivationFunction.eval(ActivationFunction.Type.GELU, val), i, j);
            }
        }
    }

    /**
     * Encoder transformer block with bidirectional self-attention.
     */
    private static class EncoderBlock {
        private final AbstractModel model;
        private final Config config;

        private final LayerNorm selfAttnLayerNorm;
        private final BidirectionalSelfAttention selfAttention;

        private final LayerNorm finalLayerNorm;
        private final AbstractTensor fc1Weight;
        private final AbstractTensor fc1Bias;
        private final AbstractTensor fc2Weight;
        private final AbstractTensor fc2Bias;

        public EncoderBlock(AbstractModel model, Config config, WeightLoader weights, int layerIndex) {
            this.model = model;
            this.config = config;

            String prefix = "model.encoder.layers." + layerIndex + ".";

            // Self-attention layer norm (applied before attention)
            this.selfAttnLayerNorm = new LayerNorm(
                model,
                weights.load(prefix + "self_attn_layer_norm.weight"),
                weights.load(prefix + "self_attn_layer_norm.bias")
            );

            // Bidirectional self-attention
            this.selfAttention = new BidirectionalSelfAttention(
                model,
                weights.load(prefix + "self_attn.q_proj.weight"),
                weights.load(prefix + "self_attn.k_proj.weight"),
                weights.load(prefix + "self_attn.v_proj.weight"),
                weights.load(prefix + "self_attn.out_proj.weight"),
                loadOptionalBias(weights, prefix + "self_attn.q_proj.bias"),
                loadOptionalBias(weights, prefix + "self_attn.k_proj.bias"),
                loadOptionalBias(weights, prefix + "self_attn.v_proj.bias"),
                loadOptionalBias(weights, prefix + "self_attn.out_proj.bias")
            );

            // Final layer norm (applied before FFN)
            this.finalLayerNorm = new LayerNorm(
                model,
                weights.load(prefix + "final_layer_norm.weight"),
                weights.load(prefix + "final_layer_norm.bias")
            );

            // FFN weights
            this.fc1Weight = weights.load(prefix + "fc1.weight");
            this.fc1Bias = weights.load(prefix + "fc1.bias");
            this.fc2Weight = weights.load(prefix + "fc2.weight");
            this.fc2Bias = weights.load(prefix + "fc2.bias");
        }

        private Optional<AbstractTensor> loadOptionalBias(WeightLoader weights, String name) {
            if (weights.isWeightPresent(name)) {
                return Optional.of(weights.load(name));
            }
            return Optional.empty();
        }

        public AbstractTensor forward(AbstractTensor x) {
            int seqLen = x.shape().first();

            // Self-attention with residual
            AbstractTensor normed = selfAttnLayerNorm.forward(x);
            AbstractTensor attnOut = selfAttention.forward(normed);
            normed.close();

            // Residual connection
            AbstractTensor afterAttn = model.makeDenseTensor(seqLen, config.embeddingLength);
            for (int i = 0; i < seqLen; i++) {
                for (int j = 0; j < config.embeddingLength; j++) {
                    afterAttn.set(x.get(i, j) + attnOut.get(i, j), i, j);
                }
            }
            attnOut.close();

            // FFN with residual
            AbstractTensor normed2 = finalLayerNorm.forward(afterAttn);

            // fc1: [seq, emb] -> [seq, hidden]
            AbstractTensor hidden = model.makeDenseTensor(seqLen, config.hiddenLength);
            VectorMath.pchunk(0, config.hiddenLength, (chunkStart, chunkLength) -> {
                TensorOperationsProvider.get()
                    .dotProductChunk(hidden, normed2, fc1Weight, 0, config.embeddingLength, chunkStart, chunkLength);
            });
            // Add bias and apply GELU
            for (int i = 0; i < seqLen; i++) {
                for (int j = 0; j < config.hiddenLength; j++) {
                    float val = hidden.get(i, j) + fc1Bias.get(0, j);
                    hidden.set(ActivationFunction.eval(ActivationFunction.Type.GELU, val), i, j);
                }
            }
            normed2.close();

            // fc2: [seq, hidden] -> [seq, emb]
            AbstractTensor ffnOut = model.makeDenseTensor(seqLen, config.embeddingLength);
            VectorMath.pchunk(0, config.embeddingLength, (chunkStart, chunkLength) -> {
                TensorOperationsProvider.get()
                    .dotProductChunk(ffnOut, hidden, fc2Weight, 0, config.hiddenLength, chunkStart, chunkLength);
            });
            hidden.close();

            // Add bias and residual
            AbstractTensor output = model.makeDenseTensor(seqLen, config.embeddingLength);
            for (int i = 0; i < seqLen; i++) {
                for (int j = 0; j < config.embeddingLength; j++) {
                    output.set(afterAttn.get(i, j) + ffnOut.get(i, j) + fc2Bias.get(0, j), i, j);
                }
            }
            ffnOut.close();
            afterAttn.close();

            return output;
        }
    }
}
