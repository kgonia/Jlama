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
import com.github.tjake.jlama.model.functions.EmbedInput;
import com.github.tjake.jlama.model.functions.FeedForward;
import com.github.tjake.jlama.model.functions.SampleOutput;
import com.github.tjake.jlama.model.whisper.WhisperConfig;
import com.github.tjake.jlama.safetensors.Config;
import com.github.tjake.jlama.safetensors.DType;
import com.github.tjake.jlama.safetensors.WeightLoader;
import com.github.tjake.jlama.safetensors.tokenizer.Tokenizer;
import com.github.tjake.jlama.tensor.AbstractTensor;
import com.github.tjake.jlama.tensor.KvBufferCache;
import com.github.tjake.jlama.tensor.operations.TensorOperationsProvider;

import java.util.Optional;
import java.util.UUID;
import java.util.concurrent.ThreadLocalRandom;
import java.util.function.BiConsumer;

/**
 * Whisper decoder model for speech-to-text transcription.
 */
public class WhisperDecoder extends AbstractModel {

    private final WhisperEncoder encoder;
    private final int numberOfEncoderLayers;
    private final int numberOfDecoderLayers;

    // Token embeddings
    private AbstractTensor tokenEmbeddings;
    // Positional embeddings for decoder
    private AbstractTensor positionalEmbeddings;

    // Current encoder output (set when transcribe is called)
    private AbstractTensor currentEncoderOutput;

    /**
     * Constructor required by ModelSupport.loadModel
     */
    public WhisperDecoder(
            InferenceType inferenceType,
            Config config,
            WeightLoader weights,
            Tokenizer tokenizer,
            DType workingDType,
            DType workingQType,
            Optional<DType> modelQType) {
        super(inferenceType, config, weights, tokenizer, workingDType, workingQType, modelQType);

        // Get encoder/decoder layer counts from WhisperConfig
        if (config instanceof WhisperConfig wc) {
            this.numberOfEncoderLayers = wc.getEncoderLayers();
            this.numberOfDecoderLayers = wc.getDecoderLayers();
        } else {
            this.numberOfEncoderLayers = config.numberOfLayers;
            this.numberOfDecoderLayers = config.numberOfLayers;
        }

        // Create encoder
        this.encoder = new WhisperEncoder(this, config, weights, numberOfEncoderLayers);
    }

    public WhisperDecoder(Config config, WeightLoader weights, Tokenizer tokenizer,
                          DType workingDType, DType workingQType, Optional<DType> modelQType) {
        this(InferenceType.FULL_GENERATION, config, weights, tokenizer, workingDType, workingQType, modelQType);
    }

    public WhisperDecoder(Config config, WeightLoader weights, Tokenizer tokenizer) {
        this(config, weights, tokenizer, weights.getModelDType(), weights.getModelDType(), Optional.empty());
    }

    @Override
    public ModelSupport.ModelType getModelType() {
        return ModelSupport.getModelType("WHISPER");
    }

    @Override
    protected EmbedInput loadInputWeights() {
        // Load token embeddings
        tokenEmbeddings = weights.load("model.decoder.embed_tokens.weight");
        // Load positional embeddings
        positionalEmbeddings = weights.load("model.decoder.embed_positions.weight");

        return new EmbedInput() {
            @Override
            public AbstractTensor inputTokenToEmbedding(int inputToken, int position) {
                AbstractTensor embedding = makeDenseTensor(1, c.embeddingLength);

                // Copy token embedding
                for (int i = 0; i < c.embeddingLength; i++) {
                    float tokenVal = tokenEmbeddings.get(inputToken, i);
                    float posVal = position < positionalEmbeddings.shape().first()
                            ? positionalEmbeddings.get(position, i) : 0.0f;
                    embedding.set(tokenVal + posVal, 0, i);
                }

                return embedding;
            }

            @Override
            public AbstractTensor batchInputsToEmbeddings(int[] inputTokens, int startPosition) {
                AbstractTensor embeddings = makeDenseTensor(inputTokens.length, c.embeddingLength);

                for (int b = 0; b < inputTokens.length; b++) {
                    int token = inputTokens[b];
                    int position = startPosition + b;

                    for (int i = 0; i < c.embeddingLength; i++) {
                        float tokenVal = tokenEmbeddings.get(token, i);
                        float posVal = position < positionalEmbeddings.shape().first()
                                ? positionalEmbeddings.get(position, i) : 0.0f;
                        embeddings.set(tokenVal + posVal, b, i);
                    }
                }

                return embeddings;
            }
        };
    }

    @Override
    protected TransformerBlock[] loadTransformerBlockWeights() {
        TransformerBlock[] blocks = new TransformerBlock[numberOfDecoderLayers];

        for (int i = 0; i < numberOfDecoderLayers; i++) {
            String prefix = "model.decoder.layers." + i + ".";

            // Self-attention
            CausalSelfAttention selfAttention = new CausalSelfAttention(
                this,
                i,
                loadOptional(prefix + "self_attn.q_proj.bias"),
                loadOptional(prefix + "self_attn.k_proj.bias"),
                loadOptional(prefix + "self_attn.v_proj.bias"),
                weights.load(prefix + "self_attn.q_proj.weight"),
                weights.load(prefix + "self_attn.k_proj.weight"),
                weights.load(prefix + "self_attn.v_proj.weight"),
                loadOptional(prefix + "self_attn.out_proj.bias"),
                weights.load(prefix + "self_attn.out_proj.weight")
            );

            LayerNorm selfAttnNorm = new LayerNorm(
                this,
                weights.load(prefix + "self_attn_layer_norm.weight"),
                weights.load(prefix + "self_attn_layer_norm.bias")
            );

            // Cross-attention
            CrossAttention crossAttention = new CrossAttention(
                this,
                weights.load(prefix + "encoder_attn.q_proj.weight"),
                weights.load(prefix + "encoder_attn.k_proj.weight"),
                weights.load(prefix + "encoder_attn.v_proj.weight"),
                weights.load(prefix + "encoder_attn.out_proj.weight"),
                loadOptionalTensor(prefix + "encoder_attn.q_proj.bias"),
                loadOptionalTensor(prefix + "encoder_attn.k_proj.bias"),
                loadOptionalTensor(prefix + "encoder_attn.v_proj.bias"),
                loadOptionalTensor(prefix + "encoder_attn.out_proj.bias")
            );

            LayerNorm crossAttnNorm = new LayerNorm(
                this,
                weights.load(prefix + "encoder_attn_layer_norm.weight"),
                weights.load(prefix + "encoder_attn_layer_norm.bias")
            );

            // FFN
            FeedForward ffBlock = new WhisperFeedForward(
                this,
                weights.load(prefix + "fc1.weight"),
                weights.load(prefix + "fc1.bias"),
                weights.load(prefix + "fc2.weight"),
                weights.load(prefix + "fc2.bias")
            );

            LayerNorm ffNorm = new LayerNorm(
                this,
                weights.load(prefix + "final_layer_norm.weight"),
                weights.load(prefix + "final_layer_norm.bias")
            );

            blocks[i] = new WhisperDecoderBlock(
                this,
                i,
                selfAttnNorm,
                selfAttention,
                crossAttnNorm,
                crossAttention,
                ffNorm,
                ffBlock,
                () -> currentEncoderOutput
            );
        }

        return blocks;
    }

    private AbstractTensor loadOptional(String name) {
        if (weights.isWeightPresent(name)) {
            return weights.load(name);
        }
        return null;
    }

    private Optional<AbstractTensor> loadOptionalTensor(String name) {
        if (weights.isWeightPresent(name)) {
            return Optional.of(weights.load(name));
        }
        return Optional.empty();
    }

    @Override
    protected SampleOutput loadOutputWeights() {
        LayerNorm outputNorm = new LayerNorm(
            this,
            weights.load("model.decoder.layer_norm.weight"),
            weights.load("model.decoder.layer_norm.bias")
        );

        // Whisper typically ties token embeddings with output projection
        AbstractTensor outputWeights = weights.isWeightPresent("proj_out.weight")
                ? weights.load("proj_out.weight")
                : tokenEmbeddings;

        return new SampleOutput() {
            @Override
            public LayerNorm getOutputLayerNorm() {
                return outputNorm;
            }

            @Override
            public AbstractTensor getOutputLogitsWeights() {
                return outputWeights;
            }
        };
    }

    /**
     * Transcribe audio from mel spectrogram features.
     *
     * @param melSpectrogram Tensor of shape [n_mels, time_frames]
     * @param maxTokens Maximum number of tokens to generate
     * @param temperature Sampling temperature (0 for greedy)
     * @param onToken Callback for each generated token
     * @return Transcribed text
     */
    public String transcribe(AbstractTensor melSpectrogram, int maxTokens,
                             float temperature, BiConsumer<String, Float> onToken) {
        // Reset cross-attention caches
        for (TransformerBlock block : transformerBlocks) {
            if (block instanceof WhisperDecoderBlock) {
                ((WhisperDecoderBlock) block).resetCrossAttentionCache();
            }
        }

        // Run encoder once
        currentEncoderOutput = encoder.encode(melSpectrogram);

        // Standard generation with special start tokens
        // Whisper uses: <|startoftranscript|><|en|><|transcribe|><|notimestamps|>
        // These tokens vary by tokenizer, we'll use a simplified approach
        int startToken = c.bosToken; // Usually <|startoftranscript|>
        int endToken = c.eosTokens.isEmpty() ? -1 : c.eosTokens.get(0);

        StringBuilder result = new StringBuilder();
        UUID sessionId = UUID.randomUUID();

        try (KvBufferCache.KvBuffer kvBuffer = kvBufferCache.getKvBuffer(sessionId);
             AbstractTensor logits = makeDenseTensor(c.vocabularySize)) {

            int position = 0;
            int currentToken = startToken;

            // Generate tokens
            for (int i = 0; i < maxTokens; i++) {
                AbstractTensor embedding = embedInput.inputTokenToEmbedding(currentToken, position);

                // Forward through decoder
                AbstractTensor output = embedding;
                for (TransformerBlock block : transformerBlocks) {
                    AbstractTensor next = block.forward(output, position, kvBuffer);
                    if (output != embedding) output.close();
                    output = next;
                }

                // Sample next token
                int nextToken = sample(output, temperature, ThreadLocalRandom.current().nextFloat(), logits);
                output.close();
                embedding.close();

                // Check for end of sequence
                if (nextToken == endToken) {
                    break;
                }

                // Decode and append
                String decoded = tokenizer.decode(nextToken);
                if (!tokenizer.getModel().isSpecialToken(nextToken)) {
                    result.append(decoded);
                    if (onToken != null) {
                        onToken.accept(decoded, 0.0f);
                    }
                }

                kvBuffer.incrementContextPosition();
                position++;
                currentToken = nextToken;
            }
        }

        // Clean up encoder output
        if (currentEncoderOutput != null) {
            currentEncoderOutput.close();
            currentEncoderOutput = null;
        }

        return result.toString();
    }

    /**
     * Simple FFN for Whisper (uses GELU activation).
     */
    private static class WhisperFeedForward implements FeedForward {
        private final AbstractModel model;
        private final AbstractTensor fc1Weight;
        private final AbstractTensor fc1Bias;
        private final AbstractTensor fc2Weight;
        private final AbstractTensor fc2Bias;

        public WhisperFeedForward(AbstractModel model,
                                   AbstractTensor fc1Weight, AbstractTensor fc1Bias,
                                   AbstractTensor fc2Weight, AbstractTensor fc2Bias) {
            this.model = model;
            this.fc1Weight = fc1Weight;
            this.fc1Bias = fc1Bias;
            this.fc2Weight = fc2Weight;
            this.fc2Bias = fc2Bias;
        }

        @Override
        public AbstractTensor forward(AbstractTensor input, Optional<java.util.function.Consumer<java.util.List<AbstractTensor>>> tensorReducer) {
            int batchSize = input.shape().first();
            int embeddingLength = model.c.embeddingLength;
            int hiddenLength = model.c.hiddenLength;

            // fc1: [batch, emb] -> [batch, hidden]
            AbstractTensor hidden = model.makeDenseTensor(batchSize, hiddenLength);
            VectorMath.pchunk(0, hiddenLength, (chunkStart, chunkLength) -> {
                TensorOperationsProvider.get()
                    .dotProductChunk(hidden, input, fc1Weight, 0, embeddingLength, chunkStart, chunkLength);
            });

            // Add bias and apply GELU
            for (int b = 0; b < batchSize; b++) {
                for (int h = 0; h < hiddenLength; h++) {
                    float val = hidden.get(b, h) + fc1Bias.get(0, h);
                    hidden.set(ActivationFunction.eval(ActivationFunction.Type.GELU, val), b, h);
                }
            }

            // fc2: [batch, hidden] -> [batch, emb]
            AbstractTensor output = model.makeDenseTensor(batchSize, embeddingLength);
            VectorMath.pchunk(0, embeddingLength, (chunkStart, chunkLength) -> {
                TensorOperationsProvider.get()
                    .dotProductChunk(output, hidden, fc2Weight, 0, hiddenLength, chunkStart, chunkLength);
            });
            hidden.close();

            // Add bias
            for (int b = 0; b < batchSize; b++) {
                for (int e = 0; e < embeddingLength; e++) {
                    float val = output.get(b, e) + fc2Bias.get(0, e);
                    output.set(val, b, e);
                }
            }

            return output;
        }
    }
}
