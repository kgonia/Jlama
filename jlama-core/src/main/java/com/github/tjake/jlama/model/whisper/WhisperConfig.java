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
package com.github.tjake.jlama.model.whisper;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.github.tjake.jlama.math.ActivationFunction;
import com.github.tjake.jlama.safetensors.Config;

import java.util.List;

/**
 * Configuration for Whisper speech-to-text models.
 */
public class WhisperConfig extends Config {

    public final int encoderLayers;
    public final int decoderLayers;
    public final int encoderAttentionHeads;
    public final int decoderAttentionHeads;
    public final int encoderFfnDim;
    public final int decoderFfnDim;
    public final int decoderStartTokenId;

    @JsonCreator
    public WhisperConfig(
            @JsonProperty("max_target_positions") int contextLength,
            @JsonProperty("d_model") int embeddingLength,
            @JsonProperty("decoder_ffn_dim") int hiddenLength,
            @JsonProperty("decoder_attention_heads") int numberOfHeads,
            @JsonProperty("encoder_layers") int encoderLayers,
            @JsonProperty("decoder_layers") int decoderLayers,
            @JsonProperty("encoder_attention_heads") int encoderAttentionHeads,
            @JsonProperty("encoder_ffn_dim") int encoderFfnDim,
            @JsonProperty("vocab_size") int vocabularySize,
            @JsonProperty("bos_token_id") int bosToken,
            @JsonProperty("eos_token_id") Object eosToken,
            @JsonProperty("decoder_start_token_id") Integer decoderStartTokenId,
            @JsonProperty("activation_function") String activationFunction,
            @JsonProperty("num_mel_bins") Integer numMelBins,
            @JsonProperty("max_source_positions") Integer maxSourcePositions,
            @JsonProperty("is_encoder_decoder") Boolean isEncoderDecoder
    ) {
        super(
            contextLength,
            embeddingLength,
            hiddenLength,
            numberOfHeads,
            numberOfHeads, // numberOfKeyValueHeads same as numberOfHeads for Whisper
            decoderLayers, // Use decoder layers as the primary layer count
            1e-5f, // Whisper uses default layer norm eps
            vocabularySize,
            bosToken,
            eosToken instanceof List<?> ? (List<Integer>) eosToken : List.of((Integer) eosToken),
            parseActivation(activationFunction),
            null, // No RoPE for Whisper (uses sinusoidal/learned embeddings)
            null,
            null,
            embeddingLength / numberOfHeads,
            null,
            null,
            null,
            null,
            null,
            null,
            numMelBins,
            maxSourcePositions,
            isEncoderDecoder
        );

        this.encoderLayers = encoderLayers;
        this.decoderLayers = decoderLayers;
        this.encoderAttentionHeads = encoderAttentionHeads;
        this.decoderAttentionHeads = numberOfHeads;
        this.encoderFfnDim = encoderFfnDim;
        this.decoderFfnDim = hiddenLength;
        this.decoderStartTokenId = decoderStartTokenId != null ? decoderStartTokenId : bosToken;
    }

    private static ActivationFunction.Type parseActivation(String activation) {
        if (activation == null) return ActivationFunction.Type.GELU;
        return switch (activation.toLowerCase()) {
            case "gelu" -> ActivationFunction.Type.GELU;
            case "silu", "swish" -> ActivationFunction.Type.SILU;
            case "tanh" -> ActivationFunction.Type.TANH;
            default -> ActivationFunction.Type.GELU;
        };
    }

    public int getEncoderLayers() {
        return encoderLayers;
    }

    public int getDecoderLayers() {
        return decoderLayers;
    }

    public int getDecoderStartTokenId() {
        return decoderStartTokenId;
    }
}
