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
package com.github.tjake.jlama.cli.commands;

import com.github.tjake.jlama.model.AbstractModel;
import com.github.tjake.jlama.model.WhisperDecoder;
import com.github.tjake.jlama.whisper.WhisperModel;
import com.github.tjake.jlama.safetensors.SafeTensorSupport;
import picocli.CommandLine.Command;
import picocli.CommandLine.Option;
import picocli.CommandLine.Parameters;

import java.io.File;
import java.io.PrintWriter;
import java.nio.file.Path;
import java.util.Optional;

import static com.github.tjake.jlama.model.ModelSupport.loadModel;

@Command(name = "transcribe", description = "Transcribe an audio file using a Whisper model")
public class TranscribeCommand extends ModelBaseCommand {

    @Parameters(index = "1", description = "The audio file to transcribe")
    private File audioFile;

    @Option(names = {"--max-tokens"}, description = "Maximum number of tokens to generate (default: ${DEFAULT-VALUE})", defaultValue = "256")
    private int maxTokens;

    @Override
    public void run() {
        Path modelPath = getModel(
            requireModelId(),
            modelDirectory,
            downloadSection.autoDownload,
            downloadSection.branch,
            downloadSection.authToken
        );

        try {
            PrintWriter out = System.console() != null ? System.console().writer() : new PrintWriter(System.out);

            out.println("Loading Whisper model from " + modelPath);
            out.flush();

            // Load the model - it should be a WhisperDecoder
            AbstractModel m = loadModel(
                modelPath.toFile(),
                workingDirectory,
                advancedSection.workingMemoryType,
                advancedSection.workingQuantizationType,
                Optional.ofNullable(advancedSection.modelQuantization),
                Optional.ofNullable(advancedSection.threadCount)
            );

            if (!(m instanceof WhisperDecoder)) {
                System.err.println("This model is not a Whisper model");
                System.exit(1);
            }

            WhisperModel whisperModel = new WhisperModel(
                m.getConfig(),
                SafeTensorSupport.loadWeights(modelPath.toFile()),
                m.getTokenizer()
            );

            out.println("Transcribing " + audioFile.getAbsolutePath());
            out.println();
            out.flush();

            String transcript = whisperModel.transcribe(
                audioFile.getAbsolutePath(),
                maxTokens,
                temperature,
                (token, timing) -> {
                    out.print(token);
                    out.flush();
                }
            );

            out.println();
            out.println();
            out.println("Full transcript: " + transcript);
            out.flush();

        } catch (Exception e) {
            System.err.println("Error transcribing audio: " + e.getMessage());
            e.printStackTrace();
            System.exit(1);
        }
    }
}
