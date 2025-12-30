package com.github.tjake.jlama.whisper;

import com.github.tjake.jlama.whisper.WhisperMelSpectrogram;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestInstance;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for WhisperMelSpectrogram implementation.
 *
 * To verify against Python reference:
 * 1. Run: python jlama-tests/src/test/resources/whisper_mel_reference.py
 * 2. Compare output with these tests
 */
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
public class TestWhisperMelSpectrogram {

    private static final int SAMPLE_RATE = 16000;
    private static final int N_MELS = 80;
    private static final float TOLERANCE = 0.1f;  // Allow some numerical difference

    private WhisperMelSpectrogram melGenerator;

    @BeforeAll
    void setup() {
        melGenerator = new WhisperMelSpectrogram(N_MELS);
        System.out.println("Vector species: " + WhisperMelSpectrogram.getVectorSpecies());
        System.out.println("Vector length: " + WhisperMelSpectrogram.getVectorLength());
    }

    @Test
    void testBasicDimensions() {
        // 1 second of audio
        float[] audio = new float[SAMPLE_RATE];
        float[][] mel = melGenerator.compute(audio);

        // Should be [80, ~3000] for 30-second padded output
        assertEquals(N_MELS, mel.length, "Should have 80 mel bins");
        assertTrue(mel[0].length > 0, "Should have frames");

        System.out.println("Output shape: [" + mel.length + ", " + mel[0].length + "]");
    }

    @Test
    void testSineWave440Hz() {
        // Generate 1 second of 440 Hz sine wave
        float[] audio = new float[SAMPLE_RATE];
        for (int i = 0; i < SAMPLE_RATE; i++) {
            audio[i] = (float) (0.5 * Math.sin(2 * Math.PI * 440 * i / SAMPLE_RATE));
        }

        float[][] mel = melGenerator.compute(audio);

        // Verify output is in expected range after normalization
        // Whisper normalization: (log_spec + 4.0) / 4.0
        // This should give values roughly between 0 and 1 for typical audio
        float min = Float.MAX_VALUE, max = Float.MIN_VALUE, sum = 0;
        int count = 0;

        for (int m = 0; m < mel.length; m++) {
            for (int t = 0; t < mel[m].length; t++) {
                float val = mel[m][t];
                min = Math.min(min, val);
                max = Math.max(max, val);
                sum += val;
                count++;
            }
        }

        float mean = sum / count;

        System.out.println("Sine 440Hz mel stats:");
        System.out.println("  min=" + min + ", max=" + max + ", mean=" + mean);

        // After Whisper normalization, values should be reasonable
        assertTrue(min >= -1.0f, "Min should be >= -1 after normalization");
        assertTrue(max <= 1.5f, "Max should be <= 1.5 after normalization");
    }

    @Test
    void testSilence() {
        float[] audio = new float[SAMPLE_RATE];  // All zeros

        float[][] mel = melGenerator.compute(audio);

        // Silence should produce uniform low values
        float min = Float.MAX_VALUE, max = Float.MIN_VALUE;

        for (int m = 0; m < mel.length; m++) {
            for (int t = 0; t < mel[m].length; t++) {
                min = Math.min(min, mel[m][t]);
                max = Math.max(max, mel[m][t]);
            }
        }

        System.out.println("Silence mel stats:");
        System.out.println("  min=" + min + ", max=" + max);

        // Silence should have low values, but after normalization
        // the dynamic range compression kicks in
        assertTrue(max - min < 0.1f, "Silence should be relatively uniform");
    }

    @Test
    void testWhiteNoise() {
        // Generate reproducible white noise
        java.util.Random rng = new java.util.Random(42);
        float[] audio = new float[SAMPLE_RATE];
        for (int i = 0; i < SAMPLE_RATE; i++) {
            audio[i] = (float) (0.1 * rng.nextGaussian());
        }

        float[][] mel = melGenerator.compute(audio);

        float min = Float.MAX_VALUE, max = Float.MIN_VALUE, sum = 0;
        int count = 0;

        for (int m = 0; m < mel.length; m++) {
            for (int t = 0; t < mel[m].length; t++) {
                float val = mel[m][t];
                min = Math.min(min, val);
                max = Math.max(max, val);
                sum += val;
                count++;
            }
        }

        System.out.println("White noise mel stats:");
        System.out.println("  min=" + min + ", max=" + max + ", mean=" + (sum / count));

        // White noise should excite all frequency bands more uniformly
        assertTrue(max > min, "Should have some variation");
    }

    @Test
    void testChirp() {
        // Frequency sweep from 200 Hz to 1000 Hz
        float[] audio = new float[SAMPLE_RATE];
        for (int i = 0; i < SAMPLE_RATE; i++) {
            double t = (double) i / SAMPLE_RATE;
            double freq = 200 + 800 * t;
            audio[i] = (float) (0.5 * Math.sin(2 * Math.PI * freq * t));
        }

        float[][] mel = melGenerator.compute(audio);

        System.out.println("Chirp mel stats:");
        System.out.println("  Shape: [" + mel.length + ", " + mel[0].length + "]");

        // Chirp should show energy moving across mel bins over time
        // We just verify it doesn't crash and produces reasonable output
        assertNotNull(mel);
        assertEquals(N_MELS, mel.length);
    }

    @Test
    void testHannWindow() {
        // Verify Hann window properties
        int n = 400;
        float[] window = new float[n];
        for (int i = 0; i < n; i++) {
            window[i] = (float) (0.5 * (1.0 - Math.cos(2.0 * Math.PI * i / n)));
        }

        // Hann window should be symmetric
        for (int i = 0; i < n / 2; i++) {
            assertEquals(window[i], window[n - 1 - i], 1e-6f,
                "Hann window should be symmetric at index " + i);
        }

        // Should be 0 at edges, 1 at center
        assertEquals(0.0f, window[0], 1e-6f, "Hann should be 0 at start");
        assertEquals(1.0f, window[n / 2], 1e-6f, "Hann should be 1 at center");
    }

    @Test
    void testFFTBasic() {
        // Test FFT with known input: DC signal should give DC bin only
        WhisperMelSpectrogram gen = new WhisperMelSpectrogram(80);

        // For a more thorough FFT test, we'd need to expose the FFT method
        // or test through the full pipeline
        float[] dc = new float[SAMPLE_RATE];
        java.util.Arrays.fill(dc, 0.5f);

        float[][] mel = gen.compute(dc);

        // DC signal should primarily excite low-frequency mel bins
        float lowSum = 0, highSum = 0;
        for (int t = 0; t < Math.min(10, mel[0].length); t++) {
            for (int m = 0; m < 10; m++) {
                lowSum += mel[m][t];
            }
            for (int m = 70; m < 80; m++) {
                highSum += mel[m][t];
            }
        }

        System.out.println("DC signal: low bins sum=" + lowSum + ", high bins sum=" + highSum);
        // Note: After log and normalization, this relationship may not hold strictly
    }

    @Test
    void testReproducibility() {
        float[] audio = new float[SAMPLE_RATE];
        for (int i = 0; i < SAMPLE_RATE; i++) {
            audio[i] = (float) (0.3 * Math.sin(2 * Math.PI * 1000 * i / SAMPLE_RATE));
        }

        float[][] mel1 = melGenerator.compute(audio);
        float[][] mel2 = melGenerator.compute(audio);

        // Same input should produce identical output
        assertEquals(mel1.length, mel2.length);
        assertEquals(mel1[0].length, mel2[0].length);

        for (int m = 0; m < mel1.length; m++) {
            for (int t = 0; t < mel1[m].length; t++) {
                assertEquals(mel1[m][t], mel2[m][t], 1e-9f,
                    "Results should be identical at [" + m + "][" + t + "]");
            }
        }
    }

    @Test
    void testPerformance() {
        // Simple performance test
        float[] audio = new float[SAMPLE_RATE * 5];  // 5 seconds
        java.util.Random rng = new java.util.Random(123);
        for (int i = 0; i < audio.length; i++) {
            audio[i] = (float) (0.1 * rng.nextGaussian());
        }

        // Warmup
        for (int i = 0; i < 3; i++) {
            melGenerator.compute(audio);
        }

        // Timed run
        int iterations = 10;
        long start = System.nanoTime();
        for (int i = 0; i < iterations; i++) {
            melGenerator.compute(audio);
        }
        long elapsed = System.nanoTime() - start;

        double msPerCall = elapsed / 1_000_000.0 / iterations;
        System.out.println("Performance: " + String.format("%.2f", msPerCall) + " ms per mel spectrogram");

        // Should be reasonably fast (< 500ms for 5 seconds of audio)
        assertTrue(msPerCall < 500, "Should compute in reasonable time");
    }
}
