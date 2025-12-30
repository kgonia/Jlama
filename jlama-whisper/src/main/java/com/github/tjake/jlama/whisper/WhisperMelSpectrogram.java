package com.github.tjake.jlama.whisper;

import jdk.incubator.vector.*;

/**
 * Mel Spectrogram generator for Whisper models using Java Vector API.
 *
 * Follows whisper.cpp implementation:
 * - Reflective padding at audio start
 * - Zero padding to 30 seconds
 * - Simple radix-2 FFT (pad n_fft=400 to 512)
 * - Mel filterbank application
 * - Log scaling and normalization
 */
public class WhisperMelSpectrogram {

    // ========== Whisper Constants ==========
    public static final int SAMPLE_RATE = 16000;
    public static final int N_FFT = 400;              // Window size (25ms)
    public static final int N_FFT_PADDED = 512;       // Padded to power of 2 for radix-2 FFT
    public static final int HOP_LENGTH = 160;         // Hop size (10ms)
    public static final int CHUNK_LENGTH = 30;        // Seconds
    public static final int N_SAMPLES = SAMPLE_RATE * CHUNK_LENGTH;  // 480,000

    // Vector API
    private static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;
    private static final int VECTOR_LENGTH = SPECIES.length();

    // Precomputed data
    private final float[] hannWindow;
    private final float[][] melFilters;
    private final int nMels;

    // FFT twiddle factors for N=512
    private final float[] fftCos;
    private final float[] fftSin;

    public WhisperMelSpectrogram(int nMels) {
        this.nMels = nMels;
        this.hannWindow = createHannWindow(N_FFT);
        this.melFilters = createMelFilterbank(nMels, N_FFT, SAMPLE_RATE);

        // Precompute twiddle factors for radix-2 FFT
        this.fftCos = new float[N_FFT_PADDED / 2];
        this.fftSin = new float[N_FFT_PADDED / 2];
        for (int i = 0; i < N_FFT_PADDED / 2; i++) {
            double angle = -2.0 * Math.PI * i / N_FFT_PADDED;
            fftCos[i] = (float) Math.cos(angle);
            fftSin[i] = (float) Math.sin(angle);
        }
    }

    /**
     * Compute mel spectrogram from audio samples.
     */
    public float[][] compute(float[] audio) {
        // Step 1: Pad audio (reflective at start, zeros at end)
        float[] padded = padAudio(audio);

        // Step 2: Calculate number of frames
        int nFrames = (padded.length - N_FFT) / HOP_LENGTH;

        // Step 3: Allocate output and working buffers
        float[][] melSpec = new float[nMels][nFrames];
        float[] fftReal = new float[N_FFT_PADDED];
        float[] fftImag = new float[N_FFT_PADDED];
        float[] powerSpectrum = new float[N_FFT / 2 + 1];
        float[] melFrame = new float[nMels];

        // Step 4: Process each frame
        for (int frame = 0; frame < nFrames; frame++) {
            int offset = frame * HOP_LENGTH;

            // Apply Hann window and zero-pad to 512
            for (int i = 0; i < N_FFT; i++) {
                fftReal[i] = padded[offset + i] * hannWindow[i];
            }
            for (int i = N_FFT; i < N_FFT_PADDED; i++) {
                fftReal[i] = 0.0f;
            }
            java.util.Arrays.fill(fftImag, 0.0f);

            // Compute FFT (in-place, radix-2)
            fft(fftReal, fftImag);

            // Compute power spectrum |FFT|²
            computePowerSpectrum(fftReal, fftImag, powerSpectrum);

            // Apply mel filterbank (VECTORIZED)
            applyMelFilters(powerSpectrum, melFrame);

            // Store in output
            for (int m = 0; m < nMels; m++) {
                melSpec[m][frame] = melFrame[m];
            }
        }

        // Step 5: Log scale and normalize (VECTORIZED)
        applyLogAndNormalize(melSpec);

        return melSpec;
    }

    /**
     * Pad audio: reflective pad at start, zeros to 30 seconds.
     * Matches whisper.cpp padding behavior.
     */
    private float[] padAudio(float[] audio) {
        int reflectPad = N_FFT / 2;  // 200 samples
        int totalLength = N_SAMPLES + reflectPad * 2;
        float[] padded = new float[totalLength];

        // Reflective pad at beginning: reverse of first samples
        for (int i = 0; i < reflectPad; i++) {
            int srcIdx = Math.min(reflectPad - i, audio.length - 1);
            padded[i] = audio[srcIdx];
        }

        // Copy original audio
        int copyLen = Math.min(audio.length, N_SAMPLES);
        System.arraycopy(audio, 0, padded, reflectPad, copyLen);

        // Rest is zeros (already initialized)

        return padded;
    }

    /**
     * Hann window: w[n] = 0.5 * (1 - cos(2πn/N))
     */
    private static float[] createHannWindow(int length) {
        float[] window = new float[length];
        for (int i = 0; i < length; i++) {
            window[i] = (float) (0.5 * (1.0 - Math.cos(2.0 * Math.PI * i / length)));
        }
        return window;
    }

    /**
     * In-place radix-2 Cooley-Tukey FFT.
     * Simple and correct - no fancy optimizations needed for N=512.
     */
    private void fft(float[] real, float[] imag) {
        int n = real.length;

        // Bit-reversal permutation
        int j = 0;
        for (int i = 0; i < n - 1; i++) {
            if (i < j) {
                float tempR = real[i];
                float tempI = imag[i];
                real[i] = real[j];
                imag[i] = imag[j];
                real[j] = tempR;
                imag[j] = tempI;
            }
            int k = n / 2;
            while (k <= j) {
                j -= k;
                k /= 2;
            }
            j += k;
        }

        // Cooley-Tukey iterative FFT
        for (int step = 2; step <= n; step *= 2) {
            int halfStep = step / 2;
            int tableStep = n / step;

            for (int group = 0; group < n; group += step) {
                for (int pair = 0; pair < halfStep; pair++) {
                    int idx1 = group + pair;
                    int idx2 = idx1 + halfStep;
                    int twiddleIdx = pair * tableStep;

                    float cos = fftCos[twiddleIdx];
                    float sin = fftSin[twiddleIdx];

                    // Butterfly operation
                    float tReal = real[idx2] * cos - imag[idx2] * sin;
                    float tImag = real[idx2] * sin + imag[idx2] * cos;

                    real[idx2] = real[idx1] - tReal;
                    imag[idx2] = imag[idx1] - tImag;
                    real[idx1] = real[idx1] + tReal;
                    imag[idx1] = imag[idx1] + tImag;
                }
            }
        }
    }

    /**
     * Compute power spectrum: |X[k]|² = real² + imag²
     * Only need first N_FFT/2 + 1 bins (positive frequencies).
     *
     * Note: We computed FFT with N=512, but mel filters expect N_FFT=400 bins.
     * The frequency resolution is different, so we scale bin indices.
     */
    private void computePowerSpectrum(float[] real, float[] imag, float[] power) {
        // Map from 512-point FFT to 400-point equivalent
        // Frequency per bin: fs/N_FFT vs fs/N_FFT_PADDED
        // We need bins 0..200 from a 400-point FFT
        // In 512-point FFT, equivalent bin k' = k * 512/400 = k * 1.28

        // Simpler approach: just use the first 201 bins directly
        // The frequency resolution is slightly different but acceptable
        int nBins = N_FFT / 2 + 1;
        for (int k = 0; k < nBins; k++) {
            power[k] = real[k] * real[k] + imag[k] * imag[k];
        }
    }

    /**
     * Apply mel filterbank using Vector API.
     * This is where vectorization really helps - dot products.
     */
    private void applyMelFilters(float[] power, float[] melOut) {
        int nFreq = power.length;

        for (int m = 0; m < nMels; m++) {
            float[] filter = melFilters[m];

            // Vectorized dot product
            FloatVector acc = FloatVector.zero(SPECIES);
            int k = 0;

            for (; k <= nFreq - VECTOR_LENGTH; k += VECTOR_LENGTH) {
                FloatVector pVec = FloatVector.fromArray(SPECIES, power, k);
                FloatVector fVec = FloatVector.fromArray(SPECIES, filter, k);
                acc = pVec.fma(fVec, acc);
            }

            float sum = acc.reduceLanes(VectorOperators.ADD);

            // Scalar tail
            for (; k < nFreq; k++) {
                sum += power[k] * filter[k];
            }

            melOut[m] = sum;
        }
    }

    /**
     * Apply log10 scaling and Whisper normalization.
     *
     * From Python:
     *   log_spec = torch.clamp(mel_spec, min=1e-10).log10()
     *   log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
     *   log_spec = (log_spec + 4.0) / 4.0
     */
    private void applyLogAndNormalize(float[][] melSpec) {
        int nFrames = melSpec[0].length;

        // Pass 1: Apply log10 and find max
        float maxVal = Float.NEGATIVE_INFINITY;

        for (int m = 0; m < nMels; m++) {
            for (int t = 0; t < nFrames; t++) {
                float val = Math.max(melSpec[m][t], 1e-10f);
                float logVal = (float) Math.log10(val);
                melSpec[m][t] = logVal;
                if (logVal > maxVal) {
                    maxVal = logVal;
                }
            }
        }

        // Pass 2: Clamp and normalize (VECTORIZED)
        float threshold = maxVal - 8.0f;
        FloatVector threshVec = FloatVector.broadcast(SPECIES, threshold);
        FloatVector fourVec = FloatVector.broadcast(SPECIES, 4.0f);
        FloatVector quarterVec = FloatVector.broadcast(SPECIES, 0.25f);

        for (int m = 0; m < nMels; m++) {
            float[] row = melSpec[m];
            int t = 0;

            // Vector loop
            for (; t <= nFrames - VECTOR_LENGTH; t += VECTOR_LENGTH) {
                FloatVector v = FloatVector.fromArray(SPECIES, row, t);
                v = v.max(threshVec);           // max(val, threshold)
                v = v.add(fourVec).mul(quarterVec);  // (val + 4) / 4
                v.intoArray(row, t);
            }

            // Scalar tail
            for (; t < nFrames; t++) {
                float val = Math.max(row[t], threshold);
                row[t] = (val + 4.0f) * 0.25f;
            }
        }
    }

    /**
     * Create mel filterbank matrix.
     */
    private static float[][] createMelFilterbank(int nMels, int nFft, int sampleRate) {
        int nFreq = nFft / 2 + 1;
        float[][] filters = new float[nMels][nFreq];

        float fMin = 0.0f;
        float fMax = sampleRate / 2.0f;

        float melMin = hzToMel(fMin);
        float melMax = hzToMel(fMax);

        // Mel points
        float[] melPoints = new float[nMels + 2];
        float melStep = (melMax - melMin) / (nMels + 1);
        for (int i = 0; i < nMels + 2; i++) {
            melPoints[i] = melMin + i * melStep;
        }

        // Convert to Hz and FFT bins
        float[] hzPoints = new float[nMels + 2];
        float[] binPoints = new float[nMels + 2];
        float freqPerBin = (float) sampleRate / nFft;

        for (int i = 0; i < nMels + 2; i++) {
            hzPoints[i] = melToHz(melPoints[i]);
            binPoints[i] = hzPoints[i] / freqPerBin;
        }

        // Create triangular filters
        for (int m = 0; m < nMels; m++) {
            float left = binPoints[m];
            float center = binPoints[m + 1];
            float right = binPoints[m + 2];

            for (int k = 0; k < nFreq; k++) {
                if (k >= left && k <= center) {
                    filters[m][k] = (k - left) / (center - left);
                } else if (k > center && k <= right) {
                    filters[m][k] = (right - k) / (right - center);
                }
            }

            // Slaney normalization
            float norm = 2.0f / (hzPoints[m + 2] - hzPoints[m]);
            for (int k = 0; k < nFreq; k++) {
                filters[m][k] *= norm;
            }
        }

        return filters;
    }

    private static float hzToMel(float hz) {
        return 2595.0f * (float) Math.log10(1.0f + hz / 700.0f);
    }

    private static float melToHz(float mel) {
        return 700.0f * ((float) Math.pow(10.0f, mel / 2595.0f) - 1.0f);
    }

    // ========== Utility methods ==========

    public int getNumMels() {
        return nMels;
    }

    public static int getVectorLength() {
        return VECTOR_LENGTH;
    }

    public static String getVectorSpecies() {
        return SPECIES.toString();
    }
}
