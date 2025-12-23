# Whisper Implementation Test Plan

## Overview

The Whisper model is producing incorrect output. This document outlines a systematic testing strategy to identify and fix issues in the implementation.

## Potential Issues Identified

### 1. Audio Preprocessing (High Priority)
**File:** `jlama-whisper/src/main/java/com/github/tjake/jlama/whisper/WhisperAudioPreprocessor.java`

- **N_FFT mismatch**: Using `512` instead of Whisper's expected `400`
- **Log function**: Using `Math.log10()` instead of natural log (`Math.log()`)
- **JLibrosa compatibility**: May produce mel spectrograms in different format than Whisper expects
- **Normalization**: Whisper applies specific normalization that may be missing

### 2. Encoder (Medium Priority)
**File:** `jlama-core/src/main/java/com/github/tjake/jlama/model/WhisperEncoder.java`

- Manual Conv1D implementation could have subtle bugs
- Positional embedding shape/indexing
- Layer norm application order

### 3. Decoder (Medium Priority)
**File:** `jlama-core/src/main/java/com/github/tjake/jlama/model/WhisperDecoder.java`

- Special token handling (SOT, language, task, notimestamps tokens)
- Cross-attention cache management
- Output projection (tied embeddings)

### 4. Attention Mechanisms (Lower Priority)
**Files:** `CrossAttention.java`, `BidirectionalSelfAttention.java`

- Attention scaling factor
- Bias tensor dimensions (loading as `[1, N]` vs `[N]`)

---

## Testing Phases

### Phase 1: Audio Preprocessing Validation

**Goal:** Verify mel spectrogram output matches OpenAI Whisper reference

**Test Cases:**

1. **Test 1.1: Mel Spectrogram Shape**
   - Input: 30 seconds of audio at 16kHz
   - Expected output shape: `[80, 3000]` (80 mels, 3000 time frames)
   - Verify dimensions match

2. **Test 1.2: Mel Spectrogram Values**
   - Generate mel spectrogram for `whisper-test.wav`
   - Compare first 10 values of first mel band against Python reference
   - Compare statistics (min, max, mean, std)

3. **Test 1.3: Log Transform**
   - Verify using natural log, not log10
   - Check clamping value (should be 1e-10 or similar)

**Python Reference Script:**
```python
import whisper
import numpy as np

# Load audio and get mel spectrogram
audio = whisper.load_audio("whisper-test.wav")
audio = whisper.pad_or_trim(audio)
mel = whisper.log_mel_spectrogram(audio)

print(f"Shape: {mel.shape}")
print(f"First 10 values [0]: {mel[0, :10]}")
print(f"Min: {mel.min()}, Max: {mel.max()}, Mean: {mel.mean()}")
np.save("reference_mel.npy", mel.numpy())
```

---

### Phase 2: Encoder Conv1D Validation

**Goal:** Verify convolutional layers produce correct output

**Test Cases:**

1. **Test 2.1: Conv1D Output Shape**
   - Input: `[80, 3000]` mel spectrogram
   - After conv1: `[d_model, 3000]`
   - After conv2 (stride=2): `[d_model, 1500]`

2. **Test 2.2: Conv1D Values with Synthetic Input**
   - Create tensor with known values (e.g., all 1.0)
   - Manually compute expected convolution output
   - Compare against implementation

3. **Test 2.3: GELU Activation**
   - Verify GELU implementation matches reference
   - Test edge cases (negative values, zero, large positive)

---

### Phase 3: Encoder Full Validation

**Goal:** Verify complete encoder output matches reference

**Test Cases:**

1. **Test 3.1: Positional Embeddings**
   - Verify shape: `[1500, d_model]`
   - Compare first few embedding vectors against reference

2. **Test 3.2: Encoder Block Output**
   - Run single encoder block with synthetic input
   - Compare layer norm, attention, FFN outputs

3. **Test 3.3: Full Encoder Output**
   - Compare encoder hidden states against Python reference
   - Use same mel spectrogram input

**Python Reference:**
```python
import whisper
model = whisper.load_model("tiny")

mel = whisper.log_mel_spectrogram(audio).unsqueeze(0)
with torch.no_grad():
    encoder_output = model.encoder(mel)

print(f"Encoder output shape: {encoder_output.shape}")
print(f"First 10 values [0,0]: {encoder_output[0, 0, :10]}")
np.save("reference_encoder_output.npy", encoder_output.numpy())
```

---

### Phase 4: Decoder Cross-Attention Validation

**Goal:** Verify decoder correctly attends to encoder output

**Test Cases:**

1. **Test 4.1: Token Embeddings**
   - Verify SOT token embedding matches reference
   - Check positional embedding addition

2. **Test 4.2: Cross-Attention Output**
   - With fixed encoder output and decoder query
   - Compare attention weights and output

3. **Test 4.3: Decoder Block Output**
   - Run single decoder block
   - Compare against reference

---

### Phase 5: End-to-End Validation

**Goal:** Verify complete transcription pipeline

**Test Cases:**

1. **Test 5.1: First Token Logits**
   - Run forward pass for first decoder step
   - Compare output logits against reference
   - Verify top-k token predictions match

2. **Test 5.2: Greedy Decoding**
   - Generate tokens with temperature=0
   - Compare token sequence against reference

3. **Test 5.3: Full Transcription**
   - Transcribe test audio file
   - Compare against expected text

---

## Implementation Priority

### Immediate Actions (Phase 1)

1. **Fix known issues in WhisperAudioPreprocessor:**
   ```java
   // Change from:
   melSpectrogram[i][j] = (float) Math.log10(val);
   // To:
   melSpectrogram[i][j] = (float) Math.log(val);
   ```

2. **Create WhisperAudioPreprocessorTest:**
   - Load reference mel spectrogram from Python
   - Compare against Java implementation

### Test File Structure

```
jlama-tests/src/test/java/com/github/tjake/jlama/whisper/
├── WhisperAudioPreprocessorTest.java
├── WhisperEncoderTest.java
├── WhisperDecoderTest.java
└── WhisperEndToEndTest.java

jlama-tests/src/test/resources/
├── whisper-test.wav
├── reference_mel.npy (generate from Python)
├── reference_encoder_output.npy
└── reference_logits.npy
```

---

## Reference Values Generation

To generate reference values, run this Python script with OpenAI Whisper:

```python
import whisper
import numpy as np
import torch

# Load model
model = whisper.load_model("tiny")  # or "base", "small", etc.

# Load and preprocess audio
audio = whisper.load_audio("whisper-test.wav")
audio = whisper.pad_or_trim(audio)

# Get mel spectrogram
mel = whisper.log_mel_spectrogram(audio)
print(f"Mel shape: {mel.shape}")
np.save("reference_mel.npy", mel.numpy())

# Get encoder output
mel_input = mel.unsqueeze(0).to(model.device)
with torch.no_grad():
    encoder_output = model.encoder(mel_input)
print(f"Encoder output shape: {encoder_output.shape}")
np.save("reference_encoder_output.npy", encoder_output.cpu().numpy())

# Get first decoder step logits
tokens = torch.tensor([[model.tokenizer.sot]]).to(model.device)
with torch.no_grad():
    logits = model.decoder(tokens, encoder_output)
print(f"Logits shape: {logits.shape}")
np.save("reference_logits.npy", logits.cpu().numpy())

# Full transcription
result = model.transcribe("whisper-test.wav")
print(f"Transcription: {result['text']}")
with open("reference_transcription.txt", "w") as f:
    f.write(result['text'])
```

---

## Success Criteria

Each phase is complete when:

1. **Phase 1:** Mel spectrogram values match within tolerance (e.g., max diff < 0.01)
2. **Phase 2:** Conv1D output matches within tolerance
3. **Phase 3:** Encoder output matches within tolerance
4. **Phase 4:** Cross-attention output matches within tolerance
5. **Phase 5:** Transcription matches reference text

---

## Notes

- Use `whisper-tiny` model for faster testing
- All tensor comparisons should use relative tolerance due to floating point differences
- Log intermediate values during debugging
- Consider adding assertions for tensor shapes throughout the pipeline
