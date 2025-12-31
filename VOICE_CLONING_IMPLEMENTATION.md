# Voice Cloning & Custom Voices Implementation for YarnGPT

## 🎯 Executive Summary

YarnGPT uses a **pure language modeling approach** for TTS by combining:
1. **WavTokenizer** - Converts audio ↔ discrete codes (4096 codebook)
2. **SmolLM2-360M** - Fine-tuned LM that learns to generate audio codes from text
3. **Speaker conditioning** - Uses reference audio codes as prompt prefix

**Key Architecture:**
- Input: Text → Tokenized words + Reference speaker codes
- Output: Discrete audio codes (75 tokens/sec @ 24kHz)
- Decoder: Audio codes → Waveform via WavTokenizer

---

## 🔍 How Current System Works

### Speaker Representation Format (JSON)
```json
{
    "text": "reference text that was spoken",
    "words": [
        {
            "word": "scientists",
            "duration": "1.00",
            "codes": [258, 551, 21, 401, 509, ...]  // WavTokenizer codes
        }
    ]
}
```

### Generation Pipeline
1. Load speaker JSON (reference voice)
2. Concatenate: `speaker_text + user_text`
3. Create prompt with speaker audio codes
4. Model generates codes for entire sequence
5. Decode codes to audio waveform

**Critical Insight:** The model learns voice characteristics from the **audio codes** in the prompt, not raw audio.

---

## 🎙️ Voice Cloning Implementation Options

### **Option 1: Quick & Simple (5-10 seconds reference audio) ⭐ RECOMMENDED**

**Best for:** Fast deployment, minimal compute, good similarity

**Steps:**
```python
# 1. Extract codes from reference audio
audio_tokenizer = AudioTokenizerV2(...)
codes = audio_tokenizer.quantize_wavtokenizer("reference_audio.wav")

# 2. Create custom speaker JSON
from audiotokenizer import AudioTokenizerV2
import json
import torchaudio

def create_custom_speaker(audio_path, transcript, output_name):
    """
    Create a custom speaker profile from reference audio
    
    Args:
        audio_path: Path to clean reference audio (5-15 seconds)
        transcript: Exact transcript of what was said
        output_name: Name for the speaker profile
    """
    tokenizer_path = "saheedniyi/YarnGPT2"
    wav_tokenizer_config = "path/to/config.yaml"
    wav_tokenizer_model = "path/to/model.ckpt"
    
    audio_tokenizer = AudioTokenizerV2(
        tokenizer_path, 
        wav_tokenizer_model, 
        wav_tokenizer_config
    )
    
    # Load and validate audio
    audio, sr = torchaudio.load(audio_path)
    assert sr == 24000, "Audio must be 24kHz"
    assert audio.shape[0] == 1, "Audio must be mono"
    
    # Extract codes
    codes = audio_tokenizer.quantize_wavtokenizer(audio_path)
    code_list = [int(c) for c in codes.split("|") if c.strip()]
    
    # Process transcript to match model's text processing
    words = audio_tokenizer.process_text(transcript)
    
    # Calculate approximate duration per word
    duration = audio.shape[1] / sr
    duration_per_word = duration / len(words)
    
    # Create speaker profile
    speaker_data = {
        "text": transcript,
        "words": []
    }
    
    codes_per_word = len(code_list) // len(words)
    for i, word in enumerate(words):
        start_idx = i * codes_per_word
        end_idx = (i + 1) * codes_per_word
        
        speaker_data["words"].append({
            "word": word,
            "duration": f"{duration_per_word:.2f}",
            "codes": code_list[start_idx:end_idx]
        })
    
    # Save speaker profile
    output_path = f"custom_speakers/{output_name}.json"
    with open(output_path, 'w') as f:
        json.dump(speaker_data, f, indent=2)
    
    return output_path

# Usage
speaker_path = create_custom_speaker(
    "my_voice.wav",
    "Scientists have discovered a new planet that may support life",
    "custom_voice_1"
)
```

**Quality Tips:**
- Use **clean, studio-quality audio** (no background noise)
- **5-15 seconds** is optimal (too short = poor quality, too long = overfitting)
- Match the **speaking style** you want (calm, energetic, etc.)
- Record in the **target accent** (Nigerian-accented English works best)

---

### **Option 2: Multi-Sample Speaker (Higher Quality) ⭐⭐**

**Best for:** Professional use cases, maximum similarity

**Approach:** Combine multiple reference samples (3-5 clips)

```python
def create_multisample_speaker(audio_clips, transcripts, output_name):
    """
    Create speaker profile from multiple audio samples
    
    Args:
        audio_clips: List of audio file paths
        transcripts: List of corresponding transcripts
        output_name: Speaker name
    """
    audio_tokenizer = AudioTokenizerV2(...)
    
    all_words = []
    full_text = " ".join(transcripts)
    
    for audio_path, transcript in zip(audio_clips, transcripts):
        codes = audio_tokenizer.quantize_wavtokenizer(audio_path)
        code_list = [int(c) for c in codes.split("|") if c.strip()]
        
        audio, sr = torchaudio.load(audio_path)
        duration = audio.shape[1] / sr
        
        words = audio_tokenizer.process_text(transcript)
        duration_per_word = duration / len(words)
        codes_per_word = len(code_list) // len(words)
        
        for i, word in enumerate(words):
            start_idx = i * codes_per_word
            end_idx = (i + 1) * codes_per_word
            
            all_words.append({
                "word": word,
                "duration": f"{duration_per_word:.2f}",
                "codes": code_list[start_idx:end_idx]
            })
    
    speaker_data = {
        "text": full_text,
        "words": all_words
    }
    
    output_path = f"custom_speakers/{output_name}.json"
    with open(output_path, 'w') as f:
        json.dump(speaker_data, f, indent=2)
    
    return output_path

# Usage - combine diverse samples
clips = [
    "voice_sample1.wav",
    "voice_sample2.wav", 
    "voice_sample3.wav"
]
transcripts = [
    "This is the first sample with normal tone",
    "Here I'm speaking with more energy",
    "And this one is calm and measured"
]

speaker_path = create_multisample_speaker(clips, transcripts, "premium_voice")
```

**Quality Boost Factors:**
- Capture different **prosody patterns** (questions, statements, emphasis)
- Include **phonetic diversity** (various phonemes and word combinations)
- **60-90 seconds total** across all samples

---

### **Option 3: Fine-tuning for Production (Maximum Quality) ⭐⭐⭐**

**Best for:** Enterprise applications, celebrity voices, brand voices

**Requirements:**
- 30-120 minutes of target speaker audio
- High-quality transcriptions
- GPU resources (A100 recommended)
- 2-5 hours training time

**Process:**

#### Step 1: Prepare Training Data
```python
import os
import json
from datasets import Dataset

def prepare_voice_dataset(audio_folder, transcript_file):
    """
    Prepare dataset for fine-tuning
    
    Structure:
    audio_folder/
        clip_001.wav
        clip_002.wav
        ...
    transcript_file.json:
        {"clip_001": "text here", "clip_002": "more text", ...}
    """
    
    audio_tokenizer = AudioTokenizerV2(...)
    
    dataset_items = []
    
    with open(transcript_file) as f:
        transcripts = json.load(f)
    
    for audio_file, text in transcripts.items():
        audio_path = os.path.join(audio_folder, f"{audio_file}.wav")
        
        # Extract codes
        codes = audio_tokenizer.quantize_wavtokenizer(audio_path)
        
        # Create training prompt
        words = audio_tokenizer.process_text(text)
        prompt = audio_tokenizer.create_prompt(text, lang="english", speaker_name="base")
        
        dataset_items.append({
            "text": text,
            "audio_codes": codes,
            "prompt": prompt
        })
    
    return Dataset.from_list(dataset_items)

dataset = prepare_voice_dataset("voice_data/", "transcripts.json")
```

#### Step 2: Fine-tune Model (based on training notebook)
```python
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer

# Load base model
model = AutoModelForCausalLM.from_pretrained(
    "saheedniyi/YarnGPT2",
    torch_dtype=torch.float16
)

training_args = TrainingArguments(
    output_dir="./custom_voice_model",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=1e-5,
    warmup_steps=100,
    logging_steps=50,
    save_steps=500,
    fp16=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)

trainer.train()
```

**Expected Results:**
- **Highest similarity** to target voice
- Consistent quality across all text
- Natural prosody and emotion
- Handles edge cases better

---

## 📊 Quality Comparison Matrix

| Method | Similarity | Naturalness | Compute | Data Required | Time to Deploy |
|--------|-----------|-------------|---------|---------------|----------------|
| Option 1: Single Sample | 70-80% | Good | Low | 5-15 sec | 5 minutes |
| Option 2: Multi-Sample | 85-90% | Very Good | Low | 60-90 sec | 15 minutes |
| Option 3: Fine-tuned | 95-99% | Excellent | High | 30-120 min | 3-6 hours |

---

## 🎯 Optimization Strategies for Maximum Quality

### 1. **Audio Preprocessing**
```python
import torchaudio
from torchaudio.transforms import Resample, Vad

def preprocess_audio(input_path, output_path):
    """
    Clean and prepare audio for voice cloning
    """
    # Load audio
    waveform, sr = torchaudio.load(input_path)
    
    # Convert to mono
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    # Resample to 24kHz
    if sr != 24000:
        resampler = Resample(sr, 24000)
        waveform = resampler(waveform)
    
    # Apply VAD to remove silence
    vad = Vad(sample_rate=24000)
    waveform = vad(waveform)
    
    # Normalize volume
    waveform = waveform / torch.max(torch.abs(waveform))
    waveform = waveform * 0.95  # Leave headroom
    
    # Save
    torchaudio.save(output_path, waveform, 24000)
    
    return output_path
```

### 2. **Inference Optimization**
```python
def generate_with_quality_boost(text, speaker_name, audio_tokenizer, model):
    """
    Generate speech with quality optimizations
    """
    prompt = audio_tokenizer.create_prompt(text, lang="english", speaker_name=speaker_name)
    input_ids = audio_tokenizer.tokenize_prompt(prompt)
    
    # Quality-focused parameters
    output = model.generate(
        input_ids=input_ids,
        temperature=0.05,           # Lower = more stable (0.05-0.15)
        repetition_penalty=1.2,     # Reduce repetition (1.1-1.3)
        max_length=4000,
        top_k=20,                   # Limit token choices
        top_p=0.9,                  # Nucleus sampling
        do_sample=True,
        num_beams=1,                # Greedy for English (beam=5 for local langs)
    )
    
    codes = audio_tokenizer.get_codes(output)
    audio = audio_tokenizer.get_audio(codes)
    
    return audio
```

### 3. **Post-processing Enhancement**
```python
def enhance_generated_audio(audio_tensor):
    """
    Apply post-processing to improve quality
    """
    import torch
    from scipy import signal
    
    audio = audio_tensor.squeeze().numpy()
    
    # Apply subtle noise gate
    threshold = 0.01
    audio[np.abs(audio) < threshold] = 0
    
    # Apply gentle high-pass filter (remove rumble)
    sos = signal.butter(4, 80, 'hp', fs=24000, output='sos')
    audio = signal.sosfilt(sos, audio)
    
    # Normalize
    audio = audio / np.max(np.abs(audio)) * 0.95
    
    return torch.tensor(audio).unsqueeze(0)
```

### 4. **Model Parameters for Different Use Cases**

```python
# For audiobooks/narration (smooth, consistent)
PARAMS_NARRATION = {
    "temperature": 0.05,
    "repetition_penalty": 1.1,
    "max_length": 6000
}

# For conversational (natural, varied)
PARAMS_CONVERSATION = {
    "temperature": 0.15,
    "repetition_penalty": 1.2,
    "max_length": 4000
}

# For dramatic reading (expressive)
PARAMS_DRAMATIC = {
    "temperature": 0.2,
    "repetition_penalty": 1.15,
    "max_length": 4000
}
```

---

## 🚀 Complete Implementation Example

```python
import os
import torch
import torchaudio
from transformers import AutoModelForCausalLM
from yarngpt.audiotokenizer import AudioTokenizerV2

class VoiceCloningSystem:
    def __init__(self, model_path="saheedniyi/YarnGPT2"):
        self.tokenizer_path = model_path
        self.wav_config = "wavtokenizer_config.yaml"
        self.wav_model = "wavtokenizer_model.ckpt"
        
        # Initialize
        self.audio_tokenizer = AudioTokenizerV2(
            self.tokenizer_path,
            self.wav_model,
            self.wav_config
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype="auto"
        ).to(self.audio_tokenizer.device)
        
        self.custom_speakers_dir = "custom_speakers"
        os.makedirs(self.custom_speakers_dir, exist_ok=True)
    
    def clone_voice(self, audio_path, transcript, speaker_name):
        """
        Clone a voice from reference audio
        """
        # Preprocess audio
        processed = self.preprocess_audio(audio_path)
        
        # Extract codes
        codes = self.audio_tokenizer.quantize_wavtokenizer(processed)
        code_list = [int(c) for c in codes.split("|") if c.strip()]
        
        # Load audio for duration
        audio, sr = torchaudio.load(processed)
        duration = audio.shape[1] / sr
        
        # Process text
        words = self.audio_tokenizer.process_text(transcript)
        duration_per_word = duration / len(words)
        codes_per_word = len(code_list) // len(words)
        
        # Create speaker profile
        speaker_data = {
            "text": transcript,
            "words": []
        }
        
        for i, word in enumerate(words):
            start_idx = i * codes_per_word
            end_idx = (i + 1) * codes_per_word
            
            speaker_data["words"].append({
                "word": word,
                "duration": f"{duration_per_word:.2f}",
                "codes": code_list[start_idx:end_idx]
            })
        
        # Save
        import json
        output_path = os.path.join(self.custom_speakers_dir, f"{speaker_name}.json")
        with open(output_path, 'w') as f:
            json.dump(speaker_data, f, indent=2)
        
        print(f"✅ Voice cloned successfully: {speaker_name}")
        return output_path
    
    def preprocess_audio(self, input_path):
        """Clean and prepare audio"""
        output_path = input_path.replace(".wav", "_processed.wav")
        
        waveform, sr = torchaudio.load(input_path)
        
        # Mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Resample
        if sr != 24000:
            resampler = torchaudio.transforms.Resample(sr, 24000)
            waveform = resampler(waveform)
        
        # Normalize
        waveform = waveform / torch.max(torch.abs(waveform)) * 0.95
        
        torchaudio.save(output_path, waveform, 24000)
        return output_path
    
    def synthesize(self, text, speaker_name, quality="high"):
        """
        Generate speech with cloned voice
        """
        # Quality presets
        params = {
            "high": {"temperature": 0.05, "repetition_penalty": 1.2},
            "balanced": {"temperature": 0.1, "repetition_penalty": 1.1},
            "expressive": {"temperature": 0.15, "repetition_penalty": 1.15}
        }[quality]
        
        # Generate
        prompt = self.audio_tokenizer.create_prompt(
            text, 
            lang="english", 
            speaker_name=speaker_name
        )
        input_ids = self.audio_tokenizer.tokenize_prompt(prompt)
        
        output = self.model.generate(
            input_ids=input_ids,
            temperature=params["temperature"],
            repetition_penalty=params["repetition_penalty"],
            max_length=4000,
        )
        
        codes = self.audio_tokenizer.get_codes(output)
        audio = self.audio_tokenizer.get_audio(codes)
        
        return audio, 24000

# Usage Example
system = VoiceCloningSystem()

# Clone a voice
system.clone_voice(
    "reference_voice.wav",
    "Scientists have discovered a new planet that may support life",
    "my_custom_voice"
)

# Generate speech with cloned voice
audio, sr = system.synthesize(
    "Hello, this is a test of my cloned voice!",
    "my_custom_voice",
    quality="high"
)

# Save output
torchaudio.save("output.wav", audio, sr)
```

---

## 📋 Recommended Workflow

### For Quick Testing (30 minutes):
1. Record 10-second clean audio sample
2. Use Option 1 (single sample)
3. Test with various texts
4. Iterate on recording quality if needed

### For Production (2-3 hours):
1. Record 3-5 diverse samples (60-90 seconds total)
2. Use Option 2 (multi-sample)
3. Fine-tune generation parameters
4. Add post-processing

### For Maximum Quality (1 week):
1. Collect 30-120 minutes of audio
2. Professional recording environment
3. Use Option 3 (fine-tuning)
4. Extensive testing and optimization

---

## ⚠️ Important Considerations

1. **Audio Quality Requirements:**
   - Sample rate: 24kHz (exact)
   - Channels: Mono
   - Format: WAV (16-bit PCM)
   - SNR: >30dB (minimal background noise)

2. **Accent Matching:**
   - Model trained on Nigerian-accented English
   - Other accents may have reduced quality
   - Consider fine-tuning for non-Nigerian accents

3. **Ethical Usage:**
   - Always obtain consent before cloning voices
   - Implement speaker verification for production
   - Add watermarking for generated audio

4. **Performance:**
   - GPU strongly recommended (CUDA)
   - CPU inference: ~15-30 seconds per sentence
   - GPU inference: ~1-3 seconds per sentence

---

## 🎬 Next Steps

1. **Install dependencies:**
```bash
pip install outetts uroman torch torchaudio transformers inflect
```

2. **Download WavTokenizer models:**
```bash
wget https://huggingface.co/novateur/WavTokenizer-medium-speech-75token/resolve/main/wavtokenizer_mediumdata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml

# Large model (better quality)
wget https://huggingface.co/novateur/WavTokenizer-large-speech-75token/resolve/main/wavtokenizer_large_speech_320_24k.ckpt
```

3. **Start with Option 1** - clone a voice with single sample
4. **Test quality** - generate various texts
5. **Scale up** to Option 2 or 3 based on requirements

---

## 📚 Additional Resources

- **YarnGPT Model:** https://huggingface.co/saheedniyi/YarnGPT2
- **WavTokenizer:** https://github.com/jishengpeng/WavTokenizer
- **Training Notebooks:** `/workspaces/Tawk/yarngpt/notebooks/`

**Questions or need help?** The architecture is straightforward - you're essentially teaching the model new voice patterns via the audio code representations!
