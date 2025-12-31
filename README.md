# Tawk - Nigerian Voice Cloning with YarnGPT 🇳🇬

A complete voice cloning system built on top of YarnGPT, enabling single-sample voice cloning with Nigerian accent preservation.

## 🎯 Features

- **Single-sample voice cloning** - Clone any voice with just 15 seconds of audio
- **Whisper-based word alignment** - Accurate word-to-code mapping for natural speech
- **Quality presets** - Choose between high, balanced, and expressive modes
- **Web dashboard** - Beautiful FastAPI + Tailwind CSS interface
- **Nigerian-inspired UI** - Green-white-green color scheme with flag 🇳🇬

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/Techdee02/Tawk.git
cd Tawk
```

### 2. Set up the environment

```bash
cd yarngpt
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install torch torchaudio transformers outetts scipy librosa soundfile pydub
pip install openai-whisper
pip install fastapi uvicorn jinja2 python-multipart
```

### 3. Download WavTokenizer models

```bash
# Download config
wget https://huggingface.co/novateur/WavTokenizer-medium-speech-75token/resolve/main/wavtokenizer_mediumdata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml -O wavtokenizer_config.yaml

# Download model checkpoint
wget https://huggingface.co/novateur/WavTokenizer-large-speech-75token/resolve/main/wavtokenizer_large_speech_320_24k.ckpt -O wavtokenizer_model.ckpt
```

### 4. Run the web app

```bash
python webapp/app.py
```

Visit `http://localhost:8000` to see the dashboard!

## 📖 Usage

### Command Line

```python
from tools.voice_cloner import VoiceCloningSystem
import soundfile as sf

# Initialize system
system = VoiceCloningSystem()

# Clone your voice
speaker_path = system.clone_voice(
    audio_path='your_audio.wav',
    transcript='What you said in the audio',
    speaker_name='my_voice'
)

# Generate speech
audio, sr = system.synthesize(
    text='Hello, this is my cloned voice!',
    speaker_name='my_voice',
    quality='expressive'  # or 'high', 'balanced'
)

# Save output
sf.write('output.wav', audio.squeeze().numpy(), sr)
```

### Web Interface

1. Your original voice sample is pre-loaded
2. Enter text to synthesize
3. Click "Generate Speech"
4. Listen and compare!

## 🔧 How It Works

### The Pipeline

1. **Audio Input** → Your voice recording (5-15 seconds recommended)
2. **Whisper Alignment** → Word-level timestamps extracted
3. **WavTokenizer** → Audio converted to discrete codes (0-4095)
4. **Code Mapping** → Codes assigned to words based on timing
5. **Speaker Profile** → JSON with word-code mappings saved
6. **Synthesis** → YarnGPT generates new audio codes from text
7. **Audio Output** → Codes decoded back to audio

### Key Insights

- **75 codes/second** - WavTokenizer operates at 75 tokens per second @ 24kHz
- **Word alignment matters** - Evenly distributing codes = gibberish. Time-based mapping = natural speech
- **Skip preprocessing** - Clean audio doesn't need noise reduction (can damage codes)
- **Test with training data** - Synthesize the exact cloning text first to verify quality

## 📊 Technical Details

### Tech Stack

- **YarnGPT** - SmolLM2-360M + WavTokenizer
- **Whisper** - Base model for forced alignment
- **PyTorch** - Deep learning framework
- **FastAPI** - Web framework
- **Tailwind CSS** - UI styling

### Model Architecture

```
SmolLM2-360M (Language Model)
    ↓
WavTokenizer (Audio Codec)
    ↓ encoding
Audio → Discrete Codes (4096 codebook)
    ↓ decoding
Synthesized Audio
```

## 🎓 Lessons Learned

1. **Documentation gaps are opportunities** - Build what's missing
2. **Audio preprocessing isn't always good** - Sometimes raw > "cleaned"
3. **Word alignment > audio quality** - Wrong alignment = garbage output
4. **Small models can surprise you** - 360M params clones Nigerian accents better than commercial solutions
5. **Test with exact data first** - Establish baseline with training text

## 🐛 Common Issues

### "Gunshot" sounds
- **Cause:** Aggressive audio preprocessing
- **Fix:** Skip preprocessing for clean audio

### Gibberish output
- **Cause:** Incorrect word alignment
- **Fix:** Use Whisper timestamps (already implemented)

### Short audio (<2s)
- **Cause:** Low max_length in generation
- **Fix:** Increased to 6000 tokens in code

## 🔮 Roadmap

- [ ] Multi-sample cloning (3-5 clips for better quality)
- [ ] Fine-tuning on custom speakers
- [ ] Real-time voice conversion
- [ ] REST API for developers
- [ ] Mobile app

## 🙏 Credits

Massive thanks to **Saheed Azee (@saheedniyi02)** for creating YarnGPT and training it on Nigerian languages!

## 📝 License

MIT License - See LICENSE file for details

## 🤝 Contributing

Contributions welcome! Open an issue or submit a PR.

## 📧 Contact

Built by TechDee - [@Techdee02](https://github.com/Techdee02)

---

*Built with ❤️ in Nigeria 🇳🇬*