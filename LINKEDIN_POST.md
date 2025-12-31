# LinkedIn Post - YarnGPT Voice Cloning Project

---

**Day 7/7 of my Project Showcase** 🎯

I just cloned my Nigerian accent with AI... and honestly? It's wild how far open-source TTS has come. 🔥

---

## The Hook

You know that feeling when you find an amazing tool but it's missing THAT one feature you desperately need? 

That was me with YarnGPT.

---

## The Problem

YarnGPT by @Saheed Azee is honestly brilliant - a pure language modeling TTS system specifically trained on Nigerian accents (Yoruba, Igbo, Hausa, and Nigerian English). 

But here's the catch: **13 preset voices only.** 

Cool for demos, but what if I want MY voice? What if I want to build personalized voice assistants? What if every Nigerian creator wants their own AI voice?

The repo had ZERO documentation on custom voice creation. Nada. Zilch.

So naturally, I decided to build it myself. 💪

---

## What I Built

A complete voice cloning pipeline for YarnGPT:

✅ Single-sample voice cloning (literally just 15 seconds of audio)  
✅ Whisper-based word alignment (the game-changer)  
✅ Quality presets (high/balanced/expressive)  
✅ Web dashboard with FastAPI + Tailwind CSS  
✅ Nigerian-inspired UI (green-white-green vibes 🇳🇬)  

Now anyone can clone their voice and generate speech in seconds.

---

## The Challenges (oh boy...)

**1. "Gunshot" Audio Syndrome**  
My first attempt? Literal gunshot sounds. One second of noise. Turns out aggressive audio preprocessing DESTROYS the audio codes. The spectral noise reduction I was so proud of? Yeah, it was creating repetitive code patterns that the model hated.

**Solution:** Skip preprocessing entirely for clean audio. Sometimes simpler is better.

**2. Word Alignment Nightmare**  
Second attempt? Gibberish. The audio was longer but made zero sense. Why? I was just dividing audio codes evenly across words like a barbarian.

Word 1: Get 65 codes  
Word 2: Get 65 codes  
Word 3: Get 65 codes  

That's not how speech works! 😅

**Solution:** Implemented Whisper forced alignment. Now each word gets codes based on its ACTUAL timing in the audio. "I" gets 65 codes (0.86s), "am" gets 16 codes (0.22s). Game changed.

**3. Audio Format Hell**  
torchaudio.load() needed torchcodec. Didn't have it. Switched to soundfile. Audio shapes were wrong (soundfile returns [samples, channels], torch expects [channels, samples]). Fixed it. Then resample() added dimensions. Fixed that too.

Classic AI engineering: 10% ML magic, 90% wrangling tensor shapes. 😂

---

## Implementation Deep Dive

**Tech Stack:**
- YarnGPT (SmolLM2-360M + WavTokenizer)
- Whisper (base model for alignment)
- FastAPI + Tailwind CSS
- WavTokenizer (discrete audio codes @ 75 tokens/sec)

**The Secret Sauce:**
The key insight? YarnGPT uses discrete audio codes (0-4095) at 75 codes/second. Each word needs the exact codes spoken during that time window:

```
Audio: "hungry" spoken from 1.5s to 2.0s
Duration: 0.5 seconds
Codes needed: int(0.5 * 75) = 37 codes
Code range: [112:149]
```

Whisper gives us those timestamps. We map codes to words. Boom. Voice cloning.

---

## What This Taught Me

1. **Documentation gaps are opportunities.** Someone has to build what's missing.

2. **Audio preprocessing isn't always good.** Sometimes raw data > "cleaned" data.

3. **Word alignment matters MORE than audio quality.** A perfectly clean audio with wrong alignment = garbage output.

4. **Small models can surprise you.** YarnGPT is 360M parameters. That's TINY compared to modern LLMs. Yet it clones Nigerian accents better than most commercial solutions.

5. **Test with exact training data first.** My best results? When I synthesized the EXACT text I used for cloning. That's the baseline. Then you know what's possible.

---

## What's Next?

This is just v1. Here's what I'm thinking:

🔮 Multi-sample cloning (combine 3-5 clips for better quality)  
🔮 Fine-tuning the LM on custom speakers  
🔮 Real-time voice conversion  
🔮 API for developers  
🔮 Mobile app (imagine cloning your voice on your phone)  

But honestly? The biggest win is proving that Nigerians can have AI voices that ACTUALLY sound like us. Not a generic "African accent". Not some colonizer's idea of how we should sound.

Our accents. Our voices. Our AI.

---

## Shoutouts

Massive props to **Saheed Azee (@saheedniyi02)** for building YarnGPT and training it on Nigerian languages. This project wouldn't exist without your groundwork. 🙏

The fact that you chose SmolLM2 (a 360M model) and still got quality results? That's some serious ML engineering. Respect.

---

## Final Thoughts

This is Day 7 of my project showcase week. Seven days, seven projects, zero chill. 😅

Learned more in this one week than in months of tutorial hell. Built things that actually work (mostly). Debugged things that absolutely didn't (frequently).

But here's the thing: **building > watching.**

Every project had bugs. Every demo had issues. Every "final version" had v2 lurking around the corner.

And that's perfectly fine.

Because the code is in production. The web app is live. Anyone can clone their voice right now.

Is it perfect? Nah.  
Is it done? Never.  
Is it out there? Hell yeah.

---

**Tech Stack:** Python, PyTorch, FastAPI, Tailwind CSS, Whisper, YarnGPT  
**GitHub:** [link to repo]  
**Live Demo:** [link to demo]  

If you've made it this far, drop a comment. What voice would YOU clone? 👇

#AI #TTS #VoiceCloning #NigerianTech #MachineLearning #OpenSource #BuildInPublic #100DaysOfCode #YarnGPT #FastAPI

---

**P.S.** - If you're a Nigerian founder/creator who needs custom AI voices for your product, let's talk. This tech is ready.

**P.P.S.** - Tomorrow I rest. Maybe. Probably not. There's this other idea I have... 😏

🚀 Keep building. Keep shipping. Keep iterating.

---

*Built with ❤️ (and a lot of debugging) in Nigeria 🇳🇬*
