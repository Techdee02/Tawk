"""
Tawk Voice Cloning Dashboard - FastAPI Backend
Nigerian-inspired voice cloning interface
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import sys
import os
from pathlib import Path
import soundfile as sf
import io

# Add yarngpt to path
sys.path.insert(0, str(Path(__file__).parent.parent / "yarngpt"))
from tools.voice_cloner import VoiceCloningSystem

app = FastAPI(title="Tawk Voice Cloning Dashboard")

# Setup directories
BASE_DIR = Path(__file__).parent
STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# Initialize voice cloning system (load once at startup)
print("🚀 Initializing Tawk Voice Cloning System...")
voice_system = VoiceCloningSystem()
print("✅ System ready!")

# Original recording info
ORIGINAL_AUDIO = Path(__file__).parent.parent / "recording_converted.wav"
ORIGINAL_TEXT = "I am hungry. I havent had anything to eat, i need to eat"
SPEAKER_NAME = "aligned_voice"


class SynthesisRequest(BaseModel):
    text: str
    quality: str = "expressive"


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Render main dashboard"""
    return templates.TemplateResponse("index.html", {
        "request": request,
        "original_text": ORIGINAL_TEXT
    })


@app.get("/audio/original")
async def get_original_audio():
    """Serve original recording"""
    if not ORIGINAL_AUDIO.exists():
        raise HTTPException(status_code=404, detail="Original audio not found")
    return FileResponse(ORIGINAL_AUDIO, media_type="audio/wav")


@app.post("/api/synthesize")
async def synthesize_speech(request: SynthesisRequest):
    """Generate cloned speech"""
    try:
        print(f"📝 Synthesizing: '{request.text}' (quality: {request.quality})")
        
        # Generate audio
        audio, sr = voice_system.synthesize(
            request.text,
            SPEAKER_NAME,
            quality=request.quality
        )
        
        # Convert to WAV bytes
        buffer = io.BytesIO()
        sf.write(buffer, audio.squeeze().numpy(), sr, format='WAV')
        buffer.seek(0)
        
        # Save to outputs folder
        output_filename = f"synthesis_{len(os.listdir(OUTPUT_DIR)) + 1}.wav"
        output_path = OUTPUT_DIR / output_filename
        sf.write(output_path, audio.squeeze().numpy(), sr)
        
        duration = len(audio.squeeze()) / sr
        print(f"✅ Generated {duration:.2f}s audio: {output_filename}")
        
        return StreamingResponse(
            buffer,
            media_type="audio/wav",
            headers={
                "Content-Disposition": f"attachment; filename={output_filename}",
                "X-Duration": str(duration)
            }
        )
        
    except Exception as e:
        print(f"❌ Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/stats")
async def get_stats():
    """Get system statistics"""
    return {
        "original_text": ORIGINAL_TEXT,
        "speaker": SPEAKER_NAME,
        "available_qualities": ["high", "balanced", "expressive"],
        "total_syntheses": len(list(OUTPUT_DIR.glob("*.wav")))
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
