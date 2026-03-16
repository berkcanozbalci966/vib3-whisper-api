from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import JSONResponse
from faster_whisper import WhisperModel
import tempfile
import os

app = FastAPI(title="Whisper API", version="1.0.0")

model: WhisperModel | None = None


def get_model() -> WhisperModel:
    global model
    if model is None:
        model_size = os.getenv("WHISPER_MODEL_SIZE", "base")
        device = os.getenv("WHISPER_DEVICE", "cpu")
        compute_type = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
        model = WhisperModel(model_size, device=device, compute_type=compute_type)
    return model


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/v1/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    language: str | None = Query(None, description="Language code (e.g. 'en', 'tr')"),
    task: str = Query("transcribe", description="'transcribe' or 'translate'"),
):
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename or ".wav")[1]) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        whisper = get_model()
        segments, info = whisper.transcribe(
            tmp_path,
            language=language,
            task=task,
            beam_size=5,
        )

        result_segments = []
        full_text = ""
        for segment in segments:
            result_segments.append({
                "start": round(segment.start, 2),
                "end": round(segment.end, 2),
                "text": segment.text.strip(),
            })
            full_text += segment.text

        return JSONResponse({
            "text": full_text.strip(),
            "language": info.language,
            "language_probability": round(info.language_probability, 4),
            "duration": round(info.duration, 2),
            "segments": result_segments,
        })
    finally:
        os.unlink(tmp_path)
