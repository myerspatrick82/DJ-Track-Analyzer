from fastapi import FastAPI, UploadFile, HTTPException
from pathlib import Path 
import tempfile

from analyzer import analyze_track

app = FastAPI(title="DJ Track Analyzer API")

MAX_SIZE = 25 * 1024 * 1024 # 25 MB

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/analyze")
async def analyze(file: UploadFile):
    # check file extension
    suffix = Path(file.filename or "").suffix.lower()

    if suffix not in {".wav", ".mp3", ".flax"}:
        raise HTTPException(status_code=400, detail="Unsupported audio file type")

    # limit upload size
    data = await file.read(MAX_SIZE + 1)

    if len(data) > MAX_SIZE:
        raise HTTPException(status_code=413, detail="File exceeds 25 MB limit")

    # process temp audio file
    with tempfile.TemporaryDirectory() as temp_dir:
        path = Path(temp_dir) / ("upload" + suffix)
        path.write_bytes(data)

        try:
            result = analyze_track(str(path))
            result["filename"] = file.filename
            return result
        
        except Exception as exc:
            raise HTTPException(status_code=422, detail="Could not analyze audio file") from exc