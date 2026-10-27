"""FastAPI server for AXU Interaction Intelligence v2."""
from __future__ import annotations
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from model_utils import predict_text, load_bundle

ROOT = Path(__file__).resolve().parent
app = FastAPI(
    title="AXU Interaction Intelligence API",
    description="Multi-label text-intent classification prototype trained by AXU.",
    version="2.0.0",
)

class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000, description="Text to classify")
    threshold: float | None = Field(default=None, ge=0.05, le=0.95, description="Optional label threshold")
    top_k: int = Field(default=8, ge=1, le=20, description="Number of candidate labels to return")

@app.get("/health")
def health():
    return {"status": "ok", "model_file_exists": (ROOT / "models" / "axu_interaction_model.joblib").exists()}

@app.get("/", response_class=HTMLResponse)
def home():
    index = ROOT / "index.html"
    if index.exists():
        return index.read_text(encoding="utf-8")
    return "<h1>AXU Interaction Intelligence API</h1><p>Open <a href='/docs'>/docs</a> to test the API.</p>"

@app.post("/predict")
def predict(req: PredictRequest):
    if not req.text.strip():
        raise HTTPException(status_code=422, detail="text must not be blank")
    try:
        return predict_text(req.text, threshold=req.threshold, top_k=req.top_k)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

@app.get("/metadata")
def metadata():
    try:
        bundle = load_bundle()
        return {
            "name": bundle.get("name"),
            "version": bundle.get("version"),
            "trained_examples": bundle.get("trained_examples"),
            "intent_count": bundle.get("intent_count"),
            "intents": bundle.get("intents", []),
            "threshold": bundle.get("threshold"),
        }
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


if __name__ == "__main__":
    import uvicorn
    # Pydroid-friendly local launcher; hosted services use their configured start command.
    uvicorn.run(app, host="127.0.0.1", port=8000)
