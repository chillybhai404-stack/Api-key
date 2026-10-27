"""Shared prediction helpers for AXU Interaction Intelligence v2."""
from __future__ import annotations
from pathlib import Path
from typing import Any
import joblib

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "axu_interaction_model.joblib"

_model_bundle: dict[str, Any] | None = None


def load_bundle() -> dict[str, Any]:
    global _model_bundle
    if _model_bundle is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model file not found at {MODEL_PATH}. Run: python train.py"
            )
        _model_bundle = joblib.load(MODEL_PATH)
    return _model_bundle


def predict_text(text: str, threshold: float | None = None, top_k: int = 8) -> dict[str, Any]:
    clean = " ".join((text or "").split()).strip()
    if not clean:
        raise ValueError("text must not be empty")
    bundle = load_bundle()
    model = bundle["model"]
    mlb = bundle["mlb"]
    used_threshold = float(bundle.get("threshold", 0.45) if threshold is None else threshold)
    probabilities = model.predict_proba([clean])[0]
    ranked = sorted(
        [(str(label), float(score)) for label, score in zip(mlb.classes_, probabilities)],
        key=lambda item: item[1], reverse=True,
    )
    selected = [(label, score) for label, score in ranked if score >= used_threshold]
    if not selected:
        selected = ranked[:1]
    selected = selected[:max(1, min(int(top_k), len(ranked)))]
    candidates = ranked[:max(1, min(int(top_k), len(ranked)))]
    return {
        "text": clean,
        "model": bundle.get("name", "AXU Interaction Intelligence"),
        "model_version": bundle.get("version", "2.0"),
        "labels": [{"label": label, "score": round(score, 4)} for label, score in selected],
        "top_prediction": ranked[0][0],
        "top_score": round(ranked[0][1], 4),
        "candidates": [{"label": label, "score": round(score, 4)} for label, score in candidates],
        "threshold": round(used_threshold, 3),
        "note": "Scores are model scores, not calibrated real-world probabilities. Validate on your own messages before relying on them.",
    }
