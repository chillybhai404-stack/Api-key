"""Train AXU Interaction Intelligence v2 from the included CSV dataset."""
from __future__ import annotations
import csv
from datetime import datetime, timezone
from pathlib import Path
import argparse
import joblib
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "interactions.csv"
ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "interactions_200k.csv"
ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "axu_dataset_1_5M.csv"
MODEL_PATH = ROOT / "models" / "axu_interaction_model.joblib"


def load_dataset(path: Path):
    texts: list[str] = []
    labels: list[list[str]] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not {"text", "labels"}.issubset(reader.fieldnames or []):
            raise ValueError("CSV must contain columns named 'text' and 'labels'.")
        for row in reader:
            text = (row.get("text") or "").strip()
            labs = [x.strip() for x in (row.get("labels") or "").split("|") if x.strip()]
            if text and labs:
                texts.append(text)
                labels.append(labs)
    if len(texts) < 100:
        raise ValueError(f"Only {len(texts)} examples found; expected a larger dataset.")
    return texts, labels


def make_pipeline():
    # Word features learn task phrases; character n-grams help with typos and variants.
    features = FeatureUnion([
        ("word_tfidf", TfidfVectorizer(
            lowercase=True, strip_accents="unicode", ngram_range=(1, 2),
            min_df=2, max_df=0.985, max_features=42000, sublinear_tf=True,
        )),
        ("char_tfidf", TfidfVectorizer(
            lowercase=True, analyzer="char_wb", ngram_range=(3, 5),
            min_df=2, max_features=28000, sublinear_tf=True,
        )),
    ])
    classifier = OneVsRestClassifier(LogisticRegression(
        C=2.5, solver="liblinear", max_iter=400, class_weight="balanced",
    ), n_jobs=1)
    return Pipeline([("features", features), ("classifier", classifier)])


def main():
    parser = argparse.ArgumentParser(description="Train AXU Interaction Intelligence v2")
    parser.add_argument("--data", default=str(DATA_PATH), help="Path to CSV with text,labels columns")
    parser.add_argument("--threshold", type=float, default=0.45, help="Default multi-label score threshold (0 to 1)")
    args = parser.parse_args()
    if not 0.05 <= args.threshold <= 0.95:
        parser.error("--threshold must be between 0.05 and 0.95")

    texts, label_lists = load_dataset(Path(args.data))
    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(label_lists)
    model = make_pipeline()
    print("AXU Interaction Intelligence v2")
    print(f"Training examples: {len(texts):,}")
    print(f"Intent categories: {len(mlb.classes_)}")
    print(f"Multi-label examples: {sum(len(set(x)) > 1 for x in label_lists):,}")
    print("Training the model (this may take a little while on a phone)...")
    model.fit(texts, y)
    bundle = {
        "name": "AXU Interaction Intelligence",
        "version": "2.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "sklearn_version": sklearn.__version__,
        "trained_examples": len(texts),
        "intent_count": int(len(mlb.classes_)),
        "intents": [str(x) for x in mlb.classes_],
        "threshold": float(args.threshold),
        "model": model,
        "mlb": mlb,
    }
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, MODEL_PATH, compress=3)
    print(f"Saved model: {MODEL_PATH}")
    print("Training finished. Next try: python predict.py")
    print("Reminder: this is a synthetic-data prototype; real-world quality depends on real, labelled examples.")

if __name__ == "__main__":
    main()
