# AXU Interaction Intelligence v2

A larger, runnable text-intent classification prototype for AXU. It predicts **one or more intent labels** for a user message and includes a terminal demo, a browser demo, and a FastAPI endpoint.

## What is included

- **52 intent categories** for support, app feedback, coding/learning, writing, questions, commands, account issues, payments, scheduling, reminders, privacy/security, and more.
- **7,000+ synthetic text examples** in `data/interactions.csv`.
- A word + character TF-IDF pipeline, which can learn phrases and tolerate some spelling variation.
- Multi-label predictions for blended messages, such as praise plus a bug report.
- A terminal tester (`predict.py`), a browser UI (`index.html`), and an HTTP API (`api.py`).
- A dataset builder (`build_dataset.py`) and model trainer (`train.py`).

## Important limitation

This is **not a large language model and does not generate replies**. It classifies the intent of text. It is a useful starting component for routing feedback, identifying bug reports, sorting support messages, or triggering workflows. The included dataset is synthetic, partly template-generated, and has class imbalance. The model may perform well on similar phrasing but badly on real messages that differ from the examples. Scores are not calibrated probabilities. For a real product, add real, permissioned messages that have been labelled correctly, then test on a separate set of unseen real messages.

## Run on Pydroid 3 / Android

1. Extract this ZIP into a folder. Keep the `data` and `models` folders next to the Python files.
2. Open Pydroid 3's Pip screen and install the packages from `requirements.txt` if you have not already installed them: `scikit-learn`, `joblib`, `fastapi`, and `uvicorn`.
3. **Try `predict.py` first.** This ZIP includes a trained model at `models/axu_interaction_model.joblib`, so you can test it without training first.
4. If you want to retrain or change the dataset, open and run `train.py`. It reads `data/interactions.csv` and saves a new model. Training can take a while on a phone.

If you see a scikit-learn version compatibility warning or error while loading the included model, run `train.py` using the scikit-learn installed on your device to rebuild it.

## Run the API locally

From a terminal in this folder, with the dependencies installed and the model file present:

```bash
uvicorn api:app --host 127.0.0.1 --port 8000
```

Or open `api.py` in Pydroid and tap Run; it starts the server locally. Then open `http://127.0.0.1:8000/` on the same device for the browser demo, or `http://127.0.0.1:8000/docs` to try the API. Keep the server process running while testing. A public hosted URL requires deploying the project to a hosting service; this code does not create a public URL by itself.

## API example

Request:

```json
{
  "text": "I love this app, but the login keeps crashing",
  "threshold": 0.45,
  "top_k": 8
}
```

Send it to `POST /predict`. Example response fields are `labels` (labels above the selected threshold), `top_prediction`, `top_score`, and `candidates` (top alternatives). Exact scores depend on the trained model file.

## Improve the dataset

Open `data/interactions.csv`. Add rows in this format, using `|` between multiple correct labels:

```csv
text,labels
"I like the update, but saving crashes", "positive_feedback|bug_report|complaint"
"Please add a compact view", "feature_request"
```

Make sure the labels match the categories in the existing data. Add varied examples for every label, including messages where two or more intents are combined. Re-run `train.py` afterwards. Do not add private customer messages without proper permission and suitable data handling.

## Hosting later

The API is FastAPI-compatible. A typical hosted start command is:

```bash
uvicorn api:app --host 0.0.0.0 --port $PORT
```

Make sure the trained model artifact is included in the deployment. Do not expose private data or secrets in the repository.
