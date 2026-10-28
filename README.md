<div align="center">

# AXU Interaction Intelligence v2

**A practical, multi-label text-intent classifier with a browser demo and REST API.**

Turn short messages into structured intent labels — useful for support routing, feedback triage, bug reporting, and lightweight workflow automation.

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/API-FastAPI-009688?logo=fastapi&logoColor=white)
![License](https://img.shields.io/badge/License-Apache--2.0-blue.svg)
![Model](https://img.shields.io/badge/Model-TF--IDF%20%2B%20Classifier-5B6ABF)

</div>

---

## At a glance

| | Details |
|---|---|
| **Intent categories** | 52 categories |
| **Dataset size** | 200K+ synthetic text examples |
| **Prediction style** | Multi-label — one message can express more than one intent |
| **Text features** | Word-level and character-level TF-IDF |
| **Interfaces** | Terminal tester, browser demo, FastAPI endpoint |
| **Mobile workflow** | Can be run locally in Pydroid 3 on Android |

> **Important:** AXU Interaction Intelligence is a text classifier, not a large language model. It predicts intent labels; it does not generate replies.

## Screenshots

### Browser demo

Try a greeting, bug report, feature request, or mixed-feedback message in the simple AXU interface.

<p align="center">
  <img src="https://i.postimg.cc/nLZyGcFB/Screenshot-2026-10-28-21-09-37-993-com-android-chrome-edit.jpg" alt="AXU Interaction Intelligence browser demo with example prompts" width="340" />
</p>

### API documentation

The FastAPI documentation lists the health, home, prediction, and metadata routes and lets you explore the endpoint schema.

<p align="center">
  <img src="https://i.postimg.cc/PJBcpdVQ/Screenshot-2026-10-28-21-10-36-021-com-android-chrome-edit.jpg" alt="AXU Interaction Intelligence API documentation" width="340" />
</p>

## What it does

- **Classifies message intent** across 52 categories, including support requests, app feedback, coding and learning, account issues, payments, scheduling, reminders, and privacy/security.
- **Supports mixed intent.** A message such as “I like the design, but search is slow” can receive multiple relevant labels.
- **Uses word and character TF-IDF features** to recognize useful terms and some spelling variation.
- **Includes an existing trained model** so you can start testing before retraining, provided the model is compatible with your installed scikit-learn version.
- **Provides three ways to try it:** a terminal script, browser interface, and HTTP API.

## Quick start

### 1. Install dependencies

Use Python in your project environment and install the packages listed in `requirements.txt`:

```bash
python -m pip install -r requirements.txt
```

The project uses `scikit-learn`, `joblib`, `fastapi`, and `uvicorn` among its dependencies.

### 2. Try a prediction

Run the terminal tester first:

```bash
python predict.py
```

The included model is expected at `models/axu_interaction_model.joblib`.

### 3. Start the browser demo and API

```bash
uvicorn api:app --host 127.0.0.1 --port 8000
```

Open these addresses on the same device while the server is running:

- Browser demo: <http://127.0.0.1:8000/>
- Interactive API docs: <http://127.0.0.1:8000/docs>

The local server does **not** create a public internet URL. Public access requires a separate hosting deployment and appropriate security configuration.

## Run on Android with Pydroid 3

1. Extract the project into one folder and keep the `data/` and `models/` directories beside the Python files.
2. Open Pydroid 3's Pip screen and install the required packages from `requirements.txt`.
3. Run `predict.py` to test the included model.
4. To change the data or retrain the model, run `train.py`.
5. To start the API from the same device, run `api.py` in Pydroid or launch the Uvicorn command above. Keep the process open while you use the browser demo.

If the model cannot be loaded because of a scikit-learn version mismatch, run `train.py` with the version installed on your device to rebuild the model.

## API example

Send a `POST` request to `/predict` with a JSON body like this:

```json
{
  "text": "I love this app, but the login keeps crashing",
  "threshold": 0.45,
  "top_k": 8
}
```

The response includes fields such as `labels`, `top_prediction`, `top_score`, and `candidates`. Exact scores depend on the trained model and settings; they should not be treated as calibrated probabilities.

Example with `curl`:

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"text":"I love this app, but the login keeps crashing","threshold":0.45,"top_k":8}'
```

## Dataset and training

The dataset lives at `data/interactions.csv`. Add representative examples with labels that match the existing categories. Use `|` to separate multiple labels.

```csv
text,labels
"I like the update, but saving crashes","positive_feedback|bug_report|complaint"
"Please add a compact view","feature_request"
"I cannot access my account after changing my password","account_issue|bug_report"
```

Then rebuild the model:

```bash
python train.py
```

If you need to rebuild the dataset itself, inspect and run `build_dataset.py` first. Keep the dataset and model in sync, and evaluate changes on messages that were not used during training.

### Dataset quality matters

The included examples are synthetic and may be template-generated or unevenly distributed across labels. That can make performance look better on familiar phrasing than on real-world messages. Before deploying this in a real product:

- Check the true row count and label distribution in `data/interactions.csv`.
- Add varied, permissioned examples that reflect real usage.
- Keep a separate, unseen evaluation set and report per-label performance.
- Do not treat model scores as calibrated confidence or guaranteed correctness.
- Do not add private customer messages without appropriate permission and data handling.

## Project layout

```text
.
├── api.py
├── build_dataset.py
├── index.html
├── predict.py
├── train.py
├── requirements.txt
├── data/
│   └── interactions.csv
├── models/
│   └── axu_interaction_model.joblib
├── assets/
│   └── license-apache-2.0.png
├── LICENSE
└── README.md
```

## Hosting

For a hosting service that provides a `PORT` environment variable, a typical start command is:

```bash
uvicorn api:app --host 0.0.0.0 --port $PORT
```

Include the trained model artifact in the deployment. Before making the API public, configure access controls, rate limiting, logging that avoids sensitive text, and any deployment-specific security measures. Never commit tokens, secrets, or private customer data to the repository.

## License

This project is distributed under the **Apache License 2.0**. Read the complete terms in [`LICENSE`](LICENSE).

<p align="center">
  <img src="assets/license-apache-2.0.png" alt="A clean visual summary of the Apache License 2.0" width="720" />
</p>

The image is a visual summary for the README; the `LICENSE` file is the authoritative legal text.

---

<div align="center">

**AXU Interaction Intelligence v2** · Text intent classification · Browser demo · FastAPI

</div>
