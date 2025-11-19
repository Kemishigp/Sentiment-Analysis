
## 🤖 Sentiment Analysis App (CS 499 Project)

A text‑processing application that classifies written input as **Positive**, **Negative**, or **Neutral**, built as part of my BYU‑Idaho CS 499 Senior Project.

## 🔍 Overview

This app uses a fine‑tuned NLP model trained on the **e‑DAIC dataset**. Users can input text, and the model returns real‑time sentiment predictions.

## 🧠 Machine Learning

* Preprocessing with Pandas
* Tokenization + sequence encoding
* Model fine‑tuning (HuggingFace Transformers)
* Deployment via REST API

## 🧰 Tech Stack

* Python
* Hugging Face Transformers
* PyTorch / TensorFlow
* Flask / FastAPI (your choice)
* Pandas / NumPy

## 📦 Installation

```bash
git clone <repo-url>
cd sentiment-analysis-app
pip install -r requirements.txt
```

Run the API:

```bash
python app.py
```

Run the front-end:

```bash
npm install
npm run dev
```

## 📈 Performance

Include metrics once evaluated:

* Accuracy: *xx%*
* F1-score: *xx%*
