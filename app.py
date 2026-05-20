"""
app.py  —  Flask REST API for the Spam Detector
------------------------------------------------
Endpoints:
  POST /predict   { "text": "your message here" }
  GET  /health    server health check
"""

import pickle
import os
import re
import numpy as np
from flask_cors import CORS
from flask import Flask, request, jsonify, render_template
# ── App setup ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)   # allow requests from the frontend (different port)

# ── Load model ────────────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "spam_model.pkl")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

print("✅ Model loaded successfully!")

# ── Helper: extract spam signals ─────────────────────────────────────────────
SPAM_KEYWORDS = [
    "win", "winner", "won", "prize", "free", "cash", "money",
    "urgent", "claim", "offer", "discount", "click here",
    "limited time", "guaranteed", "selected", "congratulations",
    "call now", "subscribe", "credit", "loan", "verify", "account suspended",
]

def extract_signals(text: str) -> dict:
    """Return human-readable clues that explain the prediction."""
    text_lower = text.lower()
    found_keywords = [kw for kw in SPAM_KEYWORDS if kw in text_lower]
    has_url        = bool(re.search(r"http[s]?://|bit\.ly|www\.", text_lower))
    has_phone      = bool(re.search(r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b|\b\d{5,}\b", text))
    all_caps_words = len(re.findall(r"\b[A-Z]{3,}\b", text))
    exclamations   = text.count("!")
    dollar_signs   = text.count("$")

    return {
        "keywords_found": found_keywords[:5],   # show top 5
        "has_url":        has_url,
        "has_phone":      has_phone,
        "all_caps_count": all_caps_words,
        "exclamation_count": exclamations,
        "dollar_signs": dollar_signs,
    }
@app.route('/')
def home():
    return render_template('index.html')
# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": "spam-detector-v1"})


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)

    # Validate input
    if not data or "text" not in data:
        return jsonify({"error": "Missing 'text' field in request body."}), 400

    text = str(data["text"]).strip()
    if not text:
        return jsonify({"error": "Text cannot be empty."}), 400
    if len(text) > 2000:
        return jsonify({"error": "Text too long (max 2000 characters)."}), 400

    # Predict
    prediction    = model.predict([text])[0]          # "spam" or "ham"
    probabilities = model.predict_proba([text])[0]    # [P(ham), P(spam)]
    classes       = model.classes_.tolist()           # ["ham", "spam"]

    spam_prob = float(probabilities[classes.index("spam")])
    ham_prob  = float(probabilities[classes.index("ham")])

    signals = extract_signals(text)

    return jsonify({
        "prediction":    prediction,
        "is_spam":       prediction == "spam",
        "confidence":    round(max(spam_prob, ham_prob) * 100, 1),
        "spam_prob":     round(spam_prob * 100, 1),
        "ham_prob":      round(ham_prob  * 100, 1),
        "signals":       signals,
        "text_length":   len(text),
        "word_count":    len(text.split()),
    })


# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("🚀 Starting Spam Detector API on http://localhost:5000")
    app.run(debug=True, port=5000)
