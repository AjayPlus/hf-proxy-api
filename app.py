from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import pipeline

# ---- configure Flask ----
app = Flask(__name__)
CORS(app)

# ---- load your model once at startup ----
# note: device=-1 forces CPU
clf = pipeline(
    "text-classification",
    model="AjayPlus/propaganda-detector-v2",
    device=-1
)

@app.route("/", methods=["GET"])
def home():
    return "✅ Self-hosted Propaganda Detector API is running!"

@app.route("/analyze", methods=["POST"])
def analyze():
    payload = request.json or {}
    text = payload.get("inputs", "")
    if not text:
        return jsonify({"error": "no 'inputs' provided"}), 400

    # run the pipeline
    preds = clf(text, truncation=True)
    # preds is a list of { label: “LABEL_#”, score: float }
    return jsonify(preds)

if __name__ == "__main__":
    # Render will respect the PORT env var; default to 7860 if you like
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
