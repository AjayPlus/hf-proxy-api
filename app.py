from flask import Flask, request, jsonify
from transformers import pipeline
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)

clf = pipeline("text-classification", model="AjayPlus/propaganda-detector-v2")

@app.route("/", methods=["GET"])
def home():
    return "✅ Flask API is running"

@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        text = request.json.get("inputs", "")
        result = clf(text)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port)

