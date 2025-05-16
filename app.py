from flask import Flask, request, jsonify
import requests
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

HF_API_URL = "https://api-inference.huggingface.co/models/AjayPlus/propaganda-detector-v2"
HF_TOKEN = os.environ.get("HF_TOKEN")

@app.route("/", methods=["GET"])
def home():
    return "✅ Hugging Face proxy is running!"

@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.json
    text = data.get("inputs", "")

    try:
        response = requests.post(
            HF_API_URL,
            headers={
                "Authorization": f"Bearer {HF_TOKEN}",
                "Content-Type": "application/json"
            },
            json={"inputs": text, "options": {"wait_for_model": True}}
        )
        response.raise_for_status()
        return jsonify(response.json())
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)
