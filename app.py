from flask import Flask, request, jsonify
from transformers import pipeline

app = Flask(__name__)

clf = pipeline("text-classification", model="AjayPlus/propaganda-detector-v2")

@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json()
    inputs = data.get("inputs", "")
    result = clf(inputs)
    return jsonify(result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)
