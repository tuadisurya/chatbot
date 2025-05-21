from flask import Flask, request, jsonify
from dotenv import load_dotenv
from backend.logic import generate_answer
import os

load_dotenv()

app = Flask(__name__)

@app.route("/api/ask", methods=["POST"])
def ask():
    data = request.get_json()
    prompt = data.get("prompt", "")

    if not prompt:
        return jsonify({"error": "Prompt kosong."}), 400

    result = generate_answer(prompt)
    return jsonify(result)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port)
