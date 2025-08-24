# server.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import tempfile
import os
import whisper

app = Flask(__name__)
CORS(app)  # allow extension to call

# Load model once (choose model size for speed/accuracy tradeoff)
# 'tiny', 'base', 'small', 'medium', 'large'
MODEL_NAME = os.environ.get("WHISPER_MODEL", "base")  # change if you want faster/slower
print("Loading Whisper model:", MODEL_NAME)
model = whisper.load_model(MODEL_NAME)
print("Model loaded.")

@app.route("/transcribe", methods=["POST"])
def transcribe():
    if 'file' not in request.files:
        return "Missing file", 400
    f = request.files['file']
    # save to tempfile
    with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp:
        f.save(tmp.name)
        tmp_path = tmp.name

    try:
        # whisper will call ffmpeg under the hood, accepts many formats
        res = model.transcribe(tmp_path)
        text = res.get("text", "")
        return jsonify({"text": text})
    except Exception as e:
        return str(e), 500
    finally:
        try:
            os.remove(tmp_path)
        except:
            pass

if __name__ == "__main__":
    #app.run(host="0.0.0.0", port=5000, debug=True)
    app.run(debug=True, port=5000)