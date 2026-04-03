from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import io
import os
import gdown

app = Flask(__name__)

MODEL_PATH = "model_best.h5"

# Auto download model if not present
if not os.path.exists(MODEL_PATH):
    print("Downloading model from Google Drive...")
    gdown.download(
        "https://drive.google.com/uc?id=1ZF0Hj0wiJcHln-5_YrPXoUFjSDBz8J0Z",
        MODEL_PATH,
        quiet=False
    )
    print("Model downloaded!")

model = load_model(MODEL_PATH)

CANCER_IF_HIGH = False
THRESHOLD = 0.3

@app.route("/", methods=["GET"])
def home():
    return "Breast Cancer API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files["file"]
    try:
        img_bytes = io.BytesIO(file.read())
        img = image.load_img(img_bytes, target_size=(224, 224))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        prediction = model.predict(img_array, verbose=0)
        score = float(prediction[0][0])
        if CANCER_IF_HIGH:
            label = "Cancer" if score > THRESHOLD else "Normal"
        else:
            label = "Normal" if score > (1 - THRESHOLD) else "Cancer"
        confidence = score if label == "Normal" else 1 - score
        return jsonify({
            "result": label,
            "score": round(score, 4),
            "confidence": round(confidence * 100, 2)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)