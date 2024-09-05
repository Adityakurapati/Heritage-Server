import logging
from flask import Flask, request, jsonify
from fastai.vision.all import *
from pathlib import Path
from flask_cors import CORS
from PIL import Image
import os
import uuid
import traceback

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Get the absolute path of the directory containing the script
script_dir = Path(__file__).resolve().parent
# Build the path to the model file
model_path = script_dir / "model.pkl"
learn = None
UPLOAD_FOLDER = script_dir / 'uploads'


def load_model(model_path):
    global learn
    if not model_path.exists():
        logger.error(f"Model file not found at {model_path}")
        return False
    try:
        logger.info(f"Attempting to load model from {model_path}")
        learn = load_learner(model_path)
        logger.info(f"Model loaded successfully from {model_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        logger.error(traceback.format_exc())
        return False


def predict_language(img_path):
    try:
        logger.info(f"Predicting language for image: {img_path}")
        img = PILImage.create(img_path)
        logger.debug(f"Image created successfully")
        pred_class, pred_idx, outputs = learn.predict(img)
        logger.info(
            f"Prediction complete. Class: {pred_class}, Confidence: {outputs.max().item()}")
        return str(pred_class), outputs.max().item()
    except Exception as e:
        logger.error(f"Error predicting image: {e}")
        logger.error(traceback.format_exc())
        return None, None


@app.route('/upload', methods=['POST'])
def upload_file():
    logger.info('Received upload request')
    logger.debug(f"Request method: {request.method}")
    logger.debug(f"Request headers: {request.headers}")
    logger.debug(f"Request files: {request.files}")
    logger.debug(f"Request form: {request.form}")

    if 'file' not in request.files:
        logger.warning("No file part in the request")
        return jsonify({"success": False, "error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        logger.warning("No selected file")
        return jsonify({"success": False, "error": "No selected file"}), 400

    if file:
        try:
            file_extension = os.path.splitext(file.filename)[1]
            unique_filename = str(uuid.uuid4()) + file_extension
            filename = UPLOAD_FOLDER / unique_filename
            logger.info(f"Saving file as: {filename}")
            file.save(filename)
            logger.info("File saved successfully")
            return jsonify({"success": True, "filename": unique_filename}), 200
        except Exception as e:
            logger.error(f"Error saving file: {e}")
            logger.error(traceback.format_exc())
            return jsonify({"success": False, "error": str(e)}), 500
    else:
        logger.warning("File object is None")
        return jsonify({"success": False, "error": "File object is None"}), 400


@app.route('/predict', methods=['POST'])
def predict():
    logger.info('Received predict request')
    logger.debug(f"Request JSON: {request.json}")

    data = request.json
    if not data or 'filename' not in data:
        logger.warning("No filename provided in the request")
        return jsonify({"error": "No filename provided"}), 400

    filename = data['filename']
    img_path = UPLOAD_FOLDER / filename
    logger.info(f"Attempting to predict language for file: {img_path}")

    if not os.path.exists(img_path):
        logger.warning(f"File not found: {img_path}")
        return jsonify({"error": "File not found"}), 404

    if learn is None:
        logger.error("Model is not loaded")
        return jsonify({"error": "Model is not loaded"}), 500

    try:
        predicted_language, confidence = predict_language(img_path)

        if predicted_language is not None:
            logger.info(
                f"Prediction successful. Language: {predicted_language}, Confidence: {confidence}")
            return jsonify({
                "predicted_language": predicted_language,
                "confidence": confidence
            })
        else:
            logger.error("Prediction failed")
            return jsonify({"error": "Prediction failed"}), 500

    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    logger.info("Starting the application")
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    logger.info(f"Upload folder created: {UPLOAD_FOLDER}")
    if load_model(model_path):
        logger.info("Model loaded successfully, starting Flask server")
        app.run(debug=True, host='0.0.0.0')
    else:
        logger.error("Failed to load the model. Exiting.")
