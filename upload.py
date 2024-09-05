from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)
CORS(app)  # This will allow CORS for all origins by default

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/upload', methods=['POST'])
def upload_file():
    logging.debug("Received request at /upload")
    
    if 'file' not in request.files:
        logging.error("No file part in the request")
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        logging.error("No selected file")
        return jsonify({'error': 'No selected file'}), 400
    
    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        try:
            file.save(file_path)
            logging.info(f"File saved successfully: {file_path}")
            return jsonify({'success': True, 'filename': file.filename}), 200
        except Exception as e:
            logging.error(f"Error saving file: {e}")
            return jsonify({'error': 'Error saving file'}), 500

if __name__ == '__main__':
    app.run(debug=True)
