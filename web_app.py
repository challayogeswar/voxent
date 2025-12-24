import os
import uuid
from flask import Flask, request, jsonify, send_from_directory, render_template, redirect, url_for
from werkzeug.utils import secure_filename
import threading
import time
from engine.batch_runner import run
import yaml

# Configuration
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'data', 'input_calls')
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'm4a', 'flac', 'ogg'}
MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50 MB limit

# Global job tracker
JOB_STATUS = {}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_files_async(job_id, config_path):
    """Process uploaded files asynchronously."""
    try:
        JOB_STATUS[job_id] = {"status": "processing", "progress": 0, "message": "Starting processing..."}

        # Run the batch processing
        result = run(config_path)

        JOB_STATUS[job_id] = {
            "status": "completed",
            "result": result,
            "message": f"Processing completed: {result.get('successful', 0)} successful, {result.get('failed', 0)} failed"
        }

    except Exception as e:
        JOB_STATUS[job_id] = {"status": "error", "error": str(e)}

@app.route('/')
def index():
    """Main page with file upload form."""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_files():
    """Handle file uploads and start processing."""
    if 'files' not in request.files:
        return jsonify({"error": "No files provided"}), 400

    files = request.files.getlist('files')
    if not files or files[0].filename == '':
        return jsonify({"error": "No files selected"}), 400

    # Save uploaded files
    uploaded_files = []
    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            uploaded_files.append(filename)
        else:
            return jsonify({"error": f"Invalid file type: {file.filename}"}), 400

    # Start processing in background
    job_id = str(uuid.uuid4())
    config_path = "config/config.yaml"

    thread = threading.Thread(target=process_files_async, args=(job_id, config_path))
    thread.daemon = True
    thread.start()

    return jsonify({
        "message": f"Uploaded {len(uploaded_files)} files successfully",
        "job_id": job_id,
        "files": uploaded_files
    })

@app.route('/status/<job_id>')
def get_status(job_id):
    """Get processing status for a job."""
    if job_id not in JOB_STATUS:
        return jsonify({"error": "Job not found"}), 404

    return jsonify(JOB_STATUS[job_id])

@app.route('/dataset/<label>')
def get_dataset_files(label):
    """Get files in a dataset category."""
    try:
        dataset_dir = f"data/voice_dataset/{label}"
        if not os.path.exists(dataset_dir):
            return jsonify({"error": "Dataset category not found"}), 404

        files = []
        for file in os.listdir(dataset_dir):
            if file.endswith('.wav'):
                file_path = os.path.join(dataset_dir, file)
                size = os.path.getsize(file_path)

                # Parse metadata from filename
                parts = file.replace('.wav', '').split('_')
                confidence = 0
                for part in parts:
                    if part.startswith('conf'):
                        confidence = int(part[4:])

                files.append({
                    "name": file,
                    "size": size,
                    "url": f"/audio/{label}/{file}",
                    "confidence": confidence
                })

        # Sort by confidence descending
        files.sort(key=lambda x: x['confidence'], reverse=True)
        return jsonify({"files": files})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/audio/<label>/<filename>')
def get_audio_file(label, filename):
    """Serve audio files for preview."""
    try:
        return send_from_directory(f"data/voice_dataset/{label}", filename)
    except Exception as e:
        return jsonify({"error": str(e)}), 404

@app.route('/train-ml', methods=['POST'])
def train_ml_classifier():
    """Train the ML classifier using current dataset."""
    try:
        from train_ml_classifier import train_ml_classifier as train_func

        # Get parameters from request
        data = request.get_json() or {}
        min_confidence = data.get('min_confidence', 70.0)
        force_retrain = data.get('force', False)

        # Check if model exists and force is not set
        model_path = "models/ml_gender_classifier.pkl"
        if os.path.exists(model_path) and not force_retrain:
            return jsonify({
                "error": "Model already exists. Use force=true to retrain."
            }), 400

        # Start training in background
        job_id = str(uuid.uuid4())

        def train_async():
            try:
                JOB_STATUS[job_id] = {
                    "status": "training",
                    "progress": 0,
                    "message": "Starting ML classifier training..."
                }

                # Train the classifier
                results = train_func(
                    min_confidence=min_confidence
                )

                JOB_STATUS[job_id] = {
                    "status": "completed",
                    "results": results,
                    "message": f"Training completed with {results['accuracy']:.1%} accuracy"
                }

            except Exception as e:
                JOB_STATUS[job_id] = {
                    "status": "error",
                    "error": str(e)
                }

        thread = threading.Thread(target=train_async)
        thread.daemon = True
        thread.start()

        return jsonify({
            "message": "ML classifier training started",
            "job_id": job_id
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/classifier-status')
def get_classifier_status():
    """Get information about the current classifier."""
    try:
        from classification import get_classifier

        classifier = get_classifier()
        info = classifier.get_classifier_info()

        return jsonify({
            "ml_available": info["ml_available"],
            "ml_model_path": info["ml_model_path"],
            "pitch_thresholds": info["pitch_thresholds"]
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
