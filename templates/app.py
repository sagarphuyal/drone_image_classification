from flask import Flask, render_template, request, send_from_directory
import os
from ultralytics import YOLO
import cv2
import numpy as np

app = Flask(__name__)

# Paths for uploads and results
UPLOAD_FOLDER = 'static/uploads'
RESULT_FOLDER = 'static/results'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

# Load YOLOv8 model (replace with your model path if different)
model = YOLO('yolov8n.pt')  # Use your pretrained model path, e.g., 'path/to/yolov8_model.pt'

# Ensure upload and result directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "No file part", 400
    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400
    
    # Save the uploaded file
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    # Process the file (image or video)
    result_path = process_file(file_path, file.filename)

    # Return the result page with the processed file
    return render_template('index.html', uploaded_file=file.filename, result_file=os.path.basename(result_path))

def process_file(file_path, filename):
    # Determine if it's an image or video
    is_video = filename.lower().endswith(('.mp4', '.avi', '.mov'))
    result_filename = f"result_{filename}"
    result_path = os.path.join(app.config['RESULT_FOLDER'], result_filename)

    if is_video:
        # Process video
        cap = cv2.VideoCapture(file_path)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(result_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            # Run YOLOv8 inference
            results = model(frame)
            annotated_frame = results[0].plot()  # Draw detections
            out.write(annotated_frame)

        cap.release()
        out.release()
    else:
        # Process image
        img = cv2.imread(file_path)
        results = model(img)
        annotated_img = results[0].plot()  # Draw detections
        cv2.imwrite(result_path, annotated_img)

    return result_path

@app.route('/static/results/<filename>')
def send_result_file(filename):
    return send_from_directory(app.config['RESULT_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)