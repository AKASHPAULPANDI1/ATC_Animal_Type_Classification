from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import time
import tensorflow as tf
from PIL import Image
import numpy as np
import torch
from torchvision import transforms
from ultralytics import YOLO
import cv2
import csv

# The Flask application setup
app = Flask(__name__)
CORS(app, resources={r"/predict-side": {"origins": "http://localhost:3000"}})

# --- CSV File Configuration ---
RESULTS_FILE_PATH = r"D:/react/Result/analysis_results.csv"

# Load models
CLASSIFIER_MODEL_PATH = r"D:/react/backend/cow_buffalo_balanced_classifier.h5"
KEYPOINT_MODEL_PATH = r'D:/react/backend/best.pt'
PIXELS_TO_CM_RATIO = 0.344018

try:
    classification_model = tf.keras.models.load_model(CLASSIFIER_MODEL_PATH, compile=False)
    expected_input_shape = classification_model.input_shape
    input_height, input_width, input_channels = expected_input_shape[1], expected_input_shape[2], expected_input_shape[3]
    print(f"Model's expected input shape: {expected_input_shape}")
    keypoint_model = YOLO(KEYPOINT_MODEL_PATH)
    print("✅ Models loaded successfully.")
except Exception as e:
    print(f"❌ Error loading models: {e}")
    print("💡 Fix: Install TensorFlow 2.12.0: pip install tensorflow==2.12.0")
    exit()

# ATC Scoring Logic
def get_stature_score(height_cm):
    MIN_H, MAX_H, STEP = 110.0, 140.0, 3.75
    if height_cm <= MIN_H: return 1
    if height_cm >= MAX_H: return 9
    return min(9, round(1 + (height_cm - MIN_H) / STEP))

def get_body_depth_score(depth_cm):
    MIN_D, MAX_D, STEP = 60.0, 85.0, 3.125
    if depth_cm <= MIN_D: return 1
    if depth_cm >= MAX_D: return 9
    return min(9, round(1 + (depth_cm - MIN_D) / STEP))

# Helper Function for Classification
def predict_class(img):
    img_resized = cv2.resize(img, (input_width, input_height))
    img_normalized = img_resized.astype(np.float32) / 255.0
    img_batch = np.expand_dims(img_normalized, axis=0)
    prediction = classification_model.predict(img_batch, verbose=0)
    score = float(prediction[0][0])
    label = "buffalo" if score > 0.5 else "cow"
    confidence = score if label == "buffalo" else 1 - score
    return label, confidence

def save_to_csv(data):
    """Saves the analysis results to a CSV file."""
    try:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(RESULTS_FILE_PATH), exist_ok=True)
        file_exists = os.path.isfile(RESULTS_FILE_PATH)
        with open(RESULTS_FILE_PATH, mode='a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=data.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(data)
        print(f"✅ Analysis results written to CSV: {RESULTS_FILE_PATH}")
    except Exception as e:
        print(f"❌ Error writing to CSV file: {e}")

@app.route('/predict-side', methods=['POST'])
def predict_side():
    try:
        if 'side' not in request.files:
            return jsonify({'error': 'No side image uploaded'}), 400

        side_image_file = request.files['side']
        side_image = Image.open(side_image_file).convert('RGB')
        side_image_np = np.array(side_image)
        
        image_filename = side_image_file.filename

        # Classification
        pred_label, confidence = predict_class(side_image_np)
        score = float(confidence * 100) # Convert to standard float

        # Keypoint Detection and Measurements
        seg_results = keypoint_model(side_image_np, conf=0.25)
        total_height_cm, total_length_cm, torso_height_cm, torso_length_cm = 0.0, 0.0, 0.0, 0.0
        stature_score, body_depth_score = 0.0, 0.0

        if seg_results and seg_results[0].keypoints and seg_results[0].keypoints.shape[1] == 8:
            keypoints = seg_results[0].keypoints.xy[0].cpu().numpy()
            c_top, c_bottom, c_left, c_right, ex_left, ex_right, ex_top, ex_bottom = keypoints

            torso_height_cm = float(abs(c_bottom[1] - c_top[1]) * PIXELS_TO_CM_RATIO)
            torso_length_cm = float(abs(c_right[0] - c_left[0]) * PIXELS_TO_CM_RATIO)
            total_height_cm = float(abs(ex_bottom[1] - ex_top[1]) * PIXELS_TO_CM_RATIO)
            total_length_cm = float(abs(ex_right[0] - ex_left[0]) * PIXELS_TO_CM_RATIO)

            stature_score = float(get_stature_score(total_height_cm))
            body_depth_score = float(get_body_depth_score(torso_height_cm))

        # Prepare data for CSV
        results_data = {
            'image_filename': image_filename,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'classification': pred_label,
            'score': score,
            'total_height_cm': total_height_cm,
            'total_length_cm': total_length_cm,
            'torso_height_cm': torso_height_cm,
            'torso_length_cm': torso_length_cm,
            'stature_score': stature_score,
            'body_depth_score': body_depth_score
        }

        # Save results to the CSV file
        save_to_csv(results_data)

        return jsonify({
            'classification': pred_label,
            'score': score,
            'total_height_cm': total_height_cm,
            'total_length_cm': total_length_cm,
            'torso_height_cm': torso_height_cm,
            'torso_length_cm': torso_length_cm,
            'stature_score': stature_score,
            'body_depth_score': body_depth_score
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# The login and signup routes are removed as they are no longer relevant without a user database.
# The database connection logic is also removed.

if __name__ == '__main__':
    app.run(debug=True, port=5000)
