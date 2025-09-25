from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import time
import tensorflow as tf
from PIL import Image
import numpy as np
from ultralytics import YOLO
import cv2
import csv

# ----------------- Flask Setup -----------------
app = Flask(__name__)
CORS(app, resources={r"/predict-side": {"origins": "http://localhost:3000"}})

# ----------------- File Paths ------------------
RESULTS_FILE_PATH = r"D:/react/Result_final/analysis_results.csv"
IMAGES_UPLOAD_PATH = r"D:/react/Result_final/images/"

CLASSIFIER_MODEL_PATH = r"D:/react/backend/MOrning/best_cow_buffalo_others.h5"
KEYPOINT_MODEL_PATH = r"D:/react/backend/best.pt"

# Pixel to cm conversion ratio
PIXELS_TO_CM_RATIO = 0.344018

# ----------------- Load Models -----------------
try:
    classification_model = tf.keras.models.load_model(CLASSIFIER_MODEL_PATH, compile=False)
    print(f"✅ Classifier loaded. Input shape: {classification_model.input_shape}")
    keypoint_model = YOLO(KEYPOINT_MODEL_PATH)
    print("✅ YOLO model loaded successfully.")
except Exception as e:
    print(f"❌ Error loading models: {e}")
    exit()

# ----------------- ATC Scoring -----------------
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

# ----------------- Utilities -----------------
def save_to_csv(data):
    try:
        os.makedirs(os.path.dirname(RESULTS_FILE_PATH), exist_ok=True)
        file_exists = os.path.isfile(RESULTS_FILE_PATH)
        with open(RESULTS_FILE_PATH, mode='a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=data.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(data)
        print(f"✅ Results written to CSV: {RESULTS_FILE_PATH}")
    except Exception as e:
        print(f"❌ Error writing CSV: {e}")

def save_uploaded_image(file):
    try:
        os.makedirs(IMAGES_UPLOAD_PATH, exist_ok=True)
        image_path = os.path.join(IMAGES_UPLOAD_PATH, file.filename)
        file.save(image_path)
        print(f"✅ Image saved: {image_path}")
    except Exception as e:
        print(f"❌ Error saving image: {e}")

def predict_class(img):
    """Run 3-class classification on resized image"""
    img_resized = cv2.resize(img, (512, 512))
    img_normalized = img_resized.astype(np.float32) / 255.0
    img_batch = np.expand_dims(img_normalized, axis=0)
    
    prediction = classification_model.predict(img_batch, verbose=0)  # shape: (1, 3)
    prediction = prediction[0]  # get array of 3 class probabilities
    
    # Find the class with the highest probability
    class_idx = np.argmax(prediction)
    label_map = {0: "cow", 1: "buffalo", 2: "others"}  # adjust if needed
    label = label_map[class_idx]
    
    confidence = float(prediction[class_idx])  # probability of predicted class
    return label, confidence


# ----------------- API -----------------
@app.route('/predict-side', methods=['POST'])
def predict_side():
    try:
        if 'side' not in request.files:
            return jsonify({'error': 'No side image uploaded'}), 400

        side_image_file = request.files['side']
        save_uploaded_image(side_image_file)
        side_image_file.seek(0)

        side_image = Image.open(side_image_file).convert('RGB')
        side_image_np = np.array(side_image)

        image_filename = side_image_file.filename

        # ---------- Resize once ----------
        resized_img = cv2.resize(side_image_np, (512, 512))

        # ---------- Classification ----------
        pred_label, confidence = predict_class(resized_img)
        score = float(confidence * 100)

        if pred_label not in ["cow", "buffalo"] or score < 80.0:
            return jsonify({'error': 'Enter a valid picture'}), 400

        # ---------- YOLO Keypoint Detection ----------
        seg_results = keypoint_model(resized_img, conf=0.80)  # high confidence

        if not seg_results or len(seg_results[0].keypoints.xy) == 0:
            return jsonify({'error': 'No valid animal detected'}), 400

        keypoints = seg_results[0].keypoints.xy[0].cpu().numpy()
        if keypoints.shape[0] < 8:
            return jsonify({'error': 'Incomplete keypoints detected'}), 400

        # ---------- Measurements ----------
        c_top, c_bottom, c_left, c_right, ex_left, ex_right, ex_top, ex_bottom = keypoints
        torso_height_cm = float(abs(c_bottom[1] - c_top[1]) * PIXELS_TO_CM_RATIO)
        torso_length_cm = float(abs(c_right[0] - c_left[0]) * PIXELS_TO_CM_RATIO)
        total_height_cm = float(abs(ex_bottom[1] - ex_top[1]) * PIXELS_TO_CM_RATIO)
        total_length_cm = float(abs(ex_right[0] - ex_left[0]) * PIXELS_TO_CM_RATIO)

        # ---------- Measurement Validation ----------
        if total_height_cm < 70.0 or total_length_cm < 100.0:
            return jsonify({'error': 'Enter a valid picture'}), 400

        stature_score = float(get_stature_score(total_height_cm))
        body_depth_score = float(get_body_depth_score(torso_height_cm))

        # ---------- Score Validation ----------
        if stature_score < 5 or body_depth_score < 5:
            return jsonify({'error': 'Enter a valid picture'}), 400

        # ---------- Save Results ----------
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
        save_to_csv(results_data)

        return jsonify(results_data)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ----------------- Run -----------------
if __name__ == '__main__':
    app.run(debug=True, port=5000)
