import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from ultralytics import YOLO
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import random
import shutil

# --- 1. CONFIGURATION ---

# --- Input/Output Paths ---
# ✅ Folder with your images, ORGANIZED IN SUBFOLDERS by class (e.g., .../Cow/, .../Buffalo/)
INPUT_FOLDER = r'D:/SIH project/Final/preprocessed/BALANCED'

# A new folder where all the results will be saved
OUTPUT_FOLDER = r'D:/Cattle_Analysis_Report/'

# --- Model Paths ---
# Path to your trained Keras classifier model (.h5 file)
CLASSIFIER_MODEL_PATH = r"D:/SIH project/Planned/Py coding/MAM/90/best_cow_buffalo_model.h5"

# Path to your custom-trained 8-point YOLO model (.pt file)
KEYPOINT_MODEL_PATH = r'D:/SIH project/Planned/Py coding/MAM/training_runs/cattle_final_8_point_model_balanced2/weights/best.pt'

# --- Calibration ---
# Your final calibrated pixel-to-centimeter ratio
PIXELS_TO_CM_RATIO = 0.344018 

# --- 2. SETUP ---
print("--- 🚀 Starting Cattle Analysis Pipeline ---")
# Create all necessary output directories
os.makedirs(os.path.join(OUTPUT_FOLDER, '1_Correctly_Classified_Images'), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_FOLDER, '2_Segmentation_Example'), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_FOLDER, '3_Edge_Detection_Example'), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_FOLDER, '4_Keypoint_Detection_Example'), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_FOLDER, '5_Final_Measurement_Visuals'), exist_ok=True)

# --- 3. LOAD MODELS ---
print("🧠 Loading all three models...")
try:
    classifier_model = tf.keras.models.load_model(CLASSIFIER_MODEL_PATH, compile=False)
    
    # ✅ DYNAMIC FIX: Get the model's expected input shape
    expected_input_shape = classifier_model.input_shape
    print(f"Model's expected input shape: {expected_input_shape}")
    if len(expected_input_shape) != 4 or expected_input_shape[0] != None:
        raise ValueError(f"Unexpected input shape: {expected_input_shape}. Expected (None, height, width, channels).")
    input_height, input_width, input_channels = expected_input_shape[1], expected_input_shape[2], expected_input_shape[3]
    print(f"  -> Will resize images to: {input_width}x{input_height} (channels: {input_channels})")
    
    if input_channels != 3:
        print(f"⚠️  Warning: Model expects {input_channels} channels. If not 3 (RGB), you may need to adjust (e.g., grayscale).")
    
    # Optional: Print model summary for full details (uncomment if needed)
    # print(classifier_model.summary())
    
    keypoint_model = YOLO(KEYPOINT_MODEL_PATH)
    segmentation_model = YOLO('yolov8n-seg.pt') 
    print("✅ Models loaded successfully.")
except Exception as e:
    print(f"❌ Error loading models. Please check the paths. Details: {e}")
    exit()

# --- 4. ATC SCORING LOGIC ---
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

# --- 5. HELPER FUNCTION for Classification ---
def predict_class(img):
    # ✅ RESIZE TO MODEL'S EXPECTED SIZE (even if original is 512x512)
    original_shape = img.shape
    img_resized = cv2.resize(img, (input_width, input_height))
    print(f"    Resizing from {original_shape} to {img_resized.shape} for model input.")  # Debug print (remove later if noisy)
    
    # Optional: Convert BGR (OpenCV) to RGB if your model was trained on RGB
    # img_resized = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)  # Uncomment if needed
    
    img_normalized = img_resized.astype(np.float32) / 255.0
    img_batch = np.expand_dims(img_normalized, axis=0)
    
    # Debug: Print batch shape before prediction
    print(f"    Batch shape before predict: {img_batch.shape}")
    
    prediction = classifier_model.predict(img_batch, verbose=0)
    score = float(prediction[0][0])
    label = "buffalo" if score > 0.5 else "cow"
    confidence = score if label == "buffalo" else 1 - score
    return label, confidence
    
# --- 6. MAIN PROCESSING ---
all_results_data = []
classifier_true_labels = []
classifier_pred_labels = []
image_files = []
class_names = keypoint_model.names

# Discover images and their true labels from subfolders
for class_folder in os.listdir(INPUT_FOLDER):
    class_path = os.path.join(INPUT_FOLDER, class_folder)
    if os.path.isdir(class_path):
        for f in os.listdir(class_path):
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_files.append((os.path.join(class_path, f), class_folder.lower()))

if not image_files:
    print("❌ Error: No images found. Make sure your INPUT_FOLDER contains 'Cow' and 'Buffalo' subfolders.")
    exit()

# --- Create Single Random Showcase Images (Tasks 2, 3, 4) ---
print("\n📸 Generating single-image showcase examples...")
random_image_path, _ = random.choice(image_files)
showcase_img = cv2.imread(random_image_path)
showcase_filename = os.path.basename(random_image_path)
if showcase_img is not None:
    seg_results = segmentation_model(showcase_img, verbose=False)
    if seg_results and seg_results[0].masks:
        largest_mask_data = max(seg_results[0].masks.data, key=lambda m: m.sum())
        h, w, _ = showcase_img.shape
        mask_resized = cv2.resize(largest_mask_data.cpu().numpy(), (w, h)).astype(np.uint8)
        segmented_img = cv2.bitwise_and(showcase_img, showcase_img, mask=mask_resized)
        
        # Save Segmentation Example
        cv2.imwrite(os.path.join(OUTPUT_FOLDER, '2_Segmentation_Example', showcase_filename), segmented_img)
        print("  -> Saved segmentation example.")
        
        # Save Edge Detection Example
        edges = cv2.Canny(mask_resized, 100, 200)
        cv2.imwrite(os.path.join(OUTPUT_FOLDER, '3_Edge_Detection_Example', showcase_filename), edges)
        print("  -> Saved edge detection example.")

        # Save Keypoint Detection Example
        kp_results = keypoint_model(segmented_img, conf=0.25)
        if kp_results and kp_results[0].keypoints:
            kp_img = kp_results[0].plot()
            cv2.imwrite(os.path.join(OUTPUT_FOLDER, '4_Keypoint_Detection_Example', showcase_filename), kp_img)
            print("  -> Saved keypoint detection example.")

# --- Process All Images for Final Report ---
print(f"\n🔄 Processing all {len(image_files)} images...")
for image_path, true_label in image_files:
    filename = os.path.basename(image_path)
    print(f"  -> Processing: {filename}")
    original_img = cv2.imread(image_path)
    if original_img is None: continue
    
    # Task 1: Classify using Keras Model
    pred_label, confidence = predict_class(original_img)
    classifier_true_labels.append(true_label)
    classifier_pred_labels.append(pred_label)
    if pred_label == true_label:
        shutil.copy(image_path, os.path.join(OUTPUT_FOLDER, '1_Correctly_Classified_Images', filename))

    # Remove background for keypoint model
    seg_results = segmentation_model(original_img, verbose=False)
    segmented_image = np.zeros_like(original_img)
    if seg_results and seg_results[0].masks:
        largest_mask_data = max(seg_results[0].masks.data, key=lambda m: m.sum())
        h, w, _ = original_img.shape
        mask_resized = cv2.resize(largest_mask_data.cpu().numpy(), (w, h)).astype(np.uint8)
        segmented_image = cv2.bitwise_and(original_img, original_img, mask=mask_resized)
    else:
        segmented_image = original_img

    # Task 5 & 6: Feature Extraction and Scoring
    kp_results = keypoint_model(segmented_image, conf=0.25)
    result = kp_results[0]
    
    torso_h_cm, torso_l_cm, total_h_cm, total_l_cm = 0, 0, 0, 0
    stature_score, body_depth_score = 0, 0
    
    if result.keypoints and result.keypoints.shape[1] == 8:
        keypoints = result.keypoints.xy[0].cpu().numpy()
        c_top,c_bottom,c_left,c_right,ex_left,ex_right,ex_top,ex_bottom = keypoints
        
        torso_h_cm = abs(c_bottom[1] - c_top[1]) * PIXELS_TO_CM_RATIO
        torso_l_cm = abs(c_right[0] - c_left[0]) * PIXELS_TO_CM_RATIO
        total_h_cm = abs(ex_bottom[1] - ex_top[1]) * PIXELS_TO_CM_RATIO
        total_l_cm = abs(ex_right[0] - ex_left[0]) * PIXELS_TO_CM_RATIO

        stature_score = get_stature_score(total_h_cm)
        body_depth_score = get_body_depth_score(torso_h_cm)

        vis_measure = original_img.copy()
        cv2.line(vis_measure, (int(c_left[0]), int(c_left[1])), (int(c_right[0]), int(c_right[1])), (0, 255, 255), 2)
        cv2.line(vis_measure, (int(c_top[0]), int(c_top[1])), (int(c_bottom[0]), int(c_bottom[1])), (255, 0, 255), 2)
        cv2.putText(vis_measure, f"Predicted: {pred_label.upper()}", (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)
        cv2.putText(vis_measure, f"Total Ht: {total_h_cm:.1f} cm (Score: {stature_score})", (15, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)
        cv2.imwrite(os.path.join(OUTPUT_FOLDER, '5_Final_Measurement_Visuals', filename), vis_measure)

    all_results_data.append({
        'Filename': filename, 'True Class': true_label, 'Predicted Class': pred_label, 'Confidence': f"{confidence:.1%}",
        'Total Height (cm)': total_h_cm, 'Total Length (cm)': total_l_cm,
        'Torso Height (cm)': torso_h_cm, 'Torso Length (cm)': torso_l_cm,
        'ATC Stature Score (1-9)': stature_score, 'ATC Body Depth Score (1-9)': body_depth_score
    })

# --- 7. GENERATE FINAL REPORT ---
print("\n📊 Generating final report files...")
df_report = pd.DataFrame(all_results_data)
df_report.to_csv(os.path.join(OUTPUT_FOLDER, '6_Final_Measurements_and_Scores.csv'), index=False)
print("  -> Saved final CSV report.")

# Generate and save Classification Report
report_str = classification_report(classifier_true_labels, classifier_pred_labels)
with open(os.path.join(OUTPUT_FOLDER, '1_Classification_Report.txt'), 'w') as f:
    f.write("--- Classification Report ---\n")
    f.write(report_str)
print("  -> Saved classification report text file.")

# Generate and save Confusion Matrix plot
cm = confusion_matrix(classifier_true_labels, classifier_pred_labels, labels=sorted(list(set(classifier_true_labels))))
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=sorted(list(set(classifier_true_labels))), 
            yticklabels=sorted(list(set(classifier_true_labels))))
plt.title('Classifier Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.savefig(os.path.join(OUTPUT_FOLDER, '1_Confusion_Matrix.png'))
print("  -> Saved confusion matrix plot.")

print("\n--- ✅ All tasks complete! Your report is ready. ---")
