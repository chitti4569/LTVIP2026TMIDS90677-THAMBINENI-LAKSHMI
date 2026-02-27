import os
import cv2
import numpy as np
import base64
from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

app = Flask(__name__)

# --- CONFIGURATION ---
MODEL_PATH = "BloodCellModel.h5"
class_labels = ['EOSINOPHIL', 'LYMPHOCYTE', 'MONOCYTE', 'NEUTROPHIL']

# Load Model
if os.path.exists(MODEL_PATH):
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
else:
    model = None
    print("❌ ERROR: Model file not found!")


def is_valid_microscope_slide(img_bgr):
    """
    Checks for the specific purple/pink staining of blood slides.
    Rejects images with high green/blue nature backgrounds.
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # 1. Define Blood Stain Range (Purples/Pinks)
    lower_stain = np.array([130, 40, 40])
    upper_stain = np.array([170, 255, 255])
    stain_mask = cv2.inRange(hsv, lower_stain, upper_stain)
    stain_pct = (np.sum(stain_mask > 0) / (img_bgr.shape[0] * img_bgr.shape[1])) * 100

    # 2. Define Nature Background Range (Greens/Cyan)
    lower_nature = np.array([35, 40, 40])
    upper_nature = np.array([90, 255, 255])
    nature_mask = cv2.inRange(hsv, lower_nature, upper_nature)
    nature_pct = (np.sum(nature_mask > 0) / (img_bgr.shape[0] * img_bgr.shape[1])) * 100

    # LOGIC: A blood slide MUST have some purple and CANNOT have much green
    if nature_pct > 8.0:
        return False, "Detected non-blood colors (Nature/Background)"
    if stain_pct < 0.8:
        return False, "Image does not match blood slide stain profile"

    return True, "Valid"


@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        file = request.files.get('file')
        if not file: return "No file."

        filepath = os.path.join("static", "uploads", file.filename)
        if not os.path.exists("static/uploads"): os.makedirs("static/uploads")
        file.save(filepath)

        img = cv2.imread(filepath)
        if img is None: return "Invalid image."

        # STEP 1: VALIDATE CONTENT
        valid, reason = is_valid_microscope_slide(img)

        if not valid:
            label = "Image not present in dataset"
            confidence = reason
        else:
            # STEP 2: PREDICT
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_res = cv2.resize(img_rgb, (224, 224))
            img_batch = np.expand_dims(preprocess_input(img_res.astype(np.float32)), axis=0)

            preds = model.predict(img_batch)
            max_conf = np.max(preds)
            pred_idx = np.argmax(preds)

            # We lower the threshold to 0.40 so Neutrophils get recognized,
            # because our color filter already blocked the parrots.
            if max_conf < 0.40:
                label = "Image not present in dataset"
                confidence = f"{round(max_conf * 100, 2)}% (Low Confidence)"
            else:
                label = class_labels[pred_idx]
                confidence = f"{round(max_conf * 100, 2)}%"

        _, buffer = cv2.imencode('.png', img)
        img_data = base64.b64encode(buffer).decode('utf-8')
        return render_template('result.html', label=label, confidence=confidence, img_data=img_data)

    return render_template('home.html')


if __name__ == "__main__":
    app.run(debug=True)