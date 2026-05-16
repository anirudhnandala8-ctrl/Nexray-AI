import os
import numpy as np
from pathlib import Path
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from google import genai  # Latest 2026 SDK
from dotenv import load_dotenv
# --- 1. INITIAL SETUP ---
app = Flask(__name__)

# Folder pathing
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_FOLDER = BASE_DIR / "static" / "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = str(UPLOAD_FOLDER)

# --- 2. LOAD MODELS ---
# Load your local CNN Model
MODEL_PATH = BASE_DIR / "xray_model.h5"
vision_model = load_model(str(MODEL_PATH))

# --- 3. AI REPORT CONFIGURATION ---
# We get the key from the system environment for security
GEMINI_KEY = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_KEY)

def get_ai_report(label, conf):
    try:
        # Check if key is being detected
        if not GEMINI_KEY:
            print("SERVER ERROR: GEMINI_API_KEY is missing from environment!")
            return "Configuration Error: API Key not found."

        # Calling the Gemini 3.1 Flash Model
        response = client.models.generate_content(
            model="gemini-3.1-flash-lite", 
            contents=f"Analyze findings: {label} with {conf} confidence. Provide 3 clean lines: One in English, one in Hindi, and one in Telugu."
        )

        if response.text:
            # Replace newlines with HTML line breaks for the web interface
            formatted_text = response.text.replace("\n", "<br>")
            return formatted_text
        
        return "Report content was empty. Please try again."

    except Exception as e:
        print(f"Detailed API Error: {e}")
        return "AI Report is synchronizing...<br>Please refresh in 10 seconds."

# --- 4. ROUTES ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded", 400
    
    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400

    if file:
        # 1. Save the uploaded X-ray
        filename = file.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # 2. Preprocess for your CNN (matching your training size 224x224)
        img = image.load_img(filepath, target_size=(224, 224))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # 3. CNN Local Prediction
        preds = vision_model.predict(img_array, verbose=0)
        prob = preds[0][0]

        # 4. Classification Logic
        if prob > 0.5:
            label = "PNEUMONIA"
            conf_val = prob * 100
        else:
            label = "NORMAL"
            conf_val = (1 - prob) * 100

        clean_conf = "{:.2f}".format(conf_val)

        # 5. Generate Multilingual Report via Gemini API
        report = get_ai_report(label, clean_conf)

        # Path for frontend display
        web_path = f"static/uploads/{filename}"

        return render_template(
            'index.html', 
            label=label, 
            confidence=f"{clean_conf}%", 
            report=report, 
            user_image=web_path
        )

if __name__ == '__main__':
    # Local testing: set your GEMINI_API_KEY in terminal first!
    app.run(debug=False)