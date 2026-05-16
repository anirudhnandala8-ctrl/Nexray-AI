import numpy as np
import google.generativeai as genai  # From your NLP report file
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# 1. SETUP: Load both "Brains"
MODEL_PATH = 'xray_model.h5'
vision_model = load_model(MODEL_PATH)

# Paste your API Key from your NLP file here
genai.configure(api_key="AIzaSyBfL1oPGV7aKuTNJaVeCmnjoYO2LW5k4l8")
nlp_model = genai.GenerativeModel('gemini-3.1-flash-lite')

def generate_nlp_report(label, confidence):
    """The 'Handshake' function that uses Gemini"""
    prompt = (
        f"Act as a radiologist. The AI found {label} with {confidence:.2f}% confidence. "
        f"Write a formal 2-sentence summary in English. "
        f"Then provide an accurate translation in Hindi and Telugu. "
        f"Add a medical disclaimer."
    )
# 1. Ask the AI for the content
    response = nlp_model.generate_content(prompt)

    # 2. SIMPLE CHECK: Did the AI actually give us a text response?
    if response.text:
        return response.text
    else:
        return "System Message: The AI could not generate a report. Please check your internet connection."

def predict_image(image_path):
    # 2. PREPROCESS (Vision)
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    final_image = np.expand_dims(img_array, axis=0)

    # 3. PREDICT (Vision)
    prediction_score = vision_model.predict(final_image, verbose=0)
    probability = prediction_score[0][0]

    # 4. LOGIC: Your 0.5 Threshold
    if probability > 0.5:
        label = "PNEUMONIA"
        confidence = probability * 100
    else:
        label = "NORMAL"
        confidence = (1 - probability) * 100

    # 5. THE HANDSHAKE (Connecting the files)
    print(f"\n--- Vision Analysis Complete: {label} ({confidence:.2f}%) ---")
    print("--- Generating Multilingual AI Report... ---\n")
    
    # This calls the NLP logic using the Vision results
    final_report = generate_nlp_report(label, confidence)
    
    print("==========================================")
    print(final_report)
    print("==========================================")

# --- START TESTING ---
# Put your image path here
image_to_test = r"C:\Users\Nandala Anirudh\OneDrive\Desktop\xray_project\outimag2.jpg"

if image_to_test:
    predict_image(image_to_test)