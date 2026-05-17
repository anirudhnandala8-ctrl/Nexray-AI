# Nexray-AI
NexRay AI – AI-Powered Chest X-ray Diagnosis and Multilingual Medical Report Generation System
🩻 NEXRAY AI — Intelligent Pneumonia Detection

A state-of-the-art medical imaging application combining deep learning and generative AI to detect Pneumonia from Chest X-rays with high precision and automated multilingual reporting.

🔗 Live Demo · 📁 Report Bug · ⭐ Star this repo if you found it useful!

📌 Overview
NEXRAY AI provides a complete end-to-end diagnostic pipeline. It doesn't just predict — it validates. Using a custom CNN for diagnosis and Gemini 2.0 Flash as an AI gatekeeper, the app ensures only medical-grade scans are analyzed.
Core Workflow
Upload X-ray → AI Gatekeeper (Gemini) → CNN Analysis → Multilingual Report

AI Gatekeeper — Filters out non-X-ray images (landscapes, objects, etc.) using Gemini's vision
Deep Learning Analysis — Processes verified scans through a TensorFlow CNN to classify Normal vs Pneumonia
Multilingual Reporting — Generates instant medical summaries in English, Hindi, and Telugu


🛠️ Tech Stack
LayerTechnologyBackendPython, Flask, GunicornDeep LearningTensorFlow, Keras (CNN)Generative AIGoogle Gemini 2.0 FlashImage ProcessingPillow (PIL), NumPyDeploymentHugging Face Spaces

📊 System Architecture
ComponentTechnologyRolePrimary ModelCNN (xray_model.h5)Classification — Normal vs PneumoniaValidation AIGemini 2.0 FlashImage guardrail & safety checkReport EngineGemini NLPMultilingual summary generationInput Resolution224 × 224 pxStandardized CNN input
🛡️ AI Gatekeeper Logic
Instead of a simple binary check, the system uses a Description-Based Validation strategy — it analyzes each image for medical anatomical markers like ribs, lung fields, and chest cavities. Non-medical images (landscapes, objects, faces) are strictly rejected to maintain diagnostic integrity.

💻 Local Installation
1. Clone the repository
bashgit clone https://github.com/Anirudh-22/Nexray-AI.git
cd Nexray-AI
2. Install dependencies
bashpip install -r requirements.txt
3. Set up environment variables
Create a .env file in the root directory:
envGEMINI_API_KEY=your_api_key_here
4. Run the app
bashpython app.py
Visit http://localhost:7860 in your browser.

👤 Author
N Anirudh — Student ID: 23951A1215
Show Image
Show Image


⚠️ Disclaimer: This project is for educational and research purposes only. It is not intended to replace professional medical diagnosis.
