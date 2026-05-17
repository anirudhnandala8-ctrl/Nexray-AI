# Nexray-AI
NexRay AI – AI-Powered Chest X-ray Diagnosis and Multilingual Medical Report Generation System

🔗 Live Demo:(https://huggingface.co/spaces/Anirudh-22/Nexray-AI)

# Overview

NEXRAY AI provides a complete end-to-end diagnostic pipeline. It doesn't just predict — it validates. Using a custom CNN for diagnosis and Gemini 2.0 Flash as an AI gatekeeper, the app ensures only medical-grade scans are analyzed.

# Core Workflow
Upload X-ray → CNN Analysis → Multilingual Report


Deep Learning Analysis — Processes verified scans through a TensorFlow CNN to classify Normal vs Pneumonia
Multilingual Reporting — Generates instant medical summaries in English, Hindi, and Telugu


# Tech Stack

LayerTechnologyBackendPython, Flask, GunicornDeep LearningTensorFlow, Keras (CNN)Generative AIGoogle Gemini 2.0 FlashImage ProcessingPillow (PIL), NumPyDeploymentHugging Face Spaces

# System Architecture

ComponentTechnologyRolePrimary ModelCNN (xray_model.h5)Classification — Normal vs PneumoniaValidation AIGemini 2.0 FlashImage guardrail & safety checkReport EngineGemini NLPMultilingual summary generationInput Resolution224 × 224 pxStandardized CNN input

# Local Installation

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
** NANDALA ANIRUDH **
   github:https://github.com/anirudhnandala8-ctrl/
   linkedin:https://www.linkedin.com/in/anirudh-nandala-974194408/


⚠️ Disclaimer: This project is for educational and research purposes only. It is not intended to replace professional medical diagnosis.
