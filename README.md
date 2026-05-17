# NEXRAY AI — Intelligent Pneumonia Detection

> A state-of-the-art medical imaging application combining deep learning and generative AI to detect Pneumonia from Chest X-rays with high precision and automated multilingual reporting.

🔗LIVE DEMO:(https://huggingface.co/spaces/Anirudh-22/Nexray-AI)
---

## Overview

NEXRAY AI provides a complete end-to-end diagnostic pipeline. It doesn't just predict — it **validates**. Using a custom CNN for diagnosis, the app ensures only medical-grade scans are analyzed.

### Core Workflow

```
Upload X-ray → CNN Analysis → Multilingual Report
```


1. **Deep Learning Analysis** — Processes verified scans through a TensorFlow CNN to classify Normal vs Pneumonia
2. **Multilingual Reporting** — Generates instant medical summaries in **English, Hindi, and Telugu**

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python, Flask, Gunicorn |
| Deep Learning | TensorFlow, Keras (CNN) |
| Generative AI | Google Gemini 2.0 Flash |
| Image Processing | Pillow (PIL), NumPy |
| Deployment | Hugging Face Spaces |

---

## 📊 System Architecture

| Component | Technology | Role |
|---|---|---|
| Primary Model | CNN (`xray_model.h5`) | Classification — Normal vs Pneumonia |
| Report Engine | Gemini NLP | Multilingual summary generation |
| Input Resolution | 224 × 224 px | Standardized CNN input |


## 💻 Local Installation

### 1. Clone the repository
```bash
git clone https://github.com/Anirudh-22/Nexray-AI.git
cd Nexray-AI
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set up environment variables
Create a `.env` file in the root directory:
```env
GEMINI_API_KEY=your_api_key_here
```

### 4. Run the app
```bash
python app.py
```

Visit `http://localhost:7860` in your browser.

---
👤 Author
** NANDALA ANIRUDH **
   github:https://github.com/anirudhnandala8-ctrl/
   linkedin:https://www.linkedin.com/in/anirudh-nandala-974194408/


⚠️ Disclaimer: This project is for educational and research purposes only. It is not intended to replace professional medical diagnosis.
