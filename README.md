# 🏥 Intelligent Skin Cancer Detection System 

An AI-powered skin cancer detection system using intelligent ensemble learning.
[view app](https://huggingface.co/spaces/RavichandraNayakar/Intelligent-Skin-Cancer-Detection-System)

# Context-Aware Ensemble AI Software for Skin Cancer Detection
[Read the full research paper (PDF)](https://acrobat.adobe.com/id/urn:aaid:sc:ap:d48153ea-64c4-45cd-976d-b7dff9071dab)

<img width="1920" height="1028" alt="image" src="https://github.com/user-attachments/assets/4348484e-42de-4a55-af10-e5121552e28b" />

A Python-based project implementing interpretable ensemble learning (KNN + RF) for melanoma detection...

## 🎯 What This Does

Detects **skin cancer** in dermoscopic lesion images by classifying them as:
- **Malignant** (cancerous - needs urgent attention)
- **Benign** (non-cancerous - safe to monitor)

## 🧠 How It Works

Uses an intelligent ensemble of:
- **KNN** (K-Nearest Neighbors) - specialized for complex, irregular lesions
- **Random Forest** - specialized for simple, symmetric lesions
- **Smart Conflict Resolution** - analyzes lesion characteristics to decide which model to trust

## 📊 Performance

- **Accuracy:** 69%
- **F1 Score:** 71.5%
- **Malignant Detection Rate:** 79%

## ⚠️ Important Notes

- Upload **dermoscopic images of skin lesions ONLY**
- Do NOT upload normal skin or other skin conditions
- This is a research/screening tool - always consult a dermatologist for proper diagnosis

## 🔬 Technical Details

- **Dataset:** ISIC 2017 (748 balanced images)
- **Features:** 7 medical features (age, sex, diameter, asymmetry, color variation, border irregularity, compactness)
- **Novel Approach:** Context-aware conflict resolution based on lesion characteristics

## 👨‍💻 Developer

Built as a research project in AI/ML medical image analysis.
