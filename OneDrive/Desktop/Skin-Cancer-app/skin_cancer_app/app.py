import gradio as gr
import sys
import os

# Add utils to path
sys.path.append('utils')

from feature_extraction import extract_features_for_prediction
from ensemble import IntelligentEnsemble

# Load ensemble model
ensemble = IntelligentEnsemble(
    model_K_path='models/model_K.pkl',
    model_R_path='models/model_R.pkl',
    scalar_path='models/scalar.pkl'
)

def predict_skin_cancer(image, age, sex):
    '''
    Main prediction function for Gradio interface

    Parameters:
    - image: uploaded image file
    - age: patient age (number)
    - sex: patient sex (dropdown)

    Returns:
    - Formatted prediction results
    '''
    try:
        # Save uploaded image temporarily
        temp_image_path = "temp_lesion.jpg"
        image.save(temp_image_path)

        # Extract features
        features = extract_features_for_prediction(temp_image_path, age, sex)

        # Get predictions
        ensemble_pred, knn_pred, rf_pred, reason = ensemble.predict_single(features)

        # Get confidence scores (if available)
        knn_proba, rf_proba = ensemble.get_confidence_scores(features)

        # Format output
        result = f"""
### 🔬 PREDICTION RESULTS

#### 🎯 Final Ensemble Prediction: **{'MALIGNANT ⚠️' if ensemble_pred == 1 else 'BENIGN ✅'}**

---

#### 📊 Individual Model Predictions:

**KNN Model:** {'Malignant' if knn_pred == 1 else 'Benign'}
**Random Forest Model:** {'Malignant' if rf_pred == 1 else 'Benign'}

---

#### 🧠 Decision Logic:
{reason}

---

#### 📈 Model Performance:
- **Ensemble Accuracy:** 69%
- **Ensemble F1 Score:** 71.5%
- **Malignant Detection Rate:** 79%

---

#### ⚠️ Medical Disclaimer:
This is an AI research tool. Always consult a dermatologist for proper diagnosis.
        """

        # Clean up temp file
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)

        return result

    except Exception as e:
        return f"Error processing image: {str(e)}"


# Create Gradio interface
demo = gr.Interface(
    fn=predict_skin_cancer,
    inputs=[
        gr.Image(type="pil", label="Upload Skin Lesion Image"),
        gr.Number(label="Patient Age", value=50, minimum=0, maximum=120),
        gr.Dropdown(choices=["male", "female"], label="Patient Sex", value="female")
    ],
    outputs=gr.Markdown(label="Prediction Results"),
    title="🏥 Intelligent Skin Cancer Detection System",
    description="""
    ### Multi-Model Ensemble with Smart Conflict Resolution

    This system uses:
    - **KNN** (specialized for complex, irregular lesions)
    - **Random Forest** (specialized for simple, symmetric lesions)
    - **Intelligent Decision Logic** (chooses which model to trust)

    ⚠️ **IMPORTANT:** 
     - Upload **dermoscopic images of skin lesions ONLY**
     - Do NOT upload normal skin images
     - This is a research tool - always consult a dermatologist

     **Upload a lesion image** and provide patient information to get a prediction.
    """,
    examples=[
        # Add example images here if you have them
    ],
    theme=gr.themes.Soft()
)

if __name__ == "__main__":
    demo.launch(share=True)
