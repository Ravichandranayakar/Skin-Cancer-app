import numpy as np
import joblib

class IntelligentEnsemble:
    '''
    Intelligent Ensemble System for Skin Cancer Detection
    Combines KNN and Random Forest with smart conflict resolution
    '''

    def __init__(self, model_K_path, model_R_path, scalar_path):
        '''Load saved models'''
        self.model_K = joblib.load(model_K_path)
        self.model_R = joblib.load(model_R_path)
        self.scalar = joblib.load(scalar_path)

    def predict_single(self, features):
        '''
        Make prediction using intelligent ensemble

        Parameters:
        - features: numpy array [age, sex_encoded, asymmetry, border, compactness]

        Returns:
        - ensemble_prediction: 0 (Benign) or 1 (Malignant)
        - knn_prediction: KNN prediction
        - rf_prediction: RF prediction
        - decision_reason: Why ensemble chose this prediction
        '''
        # Scale features
        features_scaled = self.scalar.transform(features)

        # Get predictions from both models
        knn_pred = self.model_K.predict(features_scaled)[0]
        rf_pred = self.model_R.predict(features_scaled)[0]

        # If they agree, easy decision
        if knn_pred == rf_pred:
            return int(knn_pred), int(knn_pred), int(rf_pred), "Both models agree"

        # If they disagree, use intelligent conflict resolution
        # Based on discovery: KNN better for high asymmetry, RF better for low asymmetry
        asymmetry = features_scaled[0, 2]  # asymmetry is 3rd feature (index 2)

        if asymmetry > 0:  # High asymmetry (complex lesion)
            ensemble_pred = knn_pred
            reason = "KNN trusted (high asymmetry lesion - KNN specialization)"
        else:  # Low asymmetry (simple lesion)
            ensemble_pred = rf_pred
            reason = "RF trusted (low asymmetry lesion - RF specialization)"

        return int(ensemble_pred), int(knn_pred), int(rf_pred), reason

    def get_confidence_scores(self, features):
        '''
        Get probability scores from models

        Returns:
        - knn_proba: KNN probability for each class
        - rf_proba: RF probability for each class
        '''
        features_scaled = self.scalar.transform(features)

        try:
            knn_proba = self.model_K.predict_proba(features_scaled)[0]
            rf_proba = self.model_R.predict_proba(features_scaled)[0]
            return knn_proba, rf_proba
        except:
            # If models don't support predict_proba, return None
            return None, None
