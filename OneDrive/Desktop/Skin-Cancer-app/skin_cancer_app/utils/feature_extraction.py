import cv2
import numpy as np

def preprocess_lesion(image_path):
    '''
    Preprocess skin lesion image
    Returns: original RGB image and blurred grayscale
    '''
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    return img_rgb, blurred


def extract_advanced_features(image_path):
    '''
    Extract 5 medical features from skin lesion image:
    - diameter_pixels
    - asymmetry_score
    - color_variation
    - border_irregularity
    - compactness
    '''
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Threshold and find contours
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)

        # Calculate intermediate values
        area = cv2.contourArea(largest_contour)
        perimeter = cv2.arcLength(largest_contour, True)

        # 1. Diameter
        rect = cv2.minAreaRect(largest_contour)
        diameter = max(rect[1])

        # 2. Asymmetry
        asymmetry = perimeter**2 / (4 * np.pi * area) if area > 0 else 0

        # 3. Color Variation
        mask = np.zeros(gray.shape, np.uint8)
        cv2.drawContours(mask, [largest_contour], 0, 255, -1)
        lesion_pixels = img_rgb[mask == 255]
        color_std = np.std(lesion_pixels) if len(lesion_pixels) > 0 else 0

        # 4. Border Irregularity
        hull = cv2.convexHull(largest_contour)
        hull_area = cv2.contourArea(hull)
        border_irregularity = (hull_area - area) / hull_area if hull_area > 0 else 0

        # 5. Compactness
        compactness = (4 * np.pi * area) / (perimeter**2) if perimeter > 0 else 0

        return diameter, asymmetry, color_std, border_irregularity, compactness
    else:
        return 0, 0, 0, 0, 0


def extract_features_for_prediction(image_path, age, sex):
    '''
    Extract ALL 7 features needed for model prediction

    Parameters:
    - image_path: path to skin lesion image
    - age: patient age (numeric)
    - sex: patient sex ('male' or 'female')

    Returns:
    - numpy array with 7 features: [age, sex, diameter, asymmetry, color, border, compactness]
    '''
    # Extract 5 image features
    diameter, asymmetry, color, border, compactness = extract_advanced_features(image_path)

    # Encode sex
    sex_encoded = 1 if sex.lower() == 'male' else 0

    # Return ALL 7 features in EXACT training order
    features = np.array([age, sex_encoded, diameter, asymmetry, color, border, compactness])

    return features.reshape(1, -1)  # Reshape for model input



