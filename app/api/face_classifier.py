import cv2
import numpy as np
from tensorflow import keras
import os
import base64
from typing import Tuple, Optional, List

IMG_HEIGHT = 224
IMG_WIDTH = 224
CONFIDENCE_THRESHOLD = 0.90

# Global variables to store loaded model and detector
_model = None
_face_cascade = None
_class_names = ['Alexis', 'Dimitri', 'Pallav']


def get_model_path() -> str:
    """
    Get the path to the model file relative to the API directory.
    Model is in AAI-590-Capstone root, API is in app/api.
    Structure: AAI-590-Capstone/app/api/face_classifier.py
    """
    # Get the directory where this file is located (app/api)
    api_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up two levels: app/api -> app -> AAI-590-Capstone root
    root_dir = os.path.dirname(os.path.dirname(api_dir))
    model_path = os.path.join(root_dir, 'facial recognition/face_classifier_transfer_final.keras')
    return model_path


def load_model_and_face_detector() -> Tuple[Optional[keras.Model], Optional[cv2.CascadeClassifier], List[str]]:
    """
    Load the trained face classification model and OpenCV face detector.
    
    Returns:
        Tuple of (classification_model, face_cascade, class_names)
    """
    global _model, _face_cascade, _class_names
    
    if _model is not None and _face_cascade is not None:
        return _model, _face_cascade, _class_names
    
    model_path = get_model_path()
    
    _model = keras.models.load_model(model_path, compile=False)
    _model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    _face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    return _model, _face_cascade, _class_names


def decode_base64_image(base64_string: str) -> Optional[np.ndarray]:
    """
    Decode a base64-encoded image string to a numpy array (BGR format).
    
    Args:
        base64_string: Base64-encoded image string (with or without data URL prefix)
        
    Returns:
        numpy array representing the image in BGR format, or None if decoding fails
    """
    try:
        # Remove data URL prefix if present
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        
        # Decode base64
        image_data = base64.b64decode(base64_string)
        
        # Convert to numpy array
        nparr = np.frombuffer(image_data, np.uint8)
        
        # Decode image
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        return img
    except Exception as e:
        print(f"Error decoding base64 image: {str(e)}")
        return None


def classify_face(model: keras.Model, face_img: np.ndarray, class_names: List[str]) -> Tuple[str, float, np.ndarray]:
    """
    Classify a detected face and return prediction with confidence.
    
    Args:
        model: Trained Keras model
        face_img: Cropped face image (BGR format)
        class_names: List of class names
        
    Returns:
        Tuple of (predicted_class, confidence_score, all_probabilities)
    """
    face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    face_resized = cv2.resize(face_rgb, (IMG_HEIGHT, IMG_WIDTH))
    face_array = face_resized / 255.0
    face_array = np.expand_dims(face_array, axis=0)
    
    predictions = model.predict(face_array, verbose=0)
    
    predicted_class_idx = np.argmax(predictions[0])
    confidence = float(predictions[0][predicted_class_idx])
    predicted_class = class_names[predicted_class_idx]
    
    return predicted_class, confidence, predictions[0]


def process_frames(frames: List[str]) -> Tuple[Optional[str], float, dict]:
    """
    Process multiple frames and return consensus result.
    
    Args:
        frames: List of base64-encoded image strings
        
    Returns:
        Tuple of (predicted_user, average_confidence, details_dict)
        Returns (None, 0.0, {}) if no faces detected or model not loaded
    """
    model, face_cascade, class_names = load_model_and_face_detector()
    
    if model is None or face_cascade is None:
        return None, 0.0, {"error": "Model or face detector not loaded"}
    
    all_predictions = []
    faces_detected = 0
    
    for frame_idx, frame_base64 in enumerate(frames):
        # Decode base64 image
        frame = decode_base64_image(frame_base64)
        if frame is None:
            continue
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(30, 30)
        )
        
        # Process each detected face
        for (x, y, w, h) in faces:
            faces_detected += 1
            face_img = frame[y:y+h, x:x+w]
            
            predicted_class, confidence, all_probs = classify_face(model, face_img, class_names)
            print(predicted_class)
            print(confidence)
            
            # Only consider predictions above confidence threshold
            if confidence >= CONFIDENCE_THRESHOLD:
                all_predictions.append({
                    'user': predicted_class,
                    'confidence': confidence,
                    'frame': frame_idx
                })
    
    if len(all_predictions) == 0:
        return None, 0.0, {
            "error": "No faces detected or no predictions above confidence threshold",
            "faces_detected": faces_detected
        }
    
    # Calculate consensus (majority vote)
    user_counts = {}
    user_confidences = {}
    
    for pred in all_predictions:
        user = pred['user']
        conf = pred['confidence']
        
        if user not in user_counts:
            user_counts[user] = 0
            user_confidences[user] = []
        
        user_counts[user] += 1
        user_confidences[user].append(conf)
    
    # Find user with most votes
    predicted_user = max(user_counts, key=user_counts.get)
    
    # Calculate average confidence for predicted user
    avg_confidence = np.mean(user_confidences[predicted_user])
    
    details = {
        "total_faces_detected": faces_detected,
        "total_predictions": len(all_predictions),
        "user_votes": user_counts,
        "average_confidence": float(avg_confidence)
    }
    
    return predicted_user, float(avg_confidence), details

