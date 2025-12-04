import cv2
import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing import image as keras_image
import os

MODEL_PATH = 'facial recognition/face_classifier_transfer_final.keras'
IMG_HEIGHT = 224
IMG_WIDTH = 224

CONFIDENCE_THRESHOLD = 0.90


def load_model_and_face_detector():
    """
    Load the trained face classification model and OpenCV face detector.
    
    Returns:
        Tuple of (classification_model, face_cascade, class_names)
    """
    print("Loading face classification model...")
    
    if not os.path.exists(MODEL_PATH):
        print(f"\nError: Model file '{MODEL_PATH}' not found!")
        print("\nAvailable model files in current directory:")
        model_files = [f for f in os.listdir('.') if f.endswith('.keras') or f.endswith('.h5')]
        if model_files:
            for i, f in enumerate(model_files, 1):
                print(f"  {i}. {f}")
            print("\nUpdate MODEL_PATH variable in the script to use one of these models.")
        else:
            print("  No .keras or .h5 model files found!")
            print("\nPlease train a model first using:")
            print("  python face_classifier_transfer.py")
        return None, None, None
    
    try:
        model = keras.models.load_model(MODEL_PATH, compile=False)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        print(f"Model loaded successfully from: {MODEL_PATH}")
    except Exception as e:
        print(f"\nError loading model: {str(e)}")
        print("\nTrying alternative model files...")
        
        alternative_models = [
            'face_classifier_transfer_final.keras',
            'best_model.keras',
            'face_classifier_final.keras'
        ]
        
        model = None
        for alt_model in alternative_models:
            if os.path.exists(alt_model):
                try:
                    print(f"Trying to load: {alt_model}")
                    model = keras.models.load_model(alt_model, compile=False)
                    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
                    print(f"Successfully loaded: {alt_model}")
                    break
                except:
                    continue
        
        if model is None:
            print("\nCould not load any model file!")
            print("Please retrain your model or check the model file.")
            return None, None, None
    
    class_names = ['Alexis', 'Dimitri', 'Pallav']
    
    print("Loading face detector...")
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    if face_cascade.empty():
        print("Error: Could not load face detector!")
        return None, None, None
    
    print("Models loaded successfully!")
    return model, face_cascade, class_names


def classify_face(model, face_img, class_names):
    """
    Classify a detected face and return prediction with confidence.
    
    Args:
        model: Trained Keras model
        face_img: Cropped face image (BGR format)
        class_names: List of class names
        
    Returns:
        Tuple of (predicted_class, confidence_score)
    """
    face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    face_resized = cv2.resize(face_rgb, (IMG_HEIGHT, IMG_WIDTH))
    face_array = face_resized / 255.0
    face_array = np.expand_dims(face_array, axis=0)
    
    predictions = model.predict(face_array, verbose=0)
    
    predicted_class_idx = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class_idx]
    predicted_class = class_names[predicted_class_idx]
    
    return predicted_class, confidence, predictions[0]


def draw_prediction(frame, x, y, w, h, label, confidence):
    """
    Draw bounding box and label on the frame.
    
    Args:
        frame: Video frame
        x, y, w, h: Bounding box coordinates
        label: Predicted class name
        confidence: Confidence score
    """
    if confidence > CONFIDENCE_THRESHOLD:
        color = (0, 255, 0)
        
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        
        text = f"{label}: {confidence*100:.1f}%"
        
        (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        
        cv2.rectangle(frame, (x, y - text_height - 10), (x + text_width, y), color, -1)
        
        cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


def process_image(image_path, model, face_cascade, class_names, output_path=None):
    """
    Process a single image: detect faces, classify them, and draw bounding boxes.
    
    Args:
        image_path: Path to input image
        model: Trained classification model
        face_cascade: Face detector
        class_names: List of class names
        output_path: Optional path to save output image
    """
    print(f"\nProcessing image: {image_path}")
    
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Error: Could not load image from {image_path}")
        return
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    print(f"Detected {len(faces)} face(s)")
    
    for i, (x, y, w, h) in enumerate(faces):
        face_img = frame[y:y+h, x:x+w]
        
        predicted_class, confidence, all_probs = classify_face(model, face_img, class_names)
        
        print(f"\nFace {i+1}:")
        print(f"  Predicted: {predicted_class}")
        print(f"  Confidence: {confidence*100:.1f}%")
        print(f"  All probabilities:")
        for j, class_name in enumerate(class_names):
            print(f"    {class_name}: {all_probs[j]*100:.1f}%")
        
        draw_prediction(frame, x, y, w, h, predicted_class, confidence)
    
    if len(faces) == 0:
        print("No faces detected in the image!")
    
    if output_path:
        cv2.imwrite(output_path, frame)
        print(f"\nOutput saved to: {output_path}")
    
    cv2.imshow('Face Classification', frame)
    print("\nPress any key to close the window...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def process_webcam(model, face_cascade, class_names):
    """
    Process webcam feed in real-time: detect and classify faces.
    
    Args:
        model: Trained classification model
        face_cascade: Face detector
        class_names: List of class names
    """
    print("\nStarting webcam...")
    print("Press 'q' to quit")
    print("Press 's' to save current frame")
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        for (x, y, w, h) in faces:
            face_img = frame[y:y+h, x:x+w]
            
            predicted_class, confidence, _ = classify_face(model, face_img, class_names)
            
            draw_prediction(frame, x, y, w, h, predicted_class, confidence)
        
        cv2.putText(frame, f"Faces: {len(faces)}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, "Press 'q' to quit, 's' to save", (10, frame.shape[0] - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow('Face Classification - Webcam', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            filename = f'captured_frame_{frame_count}.jpg'
            cv2.imwrite(filename, frame)
            print(f"Frame saved as: {filename}")
            frame_count += 1
    
    cap.release()
    cv2.destroyAllWindows()
    print("Webcam stopped")


def process_video(video_path, model, face_cascade, class_names, output_path=None):
    """
    Process a video file: detect and classify faces in each frame.
    
    Args:
        video_path: Path to input video
        model: Trained classification model
        face_cascade: Face detector
        class_names: List of class names
        output_path: Optional path to save output video
    """
    print(f"\nProcessing video: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video info: {width}x{height} @ {fps}fps, {total_frames} frames")
    
    out = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_num = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_num += 1
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        for (x, y, w, h) in faces:
            face_img = frame[y:y+h, x:x+w]
            
            predicted_class, confidence, _ = classify_face(model, face_img, class_names)
            
            draw_prediction(frame, x, y, w, h, predicted_class, confidence)
        
        cv2.putText(frame, f"Frame: {frame_num}/{total_frames}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        if out:
            out.write(frame)
        
        cv2.imshow('Face Classification - Video', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Stopped by user")
            break
        
        if frame_num % 30 == 0:
            print(f"Processed {frame_num}/{total_frames} frames...")
    
    cap.release()
    if out:
        out.release()
        print(f"\nOutput video saved to: {output_path}")
    cv2.destroyAllWindows()
    print("Video processing complete")


def main():
    """
    Main function to run face detection and classification.
    """
    print("=" * 60)
    print("Face Detection and Classification")
    print("=" * 60)
    
    model, face_cascade, class_names = load_model_and_face_detector()
    
    if model is None or face_cascade is None:
        print("\nCannot proceed without model and face detector.")
        print("Please fix the errors above and try again.")
        return
    
    print("\nOptions:")
    print("1. Process single image")
    print("2. Process webcam (real-time)")
    print("3. Process video file")
    
    choice = input("\nEnter your choice (1/2/3): ").strip()
    
    if choice == "1":
        image_path = input("Enter image path: ").strip()
        save_output = input("Save output? (y/n): ").strip().lower()
        
        if save_output == 'y':
            output_path = input("Enter output path (e.g., output.jpg): ").strip()
        else:
            output_path = None
        
        process_image(image_path, model, face_cascade, class_names, output_path)
    
    elif choice == "2":
        process_webcam(model, face_cascade, class_names)
    
    elif choice == "3":
        video_path = input("Enter video path: ").strip()
        save_output = input("Save output video? (y/n): ").strip().lower()
        
        if save_output == 'y':
            output_path = input("Enter output path (e.g., output.mp4): ").strip()
        else:
            output_path = None
        
        process_video(video_path, model, face_cascade, class_names, output_path)
    
    else:
        print("Invalid choice!")


if __name__ == "__main__":
    main()