import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import os

def preprocess_face(face):
    face = cv2.resize(face, (48, 48))
    face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    face = img_to_array(face)
    face = np.expand_dims(face, axis=0)
    face = face / 255.0
    return face

def main():
    # Get the current script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Load the model using relative path
    model_path = os.path.join(script_dir, 'models', 'resnet_small_48x48.keras')
    model = load_model(model_path)
    
    # Define emotion labels
    emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    
    # Load face cascade
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + '../../Data/haarcascades/haarcascade_frontalface_default.xml')
    
    # Initialize webcam with DirectShow backend
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    
    if not cap.isOpened():
        print("Error: Could not open camera. Please check:")
        print("1. Camera is enabled in Windows Settings")
        print("2. No other application is using the camera")
        print("3. Try running the script as administrator")
        return
    
    print("Camera opened successfully! Press 'q' to quit.")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break
            
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            # Process each detected face
            for (x, y, w, h) in faces:
                face_roi = frame[y:y+h, x:x+w]
                processed_face = preprocess_face(face_roi)
                
                # Predict emotion
                prediction = model.predict(processed_face)
                emotion_label = emotions[np.argmax(prediction)]
                confidence = np.max(prediction)
                
                # Draw rectangle and label
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                label = f"{emotion_label}: {confidence:.2f}"
                cv2.putText(frame, label, (x, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # Display the frame
            cv2.imshow('Real-time Emotion Detection', frame)
            
            # Break loop on 'q' press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()