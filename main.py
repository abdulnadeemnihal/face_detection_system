import cv2
import numpy as np
from face_detector import create_model, preprocess_image, detect_faces, draw_faces, init_database

def main():
    # Load the pre-trained Haar cascade for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Initialize database with Nadeem's face
    if not init_database():
        print("Error initializing database!")
        return
    
    # Create and initialize the model
    model = create_model()
    print("Model created successfully!")
    
    # Initialize video capture (0 is usually the default webcam)
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    print("Starting face detection... Press 'q' to quit")
    
    while True:
        # Read frame from webcam
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break
        
        # Detect faces using improved Haar cascade
        faces = detect_faces(frame, face_cascade)
        
        # Draw faces with improved visualization
        frame = draw_faces(frame, faces, model)
        
        # Display the frame
        cv2.imshow('Face Detection', frame)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()