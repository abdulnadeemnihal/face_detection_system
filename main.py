import cv2
import numpy as np
from face_detector import (
    create_model, preprocess_image, detect_faces, 
    draw_faces, init_database, store_face
)

def main():
    # Test SQLite connection first
    from face_detector import test_sqlite_connection
    if not test_sqlite_connection():
        print("Error: SQLite connection failed!")
        return

    # Load the pre-trained Haar cascade for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Initialize database
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
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('a'):
            # If exactly one face is detected, allow adding to database
            if len(faces) == 1:
                name = input("Enter name for this face: ")
                if name.strip():  # Make sure name is not empty
                    x, y, w, h = faces[0]
                    face_img = frame[y:y+h, x:x+w]  # Extract face region
                    if store_face(name, face_img):
                        print(f"Successfully stored face for {name}")
                    else:
                        print("Failed to store face")
            else:
                print("Please ensure exactly one face is visible to add to database")
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()