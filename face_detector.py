import tensorflow as tf
import cv2
import numpy as np
import os
import sqlite3
from datetime import datetime

# Using keras directly from tensorflow
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout

# Constants
DB_FILE = "face_recognition.db"
REFERENCE_FACE_PATH = r"C:\Users\Asus\Downloads\dummy1.jpg"

def init_database():
    """Initialize SQLite database and store reference face"""
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        
        # Create faces table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS faces (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                face_data BLOB NOT NULL,
                date_added TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Store reference face if not already in database
        reference_face = cv2.imread(REFERENCE_FACE_PATH)
        if reference_face is not None:
            # Convert image to binary
            _, img_encoded = cv2.imencode('.jpg', reference_face)
            img_bytes = img_encoded.tobytes()
            
            # Check if Nadeem's face already exists
            cursor.execute('SELECT id FROM faces WHERE name=?', ('Nadeem',))
            if not cursor.fetchone():
                cursor.execute('INSERT INTO faces (name, face_data) VALUES (?, ?)',
                             ('Nadeem', img_bytes))
                print("Successfully stored Nadeem's face in database!")
            
            conn.commit()
            return True
        else:
            print("Could not load reference face image!")
            return False
    except sqlite3.Error as e:
        print(f"Database error: {e}")
        return False
    finally:
        conn.close()

def get_face_from_db(name):
    """Retrieve face data from database"""
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute('SELECT face_data FROM faces WHERE name=?', (name,))
        result = cursor.fetchone()
        
        if result:
            # Convert binary data back to image
            nparr = np.frombuffer(result[0], np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            return img
        return None
    except sqlite3.Error as e:
        print(f"Database error: {e}")
        return None
    finally:
        conn.close()

def create_model():
    # Create a CNN model for face detection
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

def preprocess_image(img):
    # Enhanced preprocessing for better detection
    img = cv2.resize(img, (128, 128))
    # Convert to grayscale and improve contrast
    if len(img.shape) == 3:  # If color image
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.equalizeHist(gray)  # Improve contrast
    # Apply Gaussian blur to reduce noise
    img = cv2.GaussianBlur(img, (5, 5), 0)
    img = img / 255.0  # Normalize
    # Reshape for model if needed
    if len(img.shape) == 2:  # If grayscale
        img = np.stack((img,)*3, axis=-1)
    return img

def detect_faces(frame, face_cascade):
    # Convert to grayscale for Haar cascade
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Improved detection parameters
    scaleFactor = 1.1  # Smaller value = better detection but slower
    minNeighbors = 5   # Higher value = less false positives
    minSize = (30, 30) # Minimum face size to detect
    
    # Detect faces using Haar cascade with improved parameters
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=scaleFactor,
        minNeighbors=minNeighbors,
        minSize=minSize
    )
    return faces

def draw_faces(frame, faces, model=None):
    # Draw rectangles and add information
    face_count = len(faces)
    
    # Get reference face from database
    reference_face = get_face_from_db('Nadeem')
    
    # Draw human count in top-left corner
    cv2.putText(frame, f"Humans Detected: {face_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Draw rectangles and labels for each face
    for (x, y, w, h) in faces:
        # Draw blue rectangle around face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        # Extract current face for comparison
        face_roi = frame[y:y+h, x:x+w]
        
        # Compare with reference face if available
        name = "Unknown"
        if reference_face is not None and face_roi.size > 0:
            # Resize face_roi to match reference face size
            face_roi_resized = cv2.resize(face_roi, (reference_face.shape[1], reference_face.shape[0]))
            
            # Calculate similarity using template matching
            result = cv2.matchTemplate(face_roi_resized, reference_face, cv2.TM_CCOEFF_NORMED)
            similarity = result[0][0]
            
            # If similarity is above threshold, label as Nadeem
            if similarity > 0.5:  # Adjust threshold as needed
                name = "Nadeem"
        
        # Add name label below the face
        cv2.putText(frame, name, (x, y+h+20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Add face dimensions in smaller text
        dimensions = f"{w}x{h}px"
        cv2.putText(frame, dimensions, (x, y+h+45),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    
    return frame