import tensorflow as tf
import cv2
import numpy as np
import os
import sqlite3
from datetime import datetime
import sys

# Add SQLite path to system PATH
SQLITE_PATH = r"C:\Users\Asus\Downloads\sqlite"
if SQLITE_PATH not in sys.path:
    sys.path.append(SQLITE_PATH)

# Using keras directly from tensorflow
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout

# Constants
DB_FILE = "face_recognition.db"
DETECTION_CONFIDENCE = 0.6  # Minimum confidence for face detection
RECOGNITION_THRESHOLD = 0.65  # Lowered threshold for better recognition
SIMILARITY_THRESHOLD = 0.60  # Lowered threshold for initial matching
MIN_FACE_SIZE = (64, 64)  # Minimum face size to store

def init_database():
    """Initialize SQLite database"""
    try:
        # Use absolute path for database file
        db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), DB_FILE)
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Create faces table with basic structure first
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS faces (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                face_data BLOB NOT NULL,
                date_added TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_updated TIMESTAMP,
                recognition_count INTEGER DEFAULT 0
            )
        ''')
        
        # Check if face_features column exists
        cursor.execute("PRAGMA table_info(faces)")
        columns = [column[1] for column in cursor.fetchall()]
        
        # Add face_features column if it doesn't exist
        if 'face_features' not in columns:
            cursor.execute('ALTER TABLE faces ADD COLUMN face_features BLOB')
        
        conn.commit()
        print("Database initialized successfully!")
        return True
    except sqlite3.Error as e:
        print(f"Database error: {e}")
        return False
    finally:
        conn.close()

def store_face(name, face_img):
    """Store a face in the database with features"""
    conn = None
    try:
        # Ensure the image is in uint8 format
        if face_img.dtype != np.uint8:
            face_img = (face_img * 255).astype(np.uint8)
        
        # Preprocess face image
        face_img = cv2.resize(face_img, (128, 128))
        
        # Convert to grayscale for consistent feature extraction
        if len(face_img.shape) == 3:
            gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = face_img
            
        # Extract features
        features = extract_face_features(gray)
        if features is None:
            print("Failed to extract features")
            return False
            
        # Convert image to binary
        _, img_encoded = cv2.imencode('.jpg', face_img)
        img_bytes = img_encoded.tobytes()
        
        # Use absolute path for database
        db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), DB_FILE)
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Check if person already exists
        cursor.execute('SELECT id FROM faces WHERE name = ?', (name,))
        existing_face = cursor.fetchone()
        
        if existing_face:
            # Update existing face
            cursor.execute('''
                UPDATE faces 
                SET face_data = ?,
                    face_features = ?,
                    date_added = datetime('now')
                WHERE name = ?
            ''', (img_bytes, features, name))
            print(f"Updated face data for {name}")
        else:
            # Store new face
            cursor.execute('''
                INSERT INTO faces (name, face_data, face_features, date_added)
                VALUES (?, ?, ?, datetime('now'))
            ''', (name, img_bytes, features))
            print(f"Stored new face for {name}")
        
        conn.commit()
        return True
    except sqlite3.Error as e:
        print(f"Database error: {e}")
        return False
    finally:
        conn.close()

def extract_face_features(face_img):
    """Extract features from face image using multiple techniques for better recognition"""
    try:
        # Ensure image is grayscale
        if len(face_img.shape) == 3:
            gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = face_img
        
        # Convert to uint8 if needed
        if gray.dtype != np.uint8:
            gray = (gray * 255).astype(np.uint8)
        
        # Enhance image
        gray = cv2.equalizeHist(gray)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Resize to standard size
        gray = cv2.resize(gray, (128, 128))
        
        features = []
        
        # 1. Grid cell features
        cell_size = (16, 16)
        for i in range(0, gray.shape[0], cell_size[0]):
            for j in range(0, gray.shape[1], cell_size[1]):
                cell = gray[i:i+cell_size[0], j:j+cell_size[1]]
                if cell.size > 0:
                    features.append(np.mean(cell))
                    features.append(np.std(cell))
        
        # 2. Horizontal and vertical projections
        h_proj = np.mean(gray, axis=0)
        v_proj = np.mean(gray, axis=1)
        features.extend(h_proj[::4])  # Take every 4th value
        features.extend(v_proj[::4])
        
        # 3. Basic statistics of regions
        regions = [
            gray[:64, :64],   # Top-left
            gray[:64, 64:],   # Top-right
            gray[64:, :64],   # Bottom-left
            gray[64:, 64:],   # Bottom-right
        ]
        
        for region in regions:
            features.append(np.mean(region))
            features.append(np.std(region))
            features.append(np.median(region))
        
        # Convert to numpy array and normalize
        features = np.array(features, dtype=np.float64)
        if len(features) > 0:
            features = (features - np.mean(features)) / (np.std(features) + 1e-10)
        
        return features.tobytes()
        
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None

def get_all_faces_from_db():
    """Retrieve all faces and their features from database"""
    try:
        # Use absolute path for database
        db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), DB_FILE)
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get all faces with features
        cursor.execute('SELECT name, face_features FROM faces WHERE face_features IS NOT NULL')
        faces = cursor.fetchall()
        
        # Only print debug info when faces are found or added
        if len(faces) > 0 and not hasattr(get_all_faces_from_db, 'last_count'):
            print(f"Found {len(faces)} faces in database:")
            for name, _ in faces:
                print(f"- {name}")
            get_all_faces_from_db.last_count = len(faces)
        
        return faces
    except sqlite3.Error as e:
        print(f"Database error: {e}")
        return []
    finally:
        if 'conn' in locals():
            conn.close()

def compare_features(features1, features2):
    """Compare two sets of face features and return confidence score with improved accuracy"""
    try:
        # Convert binary features to numpy arrays
        if isinstance(features1, bytes):
            features1 = np.frombuffer(features1, dtype=np.float64)
        if isinstance(features2, bytes):
            features2 = np.frombuffer(features2, dtype=np.float64)
        
        # Ensure same length
        min_len = min(len(features1), len(features2))
        features1 = features1[:min_len]
        features2 = features2[:min_len]
        
        # Apply histogram normalization
        features1 = (features1 - np.min(features1)) / (np.max(features1) - np.min(features1) + 1e-10)
        features2 = (features2 - np.min(features2)) / (np.max(features2) - np.min(features2) + 1e-10)
        
        # Calculate multiple similarity metrics
        cosine_similarity = np.dot(features1, features2) / (np.linalg.norm(features1) * np.linalg.norm(features2) + 1e-10)
        euclidean_distance = 1.0 / (1.0 + np.linalg.norm(features1 - features2))
        correlation = np.corrcoef(features1, features2)[0, 1]
        
        # Combine metrics with weights
        similarity = (
            0.5 * cosine_similarity +
            0.3 * euclidean_distance +
            0.2 * (correlation if not np.isnan(correlation) else 0.0)
        )
        
        # Scale to 0-1 range
        similarity = max(0.0, min(1.0, similarity))
        
        return similarity if similarity >= SIMILARITY_THRESHOLD else 0.0
        
    except Exception as e:
        print(f"Error comparing features: {e}")
        return 0.0

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
    """Enhanced face detection with multi-scale and angle support"""
    # Convert to grayscale and enhance contrast
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    
    # Apply Gaussian blur to reduce noise
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Enhanced detection parameters for better accuracy
    scaleFactor = 1.05  # Smaller value = better detection of size variations
    minNeighbors = 6    # Increased for more reliable detection
    minSize = (30, 30)  # Minimum face size to detect
    maxSize = (300, 300)  # Maximum face size to detect
    flags = cv2.CASCADE_SCALE_IMAGE  # Use image pyramid
    
    # Detect faces with enhanced parameters
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=scaleFactor,
        minNeighbors=minNeighbors,
        minSize=minSize,
        maxSize=maxSize,
        flags=flags
    )
    
    # If no faces found, try with more lenient parameters
    if len(faces) == 0:
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(20, 20),
            flags=flags
        )
    
    # Filter out false positives and merge overlapping detections
    if len(faces) > 0:
        final_faces = []
        for (x, y, w, h) in faces:
            # Check face region properties
            face_region = gray[y:y+h, x:x+w]
            
            # Calculate standard deviation of face region
            # (helps eliminate false positives)
            if face_region.std() > 30:  # Minimum contrast threshold
                final_faces.append((x, y, w, h))
        
        faces = np.array(final_faces)
    
    return faces

def clear_database():
    """Remove all faces from the database"""
    try:
        db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), DB_FILE)
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Delete all records from faces table
        cursor.execute('DELETE FROM faces')
        
        # Reset the auto-increment counter
        cursor.execute('DELETE FROM sqlite_sequence WHERE name="faces"')
        
        conn.commit()
        print("Successfully cleared all faces from database!")
        return True
    except sqlite3.Error as e:
        print(f"Database error: {e}")
        return False
    finally:
        if 'conn' in locals():
            conn.close()

def test_sqlite_connection():
    """Test if SQLite is properly connected"""
    try:
        # Use absolute path for database file
        db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), DB_FILE)
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Try to create a test table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS test_connection (
                id INTEGER PRIMARY KEY,
                test_col TEXT
            )
        ''')
        
        # Insert a test record
        cursor.execute('INSERT INTO test_connection (test_col) VALUES (?)', ('test_successful',))
        conn.commit()
        
        # Verify the record
        cursor.execute('SELECT test_col FROM test_connection')
        result = cursor.fetchone()
        
        # Clean up test table
        cursor.execute('DROP TABLE test_connection')
        conn.commit()
        
        print("SQLite connection test successful!")
        return True
    except sqlite3.Error as e:
        print(f"SQLite connection error: {e}")
        return False
    finally:
        if 'conn' in locals():
            conn.close()

def draw_faces(frame, faces, model=None):
    # Draw rectangles and add information
    face_count = len(faces)
    
    # Get all faces from database
    known_faces = get_all_faces_from_db()
    
    # Draw human count in top-left corner with background
    count_text = f"Humans Detected: {face_count}"
    (text_width, text_height), _ = cv2.getTextSize(
        count_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2
    )
    cv2.rectangle(frame, (5, 5), (text_width + 15, 45), (0, 0, 0), -1)
    cv2.putText(frame, count_text, (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Draw rectangles and labels for each face
    for (x, y, w, h) in faces:
        # Calculate eye line (approximately 40% from top of face)
        eye_y = y + int(h * 0.4)
        
        # Draw enhanced face rectangle with eye line
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.line(frame, (x, eye_y), (x+w, eye_y), (0, 255, 255), 1)
        
        # Draw corner markers for better visualization
        corner_length = 20
        # Top left
        cv2.line(frame, (x, y), (x + corner_length, y), (0, 255, 0), 2)
        cv2.line(frame, (x, y), (x, y + corner_length), (0, 255, 0), 2)
        # Top right
        cv2.line(frame, (x + w, y), (x + w - corner_length, y), (0, 255, 0), 2)
        cv2.line(frame, (x + w, y), (x + w, y + corner_length), (0, 255, 0), 2)
        # Bottom left
        cv2.line(frame, (x, y + h), (x + corner_length, y + h), (0, 255, 0), 2)
        cv2.line(frame, (x, y + h), (x, y + h - corner_length), (0, 255, 0), 2)
        # Bottom right
        cv2.line(frame, (x + w, y + h), (x + w - corner_length, y + h), (0, 255, 0), 2)
        cv2.line(frame, (x + w, y + h), (x + w, y + h - corner_length), (0, 255, 0), 2)
        
        # Extract current face for comparison
        face_roi = frame[y:y+h, x:x+w]
        
        # Compare with known faces
        name = "Unknown"
        max_confidence = 0
        confidence_text = ""
        
        if face_roi.size > 0:
            # Extract features from current face
            face_features = extract_face_features(face_roi)
            
            # Compare with all known faces
            if face_features is not None:
                for known_name, known_features in known_faces:
                    if known_features is not None:
                        confidence = compare_features(face_features, known_features)
                        
                        if confidence > max_confidence:
                            max_confidence = confidence
                            if confidence > RECOGNITION_THRESHOLD:
                                name = known_name
                                confidence_text = f" ({confidence:.2%})"
                            else:
                                name = "Unknown"
                                confidence_text = ""
        
        # Add name and confidence label below the face
        label_text = f"{name}{confidence_text}"
        label_color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
        
        # Add background rectangle for text
        (text_width, text_height), _ = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
        )
        cv2.rectangle(frame, 
                     (x, y+h+5), 
                     (x + text_width + 10, y+h+25),
                     (0, 0, 0), -1)
        
        # Add name label
        cv2.putText(frame, label_text, (x+5, y+h+20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, label_color, 2)
        
        # Add face dimensions in smaller text
        dimensions = f"{w}x{h}px"
        cv2.putText(frame, dimensions, (x, y+h+45),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    
    return frame