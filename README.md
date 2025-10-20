# Face Detection System

A real-time face detection and recognition system using OpenCV, TensorFlow, and SQLite.

## Features

- Real-time face detection using webcam
- Face recognition with database storage
- SQLite integration for storing face data
- Person identification with name display
- Simple and intuitive UI

## Requirements

- Python 3.x
- OpenCV
- TensorFlow
- SQLite3
- NumPy

## Installation

1. Clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/face_detection_system.git
cd face_detection_system
```

2. Create a virtual environment and activate it:
```bash
python -m venv .venv
.venv\Scripts\activate  # On Windows
```

3. Install required packages:
```bash
pip install tensorflow opencv-python numpy
```

## Usage

1. Run the main script:
```bash
python main.py
```

2. The system will:
   - Initialize the database
   - Start your webcam
   - Detect and recognize faces in real-time
   - Display names of recognized people

3. Controls:
   - Press 'q' to quit the application

## File Structure

- `main.py`: Main application script
- `face_detector.py`: Core detection and recognition logic
- `face_recognition.db`: SQLite database for storing face data

## Contributing

Feel free to fork the project and submit pull requests.