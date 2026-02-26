# Face Detection & Recognition System

Real-time face detection and recognition using **OpenCV DNN** (SSD + ResNet-10) for detection and **FaceNet** (`InceptionResnetV1` / VGGFace2) for 512-dimensional embedding-based recognition. Includes both a **Flask web UI** and a **standalone desktop mode**.

---

## Tech Stack

| Layer          | Technology                                       |
| -------------- | ------------------------------------------------ |
| Detection      | OpenCV DNN (`res10_300x300_ssd_iter_140000`)     |
| Recognition    | FaceNet via `facenet-pytorch` (512-dim embeddings) |
| Backend        | Flask 3 + Flask-CORS                              |
| Database       | SQLite 3                                          |
| Frontend       | Vanilla HTML / CSS / JS (dark theme, no frameworks) |
| Production     | Gunicorn (gthread)                                |

---

## Project Structure

```
face_detection system/
├── config.py                 # All constants + model auto-download
├── requirements.txt
├── .env.example
│
├── core/
│   ├── __init__.py           # Public API exports
│   ├── database.py           # FaceDatabase (SQLite CRUD)
│   ├── detector.py           # FaceDetector (OpenCV DNN)
│   ├── recognizer.py         # FaceRecognizer (FaceNet)
│   └── pipeline.py           # FacePipeline (orchestrator)
│
├── routes/
│   ├── __init__.py
│   ├── detection.py          # POST /detect_faces, GET /faces
│   └── registration.py       # POST /register_face, DELETE /faces/<name>
│
├── app.py                    # Flask application factory
├── wsgi.py                   # WSGI entry-point
├── gunicorn_config.py        # Gunicorn settings
├── main.py                   # Standalone desktop mode (OpenCV window)
│
├── templates/
│   └── index.html            # Single-page web UI
├── static/
│   ├── js/app.js             # Client-side logic
│   └── css/style.css         # Dark-theme styles
│
├── models/                   # DNN model files (auto-downloaded)
├── data/                     # SQLite database (faces.db)
└── logs/                     # Rotating application logs
```

---

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd "face_detection system"

# Create a virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux / macOS

# Install dependencies
pip install -r requirements.txt
```

> On first run the OpenCV DNN Caffe model (~10 MB) is **automatically downloaded** to `models/`.

---

## Usage

### Desktop Mode (OpenCV window)

```bash
python main.py
```

Opens your webcam in a local window with real-time detection + recognition.

**Hotkeys:**

| Key | Action |
| --- | ------ |
| `a` | Register a face (prompts for name in console, captures 3 frames, picks sharpest) |
| `d` | Toggle debug overlay (scores, embedding distances, frame dimensions) |
| `c` | Print stats to console (total faces, FPS, device) |
| `q` | Quit gracefully |

### Web Mode (Flask server)

```bash
# Development
python wsgi.py

# Production
gunicorn -c gunicorn_config.py wsgi:application
```

Then open **http://localhost:5000** in your browser.

---

## API Endpoints

| Method | Path                | Description                           |
| ------ | ------------------- | ------------------------------------- |
| GET    | `/`                 | Serve the web UI                      |
| GET    | `/health`           | Health check + uptime + face count    |
| POST   | `/detect_faces`     | Detect & recognise faces in base64 image |
| POST   | `/register_face`    | Register a new face (name + base64 image) |
| GET    | `/faces`            | List all registered faces             |
| DELETE | `/faces/<name>`     | Delete a registered face              |

### POST /detect_faces

**Request:**
```json
{"image": "<base64 JPEG>"}
```

**Response 200:**
```json
{
  "success": true,
  "faces": [{"bbox": [x1,y1,x2,y2], "name": "Alice", "confidence": 0.98, "similarity": 0.89}],
  "count": 1,
  "frame_b64": "<annotated base64>"
}
```

### POST /register_face

**Request:**
```json
{"image": "<base64 JPEG>", "name": "Alice"}
```

**Response 201:**
```json
{"success": true, "name": "Alice", "message": "Face registered successfully"}
```

---

## Configuration

Copy `.env.example` to `.env` and adjust as needed:

```bash
cp .env.example .env
```

Key settings live in `config.py` and can be overridden via environment variables:

| Variable                | Default | Description                    |
| ----------------------- | ------- | ------------------------------ |
| `FLASK_HOST`            | 0.0.0.0 | Server bind address           |
| `FLASK_PORT`            | 5000    | Server port                    |
| `DB_PATH`               | data/faces.db | SQLite database path      |
| `DNN_CONFIDENCE`        | 0.50    | Minimum detection confidence   |
| `RECOGNITION_THRESHOLD` | 0.75    | Minimum similarity for a match |
| `LOG_LEVEL`             | INFO    | Python logging level           |

---

## Requirements

- Python 3.10+
- Webcam (for desktop mode or browser-based capture)
- Internet connection on first run (to download the DNN model)

---

## License

MIT
