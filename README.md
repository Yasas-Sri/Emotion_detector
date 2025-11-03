#  Real-Time Emotion Detection System

A sophisticated real-time emotion detection application that uses computer vision and deep learning to detect faces and classify emotions, providing personalized movie and book recommendations based on detected mood and live emotion graph

##  Table of Contents
- [Features](#features)
- [System Architecture](#system-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Models](#models)
- [API Keys](#api-keys)
- [Development](#development)
- [Docker Deployment](#docker-deployment)
  

##  Features

### Core Functionality
- **Real-time Face Detection**: 
  - Haar Cascade classifier
  - YuNet deep learning detector
  - Adjustable detection parameters
  
- **Emotion Recognition**:
  - 7 emotion classes: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral
  - Multiple model support: Keras (.h5/.keras) and PyTorch (.pth)
  - Support for CNN architectures (Mini-Xception, EfficientNet, MobileNet, VGG, ResNet)
  - Vision Transformer (ViT) support for advanced emotion detection
  
- **Live Visualization**:
  - Real-time emotion probability graphs
  - FPS counter and face count display
  - Configurable frame processing rate

- **Smart Recommendations**:
  - **Movie Recommendations**: TMDB API integration with mood-based genre matching
  - **Book Recommendations**: Open Library API integration
  - Two modes: "match" (reinforce mood) or "lift" (improve mood)

##  System Architecture

```
┌─────────────────┐
│   Web Browser   │
│   (Streamlit)   │
└────────┬────────┘
         │
┌────────▼────────┐
│  streamlit-webrtc│
│   (WebRTC)      │
└────────┬────────┘
         │
┌────────▼────────────────┐
│   FaceProcessor         │
│ ┌──────────────────┐    │
│ │ Face Detection   │    │
│ │ (Haar/YuNet)     │    │
│ └────────┬─────────┘    │
│          │              │
│ ┌────────▼─────────┐    │
│ │ Emotion Model    │    │
│ │ (CNN/ViT)        │    │
│ └────────┬─────────┘    │
│          │              │
│ ┌────────▼─────────┐    │
│ │ Recommendations  │    │
│ │ (TMDB/OpenLib)   │    │
│ └──────────────────┘    │
└─────────────────────────┘
```

##  Installation

### Prerequisites
- Python 3.9+
- Conda (recommended) or pip
- Webcam access
- TMDB API key (for movie recommendations)

### Option 1: Conda Environment (Recommended)

```bash
# Clone the repository
git clone <repository-url>
cd Emotion_detector

# Create conda environment
conda create -n ml-env python=3.9 -y
conda activate ml-env

# Install dependencies
pip install -r requirements.txt
```

### Option 2: Docker

```bash
# Build the Docker image
docker build -t emotion-detector .

# Run the container
docker run -p 8501:8501 \
  --device=/dev/video0 \
  -e TMDB_API_KEY=your_api_key_here \
  emotion-detector
```

### Option 3: Dev Container (VS Code)

1. Open the project in VS Code
2. Install the "Remote - Containers" extension
3. Press `F1` → "Remote-Containers: Reopen in Container"

##  Usage

### Starting the Application

```bash
# Activate environment
conda activate ml-env

# Run Streamlit app
streamlit run src/main.py
```

The application will open in your browser at `http://localhost:8501`

### Configuration Options

#### Face Detection
- **Detector Type**: Choose between Haar Cascade or YuNet
- **Haar Parameters**:
  - Scale Factor: 1.05 - 1.40 (detection sensitivity)
  - Min Neighbors: 1-10 (false positive reduction)
  - Min Face Size: 20-200 pixels
- **YuNet Parameters**:
  - Score Threshold: 0.1 - 1.0
  - NMS Threshold: 0.1 - 1.0
  - Top K: 100 - 10000

#### Emotion Model
- **Model Selection**: Choose from available `.h5`, `.keras`, or `.pth` models
- **Class Labels**: Configure emotion class names
- **Confidence Threshold**: Minimum confidence to display labels
- **Prediction Rate**: Predict every N frames (performance optimization)

#### Video Settings
- **Frame Width**: 320, 480, 640, 800, or 960 pixels
- **Show FPS**: Toggle FPS counter display
- **Live Graph**: Toggle real-time emotion probability visualization

### Getting Recommendations

1. Start the webcam feed
2. Wait for emotion detection (look at the camera)
3. Click **"Get Movie Recommendations"** or **"Get Book Recommendations"**
4. Choose between:
   - **Match**: Recommendations that align with current mood
   - **Lift**: Recommendations to improve/change mood

##  Project Structure

```
Emotion_detector/
├── src/
│   ├── main.py                 # Main Streamlit application
│   ├── movie.py                # TMDB movie recommendations
│   ├── books.py                # Open Library book recommendations
│   └── detectors/
│       ├── haar_detector.py    # Haar & YuNet face detectors
│       └── vit_detector.py     # Vision Transformer model wrapper
├── models/                     # Trained emotion recognition models
│   ├── effnetb0_fer.keras
│   ├── mini_xception_fer.keras
│   ├── vit_emotion_pytorch.pth
│   └── ...
├── notebooks/                  # Jupyter notebooks for training/EDA
│   ├── fc211019_Maheshi/
│   ├── FC110532_Haritha/
│   ├── FC110533_Dulith/
│   └── ...
├── Data/
│   └── images/
│       ├── train/              # Training dataset
│       └── validation/         # Validation dataset
├── Dockerfile                  # Docker configuration
├── requirements.txt            # Python dependencies
├── .env                        # Environment variables (API keys)
└── README.md                   # This file
```

##  Models

### Supported Architectures

#### Keras Models (.h5 / .keras)
- **Mini-Xception**: Lightweight architecture optimized for emotion recognition(both pretrained and custom)
- **EfficientNetB0**: Transfer learning from ImageNet
- **MobileNet**: Mobile-optimized architecture
- **VGG**: Classic deep CNN architecture
- **ResNet**: Residual network (small variant for 48×48 images)

#### PyTorch Models (.pth)
- **Vision Transformer (ViT)**: State-of-the-art transformer architecture
  - Input size: 224×224×3
  - Pre-trained on ImageNet, fine-tuned on emotion dataset

### Input Requirements
- **CNN Models**: Typically 48×48 or 64×64 grayscale/RGB images
- **ViT Model**: 224×224×3 RGB images

### Training Your Own Model

See the [`notebooks/`](notebooks/) directory for training examples:

##  API Keys

### TMDB API (Movie Recommendations)

1. Sign up at [https://www.themoviedb.org/](https://www.themoviedb.org/)
2. Get your API key from account settings
3. Create a `.env` file in the project root:

```bash
TMDB_API_KEY=your_api_key_here
```

### Open Library API (Book Recommendations)
No API key required - uses public Open Library API

##  Development

### Running Tests

```bash
# Run unit tests
pytest tests/

# Run with coverage
pytest --cov=src tests/
```

### Adding New Models

1. Place model file in [`models/`](models/) directory
2. Ensure model follows naming convention: `model_name.h5`, `.keras`, or `.pth`
3. Model will automatically appear in the UI selector

### Code Structure

- [`FaceProcessor`](src/main.py): Main video processing class
- [`HaarFaceDetector`](src/detectors/haar_detector.py): Haar Cascade implementation
- [`YuNetFaceDetector`](src/detectors/haar_detector.py): YuNet DNN implementation
- [`ViTEmotionModel`](src/detectors/vit_detector.py): Vision Transformer wrapper

##  Docker Deployment

### Build and Run

```bash
# Build image
docker build -t emotion-detector:latest .

# Run with GPU support (optional)
docker run --gpus all -p 8501:8501 \
  --device=/dev/video0 \
  -e TMDB_API_KEY=your_key \
  emotion-detector:latest

# Run without GPU
docker run -p 8501:8501 \
  --device=/dev/video0 \
  -e TMDB_API_KEY=your_key \
  emotion-detector:latest
```

### Environment Variables

- `TMDB_API_KEY`: TMDB API key for movie recommendations
- `STREAMLIT_SERVER_PORT`: Port to run Streamlit (default: 8501)
- `CONDA_DEFAULT_ENV`: Conda environment name (default: ml-env)

##  Dataset

The emotion detection models are trained on standard emotion recognition datasets:
- Training images: ~28,000+ samples
- Validation images: ~7,000+ samples
- 7 emotion classes: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral
- Image size: 48×48 pixels (grayscale) or 224×224 (RGB for ViT)



##  License

[LICENSE](LICENSE)

##  Authors

- Dulith (FC211006)
- Rikaf (FC211028)
- Haritha (FC211005)
- Yasas (FC211035)
- Maheshi (fc211019)

##  Acknowledgments

- OpenCV for computer vision functionality
- Streamlit for the web interface
- TMDB for movie data
- Open Library for book data
- Hugging Face Transformers for ViT implementation


**Note**: This project requires a webcam for real-time emotion detection. Ensure camera permissions are granted in your browser.o
