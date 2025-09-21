import os
import time
import cv2
import av
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoProcessorBase

import tensorflow as tf
from keras.models import load_model

#  Streamlit page config 
st.set_page_config(page_title="Face + Emotion Detection", page_icon="👁️", layout="centered")
st.title(" Real-Time Emotion Detection")

#  Face Cascade loader 
@st.cache_resource
def load_face_detector():
    cascade_paths = [
        "haarcascades/haarcascade_frontalface_default.xml",
        "src/haarcascades/haarcascade_frontalface_default.xml",
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml",
    ]
    for path in cascade_paths:
        if os.path.exists(path):
            c = cv2.CascadeClassifier(path)
            if not c.empty():
                return c
    return cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

CASCADE = load_face_detector()
if CASCADE.empty():
    st.error("Could not load face detection model.")
    st.stop()

# Sidebar: detector + emotion model 
st.sidebar.header("Detector Settings")
scale_factor = st.sidebar.slider("scaleFactor", 1.05, 1.40, 1.15, 0.01)
min_neighbors = st.sidebar.slider("minNeighbors", 1, 10, 5, 1)
min_size_px = st.sidebar.slider("Min face size (px)", 20, 200, 60, 10)
target_width = st.sidebar.selectbox("Frame width", [320, 480, 640, 800, 960], index=2)
show_fps = st.sidebar.checkbox("Show FPS", True)

st.sidebar.header("Emotion Model")
models_dir = "models"
os.makedirs(models_dir, exist_ok=True)
available_models = sorted([f for f in os.listdir(models_dir) if f.lower().endswith(".h5")])

if not available_models:
    st.sidebar.warning("No .h5 models found in ./models. Save your trained model first.")
selected_model_name = st.sidebar.selectbox("Pick a trained CNN (.h5)", available_models, index=0 if available_models else None)

# Default class labels (edit if your training used different names/order)
default_classes = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
labels_text = st.sidebar.text_area("Class labels (comma-separated, in model order)",
                                   ", ".join(default_classes))
EMO_CLASSES = [s.strip() for s in labels_text.split(",") if s.strip()] or default_classes

conf_threshold = st.sidebar.slider("Min confidence to show label", 0.0, 1.0, 0.30, 0.01)
predict_every_n = st.sidebar.slider("Predict every N frames (perf)", 1, 5, 2, 1)

# Store current UI values for the processor to read
st.session_state["scale_factor"] = scale_factor
st.session_state["min_neighbors"] = min_neighbors
st.session_state["min_size_px"] = min_size_px
st.session_state["target_width"] = target_width
st.session_state["show_fps"] = show_fps
st.session_state["emo_classes"] = EMO_CLASSES
st.session_state["conf_threshold"] = conf_threshold
st.session_state["predict_every_n"] = predict_every_n

@st.cache_resource(show_spinner=False)
def load_emotion_model(path):
    m = load_model(path)
    # Infer expected input (H, W, C)
    inp = m.input_shape
    # Keras can report (None, 48, 48, 1) or a list for multi-input models
    if isinstance(inp, list):
        inp = inp[0]
    _, H, W, C = inp
    return m, (H, W, C)

# ---------------- Video Processor ----------------
class FaceProcessor(VideoProcessorBase):
    def __init__(self):
        # fps helpers
        self.last_ts = None
        self.fps = 0.0
        self.frame_count = 0
        self.last_frame_bgr = None

        # detector params
        self.scale_factor = 1.15
        self.min_neighbors = 5
        self.min_size_px = 60
        self.target_width = 640
        self.show_fps = True

        # emotion model + params
        self.emotion_model = None
        self.inp_size = (48, 48, 1)
        self.emo_classes = default_classes
        self.conf_threshold = 0.30
        self.predict_every_n = 2

    def set_detector_params(self, scale_factor, min_neighbors, min_size_px, target_width, show_fps):
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.min_size_px = min_size_px
        self.target_width = target_width
        self.show_fps = show_fps

    def set_emotion_model(self, model_obj, input_shape_hw_c, emo_classes, conf_threshold, predict_every_n):
        self.emotion_model = model_obj
        self.inp_size = input_shape_hw_c  # (H, W, C)
        self.emo_classes = emo_classes
        self.conf_threshold = conf_threshold
        self.predict_every_n = max(1, int(predict_every_n))

    def _prep_face_tensor(self, face_gray):
        """
        Takes a grayscale face ROI (numpy HxW), resizes to model HxW and returns
        a batch tensor of shape (1, H, W, C) with values in [0,1].
        If model expects C=3, tiles grayscale to 3 channels.
        """
        H, W, C = self.inp_size
        face_resized = cv2.resize(face_gray, (W, H), interpolation=cv2.INTER_AREA)
        face_norm = face_resized.astype("float32") / 255.0

        if C == 1:
            x = face_norm[..., None]  # (H, W, 1)
        else:
            x = np.repeat(face_norm[..., None], C, axis=-1)  # (H, W, 3)

        x = np.expand_dims(x, axis=0)  # (1, H, W, C)
        return x

    def _predict_emotion(self, face_gray):
        if self.emotion_model is None:
            return None, None
        x = self._prep_face_tensor(face_gray)
        probs = self.emotion_model.predict(x, verbose=0)[0]  # (num_classes,)
        idx = int(np.argmax(probs))
        conf = float(probs[idx])
        return idx, conf

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")

        # Resize for stable FPS
        h, w = img.shape[:2]
        if w != self.target_width:
            s = self.target_width / float(w)
            img = cv2.resize(img, (self.target_width, int(h * s)), interpolation=cv2.INTER_AREA)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = CASCADE.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=(self.min_size_px, self.min_size_px),
        )

        # Draw detections + emotion predictions
        for i, (x, y, w, h) in enumerate(faces):
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Run emotion prediction every N frames to keep FPS stable
            do_predict = (self.frame_count % self.predict_every_n == 0)

            if do_predict and self.emotion_model is not None:
                face_roi_gray = gray[y:y+h, x:x+w]
                pred_idx, conf = self._predict_emotion(face_roi_gray)
                if pred_idx is not None:
                    label = self.emo_classes[pred_idx] if pred_idx < len(self.emo_classes) else f"class {pred_idx}"
                    if conf >= self.conf_threshold:
                        text = f"{label} ({conf*100:.1f}%)"
                    else:
                        text = f"{label}"
                    # Label background
                    (tw, th), bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(img, (x, y - th - 8), (x + tw + 6, y), (0, 255, 0), -1)
                    cv2.putText(img, text, (x + 3, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)

        # FPS overlay
        if self.show_fps:
            t = time.time()
            if self.last_ts is not None:
                dt = t - self.last_ts
                if dt > 0:
                    self.fps = 0.9 * self.fps + 0.1 * (1.0 / dt)
            self.last_ts = t
            self.frame_count += 1
            cv2.putText(img, f"FPS: {self.fps:.1f}  Faces: {len(faces)}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

        self.last_frame_bgr = img.copy()
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ---------------- UI + WebRTC ----------------
col1, col2 = st.columns(2)
with col1:
    st.write("Click **Start** below to allow camera access.")

os.makedirs("outputs", exist_ok=True)

ctx = webrtc_streamer(
    key="haar-realtime-emotion",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=FaceProcessor,
    media_stream_constraints={"video": True, "audio": False},
)

# After stream starts, push params + model into the processor
if ctx and ctx.video_processor:
    # Set detector params
    ctx.video_processor.set_detector_params(
        st.session_state["scale_factor"],
        st.session_state["min_neighbors"],
        st.session_state["min_size_px"],
        st.session_state["target_width"],
        st.session_state["show_fps"],
    )

    # Load selected model (if any) and send to processor
    if selected_model_name:
        model_path = os.path.join(models_dir, selected_model_name)
        try:
            emo_model, inp_hw_c = load_emotion_model(model_path)
            ctx.video_processor.set_emotion_model(
                emo_model,
                inp_hw_c,
                st.session_state["emo_classes"],
                st.session_state["conf_threshold"],
                st.session_state["predict_every_n"],
            )
            st.success(f"Loaded model: {selected_model_name} | Expecting input: {inp_hw_c}")
        except Exception as e:
            st.error(f"Failed to load {selected_model_name}: {e}")
else:
    st.info("Press **Start** to begin streaming.")
