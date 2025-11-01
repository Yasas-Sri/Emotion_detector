import os
import time
import cv2
import av
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoProcessorBase
from movie import discover_movies_for_emotion, recommend_from_probs, EMO_CLASSES,show_movies
from books import discover_books_for_emotion as discover_books
from books import recommend_books_from_probs, show_books as show_books_list
from detectors.haar_detector import HaarFaceDetector, YuNetFaceDetector
from detectors.vit_detector import ViTEmotionModel
import tensorflow as tf
from keras.models import load_model
from collections import deque
import pandas as pd
# Correct
from streamlit_autorefresh import st_autorefresh

#  Streamlit page config 
st.set_page_config(page_title="Face + Emotion Detection", layout="centered")
st.title(" Real-Time Emotion Detection")

#  Face Cascade loader 
@st.cache_resource
def load_face_detector():
    cascade_paths = [
        "haarcascades/haarcascade_frontalface_defaultsrcV.xml",
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




# Sidebar: Choose detector
st.sidebar.header("Face Detector")
detector_choice = st.sidebar.radio("Choose Detector", ["Haarcascade", "YuNet"], index=0)

if detector_choice == "Haarcascade":
    # Haarcascade settings
    scale_factor = st.sidebar.slider("scaleFactor", 1.05, 1.40, 1.15, 0.01)
    min_neighbors = st.sidebar.slider("minNeighbors", 1, 10, 5, 1)
    min_size_px = st.sidebar.slider("Min face size (px)", 20, 200, 60, 10)
    target_width = st.sidebar.selectbox("Frame width", [320, 480, 640, 800, 960], index=2)
    show_fps = st.sidebar.checkbox("Show FPS", True)

    detector = HaarFaceDetector(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    # Store only Haarcascade params
    st.session_state["scale_factor"] = scale_factor
    st.session_state["min_neighbors"] = min_neighbors
    st.session_state["min_size_px"] = min_size_px
    st.session_state["target_width"] = target_width
    st.session_state["show_fps"] = show_fps

elif detector_choice == "YuNet":
    # YuNet settings
    score_threshold = st.sidebar.slider("YuNet score threshold", 0.1, 1.0, 0.5, 0.05)
    nms_threshold = st.sidebar.slider("YuNet NMS threshold", 0.1, 1.0, 0.3, 0.05)
    top_k = st.sidebar.slider("YuNet top_k", 100, 10000, 1000, 100)
    target_width = st.sidebar.selectbox("Frame width", [320, 480, 640, 800, 960], index=2)
    show_fps = st.sidebar.checkbox("Show FPS", True)

    detector = YuNetFaceDetector(
        model_path="src/haarcascades/face_detection_yunet_2023mar.onnx",
        conf_threshold=score_threshold,
        nms_threshold=nms_threshold,
        top_k=top_k
    )

    # Store only YuNet params
    st.session_state["score_threshold"] = score_threshold
    st.session_state["nms_threshold"] = nms_threshold
    st.session_state["top_k"] = top_k
    # st.session_state["target_width"] = target_width
    # st.session_state["show_fps"] = show_fps
















st.sidebar.header("Emotion Model")
models_dir = "models"
os.makedirs(models_dir, exist_ok=True)
available_models = sorted([f for f in os.listdir(models_dir) if f.lower().endswith((".h5" ,".keras",".pth"))])

if not available_models:
    st.sidebar.warning("No .h5 models found in ./models. Save your trained model first.")
selected_model_name = st.sidebar.selectbox("Pick a trained CNN (.h5)", available_models, index=0 if available_models else None)

# Default class labels (edit if your training used different names/order)
default_classes = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
labels_text = st.sidebar.text_area("Class labels",
                                   ", ".join(default_classes))

# EMO_CLASSES = [s.strip() for s in labels_text.split(",") if s.strip()] or default_classes

conf_threshold = st.sidebar.slider("Min confidence to show label", 0.0, 1.0, 0.30, 0.01)
predict_every_n = st.sidebar.slider("Predict every N frames (perf)", 1, 5, 2, 1)


st.session_state["emo_classes"] = EMO_CLASSES
st.session_state["conf_threshold"] = conf_threshold
st.session_state["predict_every_n"] = predict_every_n





@st.cache_resource(show_spinner=False)
def load_emotion_model(path):
  if path.endswith((".h5", ".keras")):  
      m = load_model(path)
      # Infer expected input (H, W, C)
      inp = m.input_shape
      # Keras can report (None, 48, 48, 1) or a list for multi-input models
      if isinstance(inp, list):
        inp = inp[0]
      _, H, W, C = inp
      return "keras", m, (H, W, C)
  elif path.endswith(".pth"):
        vit_model = ViTEmotionModel(path,num_classes=7)
        # ViT always expects 224×224×3
        return ("pytorch", vit_model, (224, 224, 3)) 




# Video proccessor
class FaceProcessor(VideoProcessorBase):
    def __init__(self, detector=None):

           # 1. Initialize BOTH detectors inside the processor
        haar_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        yunet_path = "src/haarcascades/face_detection_yunet_2023mar.onnx" # Make this path robust if needed
        self.haar_detector = HaarFaceDetector(haar_path)
        self.yunet_detector = YuNetFaceDetector(model_path=yunet_path)
        
        # 2. Add a flag to control which detector is active
        self.active_detector = "Haarcascade"  # Default to Haarcascade

        #  Parameters for Haarcascade
        self.scale_factor = 1.15
        self.min_neighbors = 5
        self.min_size_px = 60
        
        

        
        self.last_probs, self.last_label = None, None
        self.last_ts, self.fps, self.frame_count = None, 0.0, 0
        self.target_width, self.show_fps = 640, True
        self.emotion_model, self.inp_size = None, (48, 48, 1)
        self.emo_classes = default_classes
        self.conf_threshold, self.predict_every_n = 0.30, 2


        self.emotion_history = deque(maxlen=60)   # store last 60 predictions
        self.time_history = deque(maxlen=60)


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

    def _predict_emotion(self, face_roi_gray,face_roi_bgr):
        if self.emotion_model is None:
            return None, None

        model_type = self.model_type
        if model_type == "keras":
            x = self._prep_face_tensor(face_roi_gray)
            probs = self.emotion_model.predict(x, verbose=0)[0]  # (num_classes,)
            
            if probs is not None:
              self.emotion_history.append(probs)
              self.time_history.append(time.time())


        elif model_type == "pytorch":
            # For ViT, use color input instead of grayscale
            face_rgb = cv2.cvtColor(face_roi_gray, cv2.COLOR_GRAY2BGR)
            probs = self.emotion_model.predict(face_roi_bgr)
            if probs is not None:
              self.emotion_history.append(probs)
              self.time_history.append(time.time())


        idx = int(np.argmax(probs))
        conf = float(probs[idx])
        return idx, conf, probs

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")

        # Resize for stable FPS
        h, w = img.shape[:2]
        if w != self.target_width:
            s = self.target_width / float(w)
            img = cv2.resize(img, (self.target_width, int(h * s)), interpolation=cv2.INTER_AREA)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)



            # 3. Use the active_detector flag to choose which detector to run
        faces = []
        if self.active_detector == "Haarcascade":
            faces = self.haar_detector.detect_faces(
                img,
                scaleFactor=self.scale_factor,
                minNeighbors=self.min_neighbors
            )
        elif self.active_detector == "YuNet":
            faces = self.yunet_detector.detect_faces(img)







        # Draw detections + emotion predictions
        for i, (x, y, w, h) in enumerate(faces):
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Run emotion prediction every N frames to keep FPS stable
            do_predict = (self.frame_count % self.predict_every_n == 0)

            if do_predict and self.emotion_model is not None:
                face_roi_gray = gray[y:y+h, x:x+w]
                face_roi_bgr = img[y:y+h, x:x+w]
                pred_idx, conf, probs = self._predict_emotion(face_roi_gray,face_roi_bgr)
                if pred_idx is not None:
                    self.last_probs = probs 
                    label = self.emo_classes[pred_idx] if pred_idx < len(self.emo_classes) else f"class {pred_idx}"
                    self.last_label = label 
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
    key="face-emotion",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=FaceProcessor,
    media_stream_constraints={"video": True, "audio": False},
)




st.markdown("###  Live Emotion Feed")

# NEW: Add the toggle and title inside the chart column
st.toggle("Show live graph", key="show_graph", value=False)

# Configuration Block 
if ctx and ctx.video_processor:
    # Get the running processor instance
    processor = ctx.video_processor
    emo_labels = st.session_state.get("emo_classes", ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"])

    # Start a live updating chart in the right column
    # st.markdown("### 📈 Real-Time Emotion Graph")
    # chart_placeholder = st.empty()



    # Tell the processor which detector to use
    processor.active_detector = detector_choice

    # Update its parameters based on the UI
    if detector_choice == "Haarcascade":
        processor.scale_factor = scale_factor
        processor.min_neighbors = min_neighbors
        processor.min_size_px = min_size_px
    else: # YuNet
        # Update the parameters on the processor's YuNet instance
        processor.yunet_detector.detector.setScoreThreshold(score_threshold)
        processor.yunet_detector.detector.setNMSThreshold(nms_threshold)
        processor.yunet_detector.detector.setTopK(top_k)

    # Update common settings
    processor.target_width = target_width
    processor.show_fps = show_fps




    if selected_model_name:
     model_path = os.path.join(models_dir, selected_model_name)
     try:
        # Load either CNN (.h5/.keras) or ViT (.pth)
        model_type, model_obj, inp_hw_c = load_emotion_model(model_path)

        # Send the model to the processor
        ctx.video_processor.emotion_model = model_obj
        ctx.video_processor.model_type = model_type
        ctx.video_processor.inp_size = inp_hw_c

        # Common emotion settings
        ctx.video_processor.set_emotion_model(
            model_obj,
            inp_hw_c,
            st.session_state["emo_classes"],
            st.session_state["conf_threshold"],
            st.session_state["predict_every_n"],
        )

        st.success(
            f"Loaded {model_type.upper()} model: {selected_model_name} | Expecting input: {inp_hw_c}"
        )

     except Exception as e:
        st.error(f"Failed to load {selected_model_name}: {e}")
        
        # Create an infinite loop to update the chart
    # NEW: Check if the 'show_graph' toggle is on
    # if st.session_state.get("show_graph", False):
        
    #     # NEW: This will re-run the script every 1 second, but ONLY if the
    #     # toggle is on. This is non-blocking.
    #     st_autorefresh(interval=1000, key="graph_refresh")

    #     if processor.emotion_history:
    #         # Get last few seconds of predictions
    #         probs = list(processor.emotion_history)
    #         times = list(processor.time_history)

    #         # Convert to pandas DataFrame
    #         df = pd.DataFrame(probs, columns=emo_labels)
    #         df["Time"] = pd.to_datetime(times, unit="s")
    #         df = df.set_index("Time")

    #         # Plot line chart in the placeholder
    #         chart_placeholder.line_chart(df)






    if st.session_state.get("show_graph", False):
        
        st_autorefresh(interval=1000, key="graph_refresh")

        if processor.emotion_history:
            # Get last few seconds of predictions
            probs = list(processor.emotion_history)
            times = list(processor.time_history)

            # Convert to pandas DataFrame
            df = pd.DataFrame(probs, columns=emo_labels)
            df["Time"] = pd.to_datetime(times, unit="s")
            df = df.set_index("Time")

            # Plot line chart DIRECTLY into the column
            st.line_chart(df) 
        
        # (Optional but good) Show a message if graph is on but no data yet
        else:
            st.write("Waiting for emotion data...")

# The 'while True:' loop is now gone.
# The script will continue to the recommendation code below.

else:
    st.info("Press **Start** to begin streaming.")





















#  Movie + Book Recommendationss
col1, col2 = st.columns(2)

with col1:
 st.subheader(" Mood-Based Movie Picks")
 rec_mode = st.radio("Movie Style", ["match", "lift"], index=0,horizontal=True)
 get_recs = st.button("Get Movie Recommendations")  # <--- BUTTON


 if get_recs:
    if ctx and ctx.video_processor:
        probs = ctx.video_processor.last_probs
        label_now = ctx.video_processor.last_label
        if probs is None:
            st.warning("No emotion detected yet. Look at the camera and try again.")
        else:
            st.info(f"Using current emotion: **{label_now}**")
            movies = recommend_from_probs(
                probs,
                class_names=st.session_state.get("emo_classes", EMO_CLASSES),
                mode=rec_mode,
                k=2,
                per_emotion=10,
                language="en-US",
                # region=(rec_region or None),
                # min_votes=min_votes,
                include_adult=False,
                recent_gte=("2000-01-01"),
                # recent_gte=("2000-01-01" if rec_recent_only else None),
            )
            if not movies:
                st.warning("No movies found. Try a different mode/region or lower Min Votes.")
            else:
                show_movies(movies, ncols=4)
    else:
        st.warning("Start the camera first, then click the button.")



with col2:
 st.subheader(" Mood-Based Book Picks")
 book_mode = st.radio("Book style", ["match", "lift"], index=0, horizontal=True)
 get_book_recs = st.button("Get Book Recommendations")

 if get_book_recs:
    if ctx and ctx.video_processor:
        probs = ctx.video_processor.last_probs
        label_now = ctx.video_processor.last_label
        if probs is None:
            st.warning("No emotion detected yet. Look at the camera and try again.")
        else:
            st.info(f"Using current emotion: **{label_now}**")
            # Blend by emotion probabilities (top-2)
            books = recommend_books_from_probs(
                probs,
                class_names=st.session_state.get("emo_classes", EMO_CLASSES),
                mode=book_mode,
                k=2,
                per_emotion=8,
                language_ol="eng",   # change to 'sin' or 'tam' if you want other languages, where available
            )
            if not books:
                st.warning("No books found. Try the other style.")
            else:
                show_books_list(books, ncols=4)
    else:
        st.warning("Start the camera first, then click the button.")
