
# import streamlit as st
# import cv2
# import numpy as np
# import os
# import time

# # Configure Streamlit page
# st.set_page_config(
#     page_title="Face Detection App",
#     page_icon="",
#     layout="centered"
# )

# @st.cache_resource
# def load_face_detector():
#     """Load and cache the face detection model"""
#     # Try to load your custom cascade first, then fall back to built-in
#     cascade_paths = [
#         "haarcascades/haarcascade_frontalface_default.xml",
#         "src/haarcascades/haarcascade_frontalface_default.xml", 
#         cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
#     ]
    
#     for path in cascade_paths:
#         if os.path.exists(path):
#             cascade = cv2.CascadeClassifier(path)
#             if not cascade.empty():
#                 return cascade
    
#     # Use OpenCV's built-in cascade as fallback
#     cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
#     return cascade

# def detect_faces(image, face_cascade):
#     """Detect faces and draw rectangles"""
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
#     faces = face_cascade.detectMultiScale(
#         gray,
#         scaleFactor=1.1,
#         minNeighbors=5,
#         minSize=(30, 30)
#     )
    
#     # Draw rectangles around faces
#     for (x, y, w, h) in faces:
#         cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
#         cv2.putText(image, f'Face {len(faces)}', (x, y-10), 
#                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
#     return image, len(faces)

# def main():
#     st.title(" Real-Time Face Detection")
#     st.write("Detect faces using your webcam in real-time!")
    
#     # Load face detector
#     face_cascade = load_face_detector()
    
#     if face_cascade.empty():
#         st.error(" Could not load face detection model")
#         return
    
#     # Controls
#     col1, col2, col3 = st.columns(3)
    
#     with col1:
#         run = st.button(" Start Camera", type="primary")
#     with col2:
#         stop = st.button(" Stop Camera")
#     with col3:
#         save_frame = st.button("📸 Save Frame")
    
#     # Placeholders for video and stats
#     video_placeholder = st.empty()
#     stats_placeholder = st.empty()
    
#     # Session state to manage camera
#     if 'camera_running' not in st.session_state:
#         st.session_state.camera_running = False
#     if 'cap' not in st.session_state:
#         st.session_state.cap = None
#     if 'frame_count' not in st.session_state:
#         st.session_state.frame_count = 0
    
#     # Start camera
#     if run:
#         if st.session_state.cap is not None:
#             st.session_state.cap.release()
        
#         st.session_state.cap = cv2.VideoCapture(0)
#         if st.session_state.cap.isOpened():
#             st.session_state.camera_running = True
#             st.session_state.frame_count = 0
#             st.success(" Camera started!")
#         else:
#             st.error("Could not access camera")
    
#     # Stop camera
#     if stop:
#         st.session_state.camera_running = False
#         if st.session_state.cap:
#             st.session_state.cap.release()
#         st.info("📷 Camera stopped")
    
#     # Main camera loop
#     if st.session_state.camera_running and st.session_state.cap:
#         # Create output directory
#         os.makedirs("outputs", exist_ok=True)
        
#         ret, frame = st.session_state.cap.read()
#         if ret:
#             st.session_state.frame_count += 1
            
#             # Detect faces
#             processed_frame, face_count = detect_faces(frame.copy(), face_cascade)
            
#             # Add frame counter
#             cv2.putText(processed_frame, f'Frame: {st.session_state.frame_count}', 
#                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
#             cv2.putText(processed_frame, f'Faces: {face_count}', 
#                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
#             # Convert BGR to RGB for Streamlit
#             frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            
#             # Display the video
#             video_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)
            
#             # Display stats
#             with stats_placeholder.container():
#                 col1, col2 = st.columns(2)
#                 col1.metric("👥 Faces Detected", face_count)
#                 col2.metric("🎬 Frame Count", st.session_state.frame_count)
            
#             # Save frame if requested
#             if save_frame:
#                 timestamp = int(time.time())
#                 filename = f"outputs/captured_frame_{timestamp}.jpg"
#                 cv2.imwrite(filename, processed_frame)
#                 st.success(f"💾 Saved frame: {filename}")
            
#             # Small delay and rerun to continue the loop
#             time.sleep(0.1)
#             st.rerun()
            
#         else:
#             st.error(" Failed to read from camera")
#             st.session_state.camera_running = False
    
#     else:
#         # Show instructions when camera is not running
#         with video_placeholder.container():
#             st.info("""
#              **Ready to detect faces!**
            
#             Click ' Start Camera' to begin real-time face detection.
            
#             **Features:**
#             - Real-time face detection
#             - Live face counting  
#             - Frame saving capability
#             - Works with any webcam
#             """)

# if __name__ == "__main__":
#     main()




import os
import time
import cv2
import av
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoProcessorBase

# Streamlit page config 
st.set_page_config(page_title="Face Detection App", page_icon="👁️", layout="centered")
st.title(" Real-Time Emotion Detection")


#  Model / Cascade loader
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

# Sidebar controls
st.sidebar.header("Detector Settings")
scale_factor = st.sidebar.slider("scaleFactor", 1.05, 1.40, 1.15, 0.01)
min_neighbors = st.sidebar.slider("minNeighbors", 1, 10, 5, 1)
min_size_px = st.sidebar.slider("Min face size (px)", 20, 200, 60, 10)
target_width = st.sidebar.selectbox("Frame width", [320, 480, 640, 800, 960], index=2)
show_fps = st.sidebar.checkbox("Show FPS", True)

# store current UI values for the processor to read
st.session_state["scale_factor"] = scale_factor
st.session_state["min_neighbors"] = min_neighbors
st.session_state["min_size_px"] = min_size_px
st.session_state["target_width"] = target_width
st.session_state["show_fps"] = show_fps

# Video processor 
class FaceProcessor(VideoProcessorBase):
    def __init__(self):
        self.last_ts = None
        self.fps = 0.0
        self.frame_count = 0
        self.last_frame_bgr = None

        # default params; will be updated from the main thread
        self.scale_factor = 1.15
        self.min_neighbors = 5
        self.min_size_px = 60
        self.target_width = 640
        self.show_fps = True

    def set_params(self, scale_factor, min_neighbors, min_size_px, target_width, show_fps):
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.min_size_px = min_size_px
        self.target_width = target_width
        self.show_fps = show_fps

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")

        # resize for stable FPS
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

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        if self.show_fps:
            import time
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

# UI: start/stop + snapshot
col1, col2 = st.columns(2)
with col1:
    st.write("Click **Start** below to allow camera access.")
# with col2:
#     save_frame = st.button(" Save Frame")

# create outputs dir
os.makedirs("outputs", exist_ok=True)

# start WebRTC streamer
ctx = webrtc_streamer(
    key="haar-realtime",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=FaceProcessor,
    media_stream_constraints={"video": True, "audio": False},
)

# Save the most recent processed frame
# if save_frame:
#     if ctx and ctx.video_processor and ctx.video_processor.last_frame_bgr is not None:
#         timestamp = int(time.time())
#         fname = f"outputs/captured_frame_{timestamp}.jpg"
#         cv2.imwrite(fname, ctx.video_processor.last_frame_bgr)
#         st.success(f"Saved: {fname}")
#     else:
#         st.warning("No frame available yet—click Start and look at the camera.")

# st.caption(
#     "Tip: Increase **minNeighbors** to reduce false positives. "
#     "Increase **Min face size** to ignore tiny detections. "
#     "Use a smaller **Frame width** for higher FPS."
# )
