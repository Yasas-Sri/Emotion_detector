import cv2
import os

class HaarFaceDetector:

    def __init__(self,haarcascade_path):
        self.face_cascade = cv2.CascadeClassifier(haarcascade_path)

    
    def detect_faces(self,frame,scaleFactor=1.1,minNeighbors=5):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=scaleFactor, minNeighbors=minNeighbors)
        return faces     


class YuNetFaceDetector:
    def __init__(self, model_path=None, input_size=(320, 320), conf_threshold=0.9, nms_threshold=0.3, top_k=5000):
        # Use OpenCV's built-in YuNet
        if model_path is None:
            # OpenCV ships YuNet .onnx inside face_detection_yunet
            # model_path = "../haarcascades/yunet.onnx"
            base_dir = os.path.dirname(os.path.abspath(__file__))  # src/detectors
            project_root = os.path.dirname(base_dir)               # src/
            model_path = os.path.join(project_root, "haarcascades", "face_detection_yunet_2023mar.onnx")


        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"YuNet ONNX model not found at {model_path}. "
                # "Download it from: https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
            )

        self.detector = cv2.FaceDetectorYN_create(
            model=model_path,
            config="",
            input_size=input_size,
            score_threshold=conf_threshold,
            nms_threshold=nms_threshold,
            top_k=top_k
        )
        self.input_size = input_size

    def detect_faces(self, frame, **kwargs):
        h, w = frame.shape[:2]
        if (w, h) != self.input_size:
            self.detector.setInputSize((w, h))
        # YuNet detect returns (num_faces, 15) where columns include [x,y,w,h,conf,...]
        _, faces = self.detector.detect(frame)
        results = []
        if faces is not None:
            for face in faces:
                x, y, w, h = face[:4].astype(int)
                results.append((x, y, w, h))
        return results    