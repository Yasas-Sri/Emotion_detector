import cv2

class HaarFaceDetector:

    def __init__(self,haarcascade_path):
        self.face_cascade = cv2.CascadeClassifier(haarcascade_path)

    
    def detect_faces(self,frame,scaleFactor=1.1,minNeighbors=5):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=scaleFactor, minNeighbors=minNeighbors)
        return faces     