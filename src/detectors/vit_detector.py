


import os
import torch
from transformers import ViTForImageClassification, ViTImageProcessor, ViTConfig
import numpy as np
import cv2

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
LOCAL_MODEL_DIR = os.path.join(PROJECT_ROOT, "vit-local")  

class ViTEmotionModel:
    def __init__(self, model_path, num_classes=7, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        
        config = ViTConfig.from_pretrained("google/vit-base-patch16-224-in21k")
        config.num_labels = num_classes

    
        self.model = ViTForImageClassification(config)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        
        self.extractor = ViTImageProcessor.from_pretrained(LOCAL_MODEL_DIR)
        self.img_size = self.extractor.size['height']
        self.num_classes = num_classes

    def predict(self, face_bgr: np.ndarray):
        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        inputs = self.extractor(images=face_rgb, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0].cpu().numpy()
        return probs