# train.py

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import joblib
import os

from preprocess import get_datasets

MODEL_PATH = "./models/model.pkls"

X_train, y_train, X_val, y_val, label_encoder = get_datasets()

print("Training RF")
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=None, 
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_val)
acc = accuracy_score(y_val, y_pred)
print(f"Validation Accuracy: {acc * 100:.2f}%")

os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
joblib.dump(rf, MODEL_PATH)
print("Model saved")
