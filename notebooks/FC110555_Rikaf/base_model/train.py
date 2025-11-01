from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import joblib
import os
import sys

from preprocess import get_datasets, get_feature_datasets

MODEL_PATH = "./models/model.pkls"

print("Select input mode:")
print("1. Train using image data")
print("2. Train using extracted features from CSV files")
mode = input("Enter 1 or 2: ").strip()

if mode == "1":
    X_train, y_train, X_val, y_val, label_encoder = get_datasets()
elif mode == "2":
    train_csv_path = input("Enter path to train CSV (default: ./train_features.csv): ").strip() or "./train_features.csv"
    val_csv_path = input("Enter path to validation CSV (default: ./validation_features.csv): ").strip() or "./validation_features.csv"
    X_train, y_train, X_val, y_val, label_encoder = get_feature_datasets(train_csv_path, val_csv_path)
else:
    print("Invalid mode. Exiting.")
    sys.exit(1)

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
