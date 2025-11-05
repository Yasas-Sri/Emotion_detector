from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from preprocess import get_feature_datasets

MODEL_PATH = "./models/model.pkls"
TRAIN_CSV_PATH = "./train_features.csv"
VAL_CSV_PATH = "./validation_features.csv"

print("Loading feature datasets...")
# Load the full datasets first
_, _, X_val_full, y_val, label_encoder = get_feature_datasets(TRAIN_CSV_PATH, VAL_CSV_PATH)

# --- CRITICAL ADDITION: REPLICATE FEATURE SELECTION ---
# You MUST use the same training data to determine feature importance.
# This ensures you select the SAME features.
print("Loading training data to replicate feature selection...")
train_df = pd.read_csv(TRAIN_CSV_PATH)
y_train_for_selection = train_df["label"].values
X_train_full_for_selection = train_df.drop(columns=["label"]).values

# Apply the same feature engineering from preprocess.py
from preprocess import engineer_features
X_train_engineered = np.array([engineer_features(x) for x in X_train_full_for_selection])

print("Running quick RF to determine feature importances...")
quick_rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
quick_rf.fit(X_train_engineered, y_train_for_selection)

# Select the SAME top 200 features
importances = quick_rf.feature_importances_
top_200_indices = np.argsort(importances)[-50:]

# Apply the selection to the VALIDATION data
X_val = X_val_full[:, top_200_indices]
print(f"Reduced validation feature shape to: {X_val.shape}")
# --- END CRITICAL ADDITION ---


model = joblib.load(MODEL_PATH)

# Feature importance analysis (now on the 200 features)
feature_importance = model.feature_importances_
top_10_indices_in_reduced = np.argsort(feature_importance)[-10:]
print("\n--- Feature Importance (on reduced set) ---")
print("Top 10 most important features (indices):")
for idx in top_10_indices_in_reduced:
    print(f"  Feature {idx}: {feature_importance[idx]:.4f}")
print("-------------------------\n")

y_pred = model.predict(X_val)

acc = accuracy_score(y_val, y_pred)
print(f"Validation Accuracy: {acc * 100:.2f}%\n")

print("Classification Report:")
print(classification_report(y_val, y_pred, target_names=label_encoder.classes_))

cm = confusion_matrix(y_val, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()