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
TEST_CSV_PATH = "./test_features.csv" 

print("Loading datasets for final testing...")
X_train_full, y_train_full, X_test_full, y_test, label_encoder = get_feature_datasets(TRAIN_CSV_PATH, TEST_CSV_PATH)

print("Replicating feature selection using training data...")
quick_rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
quick_rf.fit(X_train_full, y_train_full)

importances = quick_rf.feature_importances_
top_200_indices = np.argsort(importances)[-50:]

X_test = X_test_full[:, top_200_indices]
print(f"Reduced test feature shape to: {X_test.shape}")
# --- END CRITICAL ADDITION ---

model = joblib.load(MODEL_PATH)

y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print(f"\n=== FINAL TEST ACCURACY: {acc * 100:.2f}% ===\n")

print("Final Classification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Final Confusion Matrix on Test Set')
plt.tight_layout()
plt.show()