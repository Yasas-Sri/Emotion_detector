from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from preprocess import get_feature_datasets

MODEL_PATH = "./models/model.pkls"
TRAIN_CSV_PATH = "./train_features.csv" 
TEST_CSV_PATH = "./test_features.csv" 

print("Loading datasets for final testing...")
X_train_full, y_train_full, X_test_full, y_test, label_encoder = get_feature_datasets(TRAIN_CSV_PATH, TEST_CSV_PATH)

# Load the final, trained model
model = joblib.load(MODEL_PATH)
print(f"Model loaded. Expected feature count: {model.n_features_in_}")
print(f"Actual test feature count: {X_test_full.shape[1]}")

if model.n_features_in_ != X_test_full.shape[1]:
    raise ValueError(
        f"CRITICAL ERROR: Feature mismatch! The model was trained with {model.n_features_in_} features, "
        f"but the test data has {X_test_full.shape[1]} features. "
        "Your training and testing pipelines are catastrophically out of sync."
    )

X_test = X_test_full

y_pred = model.predict(X_test)

# Report the FINAL metrics. This is it. No more changes.
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='macro')
print(f"\n=== FINAL TEST ACCURACY: {acc * 100:.2f}% ===")
print(f"=== FINAL TEST MACRO F1-SCORE: {f1:.4f} ===\n")

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