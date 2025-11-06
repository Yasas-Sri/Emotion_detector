from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from preprocess import get_feature_datasets

MODEL_PATH = "./models/model.pkls"
TRAIN_CSV_PATH = "./train_features.csv"
VAL_CSV_PATH = "./validation_features.csv"

print("Loading feature datasets for validation..")

X_train, y_train, X_val, y_val, label_encoder = get_feature_datasets(TRAIN_CSV_PATH, VAL_CSV_PATH)

model = joblib.load(MODEL_PATH)
print(f"Model loaded. Expected feature count: {model.n_features_in_}")
print(f"Actual validation feature count: {X_val.shape[1]}")

# Sanity check: if the dimensions don't match, fail loudly and clearly.
if model.n_features_in_ != X_val.shape[1]:
    raise ValueError(
        f"Feature mismatch! The model was trained with {model.n_features_in_} features, "
        f"but the validation data has {X_val.shape[1]} features."
    )

y_pred = model.predict(X_val)

acc = accuracy_score(y_val, y_pred)
f1 = f1_score(y_val, y_pred, average='macro')
print(f"\nValidation Accuracy: {acc * 100:.2f}%")
print(f"Validation Macro F1-Score: {f1:.4f}\n")

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