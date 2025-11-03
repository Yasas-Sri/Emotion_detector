from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from preprocess import get_datasets, get_feature_datasets


MODEL_PATH = "./models/model.pkls"

print("Select input mode:")
print("1. Validate using image data")
print("2. Validate using extracted features from CSV files")
mode = input("Enter 1 or 2: ").strip()

if mode == "1":
    _, _, X_val, y_val, label_encoder = get_datasets()
elif mode == "2":
    val_csv_path = input("Enter path to validation CSV (default: ./validation_features.csv): ").strip() or "./validation_features.csv"
    train_csv_path = input("Enter path to training CSV (default: ./train_features.csv): ").strip() or "./train_features.csv"
    _, _, X_val, y_val, label_encoder = get_feature_datasets(train_csv_path, val_csv_path)
else:
    print("Invalid mode. Exiting.")
    exit(1)


model = joblib.load(MODEL_PATH)

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
