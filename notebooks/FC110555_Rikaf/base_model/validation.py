from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from preprocess import get_datasets


MODEL_PATH = "./models/model.pkls"

_, _, X_val, y_val, label_encoder = get_datasets()

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
