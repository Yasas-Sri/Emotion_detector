from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
import joblib
import os

from preprocess import get_feature_datasets

MODEL_PATH = "./models/model.pkls"
TRAIN_CSV_PATH = "./train_features.csv"
VAL_CSV_PATH = "./validation_features.csv"

print("Loading feature datasets...")
X_train, y_train, X_val, y_val, label_encoder = get_feature_datasets(TRAIN_CSV_PATH, VAL_CSV_PATH)


print(f"Training with feature shape: {X_train.shape}")
print("Training RF with Hyperparameter Tuning")

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2'],        
    'class_weight': [None, 'balanced']
}

rf_base = RandomForestClassifier(random_state=42, n_jobs=-1)

grid_search = GridSearchCV(
    estimator=rf_base, 
    param_grid=param_grid, 
    cv=3, 
    n_jobs=-1, 
    verbose=2, 
    scoring='f1_macro'
)

grid_search.fit(X_train, y_train)

best_rf = grid_search.best_estimator_

print(f"\nBest parameters found: {grid_search.best_params_}")

y_pred = best_rf.predict(X_val)
acc = accuracy_score(y_val, y_pred)
print(f"Validation Accuracy with Best Model: {acc * 100:.2f}%")

os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
joblib.dump(best_rf, MODEL_PATH)
print("Best model saved")
