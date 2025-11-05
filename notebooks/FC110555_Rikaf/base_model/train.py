from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
import joblib
import os
import numpy as np

# You are now committed to features. Act like it.
from preprocess import get_feature_datasets

MODEL_PATH = "./models/model.pkls"
TRAIN_CSV_PATH = "./train_features.csv"
VAL_CSV_PATH = "./validation_features.csv"

print("Loading feature datasets...")
# This now calls the function with your NEW feature engineering logic.
# If you didn't modify preprocess.py, this will fail. That's on you.
X_train, y_train, X_val, y_val, label_encoder = get_feature_datasets(TRAIN_CSV_PATH, VAL_CSV_PATH)

# Add this to the TOP of train.py, before the grid search
print("Running quick RF for feature selection...")
quick_rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
quick_rf.fit(X_train, y_train)

# Select top 200 features
importances = quick_rf.feature_importances_
top_200_indices = np.argsort(importances)[-50:]

X_train = X_train[:, top_200_indices]
X_val = X_val[:, top_200_indices]

print(f"Reduced feature shape to: {X_train.shape}")

print(f"Training with feature shape: {X_train.shape}")
print("Training RF with Hyperparameter Tuning")

# --- CHANGE START: Systematic Search, Not a Blind Guess ---
# You're no longer guessing parameters. You're searching for the best ones.
# This is how professionals work. They don't rely on luck.
param_grid = {
    'n_estimators': [100, 200, 300],         # Number of trees in the forest
    'max_depth': [10, 20, None],        # Max depth of the tree
    'min_samples_split': [2, 5, 10],        # Min samples required to split a node
    'min_samples_leaf': [1, 2, 4],          # Min samples required at a leaf node
    'max_features': ['sqrt', 'log2'],        # Number of features to consider at each split
    'class_weight': [None, 'balanced']
}

# Create a base model to be tuned
rf_base = RandomForestClassifier(random_state=42, n_jobs=-1)

# Instantiate the grid search model
# cv=3 means 3-fold cross-validation. It's more robust than your single validation set.
# verbose=2 will show you progress. Don't complain if it takes time.
# n_jobs=-1 uses all your CPU cores. Don't waste resources.
grid_search = GridSearchCV(
    estimator=rf_base, 
    param_grid=param_grid, 
    cv=3, 
    n_jobs=-1, 
    verbose=2, 
    scoring='accuracy'
)

# Fit the grid search to the data. This will take time. Go get coffee.
# Don't stare at it. Let it work.
grid_search.fit(X_train, y_train)

# Get the best model found during the search
best_rf = grid_search.best_estimator_

print(f"\nBest parameters found: {grid_search.best_params_}")
# --- CHANGE END ---

# Use the BEST model for prediction. Not some random guess.
y_pred = best_rf.predict(X_val)
acc = accuracy_score(y_val, y_pred)
print(f"Validation Accuracy with Best Model: {acc * 100:.2f}%")

# Save the BEST model. You worked hard to find it.
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
joblib.dump(best_rf, MODEL_PATH)
print("Best model saved")
