import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
 
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
 
CSV_PATH = "heart.csv"
df = pd.read_csv('heart.csv')
 
print("Dataset shape:", df.shape)
print("\nFirst 10 rows:\n", df.head(10))
print("\nMissing values by column:\n", df.isnull().sum())
 

target_candidates = ["target", "HeartDisease", "heart_disease", "num", "output"]
target_col = next((c for c in target_candidates if c in df.columns), None)
if target_col is None:
    raise ValueError(
        "Could not identify target column. Set target_col manually "
        "to the column that indicates heart disease."
    )
 

y_raw = df[target_col]
if target_col.lower() == "num":
    y = (pd.to_numeric(y_raw, errors="coerce").fillna(0) > 0).astype(int)
else:
    y = y_raw.copy()
    if y.dtype == "object":
        y = y.astype(str).str.strip()
        
    else:
        y = pd.to_numeric(y, errors="coerce")
 
valid = y.notna()
X = df.loc[valid].drop(columns=[target_col]).copy()
y = y.loc[valid]
print("\nTarget column:", target_col)
print("\nTarget class distribution:\n", y.value_counts())
 
numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
categorical_cols = X.select_dtypes(exclude=["number"]).columns.tolist()
 
numeric_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])
categorical_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])
preprocessor = ColumnTransformer([
    ("num", numeric_pipe, numeric_cols),
    ("cat", categorical_pipe, categorical_cols)
])
 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

baseline = Pipeline([
    ("preprocess", preprocessor),
    ("model", RandomForestClassifier(random_state=42))
])
baseline.fit(X_train, y_train)
baseline_pred = baseline.predict(X_test)
baseline_accuracy = accuracy_score(y_test, baseline_pred)
print("\nBaseline test accuracy:", round(baseline_accuracy, 4))
 
# Parameter grid required by the lab.
param_grid = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [None, 10, 20],
    "model__min_samples_split": [2, 5],
    "model__min_samples_leaf": [1, 2],
    "model__max_features": ["sqrt", "log2"]
}
grid = GridSearchCV(
    estimator=Pipeline([
        ("preprocess", preprocessor),
        ("model", RandomForestClassifier(random_state=42))
    ]),
    param_grid=param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1,
    verbose=1
)
grid.fit(X_train, y_train)
best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)
 
print("\nBest parameters:\n", grid.best_params_)
print("Best cross-validation accuracy:", round(grid.best_score_, 4))
print("Baseline test accuracy:", round(baseline_accuracy, 4))
print("Optimized test accuracy:", round(accuracy_score(y_test, y_pred), 4))
print("\nClassification report:\n",
      classification_report(y_test, y_pred, zero_division=0))
 
# Use weighted averages so this works for binary or multiclass targets.
print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
print("Precision (weighted):",
      round(precision_score(y_test, y_pred, average="weighted",
                            zero_division=0), 4))
print("Recall (weighted):",
      round(recall_score(y_test, y_pred, average="weighted",
                         zero_division=0), 4))
print("F1-score (weighted):",
      round(f1_score(y_test, y_pred, average="weighted",
                     zero_division=0), 4))
 
labels = best_model.named_steps["model"].classes_
cm = confusion_matrix(y_test, y_pred, labels=labels)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted label")
plt.ylabel("Actual label")
plt.title("Heart Disease — Confusion Matrix")
plt.tight_layout()
plt.show()
 
# Feature importances after preprocessing (one-hot encoded columns included).
feature_names = best_model.named_steps["preprocess"].get_feature_names_out()
importances = best_model.named_steps["model"].feature_importances_
importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
}).sort_values("Importance", ascending=False)
 
print("\nTop 3 influential features:\n",
      importance_df.head(3).to_string(index=False))
top = importance_df.head(15).sort_values("Importance")
plt.figure(figsize=(9, 6))
plt.barh(top["Feature"], top["Importance"])
plt.xlabel("Feature importance")
plt.title("Top Feature Importances — Heart Disease")
plt.tight_layout()
plt.show()
