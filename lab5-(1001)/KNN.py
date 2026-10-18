import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

print("Loading dataset...")

data = pd.read_csv("creditcard.csv")

print("\nFirst five rows:")
print(data.head())

print("\nDataset shape:", data.shape)

print("\nClass distribution:")
print(data["Class"].value_counts())

print("\nFraud percentage:",
      round(data["Class"].mean() * 100, 4), "%")

X = data.drop("Class", axis=1)
y = data["Class"]

print("\nFeatures shape:", X.shape)
print("Target shape:", y.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])


knn_model = Pipeline([
    ("scaler", StandardScaler()),
    ("smote", SMOTE(random_state=42)),
    ("knn", KNeighborsClassifier(
        n_neighbors=5,
        n_jobs=-1
    ))
])

print("\nTraining KNN with k=5...")

knn_model.fit(X_train, y_train)

y_pred = knn_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, zero_division=0)

print("\n========== KNN RESULTS (k=5) ==========")
print("Accuracy:", round(accuracy, 4))
print("F1-Score:", round(f1, 4))

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=["Legitimate", "Fraud"],
    zero_division=0
))


print("\nTuning K using GridSearchCV...")

param_grid = {
    "knn__n_neighbors": [3, 5, 7, 9]
}

grid_search = GridSearchCV(
    estimator=knn_model,
    param_grid=param_grid,
    scoring="f1",
    cv=3,
    n_jobs=-1,
    verbose=2
)

grid_search.fit(X_train, y_train)

print("\nBest parameters:", grid_search.best_params_)
print("Best cross-validation F1-score:",
      round(grid_search.best_score_, 4))

best_model = grid_search.best_estimator_

y_pred_best = best_model.predict(X_test)

best_accuracy = accuracy_score(y_test, y_pred_best)
best_f1 = f1_score(y_test, y_pred_best, zero_division=0)

print("\n========== TUNED KNN RESULTS ==========")
print("Best k:", grid_search.best_params_["knn__n_neighbors"])
print("Accuracy:", round(best_accuracy, 4))
print("F1-Score:", round(best_f1, 4))

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred_best,
    target_names=["Legitimate", "Fraud"],
    zero_division=0
))

results = pd.DataFrame({
    "Model": ["KNN (k=5)", "Tuned KNN"],
    "Accuracy": [accuracy, best_accuracy],
    "F1-Score": [f1, best_f1]
})

print("\n========== MODEL COMPARISON ==========")
print(results.to_string(index=False))

print("\nKNN completed successfully!")