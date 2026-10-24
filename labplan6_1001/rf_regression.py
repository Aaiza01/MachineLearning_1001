import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
 
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
 
CSV_PATH = "train.csv"
df = pd.read_csv('train.csv')
 
print("Dataset shape:", df.shape)
print("\nFirst 10 rows:\n", df.head(10))
print("\nMissing values (top 20 columns):\n",
      df.isnull().sum().sort_values(ascending=False).head(20))
print("\nSalePrice summary statistics:\n", df["SalePrice"].describe())
 

df = df.dropna(subset=["SalePrice"]).copy()
y = df["SalePrice"]
drop_cols = ["SalePrice"]
if "Id" in df.columns:
    drop_cols.append("Id")
X = df.drop(columns=drop_cols)
 
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
    X, y, test_size=0.20, random_state=42
)
 
def regression_metrics(actual, predicted):
    mse = mean_squared_error(actual, predicted)
    return {
        "MAE": mean_absolute_error(actual, predicted),
        "MSE": mse,
        "RMSE": np.sqrt(mse),
        "R2": r2_score(actual, predicted)
    }
 

baseline = Pipeline([
    ("preprocess", preprocessor),
    ("model", RandomForestRegressor(random_state=42))
])
baseline.fit(X_train, y_train)
baseline_pred = baseline.predict(X_test)
baseline_metrics = regression_metrics(y_test, baseline_pred)
 
print("\nBaseline test metrics:")
for name, value in baseline_metrics.items():
    print(f"{name}: {value:.4f}")
 
param_grid = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [None, 10, 20],
    "model__min_samples_split": [2, 5],
    "model__min_samples_leaf": [1, 2],
    "model__max_features": [0.7, 1.0]
}
grid = GridSearchCV(
    estimator=Pipeline([
        ("preprocess", preprocessor),
        ("model", RandomForestRegressor(random_state=42))
    ]),
    param_grid=param_grid,
    cv=5,
    scoring="neg_mean_squared_error",
    n_jobs=-1,
    verbose=1
)
grid.fit(X_train, y_train)
best_model = grid.best_estimator_
optimized_pred = best_model.predict(X_test)
optimized_metrics = regression_metrics(y_test, optimized_pred)
 
print("\nBest parameters:\n", grid.best_params_)
# GridSearchCV maximizes the negative MSE, so negate it to get positive MSE.
best_cv_mse = -grid.best_score_
print("Best CV MSE:", round(best_cv_mse, 4))
print("Best CV RMSE:", round(np.sqrt(best_cv_mse), 4))
 
print("\nOptimized test metrics:")
for name, value in optimized_metrics.items():
    print(f"{name}: {value:.4f}")
 
comparison = pd.DataFrame(
    [baseline_metrics, optimized_metrics],
    index=["Baseline Random Forest", "Optimized Random Forest"]
).round(4)
print("\nModel comparison:\n", comparison)

plt.figure(figsize=(7, 6))
plt.scatter(y_test, optimized_pred, alpha=0.65)
min_val = min(y_test.min(), optimized_pred.min())
max_val = max(y_test.max(), optimized_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], "r--", label="Ideal prediction")
plt.xlabel("Actual SalePrice")
plt.ylabel("Predicted SalePrice")
plt.title("Actual vs Predicted House Prices")
plt.legend()
plt.tight_layout()
plt.show()

feature_names = best_model.named_steps["preprocess"].get_feature_names_out()
importances = best_model.named_steps["model"].feature_importances_
importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
}).sort_values("Importance", ascending=False)
 
print("\nTop 5 influential features:\n",
      importance_df.head(5).to_string(index=False))
top = importance_df.head(5).sort_values("Importance")
plt.figure(figsize=(9, 5))
plt.barh(top["Feature"], top["Importance"])
plt.xlabel("Feature importance")
plt.title("Top 5 Feature Importances — House Prices")
plt.tight_layout()
plt.show()
