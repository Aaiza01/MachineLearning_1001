import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
 

LABEL_CANDIDATES = ["label", "class", "target", "y"]
 
df = pd.read_csv("fashion-mnist_train.csv")
print("Dataset shape:", df.shape)
print("\nFirst 10 rows:\n", df.head(10))
print("\nColumn names:", df.columns.tolist())
 
# Find the label column. Common Fashion-MNIST CSVs use "label".
label_col = next((c for c in LABEL_CANDIDATES if c in df.columns), None)
if label_col is None:
    # Many files put the label in the first column.
    label_col = df.columns[0]
    print(f"\nNo standard label name found; using first column: {label_col}")
 
y = df[label_col]
X_df = df.drop(columns=[label_col])
 
# Keep numeric pixel columns only; convert any numeric-looking values.
X_df = X_df.apply(pd.to_numeric, errors="coerce")
X_df = X_df.fillna(0)
X = X_df.to_numpy(dtype=np.float32)
 
# Pixel intensities are usually 0..255. Normalize to 0..1.
if X.max() > 1.0:
    X = X / 255.0
 
print("\nFeature matrix shape:", X.shape)
print("\nClass distribution:\n", y.value_counts().sort_index())
 

n_features = X.shape[1]
if n_features < 50:
    raise ValueError("This dataset has fewer than 50 pixel features.")
 
# Method 1: fixed 50 components
pca_50 = PCA(n_components=50, svd_solver="auto", random_state=42)
X_50 = pca_50.fit_transform(X)
 
# Method 2: retain at least 95% of variance
pca_95 = PCA(n_components=0.95, svd_solver="full")
X_95 = pca_95.fit_transform(X)
 
# Method 3: retain all available components
pca_all = PCA(n_components=None, svd_solver="full")
X_all = pca_all.fit_transform(X)
 
results = pd.DataFrame({
    "Method": ["Fixed 50", "95% variance", "All components"],
    "Original features": [n_features] * 3,
    "Retained components": [X_50.shape[1], X_95.shape[1], X_all.shape[1]],
    "Cumulative explained variance (%)": [
        pca_50.explained_variance_ratio_.sum() * 100,
        pca_95.explained_variance_ratio_.sum() * 100,
        pca_all.explained_variance_ratio_.sum() * 100
    ]
})
results["Cumulative explained variance (%)"] = results[
    "Cumulative explained variance (%)"
].round(2)
print("\nComparison table:\n", results.to_string(index=False))
 
# Cumulative explained variance curve using the all-components PCA.
cumulative = np.cumsum(pca_all.explained_variance_ratio_)
plt.figure(figsize=(9, 5))
plt.plot(np.arange(1, len(cumulative) + 1), cumulative, linewidth=2)
plt.axhline(0.95, linestyle="--", label="95% variance")
plt.xlabel("Number of principal components")
plt.ylabel("Cumulative explained variance")
plt.title("Fashion MNIST: Cumulative Explained Variance")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()
 
# 2D visualization: first two principal components
pca_2d = PCA(n_components=2, random_state=42)
X_2d = pca_2d.fit_transform(X)
plt.figure(figsize=(9, 6))
scatter = plt.scatter(
    X_2d[:, 0], X_2d[:, 1],
    c=pd.Categorical(y).codes, cmap="tab10", s=5, alpha=0.6
)
plt.xlabel("First principal component")
plt.ylabel("Second principal component")
plt.title("Fashion MNIST projected onto two principal components")
plt.colorbar(scatter, label="Encoded clothing class")
plt.tight_layout()
plt.show()
 
print("\nVariance retained by 2D PCA:",
      round(pca_2d.explained_variance_ratio_.sum() * 100, 2), "%")
