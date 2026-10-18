import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, adjusted_rand_score

df = pd.read_csv("Mall_Customers.csv")

print("First five rows:")
print(df.head())
print("\nDataset information:")
print(df.info())
print("\nMissing values:")
print(df.isnull().sum())

# Use Annual Income and Spending Score for clustering
income_col = "Annual Income (k$)"
score_col = "Spending Score (1-100)"
X = df[[income_col, score_col]].copy()

# Remove rows with missing values in selected features, if any
valid_rows = X.dropna().index
X = X.loc[valid_rows]
df_work = df.loc[valid_rows].copy()

kmeans_original = KMeans(n_clusters=5, random_state=42, n_init=10)
labels_kmeans_original = kmeans_original.fit_predict(X)

# DBSCAN is distance-based, so scale its inputs to make both features comparable.
# eps=0.35 is a starting point; experiment with eps and min_samples.
dbscan_scaler = StandardScaler()
X_for_dbscan = dbscan_scaler.fit_transform(X)
dbscan_model = DBSCAN(eps=0.35, min_samples=5)
labels_dbscan_original = dbscan_model.fit_predict(X_for_dbscan)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].scatter(X[income_col], X[score_col],
                c=labels_kmeans_original, cmap="viridis", s=45)
axes[0].set_title("K-Means (Original Features)")
axes[0].set_xlabel("Annual Income (k$)")
axes[0].set_ylabel("Spending Score")

axes[1].scatter(X[income_col], X[score_col],
                c=labels_dbscan_original, cmap="viridis", s=45)
axes[1].set_title("DBSCAN (Original Features; -1 = Noise)")
axes[1].set_xlabel("Annual Income (k$)")
axes[1].set_ylabel("Spending Score")

plt.tight_layout()
plt.show()

print("\nK-Means cluster counts (original features):")
print(pd.Series(labels_kmeans_original).value_counts().sort_index())
print("\nDBSCAN cluster counts (-1 means noise):")
print(pd.Series(labels_dbscan_original).value_counts().sort_index())

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans_scaled = KMeans(n_clusters=5, random_state=42, n_init=10)
labels_kmeans_scaled = kmeans_scaled.fit_predict(X_scaled)

# Plot scaled K-Means labels against original axis values for easy comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].scatter(X[income_col], X[score_col],
                c=labels_kmeans_original, cmap="viridis", s=45)
axes[0].set_title("K-Means Before Scaling")
axes[0].set_xlabel("Annual Income (k$)")
axes[0].set_ylabel("Spending Score")

axes[1].scatter(X[income_col], X[score_col],
                c=labels_kmeans_scaled, cmap="viridis", s=45)
axes[1].set_title("K-Means After StandardScaler")
axes[1].set_xlabel("Annual Income (k$)")
axes[1].set_ylabel("Spending Score")

plt.tight_layout()
plt.show()

comparison = pd.DataFrame({
    "Customer_Index": X.index,
    "KMeans_Original": labels_kmeans_original,
    "KMeans_Scaled": labels_kmeans_scaled
})
print("\nFirst 10 cluster assignment comparisons:")
print(comparison.head(10))

changed = np.sum(labels_kmeans_original != labels_kmeans_scaled)
print(f"\nRows whose numeric cluster label differs: {changed} of {len(X)}")
print("Adjusted Rand Index (1.0 means identical memberships, ignoring cluster ID order):")
print(f"{adjusted_rand_score(labels_kmeans_original, labels_kmeans_scaled):.3f}")
print("Cluster IDs can be permuted, so use this score rather than comparing IDs directly.")


inertias = []
k_values = range(1, 11)

for k in k_values:
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    model.fit(X_scaled)
    inertias.append(model.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(list(k_values), inertias, marker="o")
plt.title("Elbow Method (Scaled Features)")
plt.xlabel("Number of clusters (k)")
plt.ylabel("Inertia")
plt.xticks(list(k_values))
plt.grid(True)
plt.tight_layout()
plt.show()

cluster_summary = X.copy()
cluster_summary["Cluster"] = labels_kmeans_scaled

summary = cluster_summary.groupby("Cluster").agg(
    Customer_Count=(income_col, "size"),
    Mean_Annual_Income=(income_col, "mean"),
    Mean_Spending_Score=(score_col, "mean")
).round(2)

print("\nScaled K-Means cluster summary:")
print(summary)

print("\nInterpretation guide:")
for cluster_id, row in summary.iterrows():
    income = row["Mean_Annual_Income"]
    spending = row["Mean_Spending_Score"]

    if income >= X[income_col].median() and spending >= X[score_col].median():
        description = "High income, high spending"
    elif income >= X[income_col].median() and spending < X[score_col].median():
        description = "High income, low spending"
    elif income < X[income_col].median() and spending >= X[score_col].median():
        description = "Low income, high spending"
    else:
        description = "Low income, low spending"

    print(f"Cluster {cluster_id}: {description} "
          f"(average income={income:.1f}, average score={spending:.1f})")

rng = np.random.default_rng(42)
df_work["LuckyNumber"] = rng.integers(0, 101, size=len(df_work))

X_with_lucky = df_work[[income_col, score_col, "LuckyNumber"]]

# A: no scaling, now with LuckyNumber
kmeans_lucky_raw = KMeans(n_clusters=5, random_state=42, n_init=10)
labels_lucky_raw = kmeans_lucky_raw.fit_predict(X_with_lucky)

# B: scale all three features before clustering
scaler_lucky = StandardScaler()
X_with_lucky_scaled = scaler_lucky.fit_transform(X_with_lucky)

kmeans_lucky_scaled = KMeans(n_clusters=5, random_state=42, n_init=10)
labels_lucky_scaled = kmeans_lucky_scaled.fit_predict(X_with_lucky_scaled)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].scatter(df_work[income_col], df_work[score_col],
                c=labels_lucky_raw, cmap="viridis", s=45)
axes[0].set_title("K-Means + LuckyNumber (Unscaled)")
axes[0].set_xlabel("Annual Income (k$)")
axes[0].set_ylabel("Spending Score")

axes[1].scatter(df_work[income_col], df_work[score_col],
                c=labels_lucky_scaled, cmap="viridis", s=45)
axes[1].set_title("K-Means + LuckyNumber (All Features Scaled)")
axes[1].set_xlabel("Annual Income (k$)")
axes[1].set_ylabel("Spending Score")

plt.tight_layout()
plt.show()

print("\nLuckyNumber example rows:")
print(df_work[[income_col, score_col, "LuckyNumber"]].head())

print("\nLuckyNumber effect:")
print("A random feature can change cluster assignments because K-Means uses every feature")
print("to calculate distances. Scaling prevents a feature's numeric range from dominating,")
print("but it does not make an irrelevant feature useful; it can still add noise.")
print("Adjusted Rand Index vs. K-Means on the two original features:")
print(f"Unscaled + LuckyNumber: {adjusted_rand_score(labels_kmeans_original, labels_lucky_raw):.3f}")
print(f"Scaled + LuckyNumber:   {adjusted_rand_score(labels_kmeans_scaled, labels_lucky_scaled):.3f}")

# Optional silhouette scores; only valid when there are 2+ non-singleton clusters
def print_silhouette(name, data, labels):
    unique_labels = set(labels)
    # For DBSCAN, ignore noise label (-1) when computing the score
    if -1 in unique_labels:
        mask = labels != -1
        data_eval = np.asarray(data)[mask]
        labels_eval = np.asarray(labels)[mask]
    else:
        data_eval = np.asarray(data)
        labels_eval = np.asarray(labels)

    n_clusters = len(set(labels_eval))
    if 2 <= n_clusters < len(labels_eval):
        print(f"{name} silhouette score: {silhouette_score(data_eval, labels_eval):.3f}")
    else:
        print(f"{name} silhouette score: not defined for this clustering result.")

print("\nOptional silhouette scores:")
print_silhouette("K-Means original", X, labels_kmeans_original)
print_silhouette("K-Means scaled", X_scaled, labels_kmeans_scaled)
print_silhouette("DBSCAN original (noise excluded)", X, labels_dbscan_original)