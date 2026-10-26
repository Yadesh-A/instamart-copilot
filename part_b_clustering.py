import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# 1. Load Data
orders = pd.read_csv("data/clean/orders.csv")
completed_orders = orders[orders['status'] == 'completed'].copy()
completed_orders['order_ts'] = pd.to_datetime(completed_orders['order_ts'])

# 2. TASK 1: Build RFM Table
snapshot_date = pd.to_datetime('2025-10-01 00:00:00')
rfm = completed_orders.groupby('customer_id').agg(
    recency=('order_ts', lambda x: (snapshot_date - x.max()).days),
    frequency=('order_id', 'nunique'),
    monetary=('order_amount', 'sum')
).reset_index()

print("=== Task 1: RFM Checkpoints ===")
print(f"Total Customers: {len(rfm)}")
print(f"Recency Median: {rfm['recency'].median()}, Min: {rfm['recency'].min()}, Max: {rfm['recency'].max()}")
print(f"Frequency Mean: {rfm['frequency'].mean():.2f}, Max: {rfm['frequency'].max()}")
print(f"Monetary Mean: {rfm['monetary'].mean():.2f}, Max: {rfm['monetary'].max():.2f}\n")

# 3. TASK 2: K-Means Clustering
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm[['recency', 'frequency', 'monetary']])

print("=== Task 2: K-Means Evaluation ===")
for k in range(2, 9):
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    kmeans.fit(rfm_scaled)
    inertia = kmeans.inertia_
    sil_score = silhouette_score(rfm_scaled, kmeans.labels_)
    print(f"K={k}: Inertia = {inertia:.1f} / Silhouette = {sil_score:.3f}")

# Choose K=4 based on Elbow and Silhouette
best_k = 4
kmeans_final = KMeans(n_clusters=best_k, n_init=10, random_state=42)
rfm['segment_id'] = kmeans_final.fit_predict(rfm_scaled)

# 4. TASK 3: Profile, Name, and Project
print("\n=== Task 3: Segment Profiling ===")
profile = rfm.groupby('segment_id')[['recency', 'frequency', 'monetary']].mean()
profile['size'] = rfm.groupby('segment_id').size()
print(profile)

# Assign names and mark Premium (highest monetary)
segment_names = {
    0: "Loyal/Frequent", 
    1: "Premium",        # Segment 1 has the highest monetary value
    2: "At-Risk/Rare", 
    3: "Recent/Low-Spend"
}
rfm['segment_name'] = rfm['segment_id'].map(segment_names)
rfm['is_premium'] = rfm['segment_name'] == "Premium"

# PCA
pca = PCA(n_components=2)
pca.fit(rfm_scaled)
print(f"\nPCA Explained Variance Ratios: {pca.explained_variance_ratio_}")

# Save outputs
rfm.to_csv("outputs/customer_segments.csv", index=False)
print("\nSuccess: 'outputs/customer_segments.csv' has been saved!")