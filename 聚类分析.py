import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# 1. 读取CSV文件（请将'input.csv'替换为你的文件路径）
data = pd.read_csv(r'D:\C_data\聚类分析.csv')

# 提取前六列作为特征
X = data.iloc[:, :6]

# 2. 数据标准化（建议使用）
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. 使用肘部法则确定最佳聚类数（可选）
sse = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    sse.append(kmeans.inertia_)

plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), sse, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Sum of Squared Distances (SSE)')
plt.title('Elbow Method for Optimal Cluster Number')
plt.show()

# 4. 进行聚类分析（这里设置n_clusters=3，请根据实际需求调整）
k = 4
kmeans = KMeans(n_clusters=k, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

# 将聚类结果添加到原始数据
data['Cluster'] = clusters

# 5. 保存结果到新CSV文件
data.to_csv('clustered_result.csv', index=False)

# 6. 可视化聚类结果（使用PCA降维）
pca = PCA(n_components=2)
principal_components = pca.fit_transform(X_scaled)

plt.figure(figsize=(10, 6))
plt.scatter(principal_components[:, 0], 
            principal_components[:, 1], 
            c=clusters, 
            cmap='viridis',
            edgecolor='k',
            s=50)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title(f'Cluster Visualization (k={k})')
plt.colorbar(label='Cluster')
plt.show()

# 7. 显示聚类中心特征（逆标准化后的结果）
cluster_centers = pd.DataFrame(
    scaler.inverse_transform(kmeans.cluster_centers_),
    columns=X.columns
)
print("\n聚类中心特征分析：")
print(cluster_centers.round(2))

# 8. 显示各聚类样本数量
print("\n各聚类样本分布：")
print(data['Cluster'].value_counts().sort_index())
