# ==============================================================
# 主城2024_s属性表 — K-Prototypes 均衡聚类方案
# ==============================================================
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from kmodes.kprototypes import KPrototypes

# === 1. 读取数据 ===
file_path = r"G:\2025博三下学期\博士大论文\数据处理\主城2024_属性表.xlsx"
df_raw = pd.read_excel(file_path)

# === 2. 定义聚类变量 ===
categorical_cols = ['排队', '买单', '入围', '推荐', '订座', '团购', '优惠']
numeric_cols = ['收藏', '星级', '榜单分数', '评论数', '图片数', '平台化']

df = df_raw[categorical_cols + numeric_cols].copy()

# === 3. 数据预处理 ===
for col in categorical_cols:
    df[col] = df[col].astype(str)
df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
df[categorical_cols] = df[categorical_cols].fillna('0')

# === 4. 标准化数值变量 ===
scaler = StandardScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

categorical_idx = [df.columns.get_loc(c) for c in categorical_cols]

# ==============================================================
# 方案1: 多次运行选择最均衡的结果
# ==============================================================
print("\n🔹 方案1: 多次运行选择最均衡结果...")

best_labels = None
best_balance_score = float('inf')
best_model = None

n_runs = 30  # 运行30次
for i in range(n_runs):
    model = KPrototypes(n_clusters=4, init='Huang', random_state=i, n_init=10)
    labels = model.fit_predict(df, categorical=categorical_idx)

    # 计算均衡度指标(变异系数 CV = std/mean)
    cluster_sizes = np.bincount(labels)
    balance_score = np.std(cluster_sizes) / np.mean(cluster_sizes)

    if balance_score < best_balance_score:
        best_balance_score = balance_score
        best_labels = labels
        best_model = model

    if (i + 1) % 10 == 0:
        print(f"  已完成 {i + 1}/{n_runs} 次运行...")

df['cluster_method1'] = best_labels
print(f"\n最佳均衡度(CV): {best_balance_score:.3f}")
print("方案1聚类结果:")
print(df['cluster_method1'].value_counts().sort_index())

# ==============================================================
# 方案2: 调整gamma参数(增大gamma增加类别特征权重)
# ==============================================================
print("\n\n🔹 方案2: 调整gamma参数...")

gamma_values = [0.5, 1.0, 2.0, 5.0, 10.0]
best_gamma = None
best_gamma_score = float('inf')
best_gamma_labels = None

for gamma in gamma_values:
    model = KPrototypes(n_clusters=4, init='Huang', gamma=gamma,
                        random_state=42, n_init=10)
    labels = model.fit_predict(df, categorical=categorical_idx)

    cluster_sizes = np.bincount(labels)
    balance_score = np.std(cluster_sizes) / np.mean(cluster_sizes)

    print(f"  gamma={gamma}: CV={balance_score:.3f}, 分布={cluster_sizes}")

    if balance_score < best_gamma_score:
        best_gamma_score = balance_score
        best_gamma = gamma
        best_gamma_labels = labels

df['cluster_method2'] = best_gamma_labels
print(f"\n最佳gamma: {best_gamma}, CV: {best_gamma_score:.3f}")
print("方案2聚类结果:")
print(df['cluster_method2'].value_counts().sort_index())

# ==============================================================
# 方案3: 增加聚类数K到更大值
# ==============================================================
print("\n\n🔹 方案3: 增加聚类数到K=6或K=8...")

for k in [6, 8]:
    model = KPrototypes(n_clusters=k, init='Huang', random_state=42, n_init=10)
    labels = model.fit_predict(df, categorical=categorical_idx)

    cluster_sizes = np.bincount(labels)
    balance_score = np.std(cluster_sizes) / np.mean(cluster_sizes)

    df[f'cluster_k{k}'] = labels
    print(f"\nK={k}: CV={balance_score:.3f}")
    print(f"各类样本数: {cluster_sizes}")

# ==============================================================
# 方案4: 后处理 - 将过大簇的边缘样本重分配
# ==============================================================
print("\n\n🔹 方案4: 后处理重平衡...")

labels = best_labels.copy()
n_samples = len(df)
target_size = n_samples // 4  # 目标每类约52000样本
tolerance = 0.3  # 允许±30%偏差

# 计算每个样本到各簇中心的距离
distances_to_centers = np.zeros((n_samples, 4))
for i in range(4):
    cluster_mask = (labels == i)
    if cluster_mask.sum() > 0:
        # 简化距离计算(仅用数值特征)
        center = df.loc[cluster_mask, numeric_cols].mean()
        for idx in df.index:
            distances_to_centers[idx, i] = np.linalg.norm(
                df.loc[idx, numeric_cols] - center
            )

# 重分配策略
for large_cluster in range(4):
    cluster_size = (labels == large_cluster).sum()
    if cluster_size > target_size * (1 + tolerance):
        # 需要移出的样本数
        n_to_move = int(cluster_size - target_size)

        # 找到该簇中距离中心最远的样本
        cluster_indices = np.where(labels == large_cluster)[0]
        distances = distances_to_centers[cluster_indices, large_cluster]
        far_indices = cluster_indices[np.argsort(distances)[-n_to_move:]]

        # 重新分配到最近的其他簇
        for idx in far_indices:
            other_clusters_dist = distances_to_centers[idx].copy()
            other_clusters_dist[large_cluster] = np.inf
            labels[idx] = np.argmin(other_clusters_dist)

df['cluster_method4'] = labels
print("方案4聚类结果:")
print(pd.Series(labels).value_counts().sort_index())

# ==============================================================
# 保存所有方案结果
# ==============================================================
output_path = r"G:\2025博三下学期\博士大论文\数据处理\主城2024_多方案聚类结果.xlsx"
df_out = pd.concat([df_raw, df[[col for col in df.columns if 'cluster' in col]]], axis=1)
df_out.to_excel(output_path, index=False)

print("\n" + "=" * 60)
print("✅ 所有方案聚类完成！结果已保存")
print("=" * 60)
print("\n建议:")
print("1. 如果业务上需要严格的K=4,推荐使用方案1或方案2")
print("2. 如果可以接受更多类别,方案3(K=6或8)可能更合理")
print("3. 方案4进行了强制均衡,但可能损失聚类质量")
print("\n请根据各方案的中心特征和业务意义选择最合适的方案!")