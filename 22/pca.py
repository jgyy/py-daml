"""
Principal Component Analysis
"""
from pandas import DataFrame
from seaborn import heatmap
from matplotlib.pyplot import figure, scatter, xlabel, ylabel, show
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# The Data
cancer = load_breast_cancer()
print(cancer.keys())
print(cancer["DESCR"])
df = DataFrame(cancer["data"], columns=cancer["feature_names"])
print(df.head())

# PCA Visualisation
scaler = StandardScaler()
scaler.fit(df)
scaled_data = scaler.transform(df)
pca = PCA(n_components=2)
pca.fit(scaled_data)
x_pca = pca.transform(scaled_data)
print(scaled_data.shape)
print(x_pca.shape)

figure(figsize=(8, 6))
scatter(x_pca[:, 0], x_pca[:, 1], c=cancer["target"], cmap="plasma")
xlabel("First principal component")
ylabel("Second Principal Component")

# Interpreting the components
print(pca.components_)
df_comp = DataFrame(pca.components_, columns=cancer["feature_names"])
figure(figsize=(12, 6))
heatmap(df_comp, cmap="plasma")

# LAST STEP
show()
