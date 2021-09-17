"""
K Means Clustering Project
"""
from pandas import read_csv, DataFrame
from seaborn import set_style, lmplot, FacetGrid
from matplotlib.pyplot import show, hist
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, classification_report

# Get the Data
df = DataFrame(read_csv("College_Data", index_col=0))
print(df.head())
print(df.info())
print(df.describe())

# EDA
set_style("whitegrid")
lmplot(
    x="Room.Board",
    y="Grad.Rate",
    data=df,
    hue="Private",
    palette="coolwarm",
    height=6,
    aspect=1,
    fit_reg=False,
)
set_style("whitegrid")
lmplot(
    x="Outstate",
    y="F.Undergrad",
    data=df,
    hue="Private",
    palette="coolwarm",
    height=6,
    aspect=1,
    fit_reg=False,
)

set_style("darkgrid")
g = FacetGrid(df, hue="Private", palette="coolwarm", height=6, aspect=2)
g = g.map(hist, "Outstate", bins=20, alpha=0.7)
set_style("darkgrid")
g = FacetGrid(df, hue="Private", palette="coolwarm", height=6, aspect=2)
g = g.map(hist, "Grad.Rate", bins=20, alpha=0.7)

print(df[df["Grad.Rate"] > 100])
df.loc["Cazenovia College", "Grad.Rate"] = 100
print(df[df["Grad.Rate"] > 100])
set_style("darkgrid")
g = FacetGrid(df, hue="Private", palette="coolwarm", height=6, aspect=2)
g = g.map(hist, "Grad.Rate", bins=20, alpha=0.7)

# K Means Cluster Creation
kmeans = KMeans(n_clusters=2)
kmeans.fit(df.drop("Private", axis=1))
print(kmeans.cluster_centers_)

# Evaluation
converter = lambda x: int(x == "Yes")
df["Cluster"] = df["Private"].apply(converter)
print(df.head())
print(confusion_matrix(df["Cluster"], kmeans.labels_))
print(classification_report(df["Cluster"], kmeans.labels_))

# LAST STEP
show()
