"""
K Nearest Neighbors
"""
from numpy import mean, ndarray
from pandas import read_csv, DataFrame
from matplotlib.pyplot import figure, plot, title, xlabel, ylabel, show
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix


# Get the Data
df = DataFrame(read_csv("Classified Data", index_col=0))
print(df.head())

# Standardize the Variables
scaler = StandardScaler()
scaler.fit(df.drop("TARGET CLASS", axis=1))
scaled_features = scaler.transform(df.drop("TARGET CLASS", axis=1))
df_feat = DataFrame(scaled_features, columns=df.columns[:-1])
print(df_feat.head())

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    scaled_features, df["TARGET CLASS"], test_size=0.30
)

# Using KNN
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)
pred = knn.predict(X_test)

# Predictions and Evaluations
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))

# Choosing a K Value
error_rate = []
pred_min = ndarray([])
k = 1
# Will take some time
for i in range(1, 40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(mean(pred_i != y_test))
    if min(error_rate) == mean(pred_i != y_test):
        pred_min, k = pred_i, i

# Best fit
print("WITH K =", k)
print(confusion_matrix(y_test, pred_min))
print(classification_report(y_test, pred_min))
figure(figsize=(10, 6))
plot(
    range(1, 40),
    error_rate,
    color="blue",
    linestyle="dashed",
    marker="o",
    markerfacecolor="red",
    markersize=10,
)
title("Error Rate vs. K Value")
xlabel("K")
ylabel("Error Rate")
show()
