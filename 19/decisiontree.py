"""
Decision Trees and Random Forests
"""
from io import BytesIO
from pandas import read_csv, DataFrame
from seaborn import pairplot
from six import StringIO
from pydot import graph_from_dot_data
from matplotlib.pyplot import show, imshow, figure
from matplotlib.image import imread
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.tree import export_graphviz
from sklearn.ensemble import RandomForestClassifier

# Get the Data
df = DataFrame(read_csv("kyphosis.csv"))
print(df.head())

# EDA
pairplot(df, hue="Kyphosis", palette="Set1")

# Train Test Split
X = df.drop("Kyphosis", axis=1)
y = df["Kyphosis"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

# Decision Trees
dtree = DecisionTreeClassifier()
dtree.fit(X_train, y_train)

# Prediction and Evaluation
predictions = dtree.predict(X_test)
print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))

# Tree Visualization
features = list(df.columns[1:])
print(features)
dot_data = StringIO()
export_graphviz(
    dtree, out_file=dot_data, feature_names=features, filled=True, rounded=True
)
graph = graph_from_dot_data(dot_data.getvalue())
figure()
imshow(imread(BytesIO(graph[0].create_png())))

# Random Forests
rfc = RandomForestClassifier(n_estimators=100)
rfc.fit(X_train, y_train)
rfc_pred = rfc.predict(X_test)
print(confusion_matrix(y_test, rfc_pred))
print(classification_report(y_test, rfc_pred))

# LAST STEP
show()
