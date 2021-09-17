"""
Support Vector Machines Project
"""
from io import BytesIO
from pandas import DataFrame
from requests import get
from seaborn import load_dataset, pairplot, kdeplot
from matplotlib.pyplot import show, imshow, figure
from matplotlib.image import imread
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV

# The Iris Setosa
for url in [
    "http://upload.wikimedia.org/wikipedia/commons/5/56/Kosaciec_szczecinkowaty_Iris_setosa.jpg",
    "http://upload.wikimedia.org/wikipedia/commons/4/41/Iris_versicolor_3.jpg",
    "http://upload.wikimedia.org/wikipedia/commons/9/9f/Iris_virginica.jpg",
]:
    response = get(url, headers={"User-Agent": "Mozilla/5.0"})
    figure()
    imshow(imread(BytesIO(response.content), format="jpg"))

# Get the data
iris = DataFrame(load_dataset("iris"))

# Setosa is the most separable.
pairplot(iris, hue="species", palette="Dark2")
setosa = iris[iris["species"] == "setosa"]
figure()
kdeplot(
    x=setosa["sepal_width"],
    y=setosa["sepal_length"],
    cmap="plasma",
    shade=True,
    thresh=0.05,
)

# Train Test Split
X = iris.drop("species", axis=1)
y = iris["species"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

# Train a Model
svc_model = SVC()
svc_model.fit(X_train, y_train)

# Model Evaluation
predictions = svc_model.predict(X_test)
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))

# Gridsearch Practice
param_grid = {"C": [0.1, 1, 10, 100], "gamma": [1, 0.1, 0.01, 0.001]}
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=2)
grid.fit(X_train, y_train)
grid_predictions = grid.predict(X_test)
print(confusion_matrix(y_test, grid_predictions))
print(classification_report(y_test, grid_predictions))

# LAST STEP
show()
