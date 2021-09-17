"""
Support Vector Machines
"""
from numpy import ravel
from pandas import DataFrame
from sklearn.svm import SVC
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV

# Get the Data
cancer = load_breast_cancer()
print(cancer.keys())
print(cancer["DESCR"])
print(cancer["feature_names"])

# Set up DataFrame
df_feat = DataFrame(cancer["data"], columns=cancer["feature_names"])
df_feat.info()
print(cancer["target"])
df_target = DataFrame(cancer["target"], columns=["Cancer"])
df_target.head()

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    df_feat, ravel(df_target), test_size=0.30, random_state=101
)

# Train the Support Vector Classifier
model = SVC()
model.fit(X_train, y_train)

# Predictions and Evaluations
predictions = model.predict(X_test)
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))

# Gridsearch
param_grid = {
    "C": [0.1, 1, 10, 100, 1000],
    "gamma": [1, 0.1, 0.01, 0.001, 0.0001],
    "kernel": ["rbf"],
}
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=3)
# May take awhile!
grid.fit(X_train, y_train)
print(grid.best_params_)
print(grid.best_estimator_)
grid_predictions = grid.predict(X_test)
print(confusion_matrix(y_test, grid_predictions))
print(classification_report(y_test, grid_predictions))
