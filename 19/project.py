"""
Random Forest Project
"""
from pandas import read_csv, DataFrame, get_dummies
from matplotlib.pyplot import figure, legend, xlabel, show
from seaborn import countplot, jointplot, lmplot
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

# Get the Data
loans = DataFrame(read_csv("loan_data.csv"))
print(loans.info())
print(loans.describe())
print(loans.head())

# Exploratory Data Analysis
figure(figsize=(10, 6))
loans[loans["credit.policy"] == 1]["fico"].hist(
    alpha=0.5, color="blue", bins=30, label="Credit.Policy=1"
)
loans[loans["credit.policy"] == 0]["fico"].hist(
    alpha=0.5, color="red", bins=30, label="Credit.Policy=0"
)
legend()
xlabel("FICO")

figure(figsize=(10, 6))
loans[loans["not.fully.paid"] == 1]["fico"].hist(
    alpha=0.5, color="blue", bins=30, label="not.fully.paid=1"
)
loans[loans["not.fully.paid"] == 0]["fico"].hist(
    alpha=0.5, color="red", bins=30, label="not.fully.paid=0"
)
legend()
xlabel("FICO")

figure(figsize=(11, 7))
countplot(x="purpose", hue="not.fully.paid", data=loans, palette="Set1")
jointplot(x="fico", y="int.rate", data=loans, color="purple")
lmplot(
    y="int.rate",
    x="fico",
    data=loans,
    hue="credit.policy",
    col="not.fully.paid",
    palette="Set1",
)

# Setting up the Data
print(loans.info())

# Categorical Features
cat_feats = ["purpose"]
final_data = get_dummies(loans, columns=cat_feats, drop_first=True)
print(final_data.info())

# Train Test Split
X = final_data.drop("not.fully.paid", axis=1)
y = final_data["not.fully.paid"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=101
)

# Training a Decision Tree Model
dtree = DecisionTreeClassifier()
dtree.fit(X_train, y_train)

# Predictions and Evaluation of Decision Tree
predictions = dtree.predict(X_test)
print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))

# Training the Random Forest model
rfc = RandomForestClassifier(n_estimators=600)
rfc.fit(X_train, y_train)

# Prediction and Evaluation of Random Forest
predictions = rfc.predict(X_test)
print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))

# LAST STEP
show()
