"""
Logistic Regression
"""
from warnings import filterwarnings
from pandas import read_csv, DataFrame, isnull, get_dummies, concat
from seaborn import heatmap, set_style, countplot, distplot, boxplot
from matplotlib.pyplot import show, figure
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# The Data
filterwarnings("ignore", category=FutureWarning)
train = DataFrame(read_csv("titanic_train.csv"))
print(train.head())

# Exploratory Data Analysis
heatmap(train.isnull(), yticklabels=False, cbar=False, cmap="viridis")
figure()
set_style("whitegrid")
countplot(x="Survived", data=train, palette="RdBu_r")
figure()
set_style("whitegrid")
countplot(x="Survived", hue="Sex", data=train, palette="RdBu_r")
figure()
set_style("whitegrid")
countplot(x="Survived", hue="Pclass", data=train, palette="rainbow")
figure()
distplot(train["Age"].dropna(), kde=False, color="darkred", bins=30)
figure()
train["Age"].hist(bins=30, color="darkred", alpha=0.7)
figure()
countplot(x="SibSp", data=train)
figure()
train["Fare"].hist(color="green", bins=40, figsize=(8, 4))

# No Cufflinks
figure()
train["Fare"].plot(kind="hist", bins=30, color="green")

# Data Cleaning
figure(figsize=(12, 7))
boxplot(x="Pclass", y="Age", data=train, palette="winter")


def impute_age(cols):
    """
    Use these average age values to impute based on Pclass for Age
    """
    age = cols[0]
    pclass = cols[1]
    if isnull(age):
        if pclass == 1:
            return 37
        if pclass == 2:
            return 29
        return 24
    return age


train["Age"] = train[["Age", "Pclass"]].apply(impute_age, axis=1)
heatmap(train.isnull(), yticklabels=False, cbar=False, cmap="viridis")
train.drop("Cabin", axis=1, inplace=True)
print(train.head())
train.dropna(inplace=True)

# Converting Categorical Features
print(train.info())
sex = get_dummies(train["Sex"], drop_first=True)
embark = get_dummies(train["Embarked"], drop_first=True)
train.drop(["Sex", "Embarked", "Name", "Ticket"], axis=1, inplace=True)
train = concat([train, sex, embark], axis=1)
print(train.head())

# Building a Logistic Regression model
X_train, X_test, y_train, y_test = train_test_split(
    train.drop("Survived", axis=1), train["Survived"], test_size=0.30, random_state=101
)

# Training and Predicting
logmodel = LogisticRegression(max_iter=len(X_train))
logmodel.fit(X_train, y_train)
predictions = logmodel.predict(X_test)

# Evaluation
print(classification_report(y_test, predictions))

# LAST STEP
show()
