"""
Logistic Regression Project
"""
from pandas import read_csv, DataFrame
from seaborn import set_style, jointplot, pairplot
from matplotlib.pyplot import xlabel, figure, show
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Get the Data
ad_data = DataFrame(read_csv("advertising.csv"))
print(ad_data.head())
print(ad_data.info())
print(ad_data.describe())

# Exploratory Data Analysis
figure()
set_style("whitegrid")
ad_data["Age"].hist(bins=30)
xlabel("Age")
jointplot(x="Age", y="Area Income", data=ad_data)
jointplot(x="Age", y="Daily Time Spent on Site", data=ad_data, color="red", kind="kde")
jointplot(
    x="Daily Time Spent on Site", y="Daily Internet Usage", data=ad_data, color="green"
)
pairplot(ad_data, hue="Clicked on Ad", palette="bwr")

# Logistic Regression
X = ad_data[
    ["Daily Time Spent on Site", "Age", "Area Income", "Daily Internet Usage", "Male"]
]
y = ad_data["Clicked on Ad"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42
)
logmodel = LogisticRegression()
logmodel.fit(X_train, y_train)

# Predictions and Evaluations
predictions = logmodel.predict(X_test)
print(classification_report(y_test, predictions))

# LAST STEP
show()
