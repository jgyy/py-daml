"""
Linear Regression with Python
"""
# Import Libraries
from warnings import filterwarnings
from numpy import sqrt
from pandas import read_csv, DataFrame
from seaborn import pairplot, distplot, heatmap
from matplotlib.pyplot import show, figure, scatter
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Check out the Data
filterwarnings("ignore", category=FutureWarning)
usa_housing = DataFrame(read_csv("USA_Housing.csv"))
print(usa_housing.head())
print(usa_housing.info())
print(usa_housing.describe())
print(usa_housing.columns)

# EDA
pairplot(usa_housing)
figure()
distplot(usa_housing["Price"])
figure()
heatmap(usa_housing.corr())

# Training a Linear Regression Model
X = usa_housing[
    [
        "Avg. Area Income",
        "Avg. Area House Age",
        "Avg. Area Number of Rooms",
        "Avg. Area Number of Bedrooms",
        "Area Population",
    ]
]
y = usa_housing["Price"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=101
)

# Creating and Training the Model
lm = LinearRegression()
lm.fit(X_train, y_train)

# print the intercept
print(lm.intercept_)
coeff_df = DataFrame(lm.coef_, X.columns, columns=["Coefficient"])
print(coeff_df)

# Predictions from our Model
predictions = lm.predict(X_test)
figure()
scatter(y_test, predictions)
figure()
distplot((y_test - predictions), bins=50)

# Regression Evaluation Metrics
print("MAE:", mean_absolute_error(y_test, predictions))
print("MSE:", mean_squared_error(y_test, predictions))
print("RMSE:", sqrt(mean_squared_error(y_test, predictions)))

# LAST STEP
show()
