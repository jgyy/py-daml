"""
Linear Regression Project
"""
from warnings import filterwarnings
from numpy import sqrt
from pandas import read_csv, DataFrame
from matplotlib.pyplot import show, scatter, xlabel, ylabel, figure
from seaborn import set_palette, set_style, jointplot, pairplot, lmplot, distplot
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Get the Data
filterwarnings("ignore", category=FutureWarning)
customers = DataFrame(read_csv("Ecommerce Customers"))
print(customers.head())
print(customers.describe())
print(customers.info())

# Exploratory Data Analysis
set_palette("GnBu_d")
set_style("whitegrid")
# More time on site, more money spent.
jointplot(x="Time on Website", y="Yearly Amount Spent", data=customers)
jointplot(x="Time on App", y="Yearly Amount Spent", data=customers)
jointplot(x="Time on App", y="Length of Membership", kind="hex", data=customers)
pairplot(customers)
lmplot(x="Length of Membership", y="Yearly Amount Spent", data=customers)

# Training and Testing Data
X = customers[
    ["Avg. Session Length", "Time on App", "Time on Website", "Length of Membership"]
]
y = customers["Yearly Amount Spent"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=101
)

# Training the Model
lm = LinearRegression()
lm.fit(X_train, y_train)
# The coefficients
print("Coefficients: \n", lm.coef_)

# Predicting Test Data
predictions = lm.predict(X_test)
figure()
scatter(y_test, predictions)
xlabel("Y Test")
ylabel("Predicted Y")

# Evaluating the Model
print("MAE:", mean_absolute_error(y_test, predictions))
print("MSE:", mean_squared_error(y_test, predictions))
print("RMSE:", sqrt(mean_squared_error(y_test, predictions)))

# Residuals
figure()
distplot((y_test - predictions), bins=50)

# Coefficients
coeffecients = DataFrame(lm.coef_, X.columns)
coeffecients.columns = ["Coeffecient"]
print(coeffecients)

# LAST STEP
show()
