"""
Keras Regression
"""
from warnings import filterwarnings
from numpy import sqrt
from pandas import read_csv, DataFrame, to_datetime
from matplotlib.pyplot import figure, show, scatter, plot
from seaborn import distplot, countplot, scatterplot, boxplot
from tensorflow.keras import models, layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    explained_variance_score,
)

# The Data
filterwarnings("ignore", category=FutureWarning)
df = DataFrame(read_csv("kc_house_data.csv"))

# Exploratory Data Analysis
print(df.isnull().sum())
print(df.describe().transpose())
figure(figsize=(12, 8))
distplot(df["price"])
figure()
countplot(x=df["bedrooms"])
figure(figsize=(12, 8))
scatterplot(x="price", y="sqft_living", data=df)
figure()
boxplot(x="bedrooms", y="price", data=df)

# Geographical Properties
figure(figsize=(12, 8))
scatterplot(x="price", y="long", data=df)
figure(figsize=(12, 8))
scatterplot(x="price", y="lat", data=df)
print(df.sort_values("price", ascending=False).head(20))
print(len(df) * (0.01))
non_top_1_perc = df.sort_values("price", ascending=False).iloc[216:]
figure(figsize=(12, 8))
scatterplot(
    x="long",
    y="lat",
    data=non_top_1_perc,
    hue="price",
    palette="RdYlGn",
    edgecolor=None,
    alpha=0.2,
)
figure()
boxplot(x="waterfront", y="price", data=df)

# Working with Feature Data
print(df.head())
print(df.info())
df = df.drop("id", axis=1)
print(df.head())

# Feature Engineering from Date
df["date"] = to_datetime(df["date"])
df["month"] = df["date"].apply(lambda date: date.month)
df["year"] = df["date"].apply(lambda date: date.year)
figure()
boxplot(x="year", y="price", data=df)
figure()
boxplot(x="month", y="price", data=df)
figure()
df.groupby("month").mean()["price"].plot()
figure()
df.groupby("year").mean()["price"].plot()
df = df.drop("date", axis=1)
print(df.columns)

# May be worth considering to remove this or feature engineer categories from it
print(df["zipcode"].value_counts())
df = df.drop("zipcode", axis=1)
print(df.head())
# could make sense due to scaling, higher should correlate to more value
print(df["yr_renovated"].value_counts())
print(df["sqft_basement"].value_counts())

# Train Test Split
X = df.drop("price", axis=1)
y = df["price"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=101
)

# Scaling
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
print(X_train.shape)
print(X_test.shape)

# Creating a Model
model = models.Sequential()
for _ in range(4):
    model.add(layers.Dense(19, activation="relu"))
model.add(layers.Dense(1))
model.compile(optimizer="adam", loss="mse")

# Training the Model
model.fit(
    x=X_train,
    y=y_train.values,
    validation_data=(X_test, y_test.values),
    batch_size=128,
    epochs=400,
)
losses = DataFrame(model.history.history)
figure()
losses.plot()

# Evaluation on Test Data
print(X_test)
predictions = model.predict(X_test)
print(mean_absolute_error(y_test, predictions))
print(sqrt(mean_squared_error(y_test, predictions)))
print(explained_variance_score(y_test, predictions))
print(df["price"].mean())
print(df["price"].median())

# Our predictions
figure()
scatter(y_test, predictions)
# Perfect predictions
plot(y_test, y_test, "r")
errors = y_test.values.reshape(6480, 1) - predictions
figure()
distplot(errors)

# Predicting on a brand new house
single_house = df.drop("price", axis=1).iloc[0]
single_house = scaler.transform(single_house.values.reshape(-1, 19))
print(single_house)
print(model.predict(single_house))
print(df.iloc[0])

# LAST STEP
show()
