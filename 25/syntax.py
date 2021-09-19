"""
Keras Syntax Basics
"""
from warnings import filterwarnings
from pandas import read_csv, DataFrame, Series, concat
from seaborn import pairplot, lineplot, scatterplot, distplot
from matplotlib.pyplot import show, title, figure
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras import models as md
from tensorflow.keras import layers as ly

# Load the data
filterwarnings("ignore", category=FutureWarning)
df = DataFrame(read_csv("fake_reg.csv"))
print(df.head())

# Explore the data
pairplot(df)

# Convert Pandas to Numpy for Keras
X = df[["feature1", "feature2"]].values  # Features
y = df["price"].values  # Labels
X_train, X_test, y_train, y_test = train_test_split(  # Split
    X, y, test_size=0.3, random_state=42
)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

# Normalising and Scaling the Data
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Creating a Model
model = md.Sequential()
for _ in range(3):
    model.add(ly.Dense(4, activation="relu"))
# Final output node for prediction
model.add(ly.Dense(1))
model.compile(optimizer="rmsprop", loss="mse")

# Training
model.fit(X_train, y_train, epochs=250)

# Evaluation
print(model.history.history)
loss = model.history.history["loss"]
figure()
lineplot(x=range(len(loss)), y=loss)
title("Training Loss per Epoch")

# Compare final evaluation (MSE) on training/test set
print(model.metrics_names)
training_score = model.evaluate(X_train, y_train, verbose=0)
test_score = model.evaluate(X_test, y_test, verbose=0)
print(training_score)
print(test_score)

# Further Evaluations
test_predictions = model.predict(X_test)
print(test_predictions)
pred_df = DataFrame(y_test, columns=["Test Y"])
print(pred_df)
test_predictions = Series(test_predictions.reshape(300))
print(test_predictions)
pred_df = concat([pred_df, test_predictions], axis=1)
pred_df.columns = ["Test Y", "Model Predictions"]
print(pred_df)

# Compare to the real test labels
figure()
scatterplot(x="Test Y", y="Model Predictions", data=pred_df)
pred_df["Error"] = pred_df["Test Y"] - pred_df["Model Predictions"]
figure()
distplot(pred_df["Error"], bins=50)
print(mean_absolute_error(pred_df["Test Y"], pred_df["Model Predictions"]))
print(mean_squared_error(pred_df["Test Y"], pred_df["Model Predictions"]))
print(test_score)
print(test_score ** 0.5)

# Predicting on brand new data
new_gem = [[998, 1000]]
# Don't forget to scale!
scaler.transform(new_gem)
new_gem = scaler.transform(new_gem)
print(model.predict(new_gem))

# Saving and Loading a Model
model.save("my_model.h5")  # creates a HDF5 file 'my_model.h5'
later_model = md.load_model("my_model.h5")
later_model.predict(new_gem)

# LAST STEP
show()
