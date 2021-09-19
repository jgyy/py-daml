"""
Keras TF 2.0 Classification Project
"""
from pandas import DataFrame, read_csv
from seaborn import countplot, heatmap
from matplotlib.pyplot import figure, show
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras import models, layers, callbacks

# The Data
df = DataFrame(read_csv("cancer_classification.csv"))
print(df.info())
print(df.describe().transpose())

# EDA
figure()
countplot(x="benign_0__mal_1", data=df)
figure()
heatmap(df.corr())
print(df.corr()["benign_0__mal_1"].sort_values())
figure()
df.corr()["benign_0__mal_1"].sort_values().plot(kind="bar")
figure()
df.corr()["benign_0__mal_1"][:-1].sort_values().plot(kind="bar")

# Train Test Split
X = df.drop("benign_0__mal_1", axis=1).values
y = df["benign_0__mal_1"].values
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=101
)

# Scaling Data
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# 1. Choosing too many epochs and overfitting!
print(X_train.shape)
model = models.Sequential()
model.add(layers.Dense(units=30, activation="relu"))
model.add(layers.Dense(units=15, activation="relu"))
model.add(layers.Dense(units=1, activation="sigmoid"))
# For a binary classification problem
model.compile(loss="binary_crossentropy", optimizer="adam")
model.fit(x=X_train, y=y_train, epochs=600, validation_data=(X_test, y_test), verbose=1)
model_loss = DataFrame(model.history.history)
figure()
model_loss.plot()

# 2. Early Stopping
model = models.Sequential()
model.add(layers.Dense(units=30, activation="relu"))
model.add(layers.Dense(units=15, activation="relu"))
model.add(layers.Dense(units=1, activation="sigmoid"))
model.compile(loss="binary_crossentropy", optimizer="adam")
early_stop = callbacks.EarlyStopping(
    monitor="val_loss", mode="min", verbose=1, patience=25
)
model.fit(
    x=X_train,
    y=y_train,
    epochs=600,
    validation_data=(X_test, y_test),
    verbose=1,
    callbacks=[early_stop],
)
figure()
model_loss = DataFrame(model.history.history)
model_loss.plot()

# 3. Adding in DropOut Layers
model = models.Sequential()
model.add(layers.Dense(units=30, activation="relu"))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(units=15, activation="relu"))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(units=1, activation="sigmoid"))
model.compile(loss="binary_crossentropy", optimizer="adam")
model.fit(
    x=X_train,
    y=y_train,
    epochs=600,
    validation_data=(X_test, y_test),
    verbose=1,
    callbacks=[early_stop],
)
figure()
model_loss = DataFrame(model.history.history)
model_loss.plot()

# Model Evaluation
predictions = model.predict_classes(X_test)
# https://en.wikipedia.org/wiki/Precision_and_recall
print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))

# LAST STEP
show()
