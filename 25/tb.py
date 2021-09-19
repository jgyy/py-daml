"""
Tensorboard
"""
from os import getcwd, system
from datetime import datetime
from pandas import read_csv, DataFrame
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import models, layers, callbacks

# Data
df = DataFrame(read_csv("cancer_classification.csv"))

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

# Creating the Model
early_stop = callbacks.EarlyStopping(
    monitor="val_loss", mode="min", verbose=1, patience=25
)
print(getcwd())

# Creating the Tensorboard Callback
print(datetime.now().strftime("%Y-%m-%d--%H%M"))
log_directory = r"/tmp/logs/fit"
board = callbacks.TensorBoard(
    log_dir=log_directory,
    histogram_freq=1,
    write_graph=True,
    write_images=True,
    update_freq="epoch",
    profile_batch=2,
    embeddings_freq=1,
)
model = models.Sequential()
model.add(layers.Dense(units=30, activation="relu"))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(units=15, activation="relu"))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(units=1, activation="sigmoid"))
model.compile(loss="binary_crossentropy", optimizer="adam")

# Train the Model
model.fit(
    x=X_train,
    y=y_train,
    epochs=600,
    validation_data=(X_test, y_test),
    verbose=1,
    callbacks=[early_stop, board],
)

# Running Tensorboard
system(f"tensorboard --logdir {log_directory}")
