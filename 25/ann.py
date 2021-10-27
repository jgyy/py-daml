"""
keras syntax basics
"""
from datetime import datetime
from random import seed, randint
from os import system
from os.path import join, dirname
from pandas import read_csv, DataFrame, Series, concat, to_datetime, get_dummies
from matplotlib.pyplot import figure, show, title, plot, scatter, xlim, ylim
from tensorflow.keras import models, layers, callbacks
from numpy import sqrt, argmax, isnan
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    explained_variance_score,
    classification_report,
    confusion_matrix,
)
from seaborn import (
    pairplot,
    lineplot,
    scatterplot,
    histplot,
    countplot,
    boxplot,
    heatmap,
)

PATH = dirname(__file__)


def fake_reg():
    """
    fake_reg function
    """
    dframe = DataFrame(read_csv(join(PATH, "fake_reg.csv")))
    print(dframe.head())
    pairplot(dframe)
    x_data = dframe[["feature1", "feature2"]].values
    y_data = dframe["price"].values
    x_train, x_test, y_train, y_test = train_test_split(
        x_data, y_data, test_size=0.3, random_state=42
    )
    print(x_train.shape)
    print(x_test.shape)
    print(y_train.shape)
    print(y_test.shape)
    scaler = MinMaxScaler()
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)
    model = models.Sequential()
    model.add(layers.Dense(4, activation="relu"))
    model.add(layers.Dense(4, activation="relu"))
    model.add(layers.Dense(4, activation="relu"))
    model.add(layers.Dense(1))
    model.compile(optimizer="rmsprop", loss="mse")
    model.fit(x_train, y_train, epochs=250)
    print(model.history.history)
    loss = model.history.history["loss"]
    figure()
    lineplot(x=range(len(loss)), y=loss)
    title("Training Loss per Epoch")
    print(model.metrics_names)
    training_score = model.evaluate(x_train, y_train, verbose=0)
    test_score = model.evaluate(x_test, y_test, verbose=0)
    print(training_score)
    print(test_score)
    test_predictions = model.predict(x_test)
    print(test_predictions)
    pred_df = DataFrame(y_test, columns=["Test Y"])
    print(pred_df)
    test_predictions = Series(test_predictions.reshape(300))
    print(test_predictions)
    pred_df = concat([pred_df, test_predictions], axis=1)
    pred_df.columns = ["Test Y", "Model Predictions"]
    print(pred_df)
    return pred_df, test_score, scaler, model


def model_predictions(pred_df, test_score, scaler, model):
    """
    model_predictions function
    """
    figure()
    scatterplot(x="Test Y", y="Model Predictions", data=pred_df)
    pred_df["Error"] = pred_df["Test Y"] - pred_df["Model Predictions"]
    figure()
    histplot(pred_df["Error"], bins=50, kde=True)
    print(mean_absolute_error(pred_df["Test Y"], pred_df["Model Predictions"]))
    print(mean_squared_error(pred_df["Test Y"], pred_df["Model Predictions"]))
    print(test_score)
    print(test_score ** 0.5)
    new_gem = [[998, 1000]]
    new_gem = scaler.transform(new_gem)
    print(model.predict(new_gem))
    model.save(join(PATH, "my_model.h5"))
    later_model = models.load_model(join(PATH, "my_model.h5"))
    print(later_model.predict(new_gem))
    dframe = DataFrame(read_csv(join(PATH, "kc_house_data.csv")))
    print(dframe.isnull().sum())
    print(dframe.describe().transpose())
    figure(figsize=(12, 8))
    histplot(dframe["price"], kde=True)
    figure()
    countplot(x=dframe["bedrooms"])
    figure(figsize=(12, 8))
    scatterplot(x="price", y="sqft_living", data=dframe)
    boxplot(x="bedrooms", y="price", data=dframe)
    figure(figsize=(12, 8))
    scatterplot(x="price", y="long", data=dframe)
    figure(figsize=(12, 8))
    scatterplot(x="price", y="lat", data=dframe)
    figure(figsize=(12, 8))
    scatterplot(x="long", y="lat", data=dframe, hue="price")
    print(dframe.sort_values("price", ascending=False).head(20))
    print(len(dframe) * (0.01))
    non_top_1_perc = dframe.sort_values("price", ascending=False).iloc[216:]
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
    return dframe


def waterfront_price(dframe):
    """
    waterfront price function
    """
    figure()
    boxplot(x="waterfront", y="price", data=dframe)
    print(dframe.head())
    print(dframe.info())
    dframe = dframe.drop("id", axis=1)
    print(dframe.head())
    dframe["date"] = to_datetime(dframe["date"])
    dframe["month"] = dframe["date"].apply(lambda date: date.month)
    dframe["year"] = dframe["date"].apply(lambda date: date.year)
    figure()
    boxplot(x="year", y="price", data=dframe)
    figure()
    boxplot(x="month", y="price", data=dframe)
    figure()
    dframe.groupby("month").mean()["price"].plot()
    figure()
    dframe.groupby("year").mean()["price"].plot()
    dframe = dframe.drop("date", axis=1)
    print(dframe.columns)
    print(dframe["zipcode"].value_counts())
    dframe = dframe.drop("zipcode", axis=1)
    print(dframe.head())
    print(dframe["yr_renovated"].value_counts())
    print(dframe["sqft_basement"].value_counts())
    return dframe


def xy_data(dframe):
    """
    xy data function
    """
    x_data = dframe.drop("price", axis=1)
    y_data = dframe["price"]
    x_train, x_test, y_train, y_test = train_test_split(
        x_data, y_data, test_size=0.3, random_state=101
    )
    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    print(x_train.shape)
    print(x_test.shape)
    model = models.Sequential()
    model.add(layers.Dense(19, activation="relu"))
    model.add(layers.Dense(19, activation="relu"))
    model.add(layers.Dense(19, activation="relu"))
    model.add(layers.Dense(19, activation="relu"))
    model.add(layers.Dense(1))
    model.compile(optimizer="adam", loss="mse")
    model.fit(
        x=x_train,
        y=y_train.values,
        validation_data=(x_test, y_test.values),
        batch_size=128,
        epochs=400,
    )
    losses = DataFrame(model.history.history)
    losses.plot()
    print(x_test)
    predictions = model.predict(x_test)
    print(mean_absolute_error(y_test, predictions))
    print(sqrt(mean_squared_error(y_test, predictions)))
    print(explained_variance_score(y_test, predictions))
    print(dframe["price"].mean())
    print(dframe["price"].median())
    figure()
    scatter(y_test, predictions)
    plot(y_test, y_test, "r")
    errors = y_test.values.reshape(6480, 1) - predictions
    figure()
    histplot(errors, kde=True)
    single_house = dframe.drop("price", axis=1).iloc[0]
    single_house = scaler.transform(single_house.values.reshape(-1, 19))
    print(single_house)
    print(model.predict(single_house))
    print(dframe.iloc[0])
    dframe = DataFrame(read_csv(join(PATH, "cancer_classification.csv")))
    print(dframe.info())
    print(dframe.describe().transpose())
    figure()
    countplot(x="benign_0__mal_1", data=dframe)
    heatmap(dframe.corr())
    print(dframe.corr()["benign_0__mal_1"].sort_values())
    figure()
    dframe.corr()["benign_0__mal_1"].sort_values().plot(kind="bar")
    dframe.corr()["benign_0__mal_1"][:-1].sort_values().plot(kind="bar")
    return dframe


def benign(dframe):
    """
    benign_0__mal_1 function
    """
    x_data = dframe.drop("benign_0__mal_1", axis=1).values
    y_data = dframe["benign_0__mal_1"].values
    x_train, x_test, y_train, y_test = train_test_split(
        x_data, y_data, test_size=0.25, random_state=101
    )
    scaler = MinMaxScaler()
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)
    print(x_train.shape)
    model = models.Sequential()
    model.add(layers.Dense(units=30, activation="relu"))
    model.add(layers.Dense(units=15, activation="relu"))
    model.add(layers.Dense(units=1, activation="sigmoid"))
    model.compile(loss="binary_crossentropy", optimizer="adam")
    model.fit(
        x=x_train, y=y_train, epochs=600, validation_data=(x_test, y_test), verbose=1
    )
    model_loss = DataFrame(model.history.history)
    model_loss.plot()
    model = models.Sequential()
    model.add(layers.Dense(units=30, activation="relu"))
    model.add(layers.Dense(units=15, activation="relu"))
    model.add(layers.Dense(units=1, activation="sigmoid"))
    model.compile(loss="binary_crossentropy", optimizer="adam")
    early_stop = callbacks.EarlyStopping(
        monitor="val_loss", mode="min", verbose=1, patience=25
    )
    model.fit(
        x=x_train,
        y=y_train,
        epochs=600,
        validation_data=(x_test, y_test),
        verbose=1,
        callbacks=[early_stop],
    )
    model_loss = DataFrame(model.history.history)
    model_loss.plot()
    model = models.Sequential()
    model.add(layers.Dense(units=30, activation="relu"))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(units=15, activation="relu"))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(units=1, activation="sigmoid"))
    model.compile(loss="binary_crossentropy", optimizer="adam")
    model.fit(
        x=x_train,
        y=y_train,
        epochs=600,
        validation_data=(x_test, y_test),
        verbose=1,
        callbacks=[early_stop],
    )
    return model, x_test, y_test


def model_history(model, x_test, y_test):
    """
    model history function
    """
    model_loss = DataFrame(model.history.history)
    model_loss.plot()
    predictions = argmax(model.predict(x_test), axis=-1)
    print(classification_report(y_test, predictions))
    print(confusion_matrix(y_test, predictions))
    data_info = DataFrame(
        read_csv(join(PATH, "lending_club_info.csv"), index_col="LoanStatNew")
    )
    print(data_info.loc["revol_util"]["Description"])
    feat_info = lambda col_name: print(data_info.loc[col_name]["Description"])
    feat_info("mort_acc")
    dframe = DataFrame(read_csv(join(PATH, "lending_club_loan_two.csv")))
    print(dframe.info())
    figure()
    countplot(x="loan_status", data=dframe)
    figure(figsize=(12, 4))
    histplot(dframe["loan_amnt"], bins=40)
    xlim(0, 45000)
    print(dframe.corr())
    figure(figsize=(12, 7))
    heatmap(dframe.corr(), annot=True, cmap="viridis")
    ylim(10, 0)
    print(feat_info("installment"))
    print(feat_info("loan_amnt"))
    figure()
    scatterplot(x="installment", y="loan_amnt", data=dframe)
    figure()
    boxplot(x="loan_status", y="loan_amnt", data=dframe)
    print(dframe.groupby("loan_status")["loan_amnt"].describe())
    print(sorted(dframe["grade"].unique()))
    print(sorted(dframe["sub_grade"].unique()))
    figure()
    countplot(x="grade", data=dframe, hue="loan_status")
    figure(figsize=(12, 4))
    subgrade_order = sorted(dframe["sub_grade"].unique())
    countplot(x="sub_grade", data=dframe, order=subgrade_order, palette="coolwarm")
    figure(figsize=(12, 4))
    subgrade_order = sorted(dframe["sub_grade"].unique())
    countplot(
        x="sub_grade",
        data=dframe,
        order=subgrade_order,
        palette="coolwarm",
        hue="loan_status",
    )
    return dframe, feat_info


def fg_grade(dframe, feat_info):
    """
    f anf g function
    """
    f_and_g = dframe[(dframe["grade"] == "G") | (dframe["grade"] == "F")]
    figure(figsize=(12, 4))
    subgrade_order = sorted(f_and_g["sub_grade"].unique())
    countplot(x="sub_grade", data=f_and_g, order=subgrade_order, hue="loan_status")
    print(dframe["loan_status"].unique())
    dframe["loan_repaid"] = dframe["loan_status"].map(
        {"Fully Paid": 1, "Charged Off": 0}
    )
    print(dframe[["loan_repaid", "loan_status"]])
    figure()
    dframe.corr()["loan_repaid"].sort_values().drop("loan_repaid").plot(kind="bar")
    print(len(dframe))
    print(dframe.isnull().sum())
    print(100 * dframe.isnull().sum() / len(dframe))
    feat_info("emp_title")
    print("\n")
    feat_info("emp_length")
    print(dframe["emp_title"].nunique())
    print(dframe["emp_title"].value_counts())
    dframe = dframe.drop("emp_title", axis=1)
    print(sorted(dframe["emp_length"].dropna().unique()))
    emp_length_order = [
        "< 1 year",
        "1 year",
        "2 years",
        "3 years",
        "4 years",
        "5 years",
        "6 years",
        "7 years",
        "8 years",
        "9 years",
        "10+ years",
    ]
    figure(figsize=(12, 4))
    countplot(x="emp_length", data=dframe, order=emp_length_order)
    figure(figsize=(12, 4))
    countplot(x="emp_length", data=dframe, order=emp_length_order, hue="loan_status")
    return dframe, feat_info


def emp_func(dframe, feat_info):
    """
    emp function
    """
    emp_co = (
        dframe[dframe["loan_status"] == "Charged Off"]
        .groupby("emp_length")
        .count()["loan_status"]
    )
    emp_fp = (
        dframe[dframe["loan_status"] == "Fully Paid"]
        .groupby("emp_length")
        .count()["loan_status"]
    )
    emp_len = emp_co / emp_fp
    print(emp_len)
    figure()
    emp_len.plot(kind="bar")
    dframe = dframe.drop("emp_length", axis=1)
    print(dframe.isnull().sum())
    print(dframe["purpose"].head(10))
    print(dframe["title"].head(10))
    dframe = dframe.drop("title", axis=1)
    feat_info("mort_acc")
    print(dframe["mort_acc"].value_counts())
    print("Correlation with the mort_acc column")
    print(dframe.corr()["mort_acc"].sort_values())
    print("Mean of mort_acc column per total_acc")
    print(dframe.groupby("total_acc").mean()["mort_acc"])
    total_acc_avg = dframe.groupby("total_acc").mean()["mort_acc"]
    print(total_acc_avg[2.0])
    fill_mort_acc = (
        lambda total_acc, mort_acc: total_acc_avg[total_acc]
        if isnan(mort_acc)
        else mort_acc
    )
    dframe["mort_acc"] = dframe.apply(
        lambda x: fill_mort_acc(x["total_acc"], x["mort_acc"]), axis=1
    )
    print(dframe.isnull().sum())
    dframe = dframe.dropna()
    print(dframe.select_dtypes(["object"]).columns)
    print(dframe["term"].value_counts())
    dframe["term"] = dframe["term"].apply(lambda term: int(term[:3]))
    dframe = dframe.drop("grade", axis=1)
    subgrade_dummies = get_dummies(dframe["sub_grade"], drop_first=True)
    dframe = concat([dframe.drop("sub_grade", axis=1), subgrade_dummies], axis=1)
    print(dframe.columns)
    print(dframe.select_dtypes(["object"]).columns)
    return dframe


def dummies_func(dframe):
    """
    dummies function
    """
    dummies = get_dummies(
        dframe[
            [
                "verification_status",
                "application_type",
                "initial_list_status",
                "purpose",
            ]
        ],
        drop_first=True,
    )
    dframe = dframe.drop(
        ["verification_status", "application_type", "initial_list_status", "purpose"],
        axis=1,
    )
    dframe = concat([dframe, dummies], axis=1)
    print(dframe["home_ownership"].value_counts())
    dframe["home_ownership"] = dframe["home_ownership"].replace(
        ["NONE", "ANY"], "OTHER"
    )
    dummies = get_dummies(dframe["home_ownership"], drop_first=True)
    dframe = dframe.drop("home_ownership", axis=1)
    dframe = concat([dframe, dummies], axis=1)
    dframe["zip_code"] = dframe["address"].apply(lambda address: address[-5:])
    dummies = get_dummies(dframe["zip_code"], drop_first=True)
    dframe = dframe.drop(["zip_code", "address"], axis=1)
    dframe = concat([dframe, dummies], axis=1)
    dframe = dframe.drop("issue_d", axis=1)
    dframe["earliest_cr_year"] = dframe["earliest_cr_line"].apply(
        lambda date: int(date[-4:])
    )
    dframe = dframe.drop("earliest_cr_line", axis=1)
    print(dframe.select_dtypes(["object"]).columns)
    dframe = dframe.drop("loan_status", axis=1)
    return dframe


def loan_repaid(dframe):
    """
    loan repaid function
    """
    x_data = dframe.drop("loan_repaid", axis=1).values
    y_data = dframe["loan_repaid"].values
    print(len(dframe))
    x_train, x_test, y_train, y_test = train_test_split(
        x_data, y_data, test_size=0.20, random_state=101
    )
    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    model = models.Sequential()
    model.add(layers.Dense(78, activation="relu"))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(39, activation="relu"))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(19, activation="relu"))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(units=1, activation="sigmoid"))
    model.compile(loss="binary_crossentropy", optimizer="adam")
    model.fit(
        x=x_train,
        y=y_train,
        epochs=25,
        batch_size=256,
        validation_data=(x_test, y_test),
    )
    model.save(join(PATH, "full_data_project_model.h5"))
    losses = DataFrame(model.history.history)
    losses[["loss", "val_loss"]].plot()
    predictions = argmax(model.predict(x_test), axis=-1)
    print(classification_report(y_test, predictions))
    print(confusion_matrix(y_test, predictions))
    seed(101)
    random_ind = randint(0, len(dframe))
    new_customer = dframe.drop("loan_repaid", axis=1).iloc[random_ind]
    print(new_customer)
    print(argmax(model.predict(new_customer.values.reshape(1, 78)), axis=-1))
    print(dframe.iloc[random_ind]["loan_repaid"])


def cancer_classify():
    """
    cancer classification function
    """
    dframe = DataFrame(read_csv(join(PATH, "cancer_classification.csv")))
    x_data = dframe.drop("benign_0__mal_1", axis=1).values
    y_data = dframe["benign_0__mal_1"].values
    x_train, x_test, y_train, y_test = train_test_split(
        x_data, y_data, test_size=0.25, random_state=101
    )
    scaler = MinMaxScaler()
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)
    early_stop = callbacks.EarlyStopping(
        monitor="val_loss", mode="min", verbose=1, patience=25
    )
    print(PATH)
    print(datetime.now().strftime("%Y-%m-%d--%H%M"))
    log_directory = join(PATH, "logs", "fit")
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
    model.fit(
        x=x_train,
        y=y_train,
        epochs=600,
        validation_data=(x_test, y_test),
        verbose=1,
        callbacks=[early_stop, board],
    )


if __name__ == "__main__":
    pred_dfs, test_scores, scalers, modell = fake_reg()
    dframes = model_predictions(pred_dfs, test_scores, scalers, modell)
    dframes = waterfront_price(dframes)
    dframes = xy_data(dframes)
    modell, x_tests, y_tests = benign(dframes)
    dframes, feat_infos = model_history(modell, x_tests, y_tests)
    dframes, feat_infos = fg_grade(dframes, feat_infos)
    dframes = emp_func(dframes, feat_infos)
    dframes = dummies_func(dframes)
    loan_repaid(dframes)
    cancer_classify()
    show()
    system(f'tensorboard --logdir "{join(PATH, "logs", "fit")}"')
