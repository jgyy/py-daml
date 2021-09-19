"""
Keras API Project
"""
from random import seed, randint
from warnings import filterwarnings
from numpy import isnan, argmax
from pandas import read_csv, DataFrame, get_dummies, concat
from seaborn import countplot, distplot, heatmap, scatterplot, boxplot
from matplotlib.pyplot import figure, show, xlim, ylim
from tensorflow.keras import models, layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix


# Starter Code
filterwarnings("ignore", category=FutureWarning)
data_info = DataFrame(read_csv("lending_club_info.csv", index_col="LoanStatNew"))
feat_info = lambda x: print(data_info.loc[x]["Description"])
feat_info("revol_util")
feat_info("mort_acc")

# Loading the data
df = DataFrame(read_csv("lending_club_loan_two.csv"))
df.info()

# Exploratory Data Analysis
figure()
countplot(x="loan_status", data=df)
figure(figsize=(12, 4))
distplot(df["loan_amnt"], kde=False, bins=40)
xlim(0, 45000)

# Explore correlation between the continuous feature variables
print(df.corr())
figure(figsize=(12, 7))
heatmap(df.corr(), annot=True, cmap="viridis")
ylim(10, 0)
feat_info("installment")
feat_info("loan_amnt")
figure()
scatterplot(x="installment", y="loan_amnt", data=df)
figure()
boxplot(x="loan_status", y="loan_amnt", data=df)

# Calculate the summary statistics for the loan amount
print(df.groupby("loan_status")["loan_amnt"].describe())
print(sorted(df["grade"].unique()))
print(sorted(df["sub_grade"].unique()))
figure()
countplot(x="grade", data=df, hue="loan_status")
figure(figsize=(12, 4))
subgrade_order = sorted(df["sub_grade"].unique())
countplot(x="sub_grade", data=df, order=subgrade_order, palette="coolwarm")
figure(figsize=(12, 4))
subgrade_order = sorted(df["sub_grade"].unique())
countplot(
    x="sub_grade", data=df, order=subgrade_order, palette="coolwarm", hue="loan_status"
)
f_and_g = df[(df["grade"] == "G") | (df["grade"] == "F")]
figure(figsize=(12, 4))
subgrade_order = sorted(f_and_g["sub_grade"].unique())
countplot(x="sub_grade", data=f_and_g, order=subgrade_order, hue="loan_status")

# Create a new column called "load_repaied"
df["loan_status"].unique()
df["loan_repaid"] = df["loan_status"].map({"Fully Paid": 1, "Charged Off": 0})
print(df[["loan_repaid", "loan_status"]])
figure()
df.corr()["loan_repaid"].sort_values().drop("loan_repaid").plot(kind="bar")

# Data PreProcessing
print(df.head())

# Missing Data
print(len(df))
print(df.isnull().sum())
print(100 * df.isnull().sum() / len(df))
feat_info("emp_title")
feat_info("emp_length")
print(df["emp_title"].nunique())
print(df["emp_title"].value_counts())
df = df.drop("emp_title", axis=1)

# Create a count plot of the emp_length
print(sorted(df["emp_length"].dropna().unique()))
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
countplot(x="emp_length", data=df, order=emp_length_order)
figure(figsize=(12, 4))
countplot(x="emp_length", data=df, order=emp_length_order, hue="loan_status")
figure(figsize=(12, 4))
countplot(x="emp_length", data=df, order=emp_length_order, hue="loan_status")

# Challenge Task
emp_co = (
    df[df["loan_status"] == "Charged Off"].groupby("emp_length").count()["loan_status"]
)
emp_fp = (
    df[df["loan_status"] == "Fully Paid"].groupby("emp_length").count()["loan_status"]
)
emp_len = emp_co / emp_fp
print(emp_len)
figure()
emp_len.plot(kind="bar")
df = df.drop("emp_length", axis=1)
print(df.isnull().sum())
print(df["purpose"].head(10))
print(df["title"].head(10))
df = df.drop("title", axis=1)

# Find out what the mort_acc feature represents
feat_info("mort_acc")
print(df["mort_acc"].value_counts())
print("Correlation with the mort_acc column")
print(df.corr()["mort_acc"].sort_values())
print("Mean of mort_acc column per total_acc")
print(df.groupby("total_acc").mean()["mort_acc"])
total_acc_avg = df.groupby("total_acc").mean()["mort_acc"]
print(total_acc_avg[2.0])


def fill_mort_acc(total_acc, mort_acc):
    """
    Accepts the total_acc and mort_acc values for the row.
    Checks if the mort_acc is NaN , if so, it returns the avg mort_acc value
    for the corresponding total_acc value for that row.

    total_acc_avg here should be a Series or dictionary containing the mapping of the
    groupby averages of mort_acc per total_acc values.
    """
    if isnan(mort_acc):
        return total_acc_avg[total_acc]
    return mort_acc


df["mort_acc"] = df.apply(
    lambda x: fill_mort_acc(x["total_acc"], x["mort_acc"]), axis=1
)
print(df.isnull().sum())
df = df.dropna()
print(df.isnull().sum())

# Categorical Variables and Dummy Variables
print(df.select_dtypes(["object"]).columns)
print(df["term"].value_counts())
# Or just use .map()
df["term"] = df["term"].apply(lambda term: int(term[:3]))
df = df.drop("grade", axis=1)
subgrade_dummies = get_dummies(df["sub_grade"], drop_first=True)
df = concat([df.drop("sub_grade", axis=1), subgrade_dummies], axis=1)
print(df.columns)
print(df.select_dtypes(["object"]).columns)
dummies = get_dummies(
    df[["verification_status", "application_type", "initial_list_status", "purpose"]],
    drop_first=True,
)
df = df.drop(
    ["verification_status", "application_type", "initial_list_status", "purpose"],
    axis=1,
)
df = concat([df, dummies], axis=1)

# home_ownership
print(df["home_ownership"].value_counts())
df["home_ownership"] = df["home_ownership"].replace(["NONE", "ANY"], "OTHER")
dummies = get_dummies(df["home_ownership"], drop_first=True)
df = df.drop("home_ownership", axis=1)
df = concat([df, dummies], axis=1)

# address
df["zip_code"] = df["address"].apply(lambda address: address[-5:])
dummies = get_dummies(df["zip_code"], drop_first=True)
df = df.drop(["zip_code", "address"], axis=1)
df = concat([df, dummies], axis=1)
df = df.drop("issue_d", axis=1)
df["earliest_cr_year"] = df["earliest_cr_line"].apply(lambda date: int(date[-4:]))
df = df.drop("earliest_cr_line", axis=1)
print(df.select_dtypes(["object"]).columns)

# Train Test Split
df = df.drop("loan_status", axis=1)
X = df.drop("loan_repaid", axis=1).values
y = df["loan_repaid"].values

# Grabbing a Sample for Training Time
print(len(df))
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=101
)

# Normalizing the Data
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Creating the Model
model = models.Sequential()
# input layer
model.add(layers.Dense(78, activation="relu"))
model.add(layers.Dropout(0.5))
# hidden layer
model.add(layers.Dense(39, activation="relu"))
model.add(layers.Dropout(0.5))
# hidden layer
model.add(layers.Dense(19, activation="relu"))
model.add(layers.Dropout(0.5))
# output layer
model.add(layers.Dense(units=1, activation="sigmoid"))
# Compile model
model.compile(loss="binary_crossentropy", optimizer="adam")
model.fit(
    x=X_train, y=y_train, epochs=25, batch_size=256, validation_data=(X_test, y_test)
)
model.save("full_data_project_model.h5")

# Evaluating Model Performance
losses = DataFrame(model.history.history)
figure()
losses[["loss", "val_loss"]].plot()
predictions = argmax(model.predict(X_test), axis=-1)
print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))
seed(101)
random_ind = randint(0, len(df))
new_customer = df.drop("loan_repaid", axis=1).iloc[random_ind]
print(new_customer)
print(argmax(model.predict(new_customer.values.reshape(1, 78)), axis=-1))
print(df.iloc[random_ind]["loan_repaid"])

# LAST STEP
show()
