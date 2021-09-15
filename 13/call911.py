"""
911 Calls
"""
from matplotlib.pyplot import show, legend, figure, tight_layout, title
from pandas import read_csv, DataFrame, to_datetime
from seaborn import set_style, countplot, lmplot, heatmap, clustermap

# Read in the tcsv file as a dataframe called df
set_style("whitegrid")
df = DataFrame(read_csv("911.csv"))
print(df.info())
print(df.head(3))

# What are the top 5 zipcodes for 911 calls?
print(df["zip"].value_counts().head(5))
print(df["twp"].value_counts().head(5))
print(df["title"].nunique())

# Creating new features
df["Reason"] = df["title"].apply(lambda title: title.split(":")[0])
print(df["Reason"].value_counts())
figure()
countplot(x="Reason", data=df, palette="viridis")

# Check type of timestamp
print(type(df["timeStamp"].iloc[0]))
df["timeStamp"] = to_datetime(df["timeStamp"])
df["Hour"] = df["timeStamp"].apply(lambda time: time.hour)
df["Month"] = df["timeStamp"].apply(lambda time: time.month)
df["Day of Week"] = df["timeStamp"].apply(lambda time: time.dayofweek)
dmap = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}
df["Day of Week"] = df["Day of Week"].map(dmap)

# To relocate the legend
figure()
countplot(x="Day of Week", data=df, hue="Reason", palette="viridis")
legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
figure()
countplot(x="Month", data=df, hue="Reason", palette="viridis")
legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)

# It is missing some months! 9,10, and 11 are not there.
byMonth = df.groupby("Month").count()
print(byMonth.head())

# Could be any column
figure()
byMonth["twp"].plot()
lmplot(x="Month", y="twp", data=byMonth.reset_index())
figure()
df["Date"] = df["timeStamp"].apply(lambda t: t.date())
df.groupby("Date").count()["twp"].plot()
tight_layout()

# Recreate the plot
figure()
df[df["Reason"] == "Traffic"].groupby("Date").count()["twp"].plot()
title("Traffic")
tight_layout()
figure()
df[df["Reason"] == "Fire"].groupby("Date").count()["twp"].plot()
title("Fire")
tight_layout()
figure()
df[df["Reason"] == "EMS"].groupby("Date").count()["twp"].plot()
title("EMS")
tight_layout()

# Create heatmaps
dayHour = df.groupby(by=["Day of Week", "Hour"]).count()["Reason"].unstack()
print(dayHour.head())
figure(figsize=(12, 6))
heatmap(dayHour, cmap="viridis")
clustermap(dayHour, cmap="viridis")
dayMonth = df.groupby(by=['Day of Week','Month']).count()['Reason'].unstack()
print(dayMonth.head())
figure(figsize=(12,6))
heatmap(dayMonth,cmap='viridis')
clustermap(dayMonth,cmap='viridis')

# LAST STEP
show()
