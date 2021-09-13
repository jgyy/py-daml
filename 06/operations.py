"""
Operations
"""
from numpy import nan
from pandas import DataFrame

df = DataFrame(
    {
        "col1": [1, 2, 3, 4],
        "col2": [444, 555, 666, 444],
        "col3": ["abc", "def", "ghi", "xyz"],
    }
)
print(df.head())
print(df["col2"].unique())
print(df["col2"].nunique())
print(df["col2"].value_counts())

# Select from DataFrame using criteria from multiple columns
newdf = df[(df["col1"] > 2) & (df["col2"] == 444)]
print(newdf)
times2 = lambda x: x * 2
print(df["col1"].apply(times2))
print(df["col3"].apply(len))
print(df["col1"].sum())

# Remove column
del df["col1"]
print(df)
print(df.columns)
print(df.index)
print(df.sort_values(by="col2"))
print(df.isnull())
print(df.dropna())

# ** Filling in NaN values with something else: **
df = DataFrame(
    {
        "col1": [1, 2, 3, nan],
        "col2": [nan, 555, 666, 444],
        "col3": ["abc", "def", "ghi", "xyz"],
    }
)
print(df.head())
print(df.fillna("FILL"))

data = {
    "A": ["foo", "foo", "foo", "bar", "bar", "bar"],
    "B": ["one", "one", "two", "two", "one", "one"],
    "C": ["x", "y", "x", "y", "x", "y"],
    "D": [1, 3, 2, 5, 4, 1],
}

df = DataFrame(data)
print(df)
print(df.pivot_table(values="D", index=["A", "B"], columns=["C"]))
