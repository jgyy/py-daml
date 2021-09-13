"""
Missing Data
"""
from numpy import nan
from pandas import DataFrame

df = DataFrame({"A": [1, 2, nan], "B": [5, nan, nan], "C": [1, 2, 3]})
print(df)
print(df.dropna())
print(df.dropna(axis=1))
print(df.dropna(thresh=2))
print(df.fillna(value="FILL VALUE"))
print(df["A"].fillna(value=df["A"].mean()))
