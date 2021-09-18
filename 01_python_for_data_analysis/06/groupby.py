"""
Groupby
"""
from pandas import DataFrame

# Create dataframe
data = {
    "Company": ["GOOG", "GOOG", "MSFT", "MSFT", "FB", "FB"],
    "Person": ["Sam", "Charlie", "Amy", "Vanessa", "Carl", "Sarah"],
    "Sales": [200, 120, 340, 124, 243, 350],
}
df = DataFrame(data)
print(df)
by_comp = df.groupby("Company")
print(by_comp.mean())
print(by_comp.std())
print(by_comp.min())
print(by_comp.max())
print(by_comp.count())
print(by_comp.describe())
print(by_comp.describe().transpose())
print(by_comp.describe().transpose()["GOOG"])
