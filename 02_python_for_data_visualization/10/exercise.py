"""
Pandas Data Visualization Exercise
"""
from pandas import read_csv, DataFrame
from matplotlib.pyplot import show, figure, style, legend

# Data
df3 = DataFrame(read_csv("df3"))
print(df3.info())
print(df3.head())

# Diagram
df3.plot.scatter(x="a", y="b", c="red", s=50, figsize=(12, 3))
figure()
df3["a"].plot.hist()
figure()
style.use("ggplot")
df3["a"].plot.hist(alpha=0.5, bins=25)
df3[["a", "b"]].plot.box()
figure()
df3["d"].plot.kde()
figure()
df3["d"].plot.density(lw=5, ls="--")
df3.iloc[0:30].plot.area(alpha=0.4)

# Bonus
f = figure()
df3.iloc[0:30].plot.area(alpha=0.4, ax=f.gca())
legend(loc="center left", bbox_to_anchor=(1.0, 0.5))

# LAST STEP
show()
