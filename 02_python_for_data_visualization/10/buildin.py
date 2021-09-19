"""
Pandas Built-in Data Visualization
"""
from warnings import filterwarnings
from numpy.random import randn
from pandas import read_csv, DataFrame
from matplotlib.pyplot import show, figure, style

# Data
filterwarnings("ignore", category=RuntimeWarning)
df1 = DataFrame(read_csv("df1", index_col=0))
df2 = DataFrame(read_csv("df2"))
print(df1)
print(df2)

# Style Sheets
figure()
df1["A"].hist()
figure()
style.use("ggplot")
df1["A"].hist()
figure()
style.use("bmh")
df1["A"].hist()
figure()
style.use("dark_background")
df1["A"].hist()
figure()
style.use("fivethirtyeight")
df1["A"].hist()

# Plot Types
style.use("ggplot")
df2.plot.area(alpha=0.4)

# Barplots
print(df2.head())
df2.plot.bar()
df2.plot.bar(stacked=True)
figure()
df1["A"].plot.hist(bins=50)

# use_index=True as attributes for Line Plots
df1.plot.line(y="B", figsize=(12, 3), lw=1, use_index=True)
df1.plot.scatter(x="A", y="B")
df1.plot.scatter(x="A", y="B", c="C", cmap="coolwarm")
df1.plot.scatter(x="A", y="B", s=df1["C"] * 200)

# BoxPlots
df2.plot.box()

# Hexagonal Bin Plot
df = DataFrame(randn(1000, 2), columns=["a", "b"])
df.plot.hexbin(x="a", y="b", gridsize=25, cmap="Oranges")
figure()
df2["a"].plot.kde()
df2.plot.density()

# LAST STEP
show()
