"""
Matrix Plots
"""
from pandas import DataFrame
from matplotlib.pyplot import figure, show
from seaborn import load_dataset, heatmap, clustermap

# Datasets
flights = DataFrame(load_dataset("flights"))
tips = DataFrame(load_dataset("tips"))
print(flights.head())
print(tips.head())
print(tips.corr())

# heatmap
figure()
heatmap(tips.corr())
figure()
heatmap(tips.corr(), cmap="coolwarm", annot=True)
pvflights = flights.pivot_table(values="passengers", index="month", columns="year")
print(pvflights)
figure()
heatmap(pvflights)
figure()
heatmap(pvflights, cmap="magma", linecolor="white", linewidths=1)

# clustermap
clustermap(pvflights)

# More options to get the information a little clearer like normalization
clustermap(pvflights,cmap='coolwarm',standard_scale=1)

# LAST STEP
show()
