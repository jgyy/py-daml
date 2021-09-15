"""
Seaborn Exercise
"""
from warnings import simplefilter
from pandas import DataFrame
from matplotlib.pyplot import show, figure, title, hist
from seaborn import (
    set_style,
    load_dataset,
    jointplot,
    distplot,
    boxplot,
    swarmplot,
    countplot,
    heatmap,
    FacetGrid,
)

# Data
simplefilter(action="ignore", category=FutureWarning)
set_style("whitegrid")
titanic = DataFrame(load_dataset("titanic"))
print(titanic.head())

# Exercises joinplot no need figure
jointplot(x="fare", y="age", data=titanic)
figure()
distplot(titanic["fare"], bins=30, kde=False, color="red")
figure()
boxplot(x="class", y="age", data=titanic, palette="rainbow")
figure()
swarmplot(x="class", y="age", data=titanic, palette="Set2", size=4)
figure()
countplot(x="sex", data=titanic)
figure()
heatmap(titanic.corr(), cmap="coolwarm")
title("titanic.corr()")
g = FacetGrid(data=titanic, col="sex")
g.map(hist, "age")

# LAST STEP
show()
