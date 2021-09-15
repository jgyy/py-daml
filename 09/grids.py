"""
Grids
"""
from warnings import simplefilter
from matplotlib.pyplot import show, scatter, hist
from seaborn import (
    load_dataset,
    PairGrid,
    kdeplot,
    pairplot,
    FacetGrid,
    JointGrid,
    regplot,
    distplot,
)

# iris and tips
simplefilter(action="ignore", category=FutureWarning)
iris = load_dataset("iris")
tips = load_dataset("tips")
print(iris.head())
print(tips.head())

# Just the Grid
PairGrid(iris)

# Then you map to the grid
g = PairGrid(iris)
g.map(scatter)

# Map to upper,lower, and diagonal
g = PairGrid(iris)
g.map_diag(hist)
g.map_upper(scatter)
g.map_lower(kdeplot)

# pairplot
pairplot(iris)
pairplot(iris, hue="species", palette="rainbow")

# Just the Grid
g = FacetGrid(tips, col="time", row="smoker")
g = FacetGrid(tips, col="time", row="smoker")
g = g.map(hist, "total_bill")

# Notice hwo the arguments come after plt.scatter call
g = FacetGrid(tips, col="time", row="smoker", hue="sex")
g = g.map(scatter, "total_bill", "tip").add_legend()

# JointGrid
g = JointGrid(x="total_bill", y="tip", data=tips)
g = JointGrid(x="total_bill", y="tip", data=tips)
g = g.plot(regplot, distplot)

# LAST STEP
show()
