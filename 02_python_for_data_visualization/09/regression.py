"""
Regression Plots
"""
from matplotlib.pyplot import show
from seaborn import load_dataset, lmplot

# tips
tips = load_dataset("tips")
print(tips.head())

# lmplot
lmplot(x="total_bill", y="tip", data=tips)
lmplot(x="total_bill", y="tip", data=tips, hue="sex")
lmplot(x="total_bill", y="tip", data=tips, hue="sex", palette="coolwarm")
lmplot(
    x="total_bill",
    y="tip",
    data=tips,
    hue="sex",
    palette="coolwarm",
    markers=["o", "v"],
    scatter_kws={"s": 100},
)

# Using a Grid
lmplot(x="total_bill", y="tip", data=tips, col="sex")
lmplot(x="total_bill", y="tip", row="sex", col="time", data=tips)
lmplot(x="total_bill", y="tip", data=tips, col="day", hue="sex", palette="coolwarm")
lmplot(
    x="total_bill",
    y="tip",
    data=tips,
    col="day",
    hue="sex",
    palette="coolwarm",
    aspect=0.6,
    height=8,
)

# LAST STEP
show()
