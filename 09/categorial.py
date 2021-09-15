"""
Categorical Plots
"""
from numpy import std
from matplotlib.pyplot import show, figure
from seaborn import (
    load_dataset,
    barplot,
    countplot,
    boxplot,
    violinplot,
    stripplot,
    swarmplot,
    catplot,
)

# data
tips = load_dataset("tips")
print(tips.head())

# barplot and countplot
figure()
barplot(x="sex", y="total_bill", data=tips)
figure()
barplot(x="sex", y="total_bill", data=tips, estimator=std)
figure()
countplot(x="sex", data=tips)

# boxplot and violinplot
figure()
boxplot(x="day", y="total_bill", data=tips, palette="rainbow")
figure()
boxplot(data=tips, palette="rainbow", orient="h")
figure()
boxplot(x="day", y="total_bill", hue="smoker", data=tips, palette="coolwarm")
figure()
violinplot(x="day", y="total_bill", data=tips, palette="rainbow")
figure()
violinplot(x="day", y="total_bill", data=tips, hue="sex", dodge=True, palette="Set1")

# stripplot and swarmplot
figure()
stripplot(x="day", y="total_bill", data=tips)
figure()
stripplot(x="day", y="total_bill", data=tips, jitter=True)
figure()
stripplot(x="day", y="total_bill", data=tips, jitter=True, hue="sex", palette="Set1")
figure()
stripplot(
    x="day",
    y="total_bill",
    data=tips,
    jitter=True,
    hue="sex",
    palette="Set1",
    dodge=True,
)
figure()
swarmplot(x="day", y="total_bill", data=tips)
figure()
swarmplot(x="day", y="total_bill", hue="sex", data=tips, palette="Set1", dodge=True)

# Combining Categorical Plots
figure()
violinplot(x="tip", y="day", data=tips, palette="rainbow")
swarmplot(x="tip", y="day", data=tips, color="black", size=3)

# Factorplot renamed to catplot
catplot(x="sex", y="total_bill", data=tips, kind="bar")

# LAST STEP
show()
