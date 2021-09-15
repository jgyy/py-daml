"""
Style and Color
"""
from matplotlib.pyplot import show, figure
from seaborn import (
    countplot,
    set_style,
    load_dataset,
    despine,
    lmplot,
    set_context,
)

# tips
tips = load_dataset("tips")

# Styles
figure()
countplot(x="sex", data=tips)
figure()
set_style("white")
countplot(x="sex", data=tips)
figure()
set_style("ticks")
countplot(x="sex", data=tips, palette="deep")

# Spine Removal
figure()
countplot(x="sex", data=tips)
despine()
figure()
countplot(x="sex", data=tips)
despine(left=True)

# Non Grid Plot
figure(figsize=(12, 3))
countplot(x="sex", data=tips)

# Grid Type Plot
lmplot(x="total_bill", y="tip", height=2, aspect=4, data=tips)

# Scale and Context
figure()
set_context("poster", font_scale=4)
countplot(x="sex", data=tips, palette="coolwarm")

# LAST STEP
show()
