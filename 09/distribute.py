"""
Distribution Plots
"""
from warnings import simplefilter
from scipy import stats
from pandas import DataFrame
import numpy as npy
from numpy import linspace
from numpy.random import randn
from matplotlib.pyplot import show, plot, ylim, yticks, suptitle, figure
from seaborn import load_dataset, distplot, jointplot, pairplot, rugplot, kdeplot

# Data
simplefilter(action="ignore", category=FutureWarning)
tips = DataFrame(load_dataset("tips"))
print(tips.head())

# distplot
distplot(tips["total_bill"])
distplot(tips["total_bill"], kde=False, bins=30)

# jointplot
jointplot(x="total_bill", y="tip", data=tips, kind="scatter")
jointplot(x="total_bill", y="tip", data=tips, kind="hex")
jointplot(x="total_bill", y="tip", data=tips, kind="reg")

# pairplot
pairplot(tips)
pairplot(tips, hue="sex", palette="coolwarm")

# rugplot
figure()
rugplot(tips["total_bill"])

# kdeplot Create dataset
dataset = randn(25)
# Create another rugplot
figure()
rugplot(dataset)
# Set up the x-axis for the plot
x_min = dataset.min() - 2
x_max = dataset.max() + 2
# 100 equally spaced points from x_min to x_max
x_axis = linspace(x_min, x_max, 100)
# Set up the bandwidth
bandwidth = ((4 * dataset.std() ** 5) / (3 * len(dataset))) ** 0.2
# Create an empty kernel list
kernel_list = []
# Plot each basis function
for data_point in dataset:
    # Create a kernel for each point and append to list
    kernel = stats.norm(data_point, bandwidth).pdf(x_axis)
    kernel_list.append(kernel)
    # Scale for plotting
    kernel = kernel / kernel.max()
    kernel = kernel * 0.4
    plot(x_axis, kernel, color="grey", alpha=0.5)
ylim(0, 1)

# Plot the sum of the basis function
sum_of_kde = npy.sum(kernel_list, axis=0)
# Plot figure
fig = plot(x_axis, sum_of_kde, color="indianred")
# Add the initial rugplot
figure()
rugplot(dataset, c="indianred")
# Get rid of y-tick marks
yticks([])
# Set title
suptitle("Sum of the Basis Functions")

# Datasets
figure()
kdeplot(tips["total_bill"])
figure()
rugplot(tips["total_bill"])
figure()
kdeplot(tips['tip'])
figure()
rugplot(tips['tip'])

# LAST STEP
show()
