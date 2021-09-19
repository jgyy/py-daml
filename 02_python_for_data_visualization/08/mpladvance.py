"""
Matplotlib Advance
"""
from matplotlib import ticker, rcParams
from matplotlib.gridspec import GridSpec
from matplotlib.pyplot import subplots, show, figure, subplot2grid
from mpl_toolkits.mplot3d.axes3d import Axes3D
from numpy.random import randn
from numpy import linspace, exp, array, pi, cos, meshgrid

# Basic Example
x = linspace(0, 5, 11)
y = x ** 2
print(x)
print(y)

# Logarithmic scale
fig, axes = subplots(1, 2, figsize=(10, 4))
axes[0].plot(x, x ** 2, x, exp(x))
axes[0].set_title("Normal scale")
axes[1].plot(x, x ** 2, x, exp(x))
axes[1].set_yscale("log")
axes[1].set_title("Logarithmic scale (y)")

# Scientific notation
fig, ax = subplots(1, 1)
ax.plot(x, x ** 2, x, exp(x))
ax.set_title("scientific notation")
ax.set_yticks([0, 50, 100, 150])
formatter = ticker.ScalarFormatter(useMathText=True)
formatter.set_scientific(True)
formatter.set_powerlimits((-1, 1))
ax.yaxis.set_major_formatter(formatter)

# distance between x and y axis and the numbers on the axes
rcParams["xtick.major.pad"] = 5
rcParams["ytick.major.pad"] = 5
fig, ax = subplots(1, 1)
ax.plot(x, x ** 2, x, exp(x))
ax.set_yticks([0, 50, 100, 150])
ax.set_title("label and axis spacing")
# padding between axis label and axis numbers
ax.xaxis.labelpad = 5
ax.yaxis.labelpad = 5
ax.set_xlabel("x")
ax.set_ylabel("y")

# restore defaults
rcParams["xtick.major.pad"] = 3
rcParams["ytick.major.pad"] = 3

# Axis position adjustments
fig, ax = subplots(1, 1)
ax.plot(x, x ** 2, x, exp(x))
ax.set_yticks([0, 50, 100, 150])
ax.set_title("title")
ax.set_xlabel("x")
ax.set_ylabel("y")
fig.subplots_adjust(left=0.15, right=0.9, bottom=0.1, top=0.9)

# Axis grid
fig, axes = subplots(1, 2, figsize=(10, 3))
# default grid appearance
axes[0].plot(x, x ** 2, x, x ** 3, lw=2)
axes[0].grid(True)
# custom grid appearance
axes[1].plot(x, x ** 2, x, x ** 3, lw=2)
axes[1].grid(color="b", alpha=0.5, linestyle="dashed", linewidth=0.5)

# Axis spines
fig, ax = subplots(figsize=(6, 2))
ax.spines["bottom"].set_color("blue")
ax.spines["top"].set_color("blue")
ax.spines["left"].set_color("red")
ax.spines["left"].set_linewidth(2)
# turn off axis spine to the right
ax.spines["right"].set_color("none")
ax.yaxis.tick_left()  # only ticks on the left side

# Twin aces
fig, ax1 = subplots()
ax1.plot(x, x ** 2, lw=2, color="blue")
ax1.set_ylabel(r"area $(m^2)$", fontsize=18, color="blue")
for label in ax1.get_yticklabels():
    label.set_color("blue")
ax2 = ax1.twinx()
ax2.plot(x, x ** 3, lw=2, color="red")
ax2.set_ylabel(r"volume $(m^3)$", fontsize=18, color="red")
for label in ax2.get_yticklabels():
    label.set_color("red")

# Axes where x and y is zero
fig, ax = subplots()
ax.spines["right"].set_color("none")
ax.spines["top"].set_color("none")
ax.xaxis.set_ticks_position("bottom")
ax.spines["bottom"].set_position(("data", 0))  # set position of x spine to x=0
ax.yaxis.set_ticks_position("left")
ax.spines["left"].set_position(("data", 0))  # set position of y spine to y=0
xx = linspace(-0.75, 1.0, 100)
ax.plot(xx, xx ** 3)

# Other 2D plot styles
n = array([0, 1, 2, 3, 4, 5])
fig, axes = subplots(1, 4, figsize=(12, 3))
axes[0].scatter(xx, xx + 0.25 * randn(len(xx)))
axes[0].set_title("scatter")
axes[1].step(n, n ** 2, lw=2)
axes[1].set_title("step")
axes[2].bar(n, n ** 2, align="center", width=0.5, alpha=0.5)
axes[2].set_title("bar")
axes[3].fill_between(x, x ** 2, x ** 3, color="green", alpha=0.5)
axes[3].set_title("fill_between")

# Text annotation
fig, ax = subplots()
ax.plot(xx, xx ** 2, xx, xx ** 3)
ax.text(0.15, 0.2, r"$y=x^2$", fontsize=20, color="blue")
ax.text(0.65, 0.1, r"$y=x^3$", fontsize=20, color="green")

# subplots
fig, ax = subplots(2, 3)
fig.tight_layout()

# subplot2grid
fig = figure()
ax1 = subplot2grid((3, 3), (0, 0), colspan=3)
ax2 = subplot2grid((3, 3), (1, 0), colspan=2)
ax3 = subplot2grid((3, 3), (1, 2), rowspan=2)
ax4 = subplot2grid((3, 3), (2, 0))
ax5 = subplot2grid((3, 3), (2, 1))
fig.tight_layout()

# gridspec
fig = figure()
gs = GridSpec(2, 3, height_ratios=[2, 1], width_ratios=[1, 2, 1])
for g in gs:
    ax = fig.add_subplot(g)
fig.tight_layout()

# add_axes
fig, ax = subplots()
ax.plot(xx, xx ** 2, xx, xx ** 3)
fig.tight_layout()
# inset
inset_ax = fig.add_axes([0.2, 0.55, 0.35, 0.35])  # X, Y, width, height
inset_ax.plot(xx, xx ** 2, xx, xx ** 3)
inset_ax.set_title("zoom near origin")
# set axis range
inset_ax.set_xlim(-0.2, 0.2)
inset_ax.set_ylim(-0.005, 0.01)
# set axis tick locations
inset_ax.set_yticks([0, 0.005, 0.01])
inset_ax.set_xticks([-0.1, 0, 0.1])

# Colormap and contour figures
ALPHA = 0.7
phi_ext = 2 * pi * 0.5
flux_qubit_potential = (
    lambda phi_m, phi_p: 2
    + ALPHA
    - 2 * cos(phi_p) * cos(phi_m)
    - ALPHA * cos(phi_ext - 2 * phi_p)
)
phi_m = linspace(0, 2 * pi, 100)
phi_p = linspace(0, 2 * pi, 100)
X, Y = meshgrid(phi_p, phi_m)
Z = flux_qubit_potential(X, Y).T

# pcolor
fig, ax = subplots()
p = ax.pcolor(
    X / (2 * pi),
    Y / (2 * pi),
    Z,
    cmap="RdBu",
    vmin=abs(Z).min(),
    vmax=abs(Z).max(),
    shading="auto",
)
cb = fig.colorbar(p, ax=ax)

# imshow
fig, ax = subplots()
im = ax.imshow(
    Z,
    cmap="RdBu",
    vmin=abs(Z).min(),
    vmax=abs(Z).max(),
    extent=[0, 1, 0, 1],
)
im.set_interpolation("bilinear")
cb = fig.colorbar(im, ax=ax)

# contour
fig, ax = subplots()
cnt = ax.contour(
    Z,
    cmap="RdBu",
    vmin=abs(Z).min(),
    vmax=abs(Z).max(),
    extent=[0, 1, 0, 1],
)

# Surface plots
fig = figure(figsize=(14, 6))
# `ax` is a 3D-aware axis instance because of the projection='3d' keyword argument to add_subplot
ax = fig.add_subplot(1, 2, 1, projection="3d")
p = ax.plot_surface(X, Y, Z, rstride=4, cstride=4, linewidth=0)
# surface_plot with color grading and color bar
ax = fig.add_subplot(1, 2, 2, projection="3d")
p = ax.plot_surface(
    X,
    Y,
    Z,
    rstride=1,
    cstride=1,
    cmap="coolwarm",
    linewidth=0,
    antialiased=False,
)
cb = fig.colorbar(p, shrink=0.5)

# Wire-frame plot
fig = figure(figsize=(8, 6))
ax = fig.add_subplot(1, 1, 1, projection="3d")
p = ax.plot_wireframe(X, Y, Z, rstride=4, cstride=4)

# Contour plots with projections
fig = figure(figsize=(8, 6))
ax = fig.add_subplot(1, 1, 1, projection="3d")
ax.plot_surface(X, Y, Z, rstride=4, cstride=4, alpha=0.25)
cset = ax.contour(X, Y, Z, zdir="z", offset=-pi, cmap="coolwarm")
cset = ax.contour(X, Y, Z, zdir="x", offset=-pi, cmap="coolwarm")
cset = ax.contour(X, Y, Z, zdir="y", offset=3 * pi, cmap="coolwarm")
ax.set_xlim3d(-pi, 2 * pi)
ax.set_ylim3d(0, 3 * pi)
ax.set_zlim3d(-pi, 2 * pi)

# LAST STEP TO SHOW
show()
