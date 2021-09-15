"""
Matplotlib Exercises
"""
from numpy import arange
from matplotlib.pyplot import figure, show, subplots

# Data
x = arange(0, 100)
y = x * 2
z = x ** 2

# Exercise 1
fig = figure()
ax = fig.add_axes([0, 0, 1, 1])
ax.plot(x, y)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("title")

# Exercise 2
fig = figure()
ax1 = fig.add_axes([0, 0, 1, 1])
ax2 = fig.add_axes([0.2, 0.5, 0.2, 0.2])
ax1.plot(x, y)
ax1.set_xlabel("x")
ax1.set_ylabel("y")
ax2.plot(x, y)
ax2.set_xlabel("x")
ax2.set_ylabel("y")

# Exercise 3
fig = figure()
ax = fig.add_axes([0, 0, 1, 1])
ax2 = fig.add_axes([0.2, 0.5, 0.4, 0.4])
ax.plot(x, z)
ax.set_xlabel("X")
ax.set_ylabel("Z")
ax2.plot(x, y)
ax2.set_xlabel("X")
ax2.set_ylabel("Y")
ax2.set_title("zoom")
ax2.set_xlim(20, 22)
ax2.set_ylim(30, 50)

# Exercise 4
fig, axes = subplots(nrows=1, ncols=2)
axes[0].plot(x, y, color="blue", lw=3, ls="--")
axes[1].plot(x, z, color="red", lw=3, ls="-")

# Bonus
fig, axes = subplots(nrows=1, ncols=2, figsize=(12, 2))
axes[0].plot(x, y, color="blue", lw=5)
axes[0].set_xlabel("x")
axes[0].set_ylabel("y")
axes[1].plot(x, z, color="red", lw=3, ls="--")
axes[1].set_xlabel("x")
axes[1].set_ylabel("z")

# LAST STEP
show()
