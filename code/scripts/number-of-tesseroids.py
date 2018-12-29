"""
We will compute the number of discretization that the density-based algorithm performs
in case of an exponential density and a delta value of 0.1.
We will explore different values of the constant b to see how it affects the
computational load.
"""
from __future__ import division
import os
import warnings
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

# This is our custom tesseroid code
from tesseroid_density import tesseroid


# Check for DISPLAY variable for matplotlib
# -----------------------------------------
try:
    os.environ["DISPLAY"]
except Exception:
    plt.switch_backend('agg')


# Configure LaTeX style for plots
# -------------------------------
try:
    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = 'Computer Modern Roman'
    plt.rcParams['xtick.major.size'] = 2
    plt.rcParams['ytick.major.size'] = 2
except Exception as e:
    warnings.warn("Couldn't configure LaTeX style for plots:" + str(e))


# Define Tesseroid boundaries
# ---------------------------
w, e, s, n = -10, 10, -10, 10
top, bottom = 0, -1e3


# Initialize figure and GridSpec
# ------------------------------
fig = plt.figure(figsize=(6.66, 2.8))
outer_grid = GridSpec(ncols=3, nrows=1)
script_path = os.path.dirname(os.path.abspath(__file__))
figure_fname = os.path.join(script_path,
                            "../../manuscript/figures/number-of-tesseroids.pdf")


# Compute number of discretized tesseroids (exp)
# ----------------------------------------------
# Define denstiy related parameters
delta = 0.1
density_top, density_bottom = 0, 1
b_factors = np.array([1, 2, 5, 10, 30, 100])
n_tess = np.zeros_like(b_factors, dtype=np.int64)

# Define axe for this plot
ax = plt.subplot(outer_grid[0])

# Compute number of discretizations
heights = np.linspace(bottom, top, 101)
colors = dict(zip(b_factors, plt.cm.viridis(np.linspace(0, 0.9, len(b_factors)))))
for i, b_factor in enumerate(b_factors):
    thickness = top - bottom
    A = (density_bottom - density_top) / (1 - np.exp(-b_factor))
    C = density_bottom - A

    # Define density function
    def density_exponential(height):
        return A * np.exp(-b_factor * (height - bottom) / thickness) + C

    # Calculate number of discretizations
    subset = tesseroid._density_based_discretization([w, e, s, n, top, bottom],
                                                     density_exponential,
                                                     delta)
    n_tess[i] = len(subset)

    # Plot density profile and boundaries of discretized tesseroids
    bounds = []
    for tess in subset:
        bounds.append(tess[-1])
        bounds.append(tess[-2])
    bounds = np.unique(np.array(bounds))
    ax.plot(heights, density_exponential(heights), '-', color=colors[b_factor])
    ax.plot(bounds, density_exponential(bounds), 'o', color="C1", markersize=4)

    # Configure axes
    ax.set_yticks([])
    ax.set_xticks([bottom, top])
    ax.set_xticklabels([r"$r_1$", r"$r_2$"])
    ax.set_ylabel("Density")
    ax.set_xlabel("Depth")
    ax.set_title("(a)")


# Compute number of discretized tesseroids (sine)
# -----------------------------------------------
# Define denstiy related parameters
delta = 0.1
b_factors = [1, 2, 5, 10]
n_tess = np.zeros_like(b_factors, dtype=np.int64)

# Define axes for this plot
subgrid = GridSpecFromSubplotSpec(nrows=len(b_factors), ncols=1,
                                  subplot_spec=outer_grid[1], hspace=0)
axes = [plt.subplot(subgrid[0])]
for i in range(1, len(b_factors)):
    axes.append(plt.subplot(subgrid[i], sharex=axes[0]))

# Compute number of discretizations
heights = np.linspace(bottom, top, 501)
colors = dict(zip(b_factors, plt.cm.viridis(np.linspace(0, 0.9, len(b_factors)))))
for i, b_factor in enumerate(b_factors):
    max_density = density_bottom
    thickness = top - bottom
    k_factor = 2 * np.pi * b_factor / thickness

    # Define density function
    def density_sine(height):
        return max_density / 2 * np.sin(k_factor * height) + max_density / 2

    # Calculate number of discretizations
    subset = tesseroid._density_based_discretization([w, e, s, n, top, bottom],
                                                     density_sine,
                                                     delta)
    n_tess[i] = len(subset)

    # Plot density profile and boundaries of discretized tesseroids
    bounds = []
    for tess in subset:
        bounds.append(tess[-1])
        bounds.append(tess[-2])
    bounds = np.unique(np.array(bounds))
    axes[i].plot(heights, density_sine(heights), '-', color=colors[b_factor])
    axes[i].plot(bounds, density_sine(bounds), 'o', color="C1", markersize=4)

    # Configure axes
    axes[i].set_ylim(-0.2, 1.2)
    axes[i].set_yticks([])
    axes[i].set_xticks([bottom, top])
    axes[i].set_xticklabels([r"$r_1$", r"$r_2$"])
    axes[i].set_xlabel("Depth")
    if i == 0:
        axes[i].set_title("(b)")


# Plot number of tesseroids (sine)
# --------------------------------
ax = plt.subplot(outer_grid[2])
ax.plot(b_factors, n_tess, 'o')
ax.grid()
ax.set_ylim(2, 21)
ax.yaxis.tick_right()
ax.yaxis.set_label_position("right")
ax.set_yticks(np.arange(3, 21, 2))
ax.set_xticks(np.arange(1, 11, 2))
ax.set_xlabel(r"$b$")
ax.set_ylabel("Number of tesseroids")
ax.set_title("(c)")

outer_grid.tight_layout(fig)
plt.savefig(figure_fname, dpi=300)
plt.show()
