from __future__ import division
import os
import warnings
import numpy as np
import matplotlib.pyplot as plt

# This is our custom tesseroid code
from tesseroid_density import tesseroid


# Define functions
def density_function(height):
    return np.exp(b_factor * height / thickness)


def normalized_density(height):
    return (density_function(height) - rho_min)/(rho_max - rho_min)


def line(height, top, bottom):
    a = (normalized_density(top) - normalized_density(bottom))/(top - bottom)
    b = normalized_density(bottom)
    return a*(height - bottom) + b


# Define tesseroid boundaries
w, e, s, n, top, bottom = -10, 10, -10, 10, 0, -1
thickness = top - bottom
bounds = [w, e, s, n, top, bottom]

# Set b factor for exp density variation
b_factor = 6

# Define heights array and boundary density values
heights = np.linspace(bottom, top, 101)
density = density_function(heights)
rho_min, rho_max = density.min(), density.max()


# Calculate discretizations
# -------------------------
delta = 1
delta_step = 0.001
max_divisions = 4
subsets = [[np.array(bounds)]]
deltas = [1]

while True:
    subset = tesseroid._density_based_discretization(bounds,
                                                     density_function,
                                                     delta)
    divisions = len(subset)
    if divisions == len(subsets[-1]) + 1:
        subsets.append(subset)
        deltas.append(delta)
    elif divisions > len(subsets[-1]) + 1:
        print("More discretizations than 1 in delta {}".format(delta))
    if divisions >= max_divisions:
        break
    else:
        delta -= delta_step


# Plot discretization steps
# -------------------------

# Configure LaTeX style for plots
try:
    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = 'Computer Modern Roman'
    plt.rcParams['xtick.major.size'] = 2
    plt.rcParams['ytick.major.size'] = 2
except Exception as e:
    warnings.warn("Couldn't configure LaTeX style for plots:" + str(e))

# Initialize figure and subplots
fig, axes = plt.subplots(1, len(subsets),
                         figsize=(6.66, 2),
                         sharey=True)
labels = ["(a)", "(b)", "(c)", "(d)"]
divisions = np.array([top, bottom])

# Plot first three stages
for i in range(len(subsets) - 1):
    ax = axes[i]
    subset = subsets[i]

    tops = [tess[-2] for tess in subset]
    bottoms = [tess[-1] for tess in subset]
    divisions = np.unique(np.array(tops + bottoms))

    tops = [tess[-2] for tess in subsets[i + 1]]
    bottoms = [tess[-1] for tess in subsets[i + 1]]
    new_divisions = np.unique(np.array(tops + bottoms))

    line1 = ax.plot(heights, normalized_density(heights),
                    linewidth=2)
    dots = ax.plot(divisions, normalized_density(divisions), 'o')
    for j in range(1, len(new_divisions) - 1):
        bottom_j = new_divisions[j - 1]
        top_j = new_divisions[j + 1]
        div = new_divisions[j]
        if div not in divisions:
            line2 = ax.plot([div]*2,
                            [normalized_density(div),
                             line(div, top_j, bottom_j)],
                            '--', color="C3")
            line3 = ax.plot([bottom_j, top_j],
                            [normalized_density(bottom_j),
                             normalized_density(top_j)],
                            '-', color="C1")
            if i == 1:
                ax.plot([bottom_j, top_j],
                        [normalized_density(bottom_j) - 0.08]*2, "|--",
                        color="C7", markersize=8)
                ax.text(0.5*(bottom_j + top_j),
                        normalized_density(bottom_j) - 0.03,
                        '$L_r^{sm}$',
                        fontdict={'color': "C7"},
                        horizontalalignment='center')
            elif i == 2:
                ax.plot([bottom_j, top_j],
                        [normalized_density(top_j) + 0.08]*2, "|--",
                        color="C7", markersize=8)
                ax.text(0.5*(bottom_j + top_j),
                        normalized_density(top_j) + 0.13,
                        '$L_r^{sm}$',
                        fontdict={'color': "C7"},
                        horizontalalignment='center')

# Plot final stage
ax = axes[-1]
tops = [tess[-2] for tess in subsets[-1]]
bottoms = [tess[-1] for tess in subsets[-1]]
divisions = np.unique(np.array(tops + bottoms))
ax.plot(heights, normalized_density(heights),
        linewidth=2)
ax.plot(divisions, normalized_density(divisions), 'o')

# Configure axes
axes[0].set_ylabel("Normalized Density")
for ax, label in zip(axes, labels):
    ax.text(0.03, 0.91, label,
            fontdict={'weight': 'bold'},
            verticalalignment="center",
            transform=ax.transAxes)
    ax.set_xticks([-1, 0])
    ax.set_xticklabels([r"$r_1$", r"$r_2$"])
    ax.set_yticks([])

# Create legend
axes[-1].legend((line1[0], line3[0], line2[0], dots[0]),
                ("Norm. density",
                 "Straight Line",
                 "Max. difference",
                 "Discretizations"),
                fontsize="x-small",
                loc=(0.03, 0.46),
                )

plt.tight_layout(pad=1, h_pad=0, w_pad=0)
script_path = os.path.dirname(os.path.abspath(__file__))
figure_fname = os.path.join(
    script_path,
    "../../manuscript/figures/density-based-discretization-algorithm.pdf"
)
plt.savefig(figure_fname)
plt.show()
