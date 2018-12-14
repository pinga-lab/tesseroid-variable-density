from __future__ import division, print_function
import os
import warnings
import numpy as np
from fatiando.constants import G, MEAN_EARTH_RADIUS, SI2MGAL, SI2EOTVOS
from fatiando.mesher import TesseroidMesh
from fatiando import gridder
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
# This is our custom tesseroid code
from tesseroid_density import tesseroid


def shell_sine_density(height, top, bottom, A, k, constant_density):
    """
    Analytical solution for a spherical shell with density:

    rho(r') = A sin(k (r' - R)) + C

    Where R is the MEAN_EARTH_RADIUS
    """
    r = height + MEAN_EARTH_RADIUS
    r1 = bottom + MEAN_EARTH_RADIUS
    r2 = top + MEAN_EARTH_RADIUS
    potential = 4 * np.pi * G * A / r / k**3 * (
        (2 - k**2 * r2**2) * np.cos(k * (r2 - MEAN_EARTH_RADIUS)) +
        2 * k * r2 * np.sin(k * (r2 - MEAN_EARTH_RADIUS)) -
        (2 - k**2 * r1**2) * np.cos(k * (r1 - MEAN_EARTH_RADIUS)) -
        2 * k * r1 * np.sin(k * (r1 - MEAN_EARTH_RADIUS))
        ) + \
        4 / 3. * np.pi * G * constant_density * (r2**3 - r1**3) / r
    data = {'potential': potential,
            'gx': 0,
            'gy': 0,
            'gz': SI2MGAL*(potential/r),
            'gxx': SI2EOTVOS*(-potential/r**2),
            'gxy': 0,
            'gxz': 0,
            'gyy': SI2EOTVOS*(-potential/r**2),
            'gyz': 0,
            'gzz': SI2EOTVOS*(2*potential/r**2)}
    return data


# Create results dir if it does not exist
# ---------------------------------------
script_path = os.path.dirname(os.path.abspath(__file__))
result_dir = os.path.join(script_path, 'results/sine')
if not os.path.isdir(result_dir):
    os.makedirs(result_dir)


# Define Tesseroids models
# ------------------------
thicknesses = [100, 1e3, 1e4, 1e5, 1e6]
shape = (1, 6, 12)
models = [
    TesseroidMesh((0, 360, -90, 90, 0, -thickness), shape)
    for thickness in thicknesses
]


# Define computation grids
# ------------------------
grids = {"pole": gridder.regular((89, 90, 0, 1), (11, 11), z=0),
         "equator": gridder.regular((0, 1, 0, 1), (11, 11), z=0),
         "global": gridder.regular((-90, 90, 0, 360), (19, 13), z=0),
         "260km": gridder.regular((-90, 90, 0, 360), (19, 13), z=260e3),
         }


# Configure comparisons
# ---------------------
fields = 'potential gz'.split()
max_density = 3300.
b_factors = [1, 2, 5, 10]
delta_values = np.logspace(-4, 0, 5)


# Compute differences
# -------------------
for field in fields:
    for model in models:
        top, bottom = model.bounds[4], model.bounds[5]
        thickness = top - bottom

        for b_factor in b_factors:

            # Compute the k factor, i.e. the frequency for the sine function
            k_factor = 2 * np.pi * b_factor / thickness

            # Define density function
            def density_sine(height):
                return max_density / 2 * np.sin(k_factor * height) + max_density / 2

            # Append density function to every tesseroid of the model
            model.addprop(
                "density",
                [density_sine for i in range(model.size)]
            )

            for grid_name, grid in grids.items():
                fname = "{}-{}-{}-{}.npz".format(field, grid_name, int(thickness),
                                                 int(b_factor))
                if os.path.isfile(os.path.join(result_dir, fname)):
                    continue
                print("Thickness: {} Field: {} Grid: {} b: {}".format(
                    int(thickness), field, grid_name, b_factor)
                    )
                lats, lons, heights = grid
                analytical = shell_sine_density(heights[0], top, bottom,
                                                max_density/2, k_factor, max_density/2)
                differences = []
                for delta in delta_values:
                    result = getattr(tesseroid, field)(lons, lats, heights, model,
                                                       delta=delta)
                    diff = np.abs((result - analytical[field]) / analytical[field])
                    diff = 100 * np.max(diff)
                    differences.append(diff)
                differences = np.array(differences)
                np.savez(os.path.join(result_dir, fname),
                         delta_values=delta_values, differences=differences)


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


# Plot Densities
# --------------
bottom, top = 0, 1
thickness = top - bottom
colors = dict(zip(b_factors, plt.cm.viridis(np.linspace(0, 0.9, len(b_factors)))))

figure_fname = os.path.join(script_path,
                            "../../manuscript/figures/sine-densities.pdf")
fig, axes = plt.subplots(len(b_factors), 1, figsize=(3.33, 4), sharex=True)
for b_factor, ax in zip(b_factors, axes):

    # Compute the k factor, i.e. the frequency for the sine function
    k_factor = 2 * np.pi * b_factor / thickness

    # Define density function
    def density_sine(height):
        return max_density / 2 * np.sin(k_factor * height) + max_density / 2

    heights = np.linspace(bottom, top, 1001)
    ax.plot(heights, density_sine(heights), color=colors[b_factor],
            label="b={}".format(b_factor))
    ax.set_xticks([bottom, top])
    ax.set_xticklabels(["Inner Radius", "Outer Radius"])
    # ax.legend(loc=1)

    # Add field annotation on each axe
    ax.text(0.85, 0.8, "b={}".format(b_factor),
            fontsize=10,
            horizontalalignment='center',
            verticalalignment='center',
            bbox={'facecolor': 'w',
                  'edgecolor': '#9b9b9b',
                  'alpha': 0.7,
                  'linewidth': 0.5, 'pad': 5,
                  'boxstyle': 'square, pad=0.4'},
            transform=ax.transAxes)
fig.text(0, 0.5, r"Density [kg/m$^3$]", va='center', rotation='vertical')
plt.tight_layout(pad=1.8)
fig.subplots_adjust(hspace=0)
plt.savefig(figure_fname, dpi=300)
plt.show()


# Plot Results
# ------------
figure_fname = os.path.join(script_path,
                            "../../manuscript/figures/sine-density-diffs.pdf")
field_titles = dict(zip(fields, '$V$ $g_z$'.split()))
grid_titles = {"pole": "Pole",
               "equator": "Equator",
               "global": "Global",
               "260km": "Satellite"}
colors = dict(zip(b_factors, plt.cm.viridis(np.linspace(0, 0.9, len(b_factors)))))

# Create outer grid
fig = plt.figure(figsize=(6.66, 7))
outer_grid = GridSpec(ncols=2, nrows=2, wspace=0.001, hspace=0.1)

# Create grid specs for each grid
grid_specs = dict(
                  zip(
                      grids.keys(),
                      [GridSpecFromSubplotSpec(ncols=1, nrows=2,
                                               subplot_spec=outer_grid[j, i],
                                               hspace=0)
                       for j in range(2)
                       for i in range(2)]
                     )
                 )
positions = dict(zip(grids.keys(), [[j, i] for j in range(2) for i in range(2)]))

# Plot for each grid
for grid_name in grids:

    # Choose gridspec and create axes
    gs = grid_specs[grid_name]
    position = positions[grid_name]
    axes = [plt.subplot(gs[0, 0])]
    axes.append(plt.subplot(gs[1, 0], sharex=axes[0]))
    plt.setp(axes[0].get_xticklabels(), visible=False)
    # Remove xticks if the axes in in the middle of the figure
    if position[0] == 0:
        plt.setp(axes[1].get_xticklabels(), visible=False)
    else:
        axes[1].set_xlabel(r"$\delta$")
    # Move ticks and labels to the right if the axes are on the second column
    if position[1] == 1:
        for ax in axes:
            ax.yaxis.tick_right()
            ax.yaxis.set_label_position("right")
    # Add title to axes
    grid_title = grid_titles[grid_name]
    axes[0].set_title(grid_titles[grid_name])

    # Plot
    for ax, field in zip(axes, fields):
        field_title = field_titles[field]
        for b_factor in b_factors:
            differences_per_b = []
            color = colors[b_factor]
            for model in models:
                thickness = model.bounds[4] - model.bounds[5]
                fname = "{}-{}-{}-{}.npz".format(field, grid_name, int(thickness),
                                                 int(b_factor))
                diff_file = np.load(os.path.join(result_dir, fname))
                delta_values = diff_file["delta_values"]
                differences = diff_file["differences"]
                differences_per_b.append(differences)
            differences_per_b = np.array(differences_per_b)
            differences_per_b = np.max(differences_per_b, axis=0)
            label = "b={}".format(b_factor)
            ax.plot(delta_values, differences_per_b, "-o", color=color, label=label)

        # Add threshold line
        ax.plot([1e-5, 1e1], [1e-1, 1e-1], '--', color='k', linewidth=0.5)

        # Add field annotation on each axe
        ax.text(0.5, 0.87, field_title, fontsize=11,
                horizontalalignment='center',
                verticalalignment='center',
                bbox={'facecolor': 'w',
                      'edgecolor': '#9b9b9b',
                      'linewidth': 0.5, 'pad': 5,
                      'boxstyle': 'circle, pad=0.4'},
                transform=ax.transAxes)

        # Configure axes
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim(6.31e-5, 1.58e0)
        ax.set_yticks(ax.get_yticks()[2:-2])
        ax.set_ylabel(r'Difference (\%)')
        ax.grid(True, linewidth=0.5, color='#aeaeae')

    # Add legend
    if position == [0, 0]:
        axes[0].legend(loc=0, prop={"size": 8})

outer_grid.tight_layout(fig)
plt.savefig(figure_fname, dpi=300)
plt.show()
