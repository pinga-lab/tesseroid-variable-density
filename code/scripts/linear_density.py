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


def shell_linear_density(height, top, bottom, slope, constant_term):
    r = height + MEAN_EARTH_RADIUS
    r1 = bottom + MEAN_EARTH_RADIUS
    r2 = top + MEAN_EARTH_RADIUS
    constant = np.pi * G * slope * (r2**4 - r1**4) + \
        4/3. * np.pi * G * constant_term * (r2**3 - r1**3)
    potential = constant/r
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
result_dir = os.path.join(script_path, 'results/linear')
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


# Compute differences
# -------------------
fields = 'potential gz'.split()
D_values = np.arange(0.5, 5.5, 0.5)
for field in fields:
    for model in models:
        # Define density function
        top, bottom = model.bounds[4], model.bounds[5]
        thickness = top - bottom
        slope = (2670 - 3300) / (top - bottom)
        constant_term = 2670 - slope * (top + MEAN_EARTH_RADIUS)

        def density_linear(height):
            r = height + MEAN_EARTH_RADIUS
            return slope * r + constant_term

        # Append density function to every tesseroid of the model
        model.addprop("density", [density_linear for i in range(model.size)])

        # Compute differences
        for grid_name, grid in grids.items():
            fname = "{}-{}-{}.npz".format(field, grid_name, int(thickness))
            if os.path.isfile(os.path.join(result_dir, fname)):
                continue
            print("Thickness: {} Model size: {} Field: {} Grid: {}".format(
                int(top - bottom), model.size, field, grid_name)
                )
            lats, lons, heights = grid
            analytical = shell_linear_density(heights[0], top, bottom,
                                              slope, constant_term)
            differences = []
            for D in D_values:
                result = getattr(tesseroid, field)(lons, lats, heights, model,
                                                   ratio=D, delta=None)
                diff = np.abs((result - analytical[field]) / analytical[field])
                diff = 100 * np.max(diff)
                differences.append(diff)
            differences = np.array(differences)

            # Save results to file
            np.savez(os.path.join(result_dir, fname),
                     D_values=D_values, differences=differences)


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


# Plot Results
# ------------
# Configure plot
figure_fname = os.path.join(script_path,
                            "../../manuscript/figures/linear-density-diffs.pdf")
field_titles = dict(zip(fields, '$V$ $g_z$'.split()))
grid_titles = {"pole": "Polo",
               "equator": "Ecuador",
               "global": "Global",
               "260km": "Satelite"}
grid_names = ["pole", "equator", "global", "260km"]
colors = dict(zip(thicknesses, plt.cm.viridis(np.linspace(0, 0.9, len(thicknesses)))))
markers = dict(zip(thicknesses, ["o", "^", "s", "d", "x"]))

# Create outer grid
fig = plt.figure(figsize=(6.66, 7))
outer_grid = GridSpec(ncols=2, nrows=2, wspace=0.001, hspace=0.1)

# Create grid specs for each grid
grid_specs = dict(
                  zip(
                      grid_names,
                      [GridSpecFromSubplotSpec(ncols=1, nrows=2,
                                               subplot_spec=outer_grid[j, i],
                                               hspace=0)
                       for j in range(2)
                       for i in range(2)]
                     )
                 )
positions = dict(zip(grid_names, [[j, i] for j in range(2) for i in range(2)]))

# Plot for each grid
for grid_name in grid_names:

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
        axes[1].set_xlabel(r"$D$")
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
        for model in models:
            thickness = model.bounds[4] - model.bounds[5]
            color = colors[thickness]
            marker = markers[thickness]
            fname = "{}-{}-{}.npz".format(field, grid_name, int(thickness))
            diff_file = np.load(os.path.join(result_dir, fname))
            D_values = diff_file["D_values"]
            differences = diff_file["differences"]
            if thickness < 1e3:
                label = "{:.0f} m".format(thickness)
            else:
                label = "{:.0f} km".format(thickness*1e-3)
            ax.plot(D_values, differences, marker=marker, color=color,
                    label=label)

        # Add threshold line
        ax.plot([0, 6], [1e-1, 1e-1], '--', color='k', linewidth=0.5)

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
        ax.set_yscale('log')
        ax.set_yticks(ax.get_yticks()[2:-2])
        ax.set_ylabel(r'Diferencia (\%)')
        ax.grid(True, linewidth=0.5, color='#aeaeae')
        ax.set_xticks(np.arange(0, 6, 1))
        ax.set_xlim(0.3, 5.2)

    # Add legend
    if position == [0, 1]:
        axes[0].legend(loc=1, prop={"size": 8})

outer_grid.tight_layout(fig)
plt.savefig(figure_fname, dpi=300)
plt.show()
