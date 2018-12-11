from __future__ import division, print_function
import os
import numpy as np
from fatiando.constants import G, MEAN_EARTH_RADIUS, SI2MGAL, SI2EOTVOS
from fatiando.mesher import TesseroidMesh
from fatiando import gridder
import matplotlib.pyplot as plt
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
shapes = [(1, 1, 2), (1, 6, 12), (1, 12, 24), (1, 18, 36)]
models = [
    TesseroidMesh((0, 360, -90, 90, 0, -thickness), shape)
    for thickness in thicknesses
    for shape in shapes
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
        top, bottom = model.bounds[4], model.bounds[5]
        slope = -(3300 - 2670) / (top - bottom)
        constant_term = (3300 - 2670) * MEAN_EARTH_RADIUS + 2670

        # Define density function
        def density_linear(height):
            r = height + MEAN_EARTH_RADIUS
            return slope*r + constant_term

        model.addprop("density", [density_linear for i in range(model.size)])

        for grid_name, grid in grids.items():
            fname = "{}-{}-{}-{}.npz".format(field, grid_name, int(top - bottom),
                                             model.size)
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
            np.savez(os.path.join(result_dir, fname),
                     D_values=D_values, differences=differences)


# Plot Results
# ------------
# Plot one line per thickness using the maximum difference value for each case
field_titles = dict(zip(fields, '$V$ $g_z$'.split()))
grid_titles = {"pole": "Pole",
               "equator": "Equator",
               "global": "Global",
               "260km": "Satellite"}
colors = dict(zip(thicknesses, plt.cm.viridis(np.linspace(0, 0.9, len(thicknesses)))))

for grid_name in grids:

    fig, axes = plt.subplots(nrows=len(fields), ncols=1, sharex=True)
    fig.set_size_inches((5, 5))
    fig.subplots_adjust(hspace=0)
    grid_title = grid_titles[grid_name]

    for ax, field in zip(axes, fields):
        field_title = field_titles[field]
        for thickness in thicknesses:
            differences_per_thickness = []
            color = colors[thickness]
            for model in models:
                if model.bounds[4] - model.bounds[5] == thickness:
                    fname = "{}-{}-{}-{}.npz".format(field, grid_name, int(thickness),
                                                     model.size)
                    diff_file = np.load(os.path.join(result_dir, fname))
                    D_values = diff_file["D_values"]
                    differences = diff_file["differences"]
                    differences_per_thickness.append(differences)
            differences_per_thickness = np.array(differences_per_thickness)
            differences_per_thickness = np.max(differences_per_thickness, axis=0)
            if thickness < 1e3:
                label = "{:.0f} m".format(thickness)
            else:
                label = "{:.0f} km".format(thickness*1e-3)
            ax.plot(D_values, differences_per_thickness, '-o', color=color, label=label)

        # Add threshold line
        ax.plot([0, 10], [1e-1, 1e-1], '--', color='k', linewidth=0.5)

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
        ax.set_ylabel('Difference (%)')
        ax.grid(True, linewidth=0.5, color='#aeaeae')
        ax.set_axisbelow(True)
    ax = axes[-1]
    ax.set_xlabel(r"D")
    ax.set_xlim(0, 5.5)
    ax.set_xticks(np.arange(0, 6, 1))
    ax.legend()
    axes[0].set_title(grid_title)


# Plot single plot: one line per grid
# -----------------------------------
figure_fname = os.path.join(script_path,
                            "../../manuscript/figures/linear-density-diffs.pdf")
field_titles = dict(zip(fields, '$V$ $g_z$'.split()))
grid_titles = {"pole": "Pole",
               "equator": "Equator",
               "global": "Global",
               "260km": "Satellite"}
colors = dict(zip(grids.keys(), plt.cm.viridis(np.linspace(0, 0.9, len(grids.keys())))))
markers = dict(zip(grids.keys(), ["o", "^", "s", "d"]))

fig, axes = plt.subplots(nrows=len(fields), ncols=1, sharex=True)
fig.set_size_inches((5, 5))
fig.subplots_adjust(hspace=0)
grid_title = grid_titles[grid_name]

for ax, field in zip(axes, fields):
    field_title = field_titles[field]

    for grid_name in grids:
        color = colors[grid_name]
        differences_per_grid = []
        for model in models:
            fname = "{}-{}-{}-{}.npz".format(field, grid_name, int(thickness),
                                             model.size)
            diff_file = np.load(os.path.join(result_dir, fname))
            D_values = diff_file["D_values"]
            differences = diff_file["differences"]
            differences_per_grid.append(differences)
        differences_per_grid = np.array(differences_per_grid)
        differences_per_grid = np.max(differences_per_grid, axis=0)
        ax.plot(D_values, differences_per_grid, '-o', color=color,
                label=grid_titles[grid_name], marker=markers[grid_name])

    # Add threshold line
    ax.plot([0, 10], [1e-1, 1e-1], '--', color='k', linewidth=0.5)

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
    ax.set_ylabel('Difference (%)')
    ax.grid(True, linewidth=0.5, color='#aeaeae')
    ax.set_axisbelow(True)
ax = axes[-1]
ax.set_xlabel(r"D")
ax.set_xlim(0, 5.5)
ax.set_xticks(np.arange(0, 6, 1))
ax.legend()
plt.tight_layout()
plt.savefig(figure_fname, dpi=300)

plt.show()
