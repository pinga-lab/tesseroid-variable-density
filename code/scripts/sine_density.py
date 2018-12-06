from __future__ import division, print_function
import os
import numpy as np
from fatiando.constants import G, MEAN_EARTH_RADIUS, SI2MGAL, SI2EOTVOS
from fatiando.mesher import TesseroidMesh
from fatiando import gridder
import matplotlib.pyplot as plt
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


# Plot Densities
# --------------
bottom, top = 0, 1
thickness = top - bottom
colors = dict(zip(b_factors, plt.cm.viridis(np.linspace(0, 0.9, len(b_factors)))))
for b_factor in b_factors:

    # Compute the k factor, i.e. the frequency for the sine function
    k_factor = 2 * np.pi * b_factor / thickness

    # Define density function
    def density_sine(height):
        return max_density * np.sin(k_factor * height) + max_density

    heights = np.linspace(bottom, top, 101)
    plt.plot(heights, density_sine(heights), color=colors[b_factor],
             label="b={}".format(b_factor))
plt.ylabel(r"Density [kg/m$^3$]")
plt.xticks([bottom, top], ["Inner Radius", "Outer Radius"])
plt.legend()
plt.show()


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
                return max_density * np.sin(k_factor * height) + max_density

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
                                                max_density, k_factor, max_density)
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


# Plot Results
# ------------
field_titles = dict(zip(fields, '$V$ $g_z$'.split()))
grid_titles = {"pole": "Pole",
               "equator": "Equator",
               "global": "Global",
               "260km": "Satellite"}
colors = dict(zip(b_factors, plt.cm.viridis(np.linspace(0, 0.9, len(b_factors)))))

for grid_name in grids:

    fig, axes = plt.subplots(nrows=len(fields), ncols=1, sharex=True)
    fig.set_size_inches((5, 5))
    fig.subplots_adjust(hspace=0)
    grid_title = grid_titles[grid_name]

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
            ax.plot(delta_values, differences_per_b, "-o", color=color)

        # Add threshold line
        ax.plot([1e-3, 1e1], [1e-1, 1e-1], '--', color='k', linewidth=0.5)

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
        ax.set_yticks(ax.get_yticks()[2:-2])
        ax.set_ylabel('Difference (%)')
        ax.grid(True, linewidth=0.5, color='#aeaeae')
        ax.set_axisbelow(True)
    ax = axes[-1]
    ax.set_xlabel(r"$\delta$")
    ax.set_xlim(1e-3, 1e1)
    # ax.set_xticks(np.arange(0, 6, 1))
    ax.legend()
    axes[0].set_title(grid_title)
    plt.show()
