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
result_dir = 'results/linear-D'
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
grids = {"pole": gridder.regular((89, 90, 0, 1), (10, 10), z=0),
         "equator": gridder.regular((0, 1, 0, 1), (10, 10), z=0),
         "global": gridder.regular((-90, 90, 0, 360), (19, 13), z=0),
         "260km": gridder.regular((-90, 90, 0, 360), (19, 13), z=260e3),
         }


# Compute differences
# -------------------
fields = 'potential gz'.split()
compute = True
if compute:
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
                fname = "{}-{}-{}-{}".format(field, grid_name, int(top - bottom),
                                             model.size)
                np.savez(os.path.join(result_dir, fname),
                         D_values=D_values, differences=differences)


# Plot Results
# ------------
styles = ['-o', '-^', '-v', '-s']
colors = ["C" + str(i) for i in range(4)]
labels = [r'Pole', r'Equator', r'Global', r'260 km']
titles = '$V$ $g_z$'.split()

for model in models:
    print(model.bounds[4] - model.bounds[5], model.size)
    fig, axes = plt.subplots(nrows=len(fields), ncols=1, sharex=True)
    fig.set_size_inches((5, 5))
    fig.subplots_adjust(hspace=0)
    for ax, field, title in zip(axes, fields, titles):
        for grid_name in grids:
            thickness = model.bounds[4] - model.bounds[5]
            fname = "{}-{}-{}-{}.npz".format(field, grid_name, int(thickness),
                                             model.size)
            diff_file = np.load(os.path.join(result_dir, fname))
            D_values, differences = diff_file["D_values"], diff_file["differences"]
            ax.plot(D_values, differences)
            ax.plot([0, 10], [1e-1, 1e-1], '--', color='k', linewidth=0.5)
            ax.set_yscale('log')
            ax.set_yticks(ax.get_yticks()[2:-2])
            ax.set_ylabel('Difference (%)')
            ax.grid(True, linewidth=0.5, color='#aeaeae')
            ax.set_axisbelow(True)
        ax.set_title(title)
    ax = axes[-1]
    ax.set_xlabel(r"D")
    ax.set_xlim(0, 5.5)
    ax.set_xticks(np.arange(0, 6, 1))
    ax.legend()
    plt.show()
