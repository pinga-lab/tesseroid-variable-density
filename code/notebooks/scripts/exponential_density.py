from __future__ import division, print_function
import os
import numpy as np
from fatiando.constants import G, MEAN_EARTH_RADIUS, SI2MGAL, SI2EOTVOS
from fatiando.mesher import TesseroidMesh
from fatiando import gridder
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
# This is our custom tesseroid code
from tesseroid_density import tesseroid


def shell_exponential_density(height, top, bottom, amplitude, b_factor, constant_term):
    """
    Analytical solution for a spherical shell with inner and outer radii `bottom
    + MEAN_EARTH_RADIUS` and `top + MEAN_EARTH_RADIUS` at a computation point located at
    the radial coordinate `height + MEAN_EARTH_RADIUS` with an exponential density as:

    .. math :
        \rho(r') = A e^{-(r' - R) / b} + C

    Where $r$ is the radial coordinate where the density is going to be evaluated,
    $A$ is the amplitude, $C$ the constant term, $b$ the b factor and $R$ the mean Earth
    radius.
    """
    r = height + MEAN_EARTH_RADIUS
    r1 = bottom + MEAN_EARTH_RADIUS
    r2 = top + MEAN_EARTH_RADIUS
    potential = 4 * np.pi * G * amplitude * b_factor / r * \
        (
         (r1**2 + 2 * r1 * b_factor + 2 * b_factor**2) *
         np.exp(-(r1 - MEAN_EARTH_RADIUS) / b_factor) -
         (r2**2 + 2 * r2 * b_factor + 2 * b_factor**2) *
         np.exp(-(r2 - MEAN_EARTH_RADIUS) / b_factor)
        ) + \
        4/3 * np.pi * G * constant_term * (r2**3 - r1**3) / r
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
result_dir = 'results/exponential'
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
grids = {"pole": gridder.regular((89, 90, 0, 1), (10, 10), z=0),
         "equator": gridder.regular((0, 1, 0, 1), (10, 10), z=0),
         "global": gridder.regular((-90, 90, 0, 360), (19, 13), z=0),
         "260km": gridder.regular((-90, 90, 0, 360), (19, 13), z=260e3),
         }


# Compute differences
# -------------------
fields = 'potential gz'.split()
density_in, density_out = 3300, 2670
b_ratios = [1, 10, 100]

compute = True
if compute:
    delta_values = np.logspace(-3, -1, 5)
    for field in fields:
        for model in models:
            top, bottom = model.bounds[4], model.bounds[5]
            thickness = top - bottom

            for b_ratio in b_ratios:
                b_factor = b_ratio * thickness
                amplitude = (density_in - density_out) / \
                    (np.exp(thickness / b_factor) - 1)
                constant_term = density_out - amplitude

                # Define density function
                def density_exponential(height):
                    return amplitude*np.exp(-height / b_factor) + constant_term

                # Append density function to every tesseroid of the model
                model.addprop(
                    "density",
                    [density_exponential for i in range(model.size)]
                )

                for grid_name, grid in grids.items():
                    print("Thickness: {} Field: {} Grid: {} b: {}".format(
                        int(thickness), field, grid_name, b_factor)
                        )
                    lats, lons, heights = grid
                    analytical = shell_exponential_density(heights[0], top, bottom,
                                                           amplitude, b_factor,
                                                           constant_term)
                    differences = []
                    for delta in delta_values:
                        result = getattr(tesseroid, field)(lons, lats, heights, model,
                                                           delta=delta)
                        diff = np.abs((result - analytical[field]) / analytical[field])
                        diff = 100 * np.max(diff)
                        differences.append(diff)
                    differences = np.array(differences)
                    fname = "{}-{}-{}-{}".format(field, grid_name, int(thickness),
                                                 int(b_factor))
                    np.savez(os.path.join(result_dir, fname),
                             delta_values=delta_values, differences=differences)
