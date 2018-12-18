from __future__ import division, print_function
import os
import warnings
import numpy as np
from fatiando.constants import G, MEAN_EARTH_RADIUS, SI2MGAL, SI2EOTVOS
from fatiando.mesher import TesseroidMesh
from fatiando import gridder
import matplotlib
import matplotlib.pyplot as plt
# This is our custom tesseroid code
from tesseroid_density import tesseroid


def shell_exponential_density(height, top, bottom, amplitude, b_factor, constant_term):
    """
    Analytical solution for a spherical shell with inner and outer radii `bottom
    + MEAN_EARTH_RADIUS` and `top + MEAN_EARTH_RADIUS` at a computation point located at
    the radial coordinate `height + MEAN_EARTH_RADIUS` with an exponential density as:

    .. math :
        \rho(r') = A e^{- b * (r' - R_1) / T} + C

    Where $r'$ is the radial coordinate where the density is going to be evaluated,
    $A$ is the amplitude, $C$ the constant term, $T$ is the thickness of the shell,
    $b$ the b factor and $R_1$ the inner radius.
    """
    r = height + MEAN_EARTH_RADIUS
    r1 = bottom + MEAN_EARTH_RADIUS
    r2 = top + MEAN_EARTH_RADIUS
    thickness = float(top - bottom)
    k = b_factor / thickness
    potential = 4 * np.pi * G * amplitude / k**3 / r * \
        (
         ((r1 * k)**2 + 2*r1*k + 2) -
         ((r2 * k)**2 + 2*r2*k + 2) * np.exp(-k * thickness)
        ) + \
        4 / 3. * np.pi * G * constant_term * (r2**3 - r1**3) / r
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
result_dir = os.path.join(script_path, 'results/grid-search')
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
grid_name = "global"
grid = gridder.regular((-90, 90, 0, 360), (19, 13), z=0)


# Configure comparisons
# ---------------------
fields = 'potential gz'.split()
density_bottom, density_top = 3300, 2670
b_factor = 30
delta_values = np.logspace(-3, 1, 5)
D_values = np.arange(0.5, 4.5, 0.5)


# Compute differences
# -------------------
for field in fields:
    for model in models:
        top, bottom = model.bounds[4], model.bounds[5]
        thickness = top - bottom

        # Define density function
        A = (density_bottom - density_top) / (1 - np.exp(-b_factor))
        C = density_bottom - A

        # Define density function
        def density_exponential(height):
            return A * np.exp(-b_factor * (height - bottom) / thickness) + C

        # Append density function to every tesseroid of the model
        model.addprop(
            "density",
            [density_exponential for i in range(model.size)]
        )

        # Check if result file exists
        fname = "{}-{}-{}-{}.npz".format(field, grid_name, int(thickness),
                                         int(b_factor))
        if os.path.isfile(os.path.join(result_dir, fname)):
            continue
        print("Thickness: {} Field: {} Grid: {} b: {}".format(
            int(thickness), field, grid_name, b_factor)
            )

        # Compute differences
        lats, lons, heights = grid
        analytical = shell_exponential_density(heights[0], top, bottom,
                                               A, b_factor, C)
        differences = []
        for delta in delta_values:
            for D_ratio in D_values:
                result = getattr(tesseroid, field)(lons, lats, heights, model,
                                                   delta=delta, ratio=D_ratio)
                diff = np.abs((result - analytical[field]) / analytical[field])
                diff = 100 * np.max(diff)
                differences.append(diff)
        D_grid, delta_grid = np.meshgrid(D_values, delta_values)
        differences = np.array(differences)
        differences = differences.reshape(D_grid.shape)
        np.savez(os.path.join(result_dir, fname),
                 delta_grid=delta_grid, D_grid=D_grid,
                 differences=differences)


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
figure_fname = os.path.join(script_path, "../../manuscript/figures/grid-search.pdf")
field_titles = dict(zip(fields, '$V$ $g_z$'.split()))
D_linear = dict(zip(fields, (1, 2.5)))
fig, axes = plt.subplots(2, 1, figsize=(3.33, 3.8))
for field, ax in zip(fields, axes):
    total_differences = []

    for model in models:
        top, bottom = model.bounds[4], model.bounds[5]
        thickness = top - bottom
        fname = "{}-{}-{}-{}.npz".format(field, grid_name, int(thickness),
                                         int(b_factor))
        data = np.load(os.path.join(result_dir, fname))
        D_grid, delta_grid = data["D_grid"], data["delta_grid"]
        differences = data["differences"]
        total_differences.append(differences)

    total_differences = np.array(total_differences)
    total_differences = np.max(total_differences, axis=0)

    # Plot bigger points if error <= 1e-1
    sizes = 50
    plot_size = False
    if plot_size:
        sizes *= np.ones_like(total_differences)
        sizes[total_differences <= 1e-1] *= 3
    cm = ax.scatter(D_grid.ravel(), delta_grid.ravel(),
                    c=total_differences.ravel(), s=sizes,
                    norm=matplotlib.colors.LogNorm())

    # Add colorbar
    plt.colorbar(cm, label=r"Differences (\%)", ax=ax)

    # Bellow error points contour
    D_step = (D_values.max() - D_values.min()) / D_values.size
    delta_factor = (delta_values.max()/delta_values.min())**(1/(delta_values.size - 1))
    min_D = np.min(D_grid[total_differences <= 0.1]) - D_step/2
    max_D = D_grid.max() + D_step/8
    min_delta = delta_grid.min()*(delta_factor**(-0.125))
    max_delta = np.max(delta_grid[total_differences <= 0.1])*(delta_factor**0.5)
    ax.plot([min_D, min_D, max_D], [min_delta, max_delta, max_delta], '--', color='C7')

    # Add field title
    ax.text(-0.22, 0.88, field_titles[field], fontsize=11,
            horizontalalignment='center',
            verticalalignment='center',
            bbox={'facecolor': 'w',
                  'edgecolor': '#9b9b9b',
                  'linewidth': 0.5, 'pad': 5,
                  'boxstyle': 'circle, pad=0.4'},
            transform=ax.transAxes)

    # Configure axes
    ax.set_yscale('log')
    ax.set_ylim(3.162e-4, 3.162e1)
    if field == "potential":
        ax.set_xticks(np.arange(1, 5, 1))
    elif field == "gz":
        ax.set_xticks(np.arange(0.5, 4.5, 1))
        ax.set_xlabel(r"$D$")
    ax.set_ylabel(r"$\delta$")

    # Insert D_linear 1abel in axis ticks
    fig.canvas.draw()
    labels = [item.get_text() for item in ax.get_xticklabels()]
    for i in range(len(labels)):
        label = labels[i]
        label = label.replace("$", "")
        if float(label) == D_linear[field]:
            labels[i] = r"$D_\mathrm{linear}$"
    ax.set_xticklabels(labels)

plt.tight_layout(pad=0.5, h_pad=0, w_pad=0)
plt.savefig(figure_fname, dpi=300)
plt.show()
