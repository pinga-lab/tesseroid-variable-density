from __future__ import division, print_function
import os
import time
import warnings
import numpy as np
import pandas as pd
from fatiando.mesher import Tesseroid
import matplotlib.pyplot as plt
# This is our custom tesseroid code
from tesseroid_density import tesseroid


# Configure comparison
# --------------------
nruns = 10000
fields = "potential gz".split()


# Define Tesseroid
# ----------------
w, e, s, n = -10, 10, -10, 10
top, bottom = 0, -1e3
model = [Tesseroid(w, e, s, n, top, bottom)]


# Define computation points
# -------------------------
heights = np.array([0., 1e3, 1e4, 1e5, 1e6])
computation_points = [[np.array([0.]), np.array([0.]), np.array([height])]
                      for height in heights]


# Create results dir if it does not exist
# -------------------------
script_path = os.path.dirname(os.path.abspath(__file__))
result_dir = os.path.join(script_path, 'results/computation_time')
if not os.path.isdir(result_dir):
    os.makedirs(result_dir)


# Check if DataFrame exists
# -------------------------
csv_fname = os.path.join(result_dir, "computation_time.csv")
compute = True
if os.path.isfile(csv_fname):
    compute = False


# Compute only if DataFrame does not exists
# -----------------------------------------
if compute:

    # Create DataFrame
    # ----------------
    columns = ["{}_{}".format(density, field)
               for density in ["constant", "linear", "exp"]
               for field in fields]
    df = pd.DataFrame(data=np.zeros((heights.size, len(columns))),
                      index=heights,
                      columns=columns)

    # Compute gravitational fields for homogeneous tesseroid
    # ------------------------------------------------------
    for tess in model:
        tess.addprop('density', 3000)

    density = "constant"
    for field in fields:
        for point in computation_points:
            times = np.zeros(nruns)
            lon, lat, height = point[:]
            for run in range(nruns):
                start_time = time.time()
                getattr(tesseroid, field)(lon, lat, height, model)
                end_time = time.time()
                times[run] = start_time - end_time
            times = np.mean(times)
            df.at[height, "{}_{}".format(density, field)] = times

    # Compute gravitational fields for linear density tesseroid
    # ---------------------------------------------------------
    density_top, density_bottom = 2670, 3300
    slope = (density_top - density_bottom) / (top - bottom)
    constant_term = density_top - slope * top

    # Define density function
    def density_linear(height):
        return slope * height + constant_term

    for tess in model:
        tess.addprop('density', density_linear)

    density = "linear"
    for field in fields:
        for point in computation_points:
            times = np.zeros(nruns)
            lon, lat, height = point[:]
            for run in range(nruns):
                start_time = time.time()
                getattr(tesseroid, field)(lon, lat, height, model)
                end_time = time.time()
                times[run] = start_time - end_time
            times = np.mean(times)
            df.at[height, "{}_{}".format(density, field)] = times

    # Compute gravitational fields for exp density tesseroid
    # ------------------------------------------------------
    b_factor = 3
    density_top, density_bottom = 2670, 3300
    thickness = top - bottom

    denominator = np.exp(- bottom * b_factor / thickness) - \
        np.exp(- top * b_factor / thickness)
    amplitude = (density_bottom - density_top) / denominator
    constant_term = (
        density_top * np.exp(-bottom * b_factor / thickness) -
        density_bottom * np.exp(-top * b_factor / thickness)
        ) / denominator

    # Define density function
    def density_exponential(height):
        return amplitude * np.exp(-height * b_factor / thickness) + constant_term

    for tess in model:
        tess.addprop('density', density_exponential)

    density = "exp"
    for field in fields:
        for point in computation_points:
            times = np.zeros(nruns)
            lon, lat, height = point[:]
            for run in range(nruns):
                start_time = time.time()
                getattr(tesseroid, field)(lon, lat, height, model)
                end_time = time.time()
                times[run] = start_time - end_time
            times = np.mean(times)
            df.at[height, "{}_{}".format(density, field)] = times

    # Save DataFrame
    # --------------
    df.to_csv(csv_fname)


# Plot relative computation times
# -------------------------------
# Load computation time differences
df = pd.read_csv(csv_fname, index_col=0)

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
try:
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(3.33, 4), sharex=True)
except Exception:
    plt.switch_backend('agg')
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(3.33, 4), sharex=True)

# Plot results
ax = axes[0]
ax.plot(df["linear_potential"] / df["constant_potential"], '-o', label=r"$V$")
ax.plot(df["linear_gz"] / df["constant_gz"], '-o', label=r"$g_z$")
ax = axes[1]
ax.plot(df["exp_gz"] / df["constant_gz"], '-o', label=r"$V$")
ax.plot(df["exp_potential"] / df["constant_potential"], '-o', label=r"$g_z$")

# Configure axes
densities = 'Linear Exponential'.split()
ax.set_xlabel("Computation height [m]")
fig.text(0, 0.5, "Computation time ratio", va='center', rotation='vertical')
fig.subplots_adjust(hspace=0.1)
axes[0].legend()
for density, ax in zip(densities, axes):
    ax.grid()
    ax.set_xscale("log")

    # Add field annotation on each axe
    if density == "Linear":
        yloc = 0.8
    elif density == "Exponential":
        yloc = 0.89
    ax.text(0.5, yloc, density, fontsize=11,
            horizontalalignment='center',
            verticalalignment='center',
            bbox={'facecolor': 'w',
                  'edgecolor': '#9b9b9b',
                  'linewidth': 0.5, 'pad': 5,
                  'boxstyle': 'square, pad=0.4'},
            transform=ax.transAxes)

try:
    plt.show()
except Exception:
    pass
