from __future__ import division, print_function
import os
import time
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
    b_factor = 10
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
df = pd.read_csv(csv_fname, index_col=0)
fig, axes = plt.subplots(nrows=2, ncols=1)
ax = axes[0]
ax.plot(df["linear_potential"] / df["constant_potential"], '-o')
ax.plot(df["linear_gz"] / df["constant_gz"], '-o')
ax = axes[1]
ax.plot(df["exp_gz"] / df["constant_gz"], '-o')
ax.plot(df["exp_potential"] / df["constant_potential"], '-o')
for ax in axes:
    ax.set_xscale("log")
plt.show()
