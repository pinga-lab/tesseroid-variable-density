from __future__ import division, print_function
import time
import numpy as np
from fatiando.mesher import Tesseroid
from fatiando import gridder
import matplotlib.pyplot as plt
# This is our custom tesseroid code
from tesseroid_density import tesseroid


# Configure comparison
# --------------------
nruns = 1000
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


# Compute gravitational fields for homogeneous tesseroid
# ------------------------------------------------------
for tess in model:
    tess.addprop('density', 3000)

times_homogeneous = {}
for field in fields:
    for point in computation_points:
        times = []
        lon, lat, height = point[:]
        for run in range(nruns):
            start_time = time.time()
            getattr(tesseroid, field)(lon, lat, height, model)
            end_time = time.time()
            times.append(start_time - end_time)
        times = np.mean(times)
        if times_homogeneous[field]:
            times_homogeneous[field].append(times)
        else:
            times_homogeneous[field] = [times]


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

times_linear = {}
for field in fields:
    for point in computation_points:
        times = []
        lon, lat, height = point[:]
        for run in range(nruns):
            start_time = time.time()
            getattr(tesseroid, field)(lon, lat, height, model)
            end_time = time.time()
            times.append(start_time - end_time)
        times = np.mean(times)
        if times_linear[field]:
            times_linear[field].append(times)
        else:
            times_linear[field] = [times]


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

times_exponential = {}
for field in fields:
    for point in computation_points:
        times = []
        lon, lat, height = point[:]
        for run in range(nruns):
            start_time = time.time()
            getattr(tesseroid, field)(lon, lat, height, model)
            end_time = time.time()
            times.append(start_time - end_time)
        times = np.mean(times)
        if times_exponential[field]:
            times_exponential[field].append(times)
        else:
            times_exponential[field] = [times]


# Times comparison
# ----------------
times_ratios_linear = {}
times_ratios_exponential = {}
for field in fields:
    times_ratio = times_linear[field].mean() / times_homogeneous[field].mean()
    times_ratios_linear[field] = times_ratio
    times_ratio = times_exponential[field].mean() / times_homogeneous[field].mean()
    times_ratios_exponential[field] = times_ratio

print(times_ratios_linear)
print(times_ratios_exponential)
