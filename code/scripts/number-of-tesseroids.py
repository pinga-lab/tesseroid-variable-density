"""
We will compute the number of discretization that the density-based algorithm performs
in case of an exponential density and a delta value of 0.1.
We will explore different values of the constant b to see how it affects the
computational load.
"""
from __future__ import division
import os
import warnings
import numpy as np
import matplotlib.pyplot as plt

# This is our custom tesseroid code
from tesseroid_density import tesseroid


# Define Tesseroid boundaries
# ---------------------------
w, e, s, n = -10, 10, -10, 10
top, bottom = 0, -1e3


# Compute number of discretized tesseroids (exp)
# ----------------------------------------------
delta = 0.1
density_top, density_bottom = 0, 1
b_factors = np.array([1, 2, 5, 10, 30, 100])
n_tess = np.zeros_like(b_factors, dtype=np.int64)

fig, ax = plt.subplots()
heights = np.linspace(bottom, top, 101)
for i, b_factor in enumerate(b_factors):
    thickness = top - bottom
    A = (density_bottom - density_top) / (1 - np.exp(-b_factor))
    C = density_bottom - A

    # Define density function
    def density_exponential(height):
        return A * np.exp(-b_factor * (height - bottom) / thickness) + C

    # Calculate number of discretizations
    subset = tesseroid._density_based_discretization([w, e, s, n, top, bottom],
                                                     density_exponential,
                                                     delta)
    n_tess[i] = len(subset)

    # Plot density profile and boundaries of discretized tesseroids
    bounds = []
    for tess in subset:
        bounds.append(tess[-1])
        bounds.append(tess[-2])
    bounds = np.array(bounds)
    ax.plot(heights, density_exponential(heights), '-', color="C0")
    ax.plot(bounds, density_exponential(bounds), 'o', color="C1")

plt.show()


# Compute number of discretized tesseroids (sine)
# -----------------------------------------------
delta = 0.1
b_factors = [1, 2, 5, 10]
n_tess = np.zeros_like(b_factors, dtype=np.int64)

fig, axes = plt.subplots(nrows=len(b_factors), sharex=True)
heights = np.linspace(bottom, top, 101)
for i, b_factor in enumerate(b_factors):
    max_density = density_bottom
    thickness = top - bottom
    k_factor = 2 * np.pi * b_factor / thickness

    # Define density function
    def density_sine(height):
        return max_density / 2 * np.sin(k_factor * height) + max_density / 2

    # Calculate number of discretizations
    subset = tesseroid._density_based_discretization([w, e, s, n, top, bottom],
                                                     density_sine,
                                                     delta)
    n_tess[i] = len(subset)

    # Plot density profile and boundaries of discretized tesseroids
    bounds = []
    for tess in subset:
        bounds.append(tess[-1])
        bounds.append(tess[-2])
    bounds = np.array(bounds)
    axes[i].plot(heights, density_sine(heights), '-', color="C0")
    axes[i].plot(bounds, density_sine(bounds), 'o', color="C1")

plt.show()

# Plot number of tesseroids (sine)
plt.plot(b_factors, n_tess, 'o-')
plt.show()
