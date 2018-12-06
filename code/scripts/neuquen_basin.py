from __future__ import division
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import LightSource
from mpl_toolkits.basemap import Basemap
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from fatiando import gridder
from tesseroid_density import tesseroid
from tesseroid_model import TesseroidModel


# Create results dir if it does not exist
# ---------------------------------------
script_path = os.path.dirname(os.path.abspath(__file__))
result_dir = os.path.join(script_path, 'results/neuquen')
if not os.path.isdir(result_dir):
    os.makedirs(result_dir)


# Load digital elevation map of Neuquen, Argentina
# -----------------------------------------------
script_path = os.path.dirname(os.path.abspath(__file__))
data = np.load(os.path.join(script_path, "../../data/topography.npy"))
lat, lon, topo = data[:, 0], data[:, 1], data[:, 2]
shape = (571, 457)
area = (lat.min(), lat.max(), lon.min(), lon.max())
topography = {'lon': lon,
              'lat': lat,
              'topo': topo,
              'shape': shape,
              'area': area}


# Load sediment thickness data of Neuquen Basin
# ---------------------------------------------
lat, lon, thickness = np.loadtxt(
    os.path.join(script_path, '../../data/sediment_thickness.dat'),
    unpack=True
)
shape = (117, 91)

area = (lat.min(), lat.max(), lon.min(), lon.max())
sediments = {'lon': lon,
             'lat': lat,
             'thickness': thickness,
             'shape': shape,
             'area': area}


# Create top and bottom of sediments layer
# ----------------------------------------
# Regrid dem to sediments regular grid
topo = gridder.interp_at(topography["lat"], topography["lon"], topography["topo"],
                         sediments["lat"], sediments["lon"],
                         algorithm="linear")
sediments["top"] = topo
sediments["bottom"] = topo - sediments["thickness"]


# Create Tesseroids model of the sediments layer
# ----------------------------------------------
bottom, top = sediments["bottom"].copy(), sediments["top"].copy()
top[np.isnan(top)] = 0
bottom[np.isnan(bottom)] = 0
basin = TesseroidModel(sediments['area'], top, bottom, sediments['shape'])


# Create computation grid at 10km above the spheroid
# --------------------------------------------------
shape = (79, 81)
area = (-40.8, -33., 287, 295.)
lat, lon, height = gridder.regular(area, shape, z=10e3)
grid = {'lat': lat,
        'lon': lon,
        'height': height,
        'shape': shape,
        'area': area}


# Common variables between computations
# -------------------------------------
density_top, density_bottom = -412, -275
max_height, min_depth = basin.top.max(), basin.bottom.min()


# Compute gravitational effect of the basin with homogeneous density
# ------------------------------------------------------------------
# Compute the mean density contrast of the basin and add it to the model
mean_density = (density_top + density_bottom) / 2
basin.addprop("density", [mean_density for i in range(basin.size)])

fields = "potential gz".split()
for field in fields:
    fname = "homogeneous-{}.npz".format(field)
    if os.path.isfile(os.path.join(result_dir, fname)):
        continue
    result = getattr(tesseroid, field)(grid["lon"],
                                       grid["lat"],
                                       grid["height"],
                                       basin)
    np.savez(os.path.join(result_dir, fname),
             result=result, lon=grid["lon"], lat=grid["lat"],
             height=grid["height"], shape=grid["shape"])


# Compute gravitational effect of the basin with linear density
# -------------------------------------------------------------
slope = (density_top - density_bottom) / (max_height - min_depth)
constant_term = density_top - slope * max_height

def linear_density(h):
    return slope * h + constant_term

basin.addprop("density", [linear_density for i in range(basin.size)])

fields = "potential gz".split()
for field in fields:
    fname = "linear-{}.npz".format(field)
    if os.path.isfile(os.path.join(result_dir, fname)):
        continue
    result = getattr(tesseroid, field)(grid["lon"],
                                       grid["lat"],
                                       grid["height"],
                                       basin)
    np.savez(os.path.join(result_dir, fname),
             result=result, lon=grid["lon"], lat=grid["lat"],
             height=grid["height"], shape=grid["shape"])


# Compute gravitational effect of the basin with exponential density
# ------------------------------------------------------------------
b_factor = 10
thickness = max_height - min_depth
denominator = np.exp(- min_depth * b_factor / thickness) - \
              np.exp(- max_height * b_factor / thickness)
amplitude = (density_bottom - density_top) / denominator
constant_term = (
    density_top * np.exp(-min_depth * b_factor / thickness) -
    density_bottom * np.exp(-max_height * b_factor / thickness)
    ) / denominator

# Define density function
def density_exponential(h):
    return amplitude * np.exp(-h * b_factor / thickness) + constant_term

basin.addprop("density", [linear_density for i in range(basin.size)])

fields = "potential gz".split()
for field in fields:
    fname = "exponential-{}.npz".format(field)
    if os.path.isfile(os.path.join(result_dir, fname)):
        continue
    result = getattr(tesseroid, field)(grid["lon"],
                                       grid["lat"],
                                       grid["height"],
                                       basin)
    np.savez(os.path.join(result_dir, fname),
             result=result, lon=grid["lon"], lat=grid["lat"],
             height=grid["height"], shape=grid["shape"])


# Plot Results
# ------------
labels = {"potential": "J/kg", "gz": "mGal"}
titles = {"potential": r"$V$", "gz": r"$g_{z}$"}
densities = ["homogeneous", "linear", "exponential"]

for density in densities:
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    axes = axes.ravel()
    for field, ax in zip(fields, axes):

        bm = Basemap(projection='merc',
                     llcrnrlon=grid["area"][2],
                     llcrnrlat=grid["area"][0],
                     urcrnrlon=grid["area"][3],
                     urcrnrlat=grid["area"][1],
                     resolution='l', ax=ax)

        result = np.load(os.path.join(result_dir, "{}-{}.npz".format(density, field)))

        x, y = bm(result["lon"], result["lat"])
        shape = result["shape"]

        cb = bm.contourf(x.reshape(shape), y.reshape(shape),
                         result["result"].reshape(shape), 100)
        bm.contour(x.reshape(shape), y.reshape(shape), result["result"].reshape(shape),
                   5, colors='k', linewidths=0.7)
        bm.drawcountries()
        bm.drawstates()
        bm.drawcoastlines()
        bm.drawmeridians(np.arange(-80, -50, 2),
                         labels=[False, False, False, True])
        bm.drawparallels(np.arange(-50, -30, 2),
                         labels=[True, False, False, False])
        plt.colorbar(cb, ax=ax, label=labels[field])
        ax.set_title(titles[field])

    plt.show()


# Plot differences
# ----------------
labels = {"potential": "J/kg", "gz": "mGal"}
titles = {"potential": r"$V$", "gz": r"$g_{z}$"}

# Linear and Homogeneous
fig, axes = plt.subplots(1, 2, figsize=(15, 7))
axes = axes.ravel()
for field, ax in zip(fields, axes):

    bm = Basemap(projection='merc',
                 llcrnrlon=grid["area"][2],
                 llcrnrlat=grid["area"][0],
                 urcrnrlon=grid["area"][3],
                 urcrnrlat=grid["area"][1],
                 resolution='l', ax=ax)

    linear = np.load(os.path.join(result_dir, "linear-{}.npz".format(field)))
    homogeneous = np.load(os.path.join(result_dir,
                                       "homogeneous-{}.npz".format(field)))

    x, y = bm(linear["lon"], linear["lat"])
    shape = linear["shape"]
    differences = linear["result"] - homogeneous["result"]
    vmax = np.abs(differences).max()

    cb = bm.contourf(x.reshape(shape), y.reshape(shape), differences.reshape(shape),
                     100, vmax=vmax, vmin=-vmax, cmap="RdBu_r")
    bm.contour(x.reshape(shape), y.reshape(shape), differences.reshape(shape),
               5, colors='k', linewidths=0.7)
    bm.drawcountries()
    bm.drawstates()
    bm.drawcoastlines()
    bm.drawmeridians(np.arange(-80, -50, 2),
                     labels=[False, False, False, True])
    bm.drawparallels(np.arange(-50, -30, 2),
                     labels=[True, False, False, False])
    plt.colorbar(cb, ax=ax, label=labels[field])
    ax.set_title(titles[field])

plt.show()

# Exponential and Homogeneous
fig, axes = plt.subplots(1, 2, figsize=(15, 7))
axes = axes.ravel()
for field, ax in zip(fields, axes):

    bm = Basemap(projection='merc',
                 llcrnrlon=grid["area"][2],
                 llcrnrlat=grid["area"][0],
                 urcrnrlon=grid["area"][3],
                 urcrnrlat=grid["area"][1],
                 resolution='l', ax=ax)

    exponential = np.load(os.path.join(result_dir, "exponential-{}.npz".format(field)))
    homogeneous = np.load(os.path.join(result_dir,
                                       "homogeneous-{}.npz".format(field)))

    x, y = bm(exponential["lon"], exponential["lat"])
    shape = exponential["shape"]
    differences = exponential["result"] - homogeneous["result"]
    vmax = np.abs(differences).max()

    cb = bm.contourf(x.reshape(shape), y.reshape(shape), differences.reshape(shape),
                     100, vmax=vmax, vmin=-vmax, cmap="RdBu_r")
    bm.contour(x.reshape(shape), y.reshape(shape), differences.reshape(shape),
               5, colors='k', linewidths=0.7)
    bm.drawcountries()
    bm.drawstates()
    bm.drawcoastlines()
    bm.drawmeridians(np.arange(-80, -50, 2),
                     labels=[False, False, False, True])
    bm.drawparallels(np.arange(-50, -30, 2),
                     labels=[True, False, False, False])
    plt.colorbar(cb, ax=ax, label=labels[field])
    ax.set_title(titles[field])

plt.show()
