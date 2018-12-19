from __future__ import division
import os
import warnings
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.patches import Rectangle
from matplotlib.colors import LightSource
from mpl_toolkits.basemap import Basemap
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.gridspec import GridSpec
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
sediments_top = gridder.interp_at(topography["lat"], topography["lon"],
                                  topography["topo"], sediments["lat"],
                                  sediments["lon"], algorithm="linear")
sediments_top[np.isnan(sediments["thickness"])] = np.nan
# Compute the median of the topography on basin points
nans = np.isnan(sediments_top)
sediments_top[~nans] = np.median(sediments_top[~nans])
print("Top of sediments: {}m".format(sediments_top[~nans][0]))
# Add top and bottom arrays to sediments dictionary
sediments["top"] = sediments_top
sediments["bottom"] = sediments_top - sediments["thickness"]


# Create Tesseroids model of the sediments layer
# ----------------------------------------------
bottom, top = sediments["bottom"].copy(), sediments["top"].copy()
top[nans] = 0
bottom[nans] = 0
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
# Define density values for the top and the bottom of the sediment layer
density_top, density_bottom = -412, -275
# Define top and bottom variables as the maximum and minimum sediments'
# height and depth, respectively
top, bottom = basin.top[~nans].max(), basin.bottom[~nans].min()


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
def linear_density(h):
    slope = (density_top - density_bottom) / (top - bottom)
    return slope * (h - bottom) + density_bottom


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
def density_exponential(height):
    b_factor = 3
    thickness = top - bottom
    A = (density_bottom - density_top) / (1 - np.exp(-b_factor))
    C = density_bottom - A
    return A * np.exp(-b_factor * (height - bottom) / thickness) + C


basin.addprop("density", [density_exponential for i in range(basin.size)])

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


# ------------
# Plot Results
# ------------

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

# Create basemap and configuration
# --------------------------------
bm = Basemap(projection='merc',
             llcrnrlon=topography["area"][2],
             llcrnrlat=topography["area"][0],
             urcrnrlon=topography["area"][3],
             urcrnrlat=topography["area"][1],
             resolution='i')

config = {'countries': dict(linewidth=0.5, color='k'),
          'states': dict(linewidth=0.4, linestyle='--', color='k'),
          'coastlines': dict(linewidth=0.5, color='k'),
          'meridians': dict(meridians=np.arange(-80, -50, 2),
                            linewidth=0.5,
                            labels=[False, False, True, False],
                            labelstyle='+/-'),
          'parallels': dict(circles=np.arange(-50, -30, 2),
                            linewidth=0.5,
                            labels=[True, False, False, False],
                            labelstyle='+/-')}
config['parallels-quiet'] = config['parallels'].copy()
config['parallels-quiet']['labels'] = [False, False, False, False]

config['meridians-quiet'] = config['meridians'].copy()
config['meridians-quiet']['labels'] = [False, False, False, False]

config['parallels-right'] = config['parallels'].copy()
config['parallels-right']['labels'] = [False, True, False, False]


# Create outer grid
fig = plt.figure(figsize=(6.66, 8))
grid = GridSpec(ncols=2, nrows=2, wspace=0.001, hspace=0.1)


# Topography
# ----------
ax = plt.subplot(grid[0, 0])
bm.ax = ax
ax.set_title("(a)", y=1.08, loc='left')
ax.set_title("Neuquen Basin", y=1.08, loc="center")
x, y = bm(topography['lon'], topography['lat'])
shape = topography['shape']
vmax = np.abs([np.nanmin(topography['topo']),
               np.nanmax(topography['topo'])]).max()
vmin = -vmax
cmap = plt.cm.gist_earth

# Hillshaded topography
ls = LightSource(azdeg=120)
rgb = ls.shade(topography['topo'].reshape(shape),
               cmap, blend_mode='soft', vert_exag=1000,
               vmin=vmin, vmax=vmax)
bm.imshow(rgb)

# Proxy image for colorbar
im = bm.imshow(1e-3*topography['topo'].reshape(shape), cmap=cmap)
im.remove()
cbar = bm.colorbar(im, location='bottom', pad="3%")
tick_locator = ticker.MaxNLocator(nbins=5)
cbar.locator = tick_locator
cbar.update_ticks()
cbar.ax.tick_params(labelsize=9)

# Basemap configuration
bm.drawcountries(**config['countries'])
bm.drawstates(**config['states'])
bm.drawcoastlines(**config['coastlines'])
line = bm.drawmeridians(**config['meridians'])
line = bm.drawparallels(**config['parallels'])

# Location map
height, width = "50%", "29.375%"
axins = inset_axes(ax,
                   width=width,
                   height=height,
                   loc=1,
                   borderpad=0)
bm2 = Basemap(projection='merc',
              llcrnrlon=360 - 75, llcrnrlat=-56,
              urcrnrlon=360 - 53, urcrnrlat=-21,
              resolution='i', ax=axins)
bm2.drawmapboundary(fill_color='#7777ff')
bm2.fillcontinents(color='#ddaa66', lake_color='#7777ff')
bm2.drawcountries(linewidth=0.02)
bm2.drawcoastlines(linewidth=0.02)
x1, y1 = bm2(area[2], area[0])
x2, y2 = bm2(area[3], area[1])
rectangle = Rectangle((x1, y1),
                      abs(x2 - x1),
                      abs(y2 - y1),
                      facecolor='C3')
axins.add_patch(rectangle)


# Basin thickness
# ---------------
ax = plt.subplot(grid[0, 1])
bm.ax = ax
ax.set_title("(b)", y=1.08, loc='left')
ax.set_title("Basin Thickness", y=1.08, loc="center")
x, y = bm(sediments['lon'], sediments['lat'])
im = bm.contourf(x.reshape(sediments['shape']),
                 y.reshape(sediments['shape']),
                 sediments['thickness'].reshape(sediments['shape']),
                 100)
bm.contour(x.reshape(sediments['shape']),
           y.reshape(sediments['shape']),
           sediments['thickness'].reshape(sediments['shape']),
           5, colors='k', linewidths=0.7)
bm.drawcountries(**config['countries'])
bm.drawstates(**config['states'])
bm.drawcoastlines(**config['coastlines'])
bm.drawparallels(**config['parallels-quiet'])
bm.drawmeridians(**config['meridians'])

# Colorbar
cbar = bm.colorbar(im, location='bottom', pad="3%")
cbar.ax.tick_params(labelsize=9)
tick_locator = ticker.MaxNLocator(nbins=5)
cbar.locator = tick_locator
cbar.update_ticks()


# Results (exponential)
# ---------------------
labels = {"potential": "J/kg", "gz": "mGal"}
titles = {"potential": r"$V$", "gz": r"$g_{z}$"}
fig_titles = {"potential": "(c)", "gz": "(d)"}
grid_specs = {"potential": grid[1, 0], "gz": grid[1, 1]}

density = "exponential"
for field in fields:
    ax = plt.subplot(grid_specs[field])
    bm.ax = ax
    ax.set_title(fig_titles[field], y=1.08, loc='left')
    ax.set_title(titles[field], y=1.08, loc='center')

    # Plot result
    result = np.load(os.path.join(result_dir, "{}-{}.npz".format(density, field)))

    shape = result["shape"]
    im = bm.contourf(result["lon"].reshape(shape), result["lat"].reshape(shape),
                     result["result"].reshape(shape), 100, latlon=True)
    bm.contour(result["lon"].reshape(shape), result["lat"].reshape(shape),
               result["result"].reshape(shape), 5, colors='k', linewidths=0.7,
               latlon=True)

    # Configure basemap
    bm.drawcountries(**config['countries'])
    bm.drawstates(**config['states'])
    bm.drawcoastlines(**config['coastlines'])
    bm.drawparallels(**config['parallels'])
    bm.drawmeridians(**config['meridians'])
    # bm.drawparallels(**config['parallels-quiet'])
    # bm.drawmeridians(**config['meridians-quiet'])

    # Colorbar
    cbar = bm.colorbar(im, location='bottom', pad="3%")
    cbar.ax.tick_params(labelsize=9)
    tick_locator = ticker.MaxNLocator(nbins=5)
    cbar.locator = tick_locator
    cbar.update_ticks()

grid.tight_layout(fig)
plt.show()
