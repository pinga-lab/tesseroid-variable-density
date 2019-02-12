from __future__ import division
import os
import warnings
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
print("Top of sediments: {} \nBottom of sediments: {}".format(top, bottom))


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

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Check for DISPLAY variable for matplotlib
# -----------------------------------------
try:
    os.environ["DISPLAY"]
except Exception:
    plt.switch_backend('agg')


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

# Disable warnings
warnings.filterwarnings("ignore")


# --------------
# Plot densities
# --------------
heights = np.linspace(bottom, top, 101)

# Initialize figure and subplots
fig, ax = plt.subplots(figsize=(3.33, 3))

ax.plot(heights, linear_density(heights), label="Linear")
ax.plot(heights, density_exponential(heights), label="Exponential")
ax.set_xlabel("Height [m]")
ax.set_ylabel(r"Density [kg/m$^3$]")
ax.legend()
ax.grid()
fig.tight_layout()
figure_fname = os.path.join(script_path,
                            "../../manuscript/figures/neuquen-basin-densities.pdf")
plt.savefig(figure_fname, dpi=300)
plt.show()


# ------------
# Plot Results
# ------------

# Create basemap and configuration
# --------------------------------
bm = Basemap(projection='merc',
             llcrnrlon=grid["area"][2],
             llcrnrlat=grid["area"][0],
             urcrnrlon=grid["area"][3],
             urcrnrlat=grid["area"][1],
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


# Initiate figure and subplots
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(6.66, 7))


# Topography
# ----------
ax = axes[0, 0]
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
cbar = bm.colorbar(im, label="km")

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
ax = axes[0, 1]
bm.ax = ax
ax.set_title("(b)", y=1.08, loc='left')
ax.set_title("Basin Thickness", y=1.08, loc="center")
x, y = bm(sediments['lon'], sediments['lat'])
im = bm.pcolormesh(x.reshape(sediments['shape']),
                   y.reshape(sediments['shape']),
                   1e-3*sediments['thickness'].reshape(sediments['shape']),
                   rasterized=True)
bm.contour(x.reshape(sediments['shape']),
           y.reshape(sediments['shape']),
           sediments['thickness'].reshape(sediments['shape']),
           5, colors='k', linewidths=0.7)
bm.drawcountries(**config['countries'])
bm.drawstates(**config['states'])
bm.drawcoastlines(**config['coastlines'])
bm.drawparallels(**config['parallels'])
bm.drawmeridians(**config['meridians'])

# Colorbar
cbar = bm.colorbar(im, label="km")


# Results (exponential)
# ---------------------
units = {"potential": "J/kg", "gz": "mGal"}
titles = {"potential": r"$V$", "gz": r"$g_{z}$"}
fig_titles = {"potential": "(c)", "gz": "(d)"}
axes_dict = {"potential": axes[1, 0], "gz": axes[1, 1]}

density = "exponential"
for field in fields:
    ax = axes_dict[field]
    bm.ax = ax
    ax.set_title(fig_titles[field], y=1.08, loc='left')
    ax.set_title(titles[field], y=1.08, loc='center')

    # Plot result
    result = np.load(os.path.join(result_dir, "{}-{}.npz".format(density, field)))

    shape = result["shape"]
    im = bm.pcolormesh(result["lon"].reshape(shape), result["lat"].reshape(shape),
                       result["result"].reshape(shape), rasterized=True, latlon=True)
    bm.contour(result["lon"].reshape(shape), result["lat"].reshape(shape),
               result["result"].reshape(shape), 5, colors='k', linewidths=0.7,
               latlon=True)

    # Configure basemap
    bm.drawcountries(**config['countries'])
    bm.drawstates(**config['states'])
    bm.drawcoastlines(**config['coastlines'])
    bm.drawparallels(**config['parallels'])
    bm.drawmeridians(**config['meridians'])

    # Colorbar
    cbar = bm.colorbar(im, label=units[field])

fig.tight_layout()
figure_fname = os.path.join(script_path,
                            "../../manuscript/figures/neuquen-basin.pdf")
plt.savefig(figure_fname, dpi=300)
plt.show()


# ----------------
# Plot differences
# ----------------
# Initiate figure and subplots
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(6.66, 7))

densities = ["homogeneous", "linear"]
units = {"potential": "J/kg", "gz": "mGal"}
titles = {"potential": r"$V$", "gz": r"$g_{z}$"}
fig_titles = "(a) (b) (c) (d)".split()

# Compute differences between exponential density results with linear and homogeneous
differences = dict(zip(fields, [{}, {}]))
for field in fields:
    exponential = np.load(
        os.path.join(result_dir, "exponential-{}.npz".format(field))
    )
    shape = exponential["shape"]
    longitude = exponential["lon"].reshape(shape)
    latitude = exponential["lat"].reshape(shape)
    for density in densities:
        other = np.load(
            os.path.join(result_dir, "{}-{}.npz".format(density, field))
        )
        difference = exponential["result"] - other["result"]
        differences[field][density] = difference.reshape(shape)

min_value = {}
for field in fields:
    min_value[field] = min(
        [differences[field][density].min() for density in densities]
    )

# Plot result
for i, density in enumerate(densities):
    for j, field in enumerate(fields):
        ax = axes[i, j]
        bm.ax = ax
        ax.set_title(fig_titles[2*i + j], y=1.08, loc='left')
        ax.set_title(titles[field], y=1.08, loc='center')

        vmin = min_value[field]
        # Because max value of differences is bellow zero, we set vmax=0
        im = bm.pcolormesh(longitude, latitude, differences[field][density],
                           vmin=vmin, vmax=0,
                           cmap="Blues_r", rasterized=True, latlon=True)
        bm.contour(exponential["lon"].reshape(shape),
                   exponential["lat"].reshape(shape),
                   difference.reshape(shape),
                   5, colors='k', linewidths=0.7, latlon=True)

        # Configure basemap
        bm.drawcountries(**config['countries'])
        bm.drawstates(**config['states'])
        bm.drawcoastlines(**config['coastlines'])
        bm.drawparallels(**config['parallels'])
        bm.drawmeridians(**config['meridians'])

        # Colorbar
        cbar = bm.colorbar(im, label=units[field])

fig.tight_layout()
figure_fname = os.path.join(script_path,
                            "../../manuscript/figures/neuquen-basin-diffs.pdf")
plt.savefig(figure_fname, dpi=300)
plt.show()
