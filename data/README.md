# Neuquén Basin Data

## Digital Elevation Map

On `topography.npy` you can find a Digital Elevation Map of the Neuquén Basin
area: from 73°W to 65°W and 41°S to 33°S; obtained from
[Ryan et. al. (2009)](http://www.marine-geo.org/tools/GMRTMapTool/).

The data is saved as a 2d array into a binary file with the NumPy `.npy` format.
To access the information you must do it using the `numpy.load()` function.
The first two columns contain the latitude and longitude points, respectively;
while the third column has the elevation data (in meters).

For example, we can access the information with the following script:

```
import numpy

data = numpy.load("topography.npy")
lat, lon, topo = data[:, 0], data[:, 1], data[:, 2]
```

## Neuquen Sedimentary Basin Thickness

On `sediment_thickness.dat` you can find the Neuquén Basin thickness, obtained
by digitalizing the maps in
[Heine, C. (2007)](http://www.earthbyte.org/Resources/ICONS/index.html).

The data is saved in an ASCII file, divided in three columns containing the
latitude, longitude and sediment thickness points.
They can be read in Python using the `numpy.loadtxt()` function:

```
import numpy

lat, lon, thickness = numpy.loadtxt("sediment_thickness.dat", unpack=True)
```



Ryan, W.B.F., S.M. Carbotte, J.O. Coplan, S. O'Hara, A. Melkonian, R. Arko,
R.A. Weissel, V. Ferrini, A. Goodwillie, F. Nitsche, J. Bonczkowski, and
R. Zemsky (2009), Global Multi-Resolution Topography synthesis,
Geochem. Geophys. Geosyst., 10, Q03014, doi: 10.1029/2008GC002332,
[http://onlinelibrary.wiley.com/doi/10.1029/2008GC002332/abstract]().

Heine, Christian (2007), Formation and Evolution of intracontinental basins,
PhD Thesis, School of Geosciences, The University of Sydney,
Australia, unpublished,
[http://www.earthbyte.org/Resources/ICONS/index.html]().
