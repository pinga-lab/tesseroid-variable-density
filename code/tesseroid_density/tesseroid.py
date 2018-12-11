r"""
Forward model the gravitational fields of a tesseroid (spherical prism).

Functions in this module calculate the gravitational fields of a tesseroid with
respect to the local North-oriented coordinate system of the computation point.
See the figure below.

.. raw:: html

    <div class="row">
    <div class="col-md-3">
    </div>
    <div class="col-md-6">

.. figure:: ../_static/images/tesseroid-coord-sys.png
    :alt: A tesseroid in a geocentric coordinate system
    :width: 100%
    :align: center

    A tesseroid in a geocentric coordinate system (X, Y, Z). Point P is a
    computation point with associated local North-oriented coordinate system
    (x, y, z).
    Image by L. Uieda (doi:`10.6084/m9.figshare.1495525
    <http://dx.doi.org/10.6084/m9.figshare.1495525>`__).

.. raw:: html

    </div>
    <div class="col-md-3">
    </div>
    </div>


.. admonition:: Coordinate systems

    The gravitational attraction
    and gravity gradient tensor
    are calculated with respect to
    the local coordinate system of the computation point.
    This system has **x -> North**, **y -> East**, **z -> up**
    (radial direction).

.. warning:: The :math:`g_z` component is an **exception** to this.
    In order to conform with the regular convention
    of z-axis pointing toward the center of the Earth,
    **this component only** is calculated with **z -> Down**.
    This way, gravity anomalies of
    tesseroids with positive density
    are positive, not negative.

Gravity
-------

Forward modeling of gravitational fields is performed by functions:

:func:`~fatiando.gravmag.tesseroid.potential`,
:func:`~fatiando.gravmag.tesseroid.gx`,
:func:`~fatiando.gravmag.tesseroid.gy`,
:func:`~fatiando.gravmag.tesseroid.gz`,
:func:`~fatiando.gravmag.tesseroid.gxx`,
:func:`~fatiando.gravmag.tesseroid.gxy`,
:func:`~fatiando.gravmag.tesseroid.gxz`,
:func:`~fatiando.gravmag.tesseroid.gyy`,
:func:`~fatiando.gravmag.tesseroid.gyz`,
:func:`~fatiando.gravmag.tesseroid.gzz`

The fields are calculated using Gauss-Legendre Quadrature integration and the
adaptive discretization algorithm of Uieda et al. (2016). The accuracy of the
integration is controlled by the ``ratio`` argument. Larger values cause finer
discretization and more accuracy but slower computation times. The defaults
values are the ones suggested in the paper and guarantee an accuracy of
approximately 0.1%.

.. warning::

    The integration error may be larger than this if the computation
    points are closer than 1 km of the tesseroids. This effect is more
    significant in the gravity gradient components.

References
++++++++++

Uieda, L., V. Barbosa, and C. Braitenberg (2016), Tesseroids: Forward-modeling
gravitational fields in spherical coordinates, Geophysics, F41-F48,
doi:10.1190/geo2015-0204.1

----

"""
from __future__ import division, absolute_import
from future.builtins import range
import multiprocessing
import warnings

import numpy as np
from . import _tesseroid
from fatiando.constants import SI2MGAL, MEAN_EARTH_RADIUS, G


RATIO_V = 1
RATIO_G = 2.5
STACK_SIZE = 200
DELTA = 1e-2


def _check_input(lon, lat, height, model, ratio, njobs, pool):
    """
    Check if the inputs are as expected and generate the output array.

    Returns:

    * results : 1d-array, zero filled

    """
    assert lon.shape == lat.shape == height.shape, \
        "Input coordinate arrays must have same shape"
    assert ratio > 0, "Invalid ratio {}. Must be > 0.".format(ratio)
    assert njobs > 0, "Invalid number of jobs {}. Must be > 0.".format(njobs)
    if njobs == 1:
        assert pool is None, "njobs should be number of processes in the pool"
    result = np.zeros_like(lon)
    return result


def _convert_coords(lon, lat, height):
    """
    Convert angles to radians and heights to radius.

    Pre-compute the sine and cosine of latitude because that is what we need
    from it.
    """
    # Convert things to radians
    lon = np.radians(lon)
    lat = np.radians(lat)
    sinlat = np.sin(lat)
    coslat = np.cos(lat)
    # Transform the heights into radius
    radius = MEAN_EARTH_RADIUS + height
    return lon, sinlat, coslat, radius


def _check_tesseroid(tesseroid, dens):
    """
    Check if the tesseroid is valid and get the right density to use.

    Returns None if the tesseroid should be ignored. Else, return the density
    that should be used.
    """
    if tesseroid is None:
        return None
    if 'density' not in tesseroid.props and dens is None:
        return None
    w, e, s, n, top, bottom = tesseroid.get_bounds()
    # Check if the dimensions given are valid
    assert w <= e and s <= n and top >= bottom, \
        "Invalid tesseroid dimensions {}".format(tesseroid.get_bounds())
    if dens is not None:
        density = dens
    else:
        density = tesseroid.props['density']
    return density


def _dispatcher(field, lon, lat, height, model, **kwargs):
    """
    Dispatch the computation of *field* to the appropriate function.

    Returns:

    * result : 1d-array

    """
    njobs = kwargs.get('njobs', 1)
    pool = kwargs.get('pool', None)
    dens = kwargs['dens']
    ratio = kwargs['ratio']
    delta = kwargs['delta']
    result = _check_input(lon, lat, height, model, ratio, njobs, pool)
    if njobs > 1 and pool is None:
        pool = multiprocessing.Pool(njobs)
        created_pool = True
    else:
        created_pool = False
    if pool is None:
        _forward_model([lon, lat, height, result, model, dens, ratio,
                        field, delta])
    else:
        chunks = _split_arrays(arrays=[lon, lat, height, result],
                               extra_args=[model, dens, ratio, field,
                                           delta],
                               nparts=njobs)
        result = np.hstack(pool.map(_forward_model, chunks))
    if created_pool:
        pool.close()
    return result


def _forward_model(args):
    """
    Run the computations on the model for a given list of arguments.

    This is used because multiprocessing.Pool.map can only use functions that
    receive a single argument.

    Arguments should be, in order:

    lon, lat, height, result, model, dens, ratio, field, delta
    """
    lon, lat, height, result, model, dens, ratio, field, delta = args
    lon, sinlat, coslat, radius = _convert_coords(lon, lat, height)
    func = getattr(_tesseroid, field)
    warning_msg = (
        "Stopped dividing a tesseroid because it's horizontal dimensions would be " +
        "below the minimum numerical threshold (1e-6 degrees). " +
        "Will compute without division. Cannot guarantee the accuracy of " +
        "the solution.")
    for tesseroid in model:
        density = _check_tesseroid(tesseroid, dens)
        if density is None:
            continue
        bounds = np.array(tesseroid.get_bounds())
        if callable(density) and delta is not None:
            subset = _density_based_discretization(bounds, density, delta)
            for bounds in subset:
                error = func(bounds, density, ratio, STACK_SIZE, lon, sinlat,
                             coslat, radius, result)
                if error != 0:
                    warnings.warn(warning_msg, RuntimeWarning)
        else:
            error = func(bounds, density, ratio, STACK_SIZE, lon, sinlat,
                         coslat, radius, result)
            if error != 0:
                warnings.warn(warning_msg, RuntimeWarning)
    return result


def _density_based_discretization(bounds, density, delta):
    """
    Applies the density-based discretization algorithm.
    Parameters:
        * bounds: list or 1d-array
            List with w, e, s, n, top, bottom bounds of the tesseroid that will
            be subdivided.
        * density: function
            Density function of the tesseroid that will be subdivided.
        * delta: float
            Adimensional density variation threshold.
    Returns:
        * subset: list
            List of bounds corresponding to the subdivisions of the original
            tesseroid.
    """
    w, e, s, n, top, bottom = bounds[:]
    tesseroid_size = top - bottom
    pending, subset = [bounds], []

    # Compute maximum and minimum density for future normalization
    heights = np.linspace(bottom, top, 101)
    densities = density(heights)
    rho_min, rho_max = np.min(densities), np.max(densities)

    if np.isclose(rho_min, rho_max):
        pending, subset = subset, pending
        return subset

    # Discretization of the tesseroid
    while pending != []:
        bounds = pending.pop(0)
        top, bottom = bounds[-2], bounds[-1]
        divider, max_diff = _divider_calculation(top, bottom, density,
                                                 rho_min, rho_max)

        if divider is None:
            subset.append(np.array([w, e, s, n, top, bottom]))
            continue

        size_ratio = (top - bottom)/tesseroid_size
        if max_diff*size_ratio > delta:
            pending.append([w, e, s, n, top, divider])
            pending.append([w, e, s, n, divider, bottom])
        else:
            subset.append(np.array([w, e, s, n, top, bottom]))
    return subset


def _divider_calculation(top, bottom, density, rho_min, rho_max):
    """
    Computes the height at which the tesseroid with top and bottom boundaries
    should be divided according to the density-based discretization algorithm.
    It also computes the maximum difference between the normalised density
    and the straight reference line.
    """
    heights = np.linspace(bottom, top, 101)
    norm_density = (density(heights) - rho_min)/(rho_max - rho_min)
    line = (norm_density[-1] - norm_density[0])/(top - bottom) * \
           (heights - bottom) + norm_density[0]
    diff = np.abs(norm_density - line)
    max_diff = np.max(diff)
    divider_height = heights[np.argmax(diff)]
    return divider_height, max_diff


def _split_arrays(arrays, extra_args, nparts):
    """
    Split the coordinate arrays into nparts. Add extra_args to each part.

    Example::

    >>> chunks = _split_arrays([[1, 2, 3, 4, 5, 6]], ['meh'], 3)
    >>> chunks[0]
    [[1, 2], 'meh']
    >>> chunks[1]
    [[3, 4], 'meh']
    >>> chunks[2]
    [[5, 6], 'meh']

    """
    size = len(arrays[0])
    n = size//nparts
    strides = [(i*n, (i + 1)*n) for i in range(nparts - 1)]
    strides.append((strides[-1][-1], size))
    chunks = [[x[low:high] for x in arrays] + extra_args
              for low, high in strides]
    return chunks


def potential(lon, lat, height, model, dens=None, ratio=RATIO_V,
              njobs=1, pool=None, delta=DELTA):
    """
    Calculate the gravitational potential due to a tesseroid model.

    .. warning:: Tesseroids with dimensions < 10 cm will be ignored to avoid
        numerical errors.

    Implements the method of Uieda et al. (2016).

    Parameters:

    * lon, lat, height : arrays
        Arrays with the longitude, latitude and height coordinates of the
        computation points.
    * model : list of :class:`~fatiando.mesher.Tesseroid`
        The density model used to calculate the gravitational effect.
        Tesseroids must have the property ``'density'``. Those that don't have
        this property will be ignored in the computations. Elements that are
        None will also be ignored.
    * dens : float or None
        If not None, will use this value instead of the ``'density'`` property
        of the tesseroids. Use this, e.g., for sensitivity matrix building.
    * ratio : float
        Will divide each tesseroid until the distance between it and the
        computation points is < ratio*size of tesseroid. Used to guarantee the
        accuracy of the numerical integration.
    * njobs : int
        Split the computation into *njobs* parts and run it in parallel using
        ``multiprocessing``. If ``njobs=1`` will run the computation in serial.
    * pool : None or multiprocessing.Pool object
        If not None, will use this pool to run the computation in parallel
        instead of creating a new one. You must still specify *njobs* as the
        number of processes in the pool. Use this to avoid spawning processes
        on each call to this functions, which can have significant overhead.
    * delta : float
        Explain...

    Returns:

    * res : array
        The calculated field in SI units

    References:

    Uieda, L., V. Barbosa, and C. Braitenberg (2016), Tesseroids:
    Forward-modeling gravitational fields in spherical coordinates, Geophysics,
    F41-F48, doi:10.1190/geo2015-0204.1

    """
    field = 'potential'
    result = _dispatcher(field, lon, lat, height, model, dens=dens,
                         ratio=ratio, njobs=njobs, pool=pool,
                         delta=delta)
    result *= G
    return result


def gx(lon, lat, height, model, dens=None, ratio=RATIO_G,
       njobs=1, pool=None, delta=DELTA):
    """
    Calculate the North component of the gravitational attraction.

    .. warning:: Tesseroids with dimensions < 10 cm will be ignored to avoid
        numerical errors.

    Implements the method of Uieda et al. (2016).

    Parameters:

    * lon, lat, height : arrays
        Arrays with the longitude, latitude and height coordinates of the
        computation points.
    * model : list of :class:`~fatiando.mesher.Tesseroid`
        The density model used to calculate the gravitational effect.
        Tesseroids must have the property ``'density'``. Those that don't have
        this property will be ignored in the computations. Elements that are
        None will also be ignored.
    * dens : float or None
        If not None, will use this value instead of the ``'density'`` property
        of the tesseroids. Use this, e.g., for sensitivity matrix building.
    * ratio : float
        Will divide each tesseroid until the distance between it and the
        computation points is < ratio*size of tesseroid. Used to guarantee the
        accuracy of the numerical integration.
    * njobs : int
        Split the computation into *njobs* parts and run it in parallel using
        ``multiprocessing``. If ``njobs=1`` will run the computation in serial.
    * pool : None or multiprocessing.Pool object
        If not None, will use this pool to run the computation in parallel
        instead of creating a new one. You must still specify *njobs* as the
        number of processes in the pool. Use this to avoid spawning processes
        on each call to this functions, which can have significant overhead.
    * delta : float
        Explain...

    Returns:

    * res : array
        The calculated field in mGal

    References:

    Uieda, L., V. Barbosa, and C. Braitenberg (2016), Tesseroids:
    Forward-modeling gravitational fields in spherical coordinates, Geophysics,
    F41-F48, doi:10.1190/geo2015-0204.1

    """
    field = 'gx'
    result = _dispatcher(field, lon, lat, height, model, dens=dens,
                         ratio=ratio, njobs=njobs, pool=pool,
                         delta=delta)
    result *= SI2MGAL*G
    return result


def gy(lon, lat, height, model, dens=None, ratio=RATIO_G,
       njobs=1, pool=None, delta=DELTA):
    """
    Calculate the East component of the gravitational attraction.

    .. warning:: Tesseroids with dimensions < 10 cm will be ignored to avoid
        numerical errors.

    Implements the method of Uieda et al. (2016).

    Parameters:

    * lon, lat, height : arrays
        Arrays with the longitude, latitude and height coordinates of the
        computation points.
    * model : list of :class:`~fatiando.mesher.Tesseroid`
        The density model used to calculate the gravitational effect.
        Tesseroids must have the property ``'density'``. Those that don't have
        this property will be ignored in the computations. Elements that are
        None will also be ignored.
    * dens : float or None
        If not None, will use this value instead of the ``'density'`` property
        of the tesseroids. Use this, e.g., for sensitivity matrix building.
    * ratio : float
        Will divide each tesseroid until the distance between it and the
        computation points is < ratio*size of tesseroid. Used to guarantee the
        accuracy of the numerical integration.
    * njobs : int
        Split the computation into *njobs* parts and run it in parallel using
        ``multiprocessing``. If ``njobs=1`` will run the computation in serial.
    * pool : None or multiprocessing.Pool object
        If not None, will use this pool to run the computation in parallel
        instead of creating a new one. You must still specify *njobs* as the
        number of processes in the pool. Use this to avoid spawning processes
        on each call to this functions, which can have significant overhead.
    * delta : float
        Explain...

    Returns:

    * res : array
        The calculated field in mGal

    References:

    Uieda, L., V. Barbosa, and C. Braitenberg (2016), Tesseroids:
    Forward-modeling gravitational fields in spherical coordinates, Geophysics,
    F41-F48, doi:10.1190/geo2015-0204.1

    """
    field = 'gy'
    result = _dispatcher(field, lon, lat, height, model, dens=dens,
                         ratio=ratio, njobs=njobs, pool=pool,
                         delta=delta)
    result *= SI2MGAL*G
    return result


def gz(lon, lat, height, model, dens=None, ratio=RATIO_G,
       njobs=1, pool=None, delta=DELTA):
    """
    Calculate the radial component of the gravitational attraction.

    .. warning::
        In order to conform with the regular convention of positive density
        giving positive gz values, **this component only** is calculated
        with **z axis -> Down**.

    .. warning:: Tesseroids with dimensions < 10 cm will be ignored to avoid
        numerical errors.

    Implements the method of Uieda et al. (2016).

    Parameters:

    * lon, lat, height : arrays
        Arrays with the longitude, latitude and height coordinates of the
        computation points.
    * model : list of :class:`~fatiando.mesher.Tesseroid`
        The density model used to calculate the gravitational effect.
        Tesseroids must have the property ``'density'``. Those that don't have
        this property will be ignored in the computations. Elements that are
        None will also be ignored.
    * dens : float or None
        If not None, will use this value instead of the ``'density'`` property
        of the tesseroids. Use this, e.g., for sensitivity matrix building.
    * ratio : float
        Will divide each tesseroid until the distance between it and the
        computation points is < ratio*size of tesseroid. Used to guarantee the
        accuracy of the numerical integration.
    * njobs : int
        Split the computation into *njobs* parts and run it in parallel using
        ``multiprocessing``. If ``njobs=1`` will run the computation in serial.
    * pool : None or multiprocessing.Pool object
        If not None, will use this pool to run the computation in parallel
        instead of creating a new one. You must still specify *njobs* as the
        number of processes in the pool. Use this to avoid spawning processes
        on each call to this functions, which can have significant overhead.
    * delta : float
        Explain...

    Returns:

    * res : array
        The calculated field in mGal

    References:

    Uieda, L., V. Barbosa, and C. Braitenberg (2016), Tesseroids:
    Forward-modeling gravitational fields in spherical coordinates, Geophysics,
    F41-F48, doi:10.1190/geo2015-0204.1

    """
    field = 'gz'
    result = _dispatcher(field, lon, lat, height, model, dens=dens,
                         ratio=ratio, njobs=njobs, pool=pool,
                         delta=delta)
    result *= SI2MGAL*G
    return result
