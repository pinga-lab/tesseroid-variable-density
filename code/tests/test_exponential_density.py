from __future__ import division, absolute_import
import numpy as np
from fatiando.constants import G, MEAN_EARTH_RADIUS, SI2MGAL, SI2EOTVOS
from fatiando.mesher import TesseroidMesh
from fatiando import gridder

from tesseroid_density import tesseroid


def shell_exponential_density(height, top, bottom, a, b, c, deltah):
    r = height + MEAN_EARTH_RADIUS
    r1 = bottom + MEAN_EARTH_RADIUS
    r2 = top + MEAN_EARTH_RADIUS
    constant = 4*np.pi*G*a*b*((r1**2 + 2*r1*b + 2*b**2)*np.exp(-(r1 - deltah)/b) -
                              (r2**2 + 2*r2*b + 2*b**2)*np.exp(-(r2 - deltah)/b))
    constant += 4/3*np.pi*G*c*(r2**3 - r1**3)
    potential = constant/r
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


def test_thin_shell():

    def density_exponential(height):
        r = height + MEAN_EARTH_RADIUS
        return a*np.exp(-(r - deltah)/b) + c

    # Spherical shell model with a Tesseroid Mesh
    top, bottom = 0, -1000
    density_1, density_2 = 2670, 3300
    b = 50.
    a = (density_2 - density_1)/(np.exp((abs(top - bottom))/b) - 1)
    c = density_1 - a
    deltah = MEAN_EARTH_RADIUS
    model = TesseroidMesh((0, 360, -90, 90, top, bottom), (1, 6, 12))
    model.addprop("density", [density_exponential for i in range(model.size)])

    # Computation grids
    shape = (10, 10)
    grids = {"pole": gridder.regular((89, 90, 0, 1), shape, z=2e3),
             "equator": gridder.regular((0, 1, 0, 1), shape, z=2e3),
             "260km": gridder.regular((89, 90, 0, 1), shape, z=260e3),
             "30deg": gridder.regular((60, 90, 0, 30), shape, z=2e3)}
    fields = 'potential gx gy gz gxx gxy gxz gyy gyz gzz'.split()
    for field in fields:
        for grid in grids.keys():
            msg = "Failed linear density test for thin shell while \
                   computing {} on {} grid".format(field, grid)
            lats, lons, heights = grids[grid]
            analytical = shell_exponential_density(heights[0], top, bottom,
                                                   a, b, c, deltah)
            result = getattr(tesseroid, field)(lons, lats, heights, model)
            diff = np.abs(result - analytical[field])
            # gz gy and the off-diagonal gradients should be zero so I can't
            # calculate a relative error (in %).
            # To do that, I'll use the gz and gzz shell values to calculate the
            # percentage.
            if field in 'potential gz gxx gyy gzz'.split():
                diff /= np.abs(analytical[field])
            elif field in 'gx gy'.split():
                diff /= np.abs(analytical['gz'])
            elif field in "gxy gxz gyz".split():
                diff /= np.abs(analytical['gzz'])
            diff = 100*np.max(diff)
            assert diff < 1e-1, msg


def test_thick_shell():

    def density_exponential(height):
        r = height + MEAN_EARTH_RADIUS
        return a*np.exp(-(r - deltah)/b) + c

    # Spherical shell model with a Tesseroid Mesh
    top, bottom = 0, -35000
    density_1, density_2 = 2670, 3300
    b = 850.
    a = (density_2 - density_1)/(np.exp((abs(top - bottom))/b) - 1)
    c = density_1 - a
    deltah = MEAN_EARTH_RADIUS
    model = TesseroidMesh((0, 360, -90, 90, top, bottom), (1, 6, 12))
    model.addprop("density", [density_exponential for i in range(model.size)])

    # Computation grids
    shape = (10, 10)
    grids = {"pole": gridder.regular((89, 90, 0, 1), shape, z=2e3),
             "equator": gridder.regular((0, 1, 0, 1), shape, z=2e3),
             "260km": gridder.regular((89, 90, 0, 1), shape, z=260e3),
             "30deg": gridder.regular((60, 90, 0, 30), shape, z=2e3)}
    fields = 'potential gx gy gz gxx gxy gxz gyy gyz gzz'.split()
    for field in fields:
        for grid in grids.keys():
            msg = "Failed linear density test for thin shell while \
                   computing {} on {} grid".format(field, grid)
            lats, lons, heights = grids[grid]
            analytical = shell_exponential_density(heights[0], top, bottom,
                                                   a, b, c, deltah)
            result = getattr(tesseroid, field)(lons, lats, heights, model)
            diff = np.abs(result - analytical[field])
            # gz gy and the off-diagonal gradients should be zero so I can't
            # calculate a relative error (in %).
            # To do that, I'll use the gz and gzz shell values to calculate the
            # percentage.
            if field in 'potential gz gxx gyy gzz'.split():
                diff /= np.abs(analytical[field])
            elif field in 'gx gy'.split():
                diff /= np.abs(analytical['gz'])
            elif field in "gxy gxz gyz".split():
                diff /= np.abs(analytical['gzz'])
            diff = 100*np.max(diff)
            assert diff < 1e-1, msg

