# -*- coding: utf-8 -*-
"""
Fork of class written by Leonardo Uieda (2016)
https://github.com/pinga-lab/paper-moho-inversion-tesseroids

Available under a BSD 3-clause license.
You can freely use and modify the code, without warranty, so long as you
provide attribution to the authors. See LICENSE.md for the full license text.

Defines the classes:

* `TesseroidModel`: Class to represent a continuous geophysical structure discretized
  into tesseroids.

"""
from __future__ import division
from fatiando.mesher import Tesseroid
import numpy as np
import copy


class TesseroidModel(object):
    """
    Implements a tesseroid model of a continuous geophysical structure.

    This class behaves like a sequence of 'Tesseroid' objects, so you can pass
    it along to any function that iterates over tesseroids (like the forward
    modeling function of 'fatiando.gravmag.tesseroid').

    Parameters:

    * area : [s, n, w, e]
        The south, north, west, and east limits of the mesh in degrees.
    * shape : (nlat, nlon)
        The number of tesseroids in the latitude and longitude directions. Will
        discretize *area* into this number of tesseroids.
    * top : 1d-array
        The height-coordinates of the top boundary of the structure.
    * bottom : 1d-array
        The height-coordinates of the bottom boundary of the structure.
    * props : dict
        Dictionary with the physical properties of the mesh.

    """

    def __init__(self, area, top, bottom, shape, props=None):
        assert shape[0]*shape[1] == top.size
        assert top.size == bottom.size
        assert len(area) == 4
        assert area[0] < area[1] and area[2] < area[3]
        self.area = area
        self.shape = shape
        s, n, w, e = area
        nlat, nlon = shape
        self.lons = np.linspace(w, e, nlon, endpoint=True)
        self.lats = np.linspace(s, n, nlat, endpoint=True)
        self.lon, self.lat = np.meshgrid(self.lons, self.lats)
        self.spacing = self.lats[1] - self.lats[0], self.lons[1] - self.lons[0]
        self._top = top
        self._bottom = bottom
        self.set_top_bottom()
        if props is None:
            self.props = {}
        else:
            self.props = props
        self._i = 0

    @property
    def clons(self):
        dlon = self.spacing[1]
        return self.lons + dlon/2

    @property
    def clats(self):
        dlat = self.spacing[0]
        return self.lats + dlat/2

    @property
    def clon(self):
        dlon = self.spacing[1]
        return self.lon + dlon/2

    @property
    def clat(self):
        dlat = self.spacing[0]
        return self.lat + dlat/2

    def addprop(self, prop, values):
        """
        Add physical property values to the mesh.

        Different physical properties of the grid are stored in a dictionary.

        Parameters:

        * prop : str
            Name of the physical property.
        * values :  list or array
            Value of this physical property in each point of the grid

        """
        self.props[prop] = values

    def set_top_bottom(self):
        _top = self.top.copy()
        _bottom = self.bottom.copy()
        isbelow = self.top <= self.bottom
        _top[isbelow] = self.bottom[isbelow]
        _bottom[isbelow] = self.top[isbelow]
        self._top = _top.copy()
        self._bottom = _bottom.copy()

    @property
    def top(self):
        return self._top

    @property
    def bottom(self):
        return self._bottom

    @property
    def size(self):
        return self.shape[0]*self.shape[1]

    def __len__(self):
        return self.size

    def __iter__(self):
        self._i = 0
        return self

    def next(self):
        if self._i >= self.size:
            raise StopIteration
        cell = self.__getitem__(self._i)
        self._i += 1
        return cell

    def __getitem__(self, index):
        nlat, nlon = self.shape
        dlat, dlon = self.spacing
        i = index//nlon
        j = index - i*nlon
        w = self.lons[j] - dlon/2.
        e = w + dlon
        s = self.lats[i] - dlon/2.
        n = s + dlat
        top = self.top[index]
        bottom = self.bottom[index]
        props = {}
        for p in self.props:
            props[p] = self.props[p][index]
        cell = Tesseroid(w, e, s, n, top, bottom, props)
        return cell

    def copy(self, deep=False):
        if deep:
            other = copy.deepcopy(self)
        else:
            other = copy.copy(self)
        return other
