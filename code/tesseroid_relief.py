# -*- coding: utf-8 -*-
"""
Implements the proposed Moho inversion method.

Defines the classes:

* `TesseroidRelief`: Class to represent a relief undulating around a reference
  level and discretized into tesseroids.

"""
from __future__ import division
from fatiando import gridder
from fatiando.mesher import Tesseroid
import numpy as np
import copy


class TesseroidRelief(object):
    """
    Implements a tesseroid model of an interface undulating around a reference
    level.

    This class behaves like a sequence of 'Tesseroid' objects, so you can pass
    it along to any function that iterates over tesseroids (like the forward
    modeling function of 'fatiando.gravmag.tesseroid').

    Parameters:

    * area : [s, n, w, e]
        The south, north, west, and east limits of the mesh in degrees.
    * shape : (nlat, nlon)
        The number of tesseroids in the latitude and longitude directions. Will
        discretize *area* into this number of tesseroids.
    * relief : 1d-array
        The height-coordinates of the relief
    * reference : float
        The height-coordinate of the reference level.
    * props : dict
        Dictionary with the physical properties of the mesh.

    """

    def __init__(self, area, shape, relief, reference, props=None):
        assert shape[0]*shape[1] == relief.size
        assert len(area) == 4
        assert area[0] < area[1] and area[2] < area[3]
        self.area = area
        self.shape = shape
        s, n, w, e = area
        nlat, nlon = shape
        self.lons = np.linspace(w, e, nlon, endpoint=False)
        self.lats = np.linspace(s, n, nlat, endpoint=False)
        self.lon, self.lat = np.meshgrid(self.lons, self.lats)
        self.spacing = self.lats[1] - self.lats[0], self.lons[1] - self.lons[0]
        self._relief = relief
        self._reference = reference*np.ones_like(relief)
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
        self._top = self.relief.copy()
        self._bottom = self.reference.copy()
        isbelow = self._top <= self.reference
        self._top[isbelow] = self.reference[isbelow]
        self._bottom[isbelow] = self.relief[isbelow]

    @property
    def top(self):
        return self._top

    @property
    def bottom(self):
        return self._bottom

    @property
    def relief(self):
        return self._relief

    @relief.setter
    def relief(self, z):
        assert z.size == self.size
        self._relief = z
        self.set_top_bottom()

    @property
    def reference(self):
        return self._reference

    @reference.setter
    def reference(self, reference):
        self._reference = np.ones_like(self.relief)*reference
        self.set_top_bottom()

    @property
    def size(self):
        return self.relief.size

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
        w = self.lons[j]
        e = w + dlon
        s = self.lats[i]
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
