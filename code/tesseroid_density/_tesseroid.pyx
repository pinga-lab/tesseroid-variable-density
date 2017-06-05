"""
Cython kernels for the fatiando.gravmag.tesseroid module.

Used to optimize some slow tasks and compute the actual gravitational fields.
"""
from __future__ import division
import numpy

from fatiando import constants

from libc.math cimport sin, cos, sqrt, acos
# Import Cython definitions for numpy
cimport numpy
cimport cython

# To calculate sin and cos simultaneously
cdef extern from "math.h":
    void sincos(double x, double* sinx, double* cosx)

cdef:
    double d2r = numpy.pi/180.
    double[::1] nodes
    double MEAN_EARTH_RADIUS = constants.MEAN_EARTH_RADIUS
    ctypedef double (*kernel_func)(double, double, double, double, double,
                                   object, double[::1], double[::1],
                                   double[::1], double[::1])
nodes = numpy.array([-0.577350269189625731058868041146,
                     0.577350269189625731058868041146])


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef rediscretizer(
    numpy.ndarray[double, ndim=1] bounds,
    object density,
    double ratio,
    int STACK_SIZE,
    numpy.ndarray[double, ndim=1] lons,
    numpy.ndarray[double, ndim=1] sinlats,
    numpy.ndarray[double, ndim=1] coslats,
    numpy.ndarray[double, ndim=1] radii,
    numpy.ndarray[double, ndim=1] result,
    kernel_func kernel):
    """
    Calculate the given kernel function on the computation points by
    rediscretizing the tesseroid when needed.
    """
    cdef:
        unsigned int i, l, size
        double[::1] lonc =  numpy.empty(2, numpy.float)
        double[::1] sinlatc =  numpy.empty(2, numpy.float)
        double[::1] coslatc =  numpy.empty(2, numpy.float)
        double[::1] rc =  numpy.empty(2, numpy.float)
        double scale
        int nlon, nlat, nr, new_cells
        int stktop, error, error_code
        double[:, ::1] stack =  numpy.empty((STACK_SIZE, 6), numpy.float)
        double w, e, s, n, top, bottom
        double lon, sinlat, coslat, radius
        double Llon, Llat, Lr, distance
        double res
    size = len(result)
    error_code = 0
    for l in range(size):
        lon = lons[l]
        sinlat = sinlats[l]
        coslat = coslats[l]
        radius = radii[l]
        res = 0
        for i in range(6):
            stack[0, i] = bounds[i]
        stktop = 0
        while stktop >= 0:
            w = stack[stktop, 0]
            e = stack[stktop, 1]
            s = stack[stktop, 2]
            n = stack[stktop, 3]
            top = stack[stktop, 4]
            bottom = stack[stktop, 5]
            stktop -= 1
            distance = distance_n_size(w, e, s, n, top, bottom, lon, sinlat,
                                       coslat, radius, &Llon, &Llat, &Lr)
            # Check which dimensions I have to divide
            error = divisions(distance, Llon, Llat, Lr, ratio, &nlon,
                              &nlat, &nr, &new_cells)
            error_code += error
            if new_cells > 1:
                if stktop + nlon*nlat*nr > STACK_SIZE:
                    raise ValueError('Tesseroid stack overflow')
                stktop = split(w, e, s, n, top, bottom, nlon, nlat, nr,
                               stack, stktop)
            else:
                # Put the nodes in the current range
                scale = scale_nodes(w, e, s, n, top, bottom, lonc, sinlatc,
                                    coslatc, rc)
                res += kernel(lon, sinlat, coslat, radius, scale, density, 
                              lonc, sinlatc, coslatc, rc)
        result[l] += res
    return error_code


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef inline double distance_n_size(
    double w, double e, double s, double n, double top, double bottom,
    double lon, double sinlat, double coslat, double radius,
    double* Llon, double* Llat, double* Lr):
    cdef:
        double rt, rtop, lont, latt, sinlatt, coslatt, cospsi, distance
    # Calculate the distance to the observation point
    rt = 0.5*(top + bottom) + MEAN_EARTH_RADIUS
    lont = d2r*0.5*(w + e)
    latt = d2r*0.5*(s + n)
    sinlatt = sin(latt)
    coslatt = cos(latt)
    cospsi = sinlat*sinlatt + coslat*coslatt*cos(lon - lont)
    distance = sqrt(radius**2 + rt**2 - 2*radius*rt*cospsi)
    # Calculate the dimensions of the tesseroid in meters
    rtop = top + MEAN_EARTH_RADIUS
    Llon[0] = rtop*acos(sinlatt**2 + (coslatt**2)*cos(d2r*(e - w)))
    Llat[0] = rtop*acos(sin(d2r*n)*sin(d2r*s) + cos(d2r*n)*cos(d2r*s))
    Lr[0] = top - bottom
    return distance


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef inline int divisions(double distance, double Llon, double Llat, double Lr,
                          double ratio, int* nlon, int* nlat, int* nr,
                          int* new_cells):
    "How many divisions should be made per dimension"
    nlon[0] = 1
    nlat[0] = 1
    nr[0] = 1
    error = 0
    if distance <= ratio*Llon:
        if Llon <= 0.1:  # in meters. ~1e-6  degrees
            error = -1
        else:
            nlon[0] = 2
    if distance <= ratio*Llat:
        if Llat <= 0.1:  # in meters. ~1e-6  degrees
            error = -1
        else:
            nlat[0] = 2
    if distance <= ratio*Lr:
        if Lr <= 1e3:
            error = -1
        else:
            nr[0] = 2
    new_cells[0] = nlon[0]*nlat[0]*nr[0]
    return error


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef inline double scale_nodes(
    double w, double e, double s, double n, double top, double bottom,
    double[::1] lonc,
    double[::1] sinlatc,
    double[::1] coslatc,
    double[::1] rc):
    "Put GLQ nodes in the integration limits for a tesseroid"
    cdef:
        double dlon, dlat, dr, mlon, mlat, mr, latc, scale
        unsigned int i
    dlon = e - w
    dlat = n - s
    dr = top - bottom
    mlon = 0.5*(e + w)
    mlat = 0.5*(n + s)
    mr = 0.5*(top + bottom + 2.*MEAN_EARTH_RADIUS)
    # Scale the GLQ nodes to the integration limits
    for i in range(2):
        lonc[i] = d2r*(0.5*dlon*nodes[i] + mlon)
        latc = d2r*(0.5*dlat*nodes[i] + mlat)
        sinlatc[i] = sin(latc)
        coslatc[i] = cos(latc)
        rc[i] = (0.5*dr*nodes[i] + mr)
    scale = d2r*dlon*d2r*dlat*dr*0.125
    return scale


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef int split(double w, double e, double s, double n, double top,
               double bottom, int nlon, int nlat, int nr,
               double[:, ::1] stack, int stktop):
    cdef:
        unsigned int i, j, k
        double dlon, dlat, dr
    dlon = (e - w)/nlon
    dlat = (n - s)/nlat
    dr = (top - bottom)/nr
    for i in xrange(nlon):
        for j in xrange(nlat):
            for k in xrange(nr):
                stktop += 1
                stack[stktop, 0] = w + i*dlon
                stack[stktop, 1] = w + (i + 1)*dlon
                stack[stktop, 2] = s + j*dlat
                stack[stktop, 3] = s + (j + 1)*dlat
                stack[stktop, 4] = bottom + (k + 1)*dr
                stack[stktop, 5] = bottom + k*dr
    return stktop


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def potential(
    numpy.ndarray[double, ndim=1] bounds,
    object density,
    double ratio,
    int STACK_SIZE,
    numpy.ndarray[double, ndim=1] lons,
    numpy.ndarray[double, ndim=1] sinlats,
    numpy.ndarray[double, ndim=1] coslats,
    numpy.ndarray[double, ndim=1] radii,
    numpy.ndarray[double, ndim=1] result):
    """
    Calculate this gravity field of a tesseroid at given locations.
    """
    cdef int error
    if callable(density):
        error = rediscretizer(bounds, density, ratio, STACK_SIZE, lons,
                              sinlats, coslats, radii, result,
                              kernelV_variable)
    else:
        error = rediscretizer(bounds, density, ratio, STACK_SIZE, lons,
                              sinlats, coslats, radii, result, kernelV)
    return error


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef inline double kernelV(
    double lon, double sinlat, double coslat, double radius, double scale,
    object density, double[::1] lonc, double[::1] sinlatc, double[::1] coslatc,
    double[::1] rc):
    cdef:
        unsigned int i, j, k
        double kappa, r_sqr, coslon, l_sqr, cospsi
        double result, rc_sqr
    r_sqr = radius**2
    result = 0
    for i in range(2):
        coslon = cos(lon - lonc[i])
        for j in range(2):
            cospsi = sinlat*sinlatc[j] + coslat*coslatc[j]*coslon
            for k in range(2):
                rc_sqr = rc[k]**2
                l_sqr = r_sqr + rc_sqr - 2*radius*rc[k]*cospsi
                kappa = rc_sqr*coslatc[j]
                result += kappa/sqrt(l_sqr)
    return result*scale*density


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef inline double kernelV_variable(
    double lon, double sinlat, double coslat, double radius, double scale,
    object density, double[::1] lonc, double[::1] sinlatc, double[::1] coslatc,
    double[::1] rc):
    cdef:
        unsigned int i, j, k
        double kappa, r_sqr, coslon, l_sqr, cospsi
        double result, rc_sqr, dens
    r_sqr = radius**2
    result = 0
    for i in range(2):
        coslon = cos(lon - lonc[i])
        for j in range(2):
            cospsi = sinlat*sinlatc[j] + coslat*coslatc[j]*coslon
            for k in range(2):
                rc_sqr = rc[k]**2
                l_sqr = r_sqr + rc_sqr - 2*radius*rc[k]*cospsi
                kappa = rc_sqr*coslatc[j]
                dens = density(rc[k] - MEAN_EARTH_RADIUS)
                result += dens*kappa/sqrt(l_sqr)
    return result*scale


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def gx(
    numpy.ndarray[double, ndim=1] bounds,
    object density,
    double ratio,
    int STACK_SIZE,
    numpy.ndarray[double, ndim=1] lons,
    numpy.ndarray[double, ndim=1] sinlats,
    numpy.ndarray[double, ndim=1] coslats,
    numpy.ndarray[double, ndim=1] radii,
    numpy.ndarray[double, ndim=1] result):
    """
    Calculate this gravity field of a tesseroid at given locations.
    """
    cdef int error
    if callable(density):
        error = rediscretizer(bounds, density, ratio, STACK_SIZE, lons,
                              sinlats, coslats, radii, result,
                              kernelx_variable)
    else:
        error = rediscretizer(bounds, density, ratio, STACK_SIZE, lons,
                              sinlats, coslats, radii, result, kernelx)
    return error


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef inline double kernelx(
    double lon, double sinlat, double coslat, double radius, double scale,
    object density, double[::1] lonc, double[::1] sinlatc, double[::1] coslatc,
    double[::1] rc):
    cdef:
        unsigned int i, j, k
        double kappa, r_sqr, coslon, kphi, l_sqr, cospsi
        double result, rc_sqr
    r_sqr = radius**2
    result = 0
    for i in range(2):
        coslon = cos(lon - lonc[i])
        for j in range(2):
            kphi = coslat*sinlatc[j] - sinlat*coslatc[j]*coslon
            cospsi = sinlat*sinlatc[j] + coslat*coslatc[j]*coslon
            for k in range(2):
                rc_sqr = rc[k]**2
                l_sqr = r_sqr + rc_sqr - 2*radius*rc[k]*cospsi
                kappa = rc_sqr*coslatc[j]
                result += kappa*rc[k]*kphi/(l_sqr**1.5)
    return result*scale*density


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef inline double kernelx_variable(
    double lon, double sinlat, double coslat, double radius, double scale,
    object density, double[::1] lonc, double[::1] sinlatc, double[::1] coslatc,
    double[::1] rc):
    cdef:
        unsigned int i, j, k
        double kappa, r_sqr, coslon, kphi, l_sqr, cospsi
        double result, rc_sqr, dens
    r_sqr = radius**2
    result = 0
    for i in range(2):
        coslon = cos(lon - lonc[i])
        for j in range(2):
            kphi = coslat*sinlatc[j] - sinlat*coslatc[j]*coslon
            cospsi = sinlat*sinlatc[j] + coslat*coslatc[j]*coslon
            for k in range(2):
                rc_sqr = rc[k]**2
                l_sqr = r_sqr + rc_sqr - 2*radius*rc[k]*cospsi
                kappa = rc_sqr*coslatc[j]
                dens = density(rc[k] - MEAN_EARTH_RADIUS)
                result += dens*kappa*rc[k]*kphi/(l_sqr**1.5)
    return result*scale


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def gy(
    numpy.ndarray[double, ndim=1] bounds,
    object density,
    double ratio,
    int STACK_SIZE,
    numpy.ndarray[double, ndim=1] lons,
    numpy.ndarray[double, ndim=1] sinlats,
    numpy.ndarray[double, ndim=1] coslats,
    numpy.ndarray[double, ndim=1] radii,
    numpy.ndarray[double, ndim=1] result):
    """
    Calculate this gravity field of a tesseroid at given locations.
    """
    cdef int error
    if callable(density):
        error = rediscretizer(bounds, density, ratio, STACK_SIZE, lons,
                              sinlats, coslats, radii, result,
                              kernely_variable)
    else:
        error = rediscretizer(bounds, density, ratio, STACK_SIZE, lons,
                              sinlats, coslats, radii, result, kernely)
    return error


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef inline double kernely(
    double lon, double sinlat, double coslat, double radius, double scale,
    object density, double[::1] lonc, double[::1] sinlatc, double[::1] coslatc,
    double[::1] rc):
    cdef:
        unsigned int i, j, k
        double kappa, r_sqr, coslon, sinlon, l_sqr, cospsi
        double result, rc_sqr
    r_sqr = radius**2
    result = 0
    for i in range(2):
        coslon = cos(lon - lonc[i])
        sinlon = sin(lonc[i] - lon)
        for j in range(2):
            cospsi = sinlat*sinlatc[j] + coslat*coslatc[j]*coslon
            for k in range(2):
                rc_sqr = rc[k]**2
                l_sqr = r_sqr + rc_sqr - 2*radius*rc[k]*cospsi
                kappa = rc_sqr*coslatc[j]
                result += kappa*(rc[k]*coslatc[j]*sinlon/(l_sqr**1.5))
    return result*scale*density


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef inline double kernely_variable(
    double lon, double sinlat, double coslat, double radius, double scale,
    object density, double[::1] lonc, double[::1] sinlatc, double[::1] coslatc,
    double[::1] rc):
    cdef:
        unsigned int i, j, k
        double kappa, r_sqr, coslon, sinlon, l_sqr, cospsi
        double result, rc_sqr, dens
    r_sqr = radius**2
    result = 0
    for i in range(2):
        coslon = cos(lon - lonc[i])
        sinlon = sin(lonc[i] - lon)
        for j in range(2):
            cospsi = sinlat*sinlatc[j] + coslat*coslatc[j]*coslon
            for k in range(2):
                rc_sqr = rc[k]**2
                l_sqr = r_sqr + rc_sqr - 2*radius*rc[k]*cospsi
                kappa = rc_sqr*coslatc[j]
                dens = density(rc[k] - MEAN_EARTH_RADIUS)
                result += dens*kappa*(rc[k]*coslatc[j]*sinlon/(l_sqr**1.5))
    return result*scale


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def gz(
    numpy.ndarray[double, ndim=1] bounds,
    object density,
    double ratio,
    int STACK_SIZE,
    numpy.ndarray[double, ndim=1] lons,
    numpy.ndarray[double, ndim=1] sinlats,
    numpy.ndarray[double, ndim=1] coslats,
    numpy.ndarray[double, ndim=1] radii,
    numpy.ndarray[double, ndim=1] result):
    """
    Calculate this gravity field of a tesseroid at given locations.
    """
    cdef int error
    if callable(density):
        error = rediscretizer(bounds, density, ratio, STACK_SIZE, lons,
                              sinlats, coslats, radii, result,
                              kernelz_variable)
    else:
        error = rediscretizer(bounds, density, ratio, STACK_SIZE, lons,
                              sinlats, coslats, radii, result, kernelz)
    return error


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef inline double kernelz(
    double lon, double sinlat, double coslat, double radius, double scale,
    object density, double[::1] lonc, double[::1] sinlatc, double[::1] coslatc,
    double[::1] rc):
    cdef:
        unsigned int i, j, k
        double kappa, r_sqr, coslon, l_sqr, cospsi
        double result, rc_sqr
    r_sqr = radius**2
    result = 0
    for i in range(2):
        coslon = cos(lon - lonc[i])
        for j in range(2):
            cospsi = sinlat*sinlatc[j] + coslat*coslatc[j]*coslon
            for k in range(2):
                rc_sqr = rc[k]**2
                l_sqr = r_sqr + rc_sqr - 2*radius*rc[k]*cospsi
                kappa = rc_sqr*coslatc[j]
                result += kappa*(rc[k]*cospsi - radius)/(l_sqr**1.5)
    # Multiply by -1 so that z is pointing down for gz and the gravity anomaly
    # doesn't look inverted (ie, negative for positive density)
    result *= -1
    return result*scale*density


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef inline double kernelz_variable(
    double lon, double sinlat, double coslat, double radius, double scale,
    object density, double[::1] lonc, double[::1] sinlatc, double[::1] coslatc,
    double[::1] rc):
    cdef:
        unsigned int i, j, k
        double kappa, r_sqr, coslon, l_sqr, cospsi
        double result, rc_sqr, dens
    r_sqr = radius**2
    result = 0
    for i in range(2):
        coslon = cos(lon - lonc[i])
        for j in range(2):
            cospsi = sinlat*sinlatc[j] + coslat*coslatc[j]*coslon
            for k in range(2):
                rc_sqr = rc[k]**2
                l_sqr = r_sqr + rc_sqr - 2*radius*rc[k]*cospsi
                kappa = rc_sqr*coslatc[j]
                dens = density(rc[k] - MEAN_EARTH_RADIUS)
                result += dens*kappa*(rc[k]*cospsi - radius)/(l_sqr**1.5)
    # Multiply by -1 so that z is pointing down for gz and the gravity anomaly
    # doesn't look inverted (ie, negative for positive density)
    result *= -1
    return result*scale


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def gxx(
    numpy.ndarray[double, ndim=1] bounds,
    object density,
    double ratio,
    int STACK_SIZE,
    numpy.ndarray[double, ndim=1] lons,
    numpy.ndarray[double, ndim=1] sinlats,
    numpy.ndarray[double, ndim=1] coslats,
    numpy.ndarray[double, ndim=1] radii,
    numpy.ndarray[double, ndim=1] result):
    """
    Calculate this gravity field of a tesseroid at given locations.
    """
    cdef int error
    if callable(density):
        error = rediscretizer(bounds, density, ratio, STACK_SIZE, lons,
                              sinlats, coslats, radii, result,
                              kernelxx_variable)
    else:
        error = rediscretizer(bounds, density, ratio, STACK_SIZE, lons,
                              sinlats, coslats, radii, result, kernelxx)
    return error


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef inline double kernelxx(
    double lon, double sinlat, double coslat, double radius, double scale,
    object density, double[::1] lonc, double[::1] sinlatc, double[::1] coslatc,
    double[::1] rc):
    cdef:
        unsigned int i, j, k
        double kappa, r_sqr, coslon, l_sqr, cospsi, kphi
        double result, rc_sqr
    r_sqr = radius**2
    result = 0
    for i in range(2):
        coslon = cos(lon - lonc[i])
        for j in range(2):
            kphi = coslat*sinlatc[j] - sinlat*coslatc[j]*coslon
            cospsi = sinlat*sinlatc[j] + coslat*coslatc[j]*coslon
            for k in range(2):
                rc_sqr = rc[k]**2
                l_sqr = r_sqr + rc_sqr - 2*radius*rc[k]*cospsi
                kappa = rc_sqr*coslatc[j]
                result += kappa*(3*((rc[k]*kphi)**2) - l_sqr)/(l_sqr**2.5)
    return result*scale*density


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef inline double kernelxx_variable(
    double lon, double sinlat, double coslat, double radius, double scale,
    object density, double[::1] lonc, double[::1] sinlatc, double[::1] coslatc,
    double[::1] rc):
    cdef:
        unsigned int i, j, k
        double kappa, r_sqr, coslon, l_sqr, cospsi, kphi
        double result, rc_sqr, dens
    r_sqr = radius**2
    result = 0
    for i in range(2):
        coslon = cos(lon - lonc[i])
        for j in range(2):
            kphi = coslat*sinlatc[j] - sinlat*coslatc[j]*coslon
            cospsi = sinlat*sinlatc[j] + coslat*coslatc[j]*coslon
            for k in range(2):
                rc_sqr = rc[k]**2
                l_sqr = r_sqr + rc_sqr - 2*radius*rc[k]*cospsi
                kappa = rc_sqr*coslatc[j]
                dens = density(rc[k] - MEAN_EARTH_RADIUS)
                result += dens*kappa*(3*((rc[k]*kphi)**2) - l_sqr)/(l_sqr**2.5)
    return result*scale


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def gxy(
    numpy.ndarray[double, ndim=1] bounds,
    object density,
    double ratio,
    int STACK_SIZE,
    numpy.ndarray[double, ndim=1] lons,
    numpy.ndarray[double, ndim=1] sinlats,
    numpy.ndarray[double, ndim=1] coslats,
    numpy.ndarray[double, ndim=1] radii,
    numpy.ndarray[double, ndim=1] result):
    """
    Calculate this gravity field of a tesseroid at given locations.
    """
    cdef int error
    if callable(density):
        error = rediscretizer(bounds, density, ratio, STACK_SIZE, lons,
                              sinlats, coslats, radii, result,
                              kernelxy_variable)
    else:
        error = rediscretizer(bounds, density, ratio, STACK_SIZE, lons,
                              sinlats, coslats, radii, result, kernelxy)
    return error


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef inline double kernelxy(
    double lon, double sinlat, double coslat, double radius, double scale,
    object density, double[::1] lonc, double[::1] sinlatc, double[::1] coslatc,
    double[::1] rc):
    cdef:
        unsigned int i, j, k
        double kappa, r_sqr, rc_sqr, coslon, sinlon, l_sqr, cospsi, kphi
        double result
    r_sqr = radius**2
    result = 0
    for i in range(2):
        sincos(lonc[i] - lon, &sinlon, &coslon)
        for j in range(2):
            kphi = coslat*sinlatc[j] - sinlat*coslatc[j]*coslon
            cospsi = sinlat*sinlatc[j] + coslat*coslatc[j]*coslon
            for k in range(2):
                rc_sqr = rc[k]**2
                l_sqr = r_sqr + rc_sqr - 2*radius*rc[k]*cospsi
                kappa = rc_sqr*coslatc[j]
                result += kappa*3*rc_sqr*kphi*coslatc[j]*sinlon/(l_sqr**2.5)
    return result*scale*density


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef inline double kernelxy_variable(
    double lon, double sinlat, double coslat, double radius, double scale,
    object density, double[::1] lonc, double[::1] sinlatc, double[::1] coslatc,
    double[::1] rc):
    cdef:
        unsigned int i, j, k
        double kappa, r_sqr, rc_sqr, coslon, sinlon, l_sqr, cospsi, kphi
        double result, dens
    r_sqr = radius**2
    result = 0
    for i in range(2):
        sincos(lonc[i] - lon, &sinlon, &coslon)
        for j in range(2):
            kphi = coslat*sinlatc[j] - sinlat*coslatc[j]*coslon
            cospsi = sinlat*sinlatc[j] + coslat*coslatc[j]*coslon
            for k in range(2):
                rc_sqr = rc[k]**2
                l_sqr = r_sqr + rc_sqr - 2*radius*rc[k]*cospsi
                kappa = rc_sqr*coslatc[j]
                dens = density(rc[k] - MEAN_EARTH_RADIUS)
                result += dens*kappa * \
                    3*rc_sqr*kphi*coslatc[j]*sinlon/(l_sqr**2.5)
    return result*scale


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def gxz(
    numpy.ndarray[double, ndim=1] bounds,
    object density,
    double ratio,
    int STACK_SIZE,
    numpy.ndarray[double, ndim=1] lons,
    numpy.ndarray[double, ndim=1] sinlats,
    numpy.ndarray[double, ndim=1] coslats,
    numpy.ndarray[double, ndim=1] radii,
    numpy.ndarray[double, ndim=1] result):
    """
    Calculate this gravity field of a tesseroid at given locations.
    """
    cdef int error
    if callable(density):
        error = rediscretizer(bounds, density, ratio, STACK_SIZE, lons,
                              sinlats, coslats, radii, result,
                              kernelxz_variable)
    else:
        error = rediscretizer(bounds, density, ratio, STACK_SIZE, lons,
                              sinlats, coslats, radii, result, kernelxz)
    return error


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef inline double kernelxz(
    double lon, double sinlat, double coslat, double radius, double scale,
    object density, double[::1] lonc, double[::1] sinlatc, double[::1] coslatc,
    double[::1] rc):
    cdef:
        unsigned int i, j, k
        double kappa, r_sqr, rc_sqr, coslon, l_5, cospsi, kphi
        double result
    r_sqr = radius**2
    result = 0
    for i in range(2):
        coslon = cos(lon - lonc[i])
        for j in range(2):
            kphi = coslat*sinlatc[j] - sinlat*coslatc[j]*coslon
            cospsi = sinlat*sinlatc[j] + coslat*coslatc[j]*coslon
            for k in range(2):
                rc_sqr = rc[k]**2
                l_5 = (r_sqr + rc_sqr - 2*radius*rc[k]*cospsi)**2.5
                kappa = rc_sqr*coslatc[j]
                result += kappa*3*rc[k]*kphi*(rc[k]*cospsi - radius)/l_5
    return result*scale*density


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef inline double kernelxz_variable(
    double lon, double sinlat, double coslat, double radius, double scale,
    object density, double[::1] lonc, double[::1] sinlatc, double[::1] coslatc,
    double[::1] rc):
    cdef:
        unsigned int i, j, k
        double kappa, r_sqr, rc_sqr, coslon, l_5, cospsi, kphi
        double result, dens
    r_sqr = radius**2
    result = 0
    for i in range(2):
        coslon = cos(lon - lonc[i])
        for j in range(2):
            kphi = coslat*sinlatc[j] - sinlat*coslatc[j]*coslon
            cospsi = sinlat*sinlatc[j] + coslat*coslatc[j]*coslon
            for k in range(2):
                rc_sqr = rc[k]**2
                l_5 = (r_sqr + rc_sqr - 2*radius*rc[k]*cospsi)**2.5
                kappa = rc_sqr*coslatc[j]
                dens = density(rc[k] - MEAN_EARTH_RADIUS)
                result += dens*kappa*3*rc[k]*kphi*(rc[k]*cospsi - radius)/l_5
    return result*scale


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def gyy(
    numpy.ndarray[double, ndim=1] bounds,
    object density,
    double ratio,
    int STACK_SIZE,
    numpy.ndarray[double, ndim=1] lons,
    numpy.ndarray[double, ndim=1] sinlats,
    numpy.ndarray[double, ndim=1] coslats,
    numpy.ndarray[double, ndim=1] radii,
    numpy.ndarray[double, ndim=1] result):
    """
    Calculate this gravity field of a tesseroid at given locations.
    """
    cdef int error
    if callable(density):
        error = rediscretizer(bounds, density, ratio, STACK_SIZE, lons,
                              sinlats, coslats, radii, result,
                              kernelyy_variable)
    else:
        error = rediscretizer(bounds, density, ratio, STACK_SIZE, lons,
                              sinlats, coslats, radii, result, kernelyy)
    return error


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef inline double kernelyy(
    double lon, double sinlat, double coslat, double radius, double scale,
    object density, double[::1] lonc, double[::1] sinlatc, double[::1] coslatc,
    double[::1] rc):
    cdef:
        unsigned int i, j, k
        double kappa, r_sqr, rc_sqr, coslon, sinlon, l_sqr, cospsi, deltay
        double result
    r_sqr = radius**2
    result = 0
    for i in range(2):
        sincos(lonc[i] - lon, &sinlon, &coslon)
        for j in range(2):
            cospsi = sinlat*sinlatc[j] + coslat*coslatc[j]*coslon
            for k in range(2):
                rc_sqr = rc[k]**2
                l_sqr = r_sqr + rc_sqr - 2*radius*rc[k]*cospsi
                kappa = rc_sqr*coslatc[j]
                deltay = rc[k]*coslatc[j]*sinlon
                result += kappa*(3*(deltay**2) - l_sqr)/(l_sqr**2.5)
    return result*scale*density


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef inline double kernelyy_variable(
    double lon, double sinlat, double coslat, double radius, double scale,
    object density, double[::1] lonc, double[::1] sinlatc, double[::1] coslatc,
    double[::1] rc):
    cdef:
        unsigned int i, j, k
        double kappa, r_sqr, rc_sqr, coslon, sinlon, l_sqr, cospsi, deltay
        double result, dens
    r_sqr = radius**2
    result = 0
    for i in range(2):
        sincos(lonc[i] - lon, &sinlon, &coslon)
        for j in range(2):
            cospsi = sinlat*sinlatc[j] + coslat*coslatc[j]*coslon
            for k in range(2):
                rc_sqr = rc[k]**2
                l_sqr = r_sqr + rc_sqr - 2*radius*rc[k]*cospsi
                kappa = rc_sqr*coslatc[j]
                deltay = rc[k]*coslatc[j]*sinlon
                dens = density(rc[k] - MEAN_EARTH_RADIUS)
                result += dens*kappa*(3*(deltay**2) - l_sqr)/(l_sqr**2.5)
    return result*scale


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def gyz(
    numpy.ndarray[double, ndim=1] bounds,
    object density,
    double ratio,
    int STACK_SIZE,
    numpy.ndarray[double, ndim=1] lons,
    numpy.ndarray[double, ndim=1] sinlats,
    numpy.ndarray[double, ndim=1] coslats,
    numpy.ndarray[double, ndim=1] radii,
    numpy.ndarray[double, ndim=1] result):
    """
    Calculate this gravity field of a tesseroid at given locations.
    """
    cdef int error
    if callable(density):
        error = rediscretizer(bounds, density, ratio, STACK_SIZE, lons,
                              sinlats, coslats, radii, result,
                              kernelyz_variable)
    else:
        error = rediscretizer(bounds, density, ratio, STACK_SIZE, lons,
                              sinlats, coslats, radii, result, kernelyz)
    return error


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef inline double kernelyz(
    double lon, double sinlat, double coslat, double radius, double scale,
    object density, double[::1] lonc, double[::1] sinlatc, double[::1] coslatc,
    double[::1] rc):
    cdef:
        unsigned int i, j, k
        double kappa, r_sqr, rc_sqr, coslon, sinlon, l_sqr, cospsi, deltay
        double deltaz, result
    r_sqr = radius**2
    result = 0
    for i in range(2):
        sincos(lonc[i] - lon, &sinlon, &coslon)
        for j in range(2):
            cospsi = sinlat*sinlatc[j] + coslat*coslatc[j]*coslon
            for k in range(2):
                rc_sqr = rc[k]**2
                l_sqr = r_sqr + rc_sqr - 2*radius*rc[k]*cospsi
                kappa = rc_sqr*coslatc[j]
                deltay = rc[k]*coslatc[j]*sinlon
                deltaz = rc[k]*cospsi - radius
                result += kappa*3.*deltay*deltaz/(l_sqr**2.5)
    return result*scale*density


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef inline double kernelyz_variable(
    double lon, double sinlat, double coslat, double radius, double scale,
    object density, double[::1] lonc, double[::1] sinlatc, double[::1] coslatc,
    double[::1] rc):
    cdef:
        unsigned int i, j, k
        double kappa, r_sqr, rc_sqr, coslon, sinlon, l_sqr, cospsi, deltay
        double deltaz, result, dens
    r_sqr = radius**2
    result = 0
    for i in range(2):
        sincos(lonc[i] - lon, &sinlon, &coslon)
        for j in range(2):
            cospsi = sinlat*sinlatc[j] + coslat*coslatc[j]*coslon
            for k in range(2):
                rc_sqr = rc[k]**2
                l_sqr = r_sqr + rc_sqr - 2*radius*rc[k]*cospsi
                kappa = rc_sqr*coslatc[j]
                deltay = rc[k]*coslatc[j]*sinlon
                deltaz = rc[k]*cospsi - radius
                dens = density(rc[k] - MEAN_EARTH_RADIUS)
                result += dens*kappa*3.*deltay*deltaz/(l_sqr**2.5)
    return result*scale


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def gzz(
    numpy.ndarray[double, ndim=1] bounds,
    object density,
    double ratio,
    int STACK_SIZE,
    numpy.ndarray[double, ndim=1] lons,
    numpy.ndarray[double, ndim=1] sinlats,
    numpy.ndarray[double, ndim=1] coslats,
    numpy.ndarray[double, ndim=1] radii,
    numpy.ndarray[double, ndim=1] result):
    """
    Calculate this gravity field of a tesseroid at given locations.
    """
    cdef int error
    if callable(density):
        error = rediscretizer(bounds, density, ratio, STACK_SIZE, lons,
                              sinlats, coslats, radii, result,
                              kernelzz_variable)
    else:
        error = rediscretizer(bounds, density, ratio, STACK_SIZE, lons,
                              sinlats, coslats, radii, result, kernelzz)
    return error


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef inline double kernelzz(
    double lon, double sinlat, double coslat, double radius, double scale,
    object density, double[::1] lonc, double[::1] sinlatc, double[::1] coslatc,
    double[::1] rc):
    cdef:
        unsigned int i, j, k
        double kappa, r_sqr, coslon, rc_sqr, l_sqr, l_5, cospsi, deltaz
        double result
    r_sqr = radius**2
    result = 0
    for i in range(2):
        coslon = cos(lon - lonc[i])
        for j in range(2):
            cospsi = sinlat*sinlatc[j] + coslat*coslatc[j]*coslon
            for k in range(2):
                rc_sqr = rc[k]**2
                l_sqr = r_sqr + rc_sqr - 2*radius*rc[k]*cospsi
                l_5 = l_sqr**2.5
                kappa = rc_sqr*coslatc[j]
                deltaz = rc[k]*cospsi - radius
                result += kappa*(3*deltaz**2 - l_sqr)/l_5
    return result*scale*density


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef inline double kernelzz_variable(
    double lon, double sinlat, double coslat, double radius, double scale,
    object density, double[::1] lonc, double[::1] sinlatc, double[::1] coslatc,
    double[::1] rc):
    cdef:
        unsigned int i, j, k
        double kappa, r_sqr, coslon, rc_sqr, l_sqr, l_5, cospsi, deltaz
        double result, dens
    r_sqr = radius**2
    result = 0
    for i in range(2):
        coslon = cos(lon - lonc[i])
        for j in range(2):
            cospsi = sinlat*sinlatc[j] + coslat*coslatc[j]*coslon
            for k in range(2):
                rc_sqr = rc[k]**2
                l_sqr = r_sqr + rc_sqr - 2*radius*rc[k]*cospsi
                l_5 = l_sqr**2.5
                kappa = rc_sqr*coslatc[j]
                deltaz = rc[k]*cospsi - radius
                dens = density(rc[k] - MEAN_EARTH_RADIUS)
                result += dens*kappa*(3*deltaz**2 - l_sqr)/l_5
    return result*scale
