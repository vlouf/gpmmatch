"""
Various utilities for correction and conversion of satellite data.

@title: correct
@author: Valentin Louf <valentin.louf@bom.gov.au>
@institutions: Monash University and the Australian Bureau of Meteorology
@creation: 17/02/2020
@date: 29/05/2025

.. autosummary::
    :toctree: generated/

    compute_gaussian_curvature
    convert_sat_refl_to_gr_band
    correct_attenuation
    correct_parallax
    get_offset
    grid_displacement
"""

from typing import Tuple
import numpy as np
import xarray as xr


def compute_gaussian_curvature(lat0: float) -> float:
    """
    Determine the Earth's Gaussian radius of curvature at the radar
    https://en.wikipedia.org/wiki/Earth_radius#Radii_of_curvature

    Parameter:
    ----------
    lat0: float
        Ground radar latitude.

    Returns:
    --------
    ae: float
        Scaled Gaussian radius.
    """
    # Major and minor radii of the Ellipsoid
    a = 6378137.0  # Earth radius in meters
    e2 = 0.0066943800
    b = a * np.sqrt(1 - e2)

    tmp = (a * np.cos(np.pi / 180 * lat0)) ** 2 + (b * np.sin(np.pi / 180 * lat0)) ** 2  # Denominator
    an = (a**2) / np.sqrt(tmp)  # Radius of curvature in the prime vertical (east–west direction)
    am = (a * b) ** 2 / tmp**1.5  # Radius of curvature in the north–south meridian
    ag = np.sqrt(an * am)  # Earth's Gaussian radius of curvature
    ae = (4 / 3.0) * ag

    return ae


def convert_gpmrefl_grband_dfr(refl_gpm: np.ndarray, radar_band: str) -> np.ndarray:
    """
    Convert GPM reflectivity to ground radar band using the DFR relationship
    found in Louf et al. (2019) paper.

    Parameters:
    ===========
    refl_gpm:
        Satellite reflectivity field.
    radar_band: str
        Possible values are 'S', 'C', or 'X'

    Return:
    =======
    refl:
        Reflectivity conversion from Ku-band to ground radar band
    """
    if radar_band == "S":
        cof = np.array([2.01236803e-07, -6.50694273e-06, 1.10885533e-03, -6.47985914e-02, -7.46518423e-02])
        dfr = np.poly1d(cof)
    elif radar_band == "C":
        cof = np.array([1.21547932e-06, -1.23266138e-04, 6.38562875e-03, -1.52248868e-01, 5.33556919e-01])
        dfr = np.poly1d(cof)
    elif radar_band == "X":
        # Use of C band DFR relationship multiply by ratio
        cof = np.array([1.21547932e-06, -1.23266138e-04, 6.38562875e-03, -1.52248868e-01, 5.33556919e-01])
        dfr = 3.2 / 5.5 * np.poly1d(cof)
    else:
        raise ValueError(f"Radar reflectivity band ({radar_band}) not supported.")

    return refl_gpm + dfr(refl_gpm)


def correct_attenuation(reflectivity: np.ndarray, radar_band: str) -> np.ndarray:
    """
    Correct from C- or X-Band attenuation using a Z-A relationship derived from
    T-matrix calculations using the Meteor disdrometer.

    Parameters:
    ===========
    reflectivity: ndarray
        Input attenuated reflectivity

    Returns:
    ========
    corr_refl: ndarray
        Attenuation-corrected reflectivity.
    """
    ze = 10 ** (reflectivity / 10)
    if radar_band == "X":
        atten = 3.30240183e-6 * ze + 9.67774379e-2
    elif radar_band == "C":
        atten = 1.31885e-6 * ze + 1.8041e-3
    else:
        # Doesnt correct.
        return reflectivity

    corr_refl = reflectivity + 2 * np.cumsum(atten, axis=1)
    return corr_refl


def correct_parallax(
    sr_x: np.ndarray, sr_y: np.ndarray, gpmset: xr.Dataset
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Adjust the geo-locations of the SR pixels. The `sr_xy` coordinates of the
    SR beam footprints need to be in the azimuthal equidistant projection of
    the ground radar. This ensures that the ground radar is fixed at
    xy-coordinate (0, 0), and every SR bin has its relative xy-coordinates
    with respect to the ground radar site.

    Parameters:
    -----------
    sr_x: ndarray
        Array of x-coordinates of shape (nscans, nbeams)
    sr_y: ndarray
        Array of y-coordinates of shape (nscans, nbeams)
    gpmset: xarray

    Returns:
    --------
    sr_xyp : :class:`numpy:numpy.ndarray`
        Array of parallax corrected coordinates
        of shape (nscans, nbeams, nbins).
    z_sr : :class:`numpy:numpy.ndarray`
        Array of SR bin altitudes of shape (nscans, nbeams, nbins).
    """
    r_sr_inv, alpha = gpmset.nbin.values, gpmset.nray.values
    # calculate height of bin
    z = r_sr_inv * np.cos(np.deg2rad(alpha))[..., np.newaxis]
    z_sr = np.repeat(z[np.newaxis, :, :], len(gpmset.nscan), axis=0)
    # calculate bin ground xy-displacement length
    ds = r_sr_inv * np.sin(np.deg2rad(alpha))[..., np.newaxis]

    # calculate x,y-differences between ground coordinate
    # and center ground coordinate [25th element]
    center = int(np.floor(len(sr_x[-1]) / 2.0))
    xdiff = sr_x - sr_x[:, center][:, np.newaxis]
    ydiff = sr_y - sr_y[:, center][:, np.newaxis]

    # assuming ydiff and xdiff being a triangles adjacent and
    # opposite this calculates the xy-angle of the SR scan
    ang = np.arctan2(ydiff, xdiff)

    # calculate displacement dx, dy from displacement length
    dx = ds * np.cos(ang)[..., np.newaxis]
    dy = ds * np.sin(ang)[..., np.newaxis]

    # subtract displacement from SR ground coordinates
    sr_xp = sr_x[..., np.newaxis] - dx
    sr_yp = sr_y[..., np.newaxis] - dy

    return sr_xp, sr_yp, z_sr


def get_offset(matchset: xr.Dataset, dr: float, nbins: int = 200) -> float:
    """
    Compute the Offset between GR and GPM. It will try to compute the mode of
    the distribution and if it fails, then it will use the mean.

    Parameter:
    ==========
    matchset: xr.Dataset
        Dataset of volume matching.
    dr: float
        Ground radar gate spacing (m).
    nbins: int
        Defines the number of equal-width bins in the distribution.

    Returns:
    ========
    offset: float
        Offset between GR and GPM
    """
    offset = np.arange(-15, 15, 0.2, dtype=np.float64)
    area = np.zeros_like(offset)

    refl_gpm = matchset.refl_gpm_grband.values.flatten().copy()
    refl_gr = matchset.refl_gr_weigthed.values.flatten().copy()
    fmin = matchset.fmin_gr.values.flatten().copy()

    pos = (refl_gpm > 36) | (refl_gr > 36) | (fmin != 1)
    if (~pos).sum() == 0:  # Relaxing fmin parameter.
        pos = (refl_gpm > 36) | (refl_gr > 36) | (fmin < 0.9)
    if (~pos).sum() == 0:
        pos = (refl_gpm > 36) | (refl_gr > 36) | (fmin < 0.7)

    refl_gpm[pos] = np.nan
    refl_gr[pos] = np.nan

    pdf_gpm, _ = np.histogram(refl_gpm, range=(0, 50), bins=nbins, density=True)
    for idx, a in enumerate(offset):
        pdf_gr, _ = np.histogram(refl_gr - a, range=(0, 50), bins=nbins, density=True)
        diff = np.min([pdf_gr, pdf_gpm], axis=0)
        area[idx] = np.sum(diff)

    smoothed_area = np.convolve([1] * 12, area, "same")
    maxpos = np.argmax(smoothed_area)
    gr_offset = offset[maxpos]
    return gr_offset


def grid_displacement(field1: np.ndarray, field2: np.ndarray) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """
    Calculate the grid displacement using Phase correlation.
    http://en.wikipedia.org/wiki/Phase_correlation

    Parameters:
    -----------
    field1, field2 : ndarray
       Fields separated in time.

    Returns:
    --------
    displacement : two-tuple
         integers if pixels, otherwise floats. Result of the calculation
    """
    # create copies of the data
    ige1 = np.ma.masked_invalid(10 ** (field1 / 10)).filled(0)
    ige2 = np.ma.masked_invalid(10 ** (field2 / 10)).filled(0)

    # discrete fast fourier transformation and complex conjugation of image 2
    image1FFT = np.fft.fft2(ige1)
    image2FFT = np.conjugate(np.fft.fft2(ige2))

    # inverse fourier transformation of product -> equal to cross correlation
    imageCCor = np.real(np.fft.ifft2((image1FFT * image2FFT)))

    # Shift the zero-frequency component to the center of the spectrum
    imageCCorShift = np.fft.fftshift(imageCCor)
    row, col = ige1.shape

    # find the peak in the correlation
    yShift, xShift = np.unravel_index(np.argmax(imageCCorShift), (row, col))
    yShift -= row // 2
    xShift -= col // 2

    return (xShift, yShift)
