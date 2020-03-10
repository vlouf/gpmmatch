'''
Various utilities for correction and conversion of satellite data.

@title: correct
@author: Valentin Louf <valentin.louf@bom.gov.au>
@institutions: Monash University and the Australian Bureau of Meteorology
@creation: 17/02/2020
@date: 04/03/2020
    correct_parallax
    convert_sat_refl_to_gr_band
    compute_gaussian_curvature
    grid_displacement
'''
import numpy as np


def correct_parallax(sr_x, sr_y, gpmset):
    '''
    Adjust the geo-locations of the SR pixels.
    The `sr_xy` coordinates of the SR beam footprints need to be in the
    azimuthal equidistant projection of the ground radar. This ensures that the
    ground radar is fixed at xy-coordinate (0, 0), and every SR bin has its
    relative xy-coordinates with respect to the ground radar site.

    Parameters
    ----------
    sr_xy : :class:`numpy:numpy.ndarray`
        Array of xy-coordinates of shape (nscans, nbeams, 2)
    gpmset: xarray

    Returns
    -------
    sr_xyp : :class:`numpy:numpy.ndarray`
        Array of parallax corrected coordinates
        of shape (nscans, nbeams, nbins).
    z_sr : :class:`numpy:numpy.ndarray`
        Array of SR bin altitudes of shape (nscans, nbeams, nbins).
    '''
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


def convert_sat_refl_to_gr_band(refp, zp, zbb, bbwidth, radar_band=None):
    """
    Convert the satellite reflectivity to S, C, or X-band using the Cao et al.
    (2013) method.

    Parameters
    ==========
    refp:
        Satellite reflectivity field.
    zp:
        Altitude.
    zbb:
        Bright band height.
    bbwidth:
        Bright band width.
    radar_band: str
        Possible values are 'S', 'C', or 'X'

    Return
    ======
    refp_ss:
        Stratiform reflectivity conversion from Ku-band to S-band
    refp_sh:
        Convective reflectivity conversion from Ku-band to S-band
    """
    if radar_band not in ['S', 'C', 'X']:
        raise ValueError(f'Radar reflectivity band ({radar_band}) not supported.')

    refp_ss = np.zeros_like(refp) # snow
    refp_sh = np.zeros_like(refp) # hail

    # Set coefficients for conversion from Ku-band to S-band
    #        Rain      90%      80%      70%      60%      50%      40%      30%      20%      10%     Snow
    as0 = [ 4.78e-2, 4.12e-2, 8.12e-2, 1.59e-1, 2.87e-1, 4.93e-1, 8.16e-1, 1.31e+0, 2.01e+0, 2.82e+0, 1.74e-1]
    as1 = [ 1.23e-2, 3.66e-3, 2.00e-3, 9.42e-4, 5.29e-4, 5.96e-4, 1.22e-3, 2.11e-3, 3.34e-3, 5.33e-3, 1.35e-2]
    as2 = [-3.50e-4, 1.17e-3, 1.04e-3, 8.16e-4, 6.59e-4, 5.85e-4, 6.13e-4, 7.01e-4, 8.24e-4, 1.01e-3,-1.38e-3]
    as3 = [-3.30e-5,-8.08e-5,-6.44e-5,-4.97e-5,-4.15e-5,-3.89e-5,-4.15e-5,-4.58e-5,-5.06e-5,-5.78e-5, 4.74e-5]
    as4 = [ 4.27e-7, 9.25e-7, 7.41e-7, 6.13e-7, 5.80e-7, 6.16e-7, 7.12e-7, 8.22e-7, 9.39e-7, 1.10e-6, 0]
    #        Rain      90%      80%      70%      60%      50%      40%      30%      20%      10%     Hail
    ah0 = [ 4.78e-2, 1.80e-1, 1.95e-1, 1.88e-1, 2.36e-1, 2.70e-1, 2.98e-1, 2.85e-1, 1.75e-1, 4.30e-2, 8.80e-2]
    ah1 = [ 1.23e-2,-3.73e-2,-3.83e-2,-3.29e-2,-3.46e-2,-2.94e-2,-2.10e-2,-9.96e-3,-8.05e-3,-8.27e-3, 5.39e-2]
    ah2 = [-3.50e-4, 4.08e-3, 4.14e-3, 3.75e-3, 3.71e-3, 3.22e-3, 2.44e-3, 1.45e-3, 1.21e-3, 1.66e-3,-2.99e-4]
    ah3 = [-3.30e-5,-1.59e-4,-1.54e-4,-1.39e-4,-1.30e-4,-1.12e-4,-8.56e-5,-5.33e-5,-4.66e-5,-7.19e-5, 1.90e-5]
    ah4 = [ 4.27e-7, 1.59e-6, 1.51e-6, 1.37e-6, 1.29e-6, 1.15e-6, 9.40e-7, 6.71e-7, 6.33e-7, 9.52e-7, 0]

    zbb = np.repeat(zbb[:, :, np.newaxis], zp.shape[2], axis=2)
    bbwidth = np.repeat(bbwidth[:, :, np.newaxis], zp.shape[2], axis=2)

    zmlt = zbb + bbwidth / 2.0  # APPROXIMATION!
    zmlb = zbb - bbwidth / 2.0  # APPROXIMATION!
    ratio = (zp - zmlb) / (zmlt - zmlb)

    pos = ratio >= 1
    # above melting layer
    if pos.sum() > 0:
        dfrs = (
            as0[10]
            + as1[10] * refp[pos]
            + as2[10] * refp[pos] ** 2
            + as3[10] * refp[pos] ** 3
            + as4[10] * refp[pos] ** 4
        )
        dfrh = (
            ah0[10]
            + ah1[10] * refp[pos]
            + ah2[10] * refp[pos] ** 2
            + ah3[10] * refp[pos] ** 3
            + ah4[10] * refp[pos] ** 4
        )
        refp_ss[pos] = refp[pos] + dfrs
        refp_sh[pos] = refp[pos] + dfrh

    pos = ratio <= 0
    if pos.sum() > 0:  # below the melting layer
        dfrs = (
            as0[0]
            + as1[0] * refp[pos]
            + as2[0] * refp[pos] ** 2
            + as3[0] * refp[pos] ** 3
            + as4[0] * refp[pos] ** 4
        )
        dfrh = (
            ah0[0]
            + ah1[0] * refp[pos]
            + ah2[0] * refp[pos] ** 2
            + ah3[0] * refp[pos] ** 3
            + ah4[0] * refp[pos] ** 4
        )
        refp_ss[pos] = refp[pos] + dfrs
        refp_sh[pos] = refp[pos] + dfrh

    pos = (ratio > 0) & (ratio < 1)
    if pos.sum() > 0:  # within the melting layer
        ind = np.round(ratio[pos]).astype(int)[0]
        dfrs = (
            as0[ind]
            + as1[ind] * refp[pos]
            + as2[ind] * refp[pos] ** 2
            + as3[ind] * refp[pos] ** 3
            + as4[ind] * refp[pos] ** 4
        )
        dfrh = (
            ah0[ind]
            + ah1[ind] * refp[pos]
            + ah2[ind] * refp[pos] ** 2
            + ah3[ind] * refp[pos] ** 3
            + ah4[ind] * refp[pos] ** 4
        )
        refp_ss[pos] = refp[pos] + dfrs
        refp_sh[pos] = refp[pos] + dfrh

    # Jackson Tan's fix for C-band
    if radar_band == "C":
        deltas = 5.3 / 10.0 * (refp_ss - refp)
        refp_ss = refp + deltas
        deltah = 5.3 / 10.0 * (refp_sh - refp)
        refp_sh = refp + deltah
    elif radar_band == "X":
        deltas = 3.2 / 10.0 * (refp_ss - refp)
        refp_ss = refp + deltas
        deltah = 3.2 / 10.0 * (refp_sh - refp)
        refp_sh = refp + deltah

    return np.ma.masked_invalid(refp_ss), np.ma.masked_invalid(refp_sh)


def compute_gaussian_curvature(lat0):
    '''
    Determine the Earth's Gaussian radius of curvature at the radar
    https://en.wikipedia.org/wiki/Earth_radius#Radii_of_curvature

    Parameter:
    ----------
    lat0: float
        Ground radar latitude.

    Returns:
    --------
    ae: float
        Earth's Gaussian radius.
    '''
    # Major and minor radii of the Ellipsoid
    a = 6378137.0  # Earth radius in meters
    e2 = 0.0066943800
    b = a * np.sqrt(1 - e2)

    tmp = (a * np.cos(np.pi / 180 * lat0))**2 + (b * np.sin(np.pi / 180 * lat0))**2   # Denominator
    an = (a**2) / np.sqrt(tmp)  # Radius of curvature in the prime vertical (east–west direction)
    am = (a * b)**2 / tmp ** 1.5  # Radius of curvature in the north–south meridian
    ag = np.sqrt(an * am)  # Earth's Gaussian radius of curvature
    ae = (4 / 3.) * ag

    return ae


def grid_displacement(field1, field2):
    """
    Calculate the grid displacement using Phase correlation.
    http://en.wikipedia.org/wiki/Phase_correlation

    Parameters
    ----------
    field1, field2 : ndarray
       Fields separated in time.

    Returns
    -------
    displacement : two-tuple
         integers if pixels, otherwise floats. Result of the calculation
    """
    #create copies of the data
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

    #find the peak in the correlation
    yShift, xShift = np.unravel_index(np.argmax(imageCCorShift), (row,col))
    yShift -= row // 2
    xShift -= col // 2

    return (xShift, yShift)