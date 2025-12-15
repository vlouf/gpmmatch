"""
Various utilities for correction and conversion of satellite data.

@title: correct
@author: Valentin Louf <valentin.louf@bom.gov.au>
@institutions: Monash University and the Australian Bureau of Meteorology
@creation: 17/02/2020
@date: 12/05/2025

.. autosummary::
    :toctree: generated/

    compute_gaussian_curvature
    convert_gpmrefl_grband_dfr
    convert_gpmrefl_grband_phase_aware
    correct_parallax
    attenuation_correction_zphi
    attenuation_correction_gunn_east
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


def convert_gpmrefl_grband_phase_aware(
    refl_gpm: np.ndarray,
    height_gpm: np.ndarray,
    height_bb: np.ndarray,
    radar_band: str,
    bb_margin: float = 500.0,
) -> np.ndarray:
    """
    Convert GPM Ku-band reflectivity to ground radar band using phase-aware DFR.

    This function accounts for hydrometeor phase (ice, melting layer, or liquid)
    by using the GPM bright band (melting layer) height. Different DFR relationships
    are applied based on whether the radar gate is above, within, or below the
    melting layer.

    Parameters:
    ===========
    refl_gpm: np.ndarray
        GPM Ku-band reflectivity field (dBZ). Shape: (nscan, nray, nbin)
    height_gpm: np.ndarray
        Height of each GPM gate (meters). Shape: (nscan, nray, nbin)
    height_bb: np.ndarray
        Bright band (melting layer) height from GPM (meters). Shape: (nscan, nray)
        Values < 0 indicate no bright band detected (all rain or all ice)
    radar_band: str
        Ground radar frequency band. Possible values: 'S', 'C', or 'X'
    bb_margin: float, optional
        Vertical margin around bright band height to define melting layer (meters).
        Default is 500m (±500m around heightBB defines the melting layer).

    Returns:
    ========
    refl_grband: np.ndarray
        Reflectivity converted from Ku-band to ground radar band (dBZ).
        Same shape as refl_gpm.

    References:
    ===========
    - Rain DFR coefficients from Louf et al. (2019) J. Atmos. Oceanic Technol.
    - Ice scattering theory: Tyynelä et al. (2011), Leinonen (2014)
    - GPM bright band detection: Awaka et al. (2016)
    """
    if radar_band not in ["S", "C", "X"]:
        raise ValueError(f"Radar reflectivity band ({radar_band}) not supported. Use 'S', 'C', or 'X'.")

    # Define rain DFR polynomial coefficients (from Louf et al. 2019)
    if radar_band == "S":
        rain_cof = np.array([2.01236803e-07, -6.50694273e-06, 1.10885533e-03, -6.47985914e-02, -7.46518423e-02])
    elif radar_band == "C":
        rain_cof = np.array([1.21547932e-06, -1.23266138e-04, 6.38562875e-03, -1.52248868e-01, 5.33556919e-01])
    elif radar_band == "X":
        rain_cof = np.array([1.21547932e-06, -1.23266138e-04, 6.38562875e-03, -1.52248868e-01, 5.33556919e-01])
        rain_cof = rain_cof * (3.2 / 5.5)

    rain_dfr = np.poly1d(rain_cof)

    # Ice DFR - smaller frequency dependence for ice particles
    # Based on Mie scattering theory for ice spheres
    # Ice has less frequency-dependent scattering than rain
    ice_dfr_offset = {
        'S': 0.3,   # dB - small positive offset for S-band
        'C': 0.5,   # dB - moderate offset for C-band
        'X': 0.8    # dB - larger offset for X-band
    }

    refl_grband = np.full_like(refl_gpm, np.nan, dtype=np.float64)
    # Expand heightBB to match 3D shape (nscan, nray, nbin)
    height_bb_3d = np.broadcast_to(height_bb[..., np.newaxis], refl_gpm.shape)

    no_bb_mask = height_bb_3d < 0
    liquid_mask = (height_bb_3d >= 0) & (height_gpm < (height_bb_3d - bb_margin))
    ice_mask = (height_bb_3d >= 0) & (height_gpm > (height_bb_3d + bb_margin))
    melting_mask = (height_bb_3d >= 0) & (height_gpm >= (height_bb_3d - bb_margin)) & (height_gpm <= (height_bb_3d + bb_margin))

    if np.any(no_bb_mask):
        refl_grband[no_bb_mask] = refl_gpm[no_bb_mask] + rain_dfr(refl_gpm[no_bb_mask])

    if np.any(liquid_mask):
        refl_grband[liquid_mask] = refl_gpm[liquid_mask] + rain_dfr(refl_gpm[liquid_mask])

    if np.any(ice_mask):
        refl_grband[ice_mask] = refl_gpm[ice_mask] + ice_dfr_offset[radar_band]

    if np.any(melting_mask):
        # Calculate normalized height within melting layer (0 = bottom, 1 = top)
        height_in_ml = height_gpm[melting_mask]
        bb_height = height_bb_3d[melting_mask]

        # Fraction of liquid (1 at bottom, 0 at top of melting layer)
        liquid_fraction = 1.0 - (height_in_ml - (bb_height - bb_margin)) / (2 * bb_margin)
        liquid_fraction = np.clip(liquid_fraction, 0, 1)

        # Interpolate between ice and rain DFR
        dfr_ice = ice_dfr_offset[radar_band]
        dfr_rain = rain_dfr(refl_gpm[melting_mask])
        dfr_mixed = liquid_fraction * dfr_rain + (1 - liquid_fraction) * dfr_ice

        refl_grband[melting_mask] = refl_gpm[melting_mask] + dfr_mixed

    return refl_grband


def attenuation_correction_zphi(zh: np.ndarray, kdp: np.ndarray, dr: float = 1, wavelength: str = "C") -> np.ndarray:
    """
    Attenuation correction using the ZPHI method.

    Parameters:
    -----------
    zh : array-like
        Horizontal reflectivity (dBZ)
    kdp : array-like
        Specific differential phase (deg/km)
    dr : float
        Range gate spacing in km
    wavelength : str
        'C' for C-band (~5.3 cm) or 'X' for X-band (~3.2 cm)

    Returns:
    --------
    zh_corrected : array
        Attenuation-corrected horizontal reflectivity (dBZ)
    """
    if dr > 10:
        raise ValueError("Range gate spacing 'dr' seems too large (>10 km). Check input.")

    if wavelength.upper() == "C":
        # C-band coefficients (Bringi et al. 2001)
        alpha_h = 0.08  # coefficient for Ah = alpha * Kdp^beta
        beta_h = 0.93

    elif wavelength.upper() == "X":
        # X-band coefficients (Park et al. 2005)
        alpha_h = 0.28
        beta_h = 0.95
    else:
        raise ValueError("Wavelength must be 'C' or 'X'")

    # Calculate specific attenuation
    ah = alpha_h * np.power(np.abs(kdp), beta_h) * np.sign(kdp)
    PIA_h = np.nancumsum(ah * dr, axis=1)  # Path Integrated Attenuation for Zh
    zh_corrected = zh + 2 * PIA_h  # Factor of 2 for two-way path

    return zh_corrected


def attenuation_correction_gunn_east(
    zh: np.ndarray, dr: float = 1, wavelength: str = "C", max_thld: float = 10
) -> np.ndarray:
    """
    Attenuation correction using Gunn and East (1954) method.
    This is based on the power-law relationship.
    Eq. 2.66 p 106 of radar meteorology by Henri Sauvagot.

    Parameters:
    -----------
    zh : array-like
        Horizontal reflectivity in dBZ along a radial
    dr : float
        Range gate spacing in km
    wavelength : str
        'C' for C-band or 'X' for X-band
    max_thld: float
        Capped-attenuation correction to avoid blowing up the refl in dB.

    Returns:
    --------
    zh_corrected : array
        Attenuation-corrected reflectivity (dBZ)
    """
    if dr > 10:
        raise ValueError("Range gate spacing 'dr' seems too large (>10 km). Check input.")
    if wavelength.upper() == "C":
        l = 5.5
    elif wavelength.upper() == "X":
        l = 3.2
    else:
        raise ValueError("Wavelength must be 'C' or 'X'")

    # Convert dBZ to linear Z (mm^6/m^3)
    Z_linear = 10 ** (zh / 10.0)
    R = (Z_linear / 200) ** (1 / 1.6)
    ap = 0.35e-2 * R**1.6 / (l**4) + 0.22e-2 * R / l

    PIA_h = np.nancumsum(ap * dr, axis=1)
    PIA_h[PIA_h > max_thld / 2] = max_thld / 2
    Z_corrected = zh + 2 * PIA_h

    return Z_corrected


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
