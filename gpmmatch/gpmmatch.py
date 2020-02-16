import gc
import os
import re
import time
import glob
import uuid
import traceback
import datetime
import itertools

import h5py
import pyart
import pyproj
import netCDF4
import numpy as np
import pandas as pd
import xarray as xr


class NoPrecipitationError(Exception):
    pass


def get_metadata():
    metadata = {'refl_gpm_raw': {'units': 'dBZ', 'long_name': 'GPM_reflectivity', 'description': 'GPM reflectivity volume-matched to ground radar.'},
            'refl_gpm_strat': {'units': 'dBZ', 'long_name': 'GPM_reflectivity_grband_stratiform', 'description': 'GR-frequency band converted GPM reflectivity volume-matched to ground radar - stratiform approximation.'},
            'refl_gpm_conv': {'units': 'dBZ', 'long_name': 'GPM_reflectivity_grband_convective', 'description': 'GR-frequency band converted GPM reflectivity volume-matched to ground radar - convective approximation.'},
            'refl_gr_raw': {'units': 'dBZ', 'long_name': 'reflectivity', 'description': 'Ground radar reflectivity volume matched using a `normal` average.'},
            'refl_gr_weigthed': {'units': 'dBZ', 'long_name': 'reflectivity', 'description': 'Ground radar reflectivity volume matched using a distance-weighted average.'},
            'std_refl_gpm': {'units': 'dB', 'long_name': 'standard_deviation_reflectivity', 'description': 'GPM reflectivity standard deviation of the volume-matched sample.'},
            'std_refl_gr': {'units': 'dB', 'long_name': 'standard_deviation_reflectivity', 'description': 'Ground radar reflectivity standard deviation of the volume-matched sample.'},
            'zrefl_gpm_raw': {'units': 'dBZ', 'long_name': 'GPM_reflectivity', 'description': 'GPM reflectivity volume-matched to ground radar computed in linear units.'},
            'zrefl_gpm_strat': {'units': 'dBZ', 'long_name': 'GPM_reflectivity_grband_stratiform', 'description': 'GR-frequency band converted GPM reflectivity volume-matched to ground radar - stratiform approximation computed in linear units.'},
            'zrefl_gpm_conv': {'units': 'dBZ', 'long_name': 'GPM_reflectivity_grband_convective', 'description': 'GR-frequency band converted GPM reflectivity volume-matched to ground radar - convective approximation computed in linear units.'},
            'zrefl_gr_raw': {'units': 'dBZ', 'long_name': 'reflectivity', 'description': 'Ground radar reflectivity volume matched using a `normal` average computed in linear units.'},
            'zrefl_gr_weigthed': {'units': 'dBZ', 'long_name': 'reflectivity', 'description': 'Ground radar reflectivity volume matched using a distance-weighted average computed in linear units.'},
            'std_zrefl_gpm': {'units': 'dB', 'long_name': 'standard_deviation_reflectivity', 'description': 'GPM reflectivity standard deviation of the volume-matched sample computed in linear units.'},
            'std_zrefl_gr': {'units': 'dB', 'long_name': 'standard_deviation_reflectivity', 'description': 'Ground radar reflectivity standard deviation of the volume-matched sample computed in linear units.'},
            'sample_gpm': {'units': '1', 'long_name': 'sample_size', 'description': 'Number of GPM bins used to compute the volume-matched pixels at a given points'},
            'reject_gpm': {'units': '1', 'long_name': 'rejected_sample_size', 'description': 'Number of GPM bins rejected to compute the volume-matched pixels at a given points'},
            'sample_gr': {'units': '1', 'long_name': 'sample_size', 'description': 'Number of ground-radar bins used to compute the volume-matched pixels at a given points'},
            'reject_gr': {'units': '1', 'long_name': 'rejected_sample_size', 'description': 'Number of ground-radar bins rejected to compute the volume-matched pixels at a given points'},
            'volume_match_gpm': {'units': 'm^3', 'long_name': 'volume', 'description': 'Volume of the GPM sample for each match points.'},
            'volume_match_gr': {'units': 'm^3', 'long_name': 'volume', 'description': 'Volume of the ground radar sample for each match points.'},
            'x': {'units': 'm', 'long_name': 'projected_x_axis_coordinates', 'projection': 'Azimuthal Equidistant from ground radar.'},
            'y': {'units': 'm', 'long_name': 'projected_y_axis_coordinates', 'projection': 'Azimuthal Equidistant from ground radar.'},
            'z': {'units': 'm', 'long_name': 'projected_z_axis_coordinates', 'projection': 'Azimuthal Equidistant from ground radar.'},
            'r': {'units': 'm', 'long_name': 'range', 'description': 'Range from ground radar.'},
            'elevation_gr': {'units': 'degrees', 'long_name': 'elevation', 'description': 'Ground radar reference elevation.'},
            'ntilt': {'units': '1', 'long_name': 'ground_radar_tilt_number', 'description': 'Number of ground radar tilts used for volume matching.'},
            'nprof': {'units': '1', 'long_name': 'gpm_profile_number', 'description': 'Number of GPM profiles (nrays x nscan) used for volume matching.'},
           }
    return metadata


def get_gpm_orbit(gpmfile):
    '''
    Parameters:
    ----------
    gpmfile: str
        GPM data file.

    Returns:
    --------
    orbit: int
        GPM Granule Number.
    '''
    try:
        with h5py.File(gpmfile) as hid:
            grannb = [s for s in hid.attrs['FileHeader'].split() if b'GranuleNumber' in s][0].decode('utf-8')
            orbit = re.findall('[0-9]{3,}', grannb)[0]
    except Exception:
        return 0

    return int(orbit)


def read_GPM(infile, refl_min_thld):
    '''
    Read GPM data and organize them into a Dataset.

    Parameters:
    ----------
    gpmfile: str
        GPM data file.
    refl_min_thld: float
        Minimum threshold applied to GPM reflectivity.

    Returns:
    --------
    dset: xr.Dataset
        GPM dataset.
    '''
    data = dict()
    date = dict()
    with h5py.File(infile, 'r') as hid:
        keys = hid['/NS'].keys()
        for k in keys:
            if k == 'Latitude' or k == 'Longitude':
                dims = tuple(hid[f'/NS/{k}'].attrs['DimensionNames'].decode('UTF-8').split(','))
                fv =  hid[f'/NS/{k}'].attrs['_FillValue']
                data[k] = (dims, np.ma.masked_equal(hid[f'/NS/{k}'][:], fv))
            else:
                subkeys = hid[f'/NS/{k}'].keys()
                for sk in subkeys:
                    dims = tuple(hid[f'/NS/{k}/{sk}'].attrs['DimensionNames'].decode('UTF-8').split(','))
                    fv =  hid[f'/NS/{k}/{sk}'].attrs['_FillValue']

                    if sk in ['Year', 'Month', 'DayOfMonth', 'Hour', 'Minute', 'Second', 'MilliSecond']:
                        date[sk] = np.ma.masked_equal(hid[f'/NS/{k}/{sk}'][:], fv)
                    elif sk in ['DayOfYear', 'SecondOfDay']:
                        continue
                    elif sk == 'typePrecip':
                        # Simplify precipitation type
                        data[sk] = (dims, hid[f'/NS/{k}/{sk}'][:] / 10000000)
                    elif sk == 'zFactorCorrected':
                        # Reverse direction along the beam.
                        data[sk] = (dims, np.ma.masked_less_equal(hid[f'/NS/{k}/{sk}'][:][:, :, ::-1], refl_min_thld))
                    else:
                        data[sk] = (dims, np.ma.masked_equal(hid[f'/NS/{k}/{sk}'][:], fv))

    try:
        data['zFactorCorrected']
    except Exception:
        raise KeyError(f"GPM Reflectivity not found in {infile}")

    # Create Quality indicator.
    quality = np.zeros(data['heightBB'][-1].shape, dtype=np.int32)
    quality[((data['qualityBB'][-1] == 0) | (data['qualityBB'][-1] == 1)) & (data['qualityTypePrecip'][-1] == 1)] = 1
    quality[(data['qualityBB'][-1] > 1) | (data['qualityTypePrecip'][-1] > 1)] = 2
    data['quality'] = (data['heightBB'][0], quality)

    # Generate dimensions.
    nray = np.linspace(-17.04, 17.04, 49)
    nbin = np.arange(0, 125 * 176, 125)

    R, A = np.meshgrid(nbin, nray)
    distance_from_sr = 407000 / np.cos(np.deg2rad(A)) - R  # called rt in IDL code.
    data['distance_from_sr'] = (('nray', 'nbin'), distance_from_sr)

    dtime = np.array([datetime.datetime(*d) for d in zip(date['Year'],
                                                         date['Month'],
                                                         date['DayOfMonth'],
                                                         date['Hour'],
                                                         date['Minute'],
                                                         date['Second'],
                                                         date['MilliSecond'])], dtype='datetime64')

    data['nscan'] = (('nscan'), dtime)
    data['nray'] = (('nray'), nray)
    data['nbin'] = (('nbin'), nbin)

    dset = xr.Dataset(data)

    dset.nray.attrs = {'units': 'degree', 'description':'Deviation from Nadir'}
    dset.nbin.attrs = {'units': 'm', 'description':'Downward from 0: TOA to Earth ellipsoid.'}
    dset.attrs['altitude'] = 407000
    dset.attrs['altitude_units'] = 'm'
    dset.attrs['altitude_description'] = "GPM orbit"
    dset.attrs['beamwidth'] = 0.71
    dset.attrs['beamwidth_units'] = 'degree'
    dset.attrs['beamwidth_description'] = "GPM beamwidth"
    dset.attrs['dr'] = 125
    dset.attrs['dr_units'] = 'm'
    dset.attrs['dr_description'] = "GPM gate spacing"
    dset.attrs['orbit'] = get_gpm_orbit(infile)

    return dset


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
    center = int(np.floor(len(sr_x[-1]) / 2.))
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


def convert_sat_refl_to_gr_band(refp, zp, zbb, bbwidth, radar_band='S'):
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

    zmlt = zbb + bbwidth / 2.  # APPROXIMATION!
    zmlb = zbb - bbwidth / 2.  # APPROXIMATION!
    ratio = (zp - zmlb) / (zmlt - zmlb)

    pos = (ratio >= 1)
    # above melting layer
    if pos.sum() > 0:
        dfrs = as0[10] + as1[10] * refp[pos] + as2[10] * refp[pos] ** 2 + as3[10] * refp[pos] ** 3 + as4[10] * refp[pos] ** 4
        dfrh = ah0[10] + ah1[10] * refp[pos] + ah2[10] * refp[pos] ** 2 + ah3[10] * refp[pos] ** 3 + ah4[10] * refp[pos] ** 4
        refp_ss[pos] = refp[pos] + dfrs
        refp_sh[pos] = refp[pos] + dfrh

    pos = (ratio <= 0)
    if pos.sum() > 0: # below the melting layer
        dfrs = as0[0] + as1[0] * refp[pos] + as2[0] * refp[pos]**2 + as3[0] * refp[pos]**3 + as4[0] * refp[pos]**4
        dfrh = ah0[0] + ah1[0] * refp[pos] + ah2[0] * refp[pos]**2 + ah3[0] * refp[pos]**3 + ah4[0] * refp[pos]**4
        refp_ss[pos] = refp[pos] + dfrs
        refp_sh[pos] = refp[pos] + dfrh

    pos = ((ratio > 0) & (ratio < 1))
    if pos.sum() > 0:  # within the melting layer
        ind = np.round(ratio[pos]).astype(int)[0]
        dfrs = as0[ind] + as1[ind] * refp[pos] + as2[ind] * refp[pos] ** 2 + as3[ind] * refp[pos] ** 3 + as4[ind] * refp[pos] ** 4
        dfrh = ah0[ind] + ah1[ind] * refp[pos] + ah2[ind] * refp[pos] ** 2 + ah3[ind] * refp[pos] ** 3 + ah4[ind] * refp[pos] ** 4
        refp_ss[pos] = refp[pos] + dfrs
        refp_sh[pos] = refp[pos] + dfrh

    # Jackson Tan's fix for C-band
    if radar_band == 'C':
        deltas = 5.3 / 10.0 * (refp_ss - refp)
        refp_ss = refp + deltas
        deltah = 5.3 / 10.0 * (refp_sh - refp)
        refp_sh = refp + deltah
    elif radar_band == 'X':
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


def data_load_and_checks(gpmfile, grfile, *, refl_name, gpm_refl_threshold):
    '''
    Load GPM and Ground radar files and perform some initial checks:
    domains intersect, precipitation, time difference.

    Parameters:
    ----------
    gpmfile: str
        GPM data file.
    grfile: str
        Ground radar input file.
    refl_name: str
        Name of the reflectivity field in the ground radar data.
    refl_min_thld: float
        Minimum threshold applied to GPM reflectivity.

    Returns:
    --------
    gpmset: xarray.Dataset
        Dataset containing the input datas.
    radar: pyart.core.Radar
        Pyart radar dataset.
    '''
    gpmset = read_GPM(gpmfile, gpm_refl_threshold)
    try:
        radar = pyart.io.read_cfradial(grfile, delay_field_loading='True')
    except Exception:
        radar = pyart.aux_io.read_odim_h5(grfile, delay_field_loading='True')

    rmax = radar.range['data'].max()
    rmin = 20e3
    grlon = radar.longitude['data'][0]
    grlat = radar.latitude['data'][0]

    xg = radar.gate_x['data']
    yg = radar.gate_y['data']
    zg = radar.gate_z['data'] + radar.altitude['data']
    rg = radar.range['data']
    dr_gr = 1e-3 * (rg[1] - rg[0])
    try:
        reflectivity_gr = radar.fields[refl_name]['data']
    except KeyError:
        raise KeyError('Name of the reflectivity field not found in the input ground radar file. ' +
                       f'Name of the field given: {refl_name}. ' +
                       f'Fields present in radar file: {radar.fields.keys()}.')
    elevation_gr = radar.elevation['data']

    # Reproject satellite coordinates onto ground radar
    georef = pyproj.Proj(f"+proj=aeqd +lon_0={grlon} +lat_0={grlat} +ellps=WGS84")
    gpmlat = gpmset.Latitude.values
    gpmlon = gpmset.Longitude.values

    xgpm, ygpm = georef(gpmlon, gpmlat)
    rproj_gpm = (xgpm ** 2 + ygpm ** 2) ** 0.5

    # Checks.
#     if gpmlon.shape != rproj_gpm.shape:
#         print(gpmlon.shape, rproj_gpm.shape)
#         return None

    gr_domain = (rproj_gpm <= rmax) & (rproj_gpm >= rmin)
    if gr_domain.sum() < 10:
        info = f'The closest satellite measurement is {np.min(rproj_gpm / 1e3):0.4} km away from ground radar.'
        if gr_domain.sum() == 0:
            raise NoPrecipitationError('GPM swath does not go through the radar domain. ' + info)
        else:
            raise NoPrecipitationError('GPM swath is on the edge of the ground radar domain and there is not enough measurements inside it. ' + info)

    nprof = np.sum(gpmset.flagPrecip.values[gr_domain])
    if nprof < 10:
        raise NoPrecipitationError('No precipitation measured by GPM inside radar domain.')

    brightband_domain = ((gpmset.heightBB.values[gr_domain] > 0) &
                         (gpmset.widthBB.values[gr_domain] > 0) &
                         (gpmset.quality.values[gr_domain] == 1))
    if brightband_domain.sum() < 10:
        raise NoPrecipitationError('Insufficient bright band rays')

    # Parallax correction
    sr_xp, sr_yp, z_sr = correct_parallax(xgpm, ygpm, gpmset)

    # Compute the elevation of the satellite bins with respect to the ground radar.
    gr_gaussian_radius = compute_gaussian_curvature(grlat)
    gamma = np.sqrt(sr_xp ** 2 + sr_yp ** 2) / gr_gaussian_radius
    elev_sr_grref = np.rad2deg(np.arctan2(np.cos(gamma) - (gr_gaussian_radius + radar.altitude['data']) / (gr_gaussian_radius + z_sr), np.sin(gamma)))

    # Convert reflectivity band correction
    refp_strat, refp_conv = convert_sat_refl_to_gr_band(gpmset.zFactorCorrected.values,
                                                        z_sr,
                                                        gpmset.heightBB.values,
                                                        gpmset.widthBB.values,
                                                        radar_band='C')

    gpmset = gpmset.merge({'precip_in_gr_domain':  (('nscan', 'nray'), gpmset.flagPrecip.values & gr_domain),
                           'range_from_gr': (('nscan', 'nray'), rproj_gpm),
                           'elev_from_gr': (('nscan', 'nray', 'nbin'), elev_sr_grref),
                           'x': (('nscan', 'nray', 'nbin'), sr_xp),
                           'y': (('nscan', 'nray', 'nbin'), sr_yp),
                           'z': (('nscan', 'nray', 'nbin'), z_sr),
                           'strat_reflectivity_grband': (('nscan', 'nray', 'nbin'), refp_strat),
                           'conv_reflectivity_grband': (('nscan', 'nray', 'nbin'), refp_conv)})

    gpmset.x.attrs['units'] = 'm'
    gpmset.x.attrs['description'] = 'Cartesian distance along x-axis of satellite bin in relation to ground radar (0, 0), parallax corrected'
    gpmset.y.attrs['units'] = 'm'
    gpmset.y.attrs['description'] = 'Cartesian distance along y-axis of satellite bin in relation to ground radar (0, 0), parallax corrected'
    gpmset.z.attrs['units'] = 'm'
    gpmset.z.attrs['description'] = 'Cartesian distance along z-axis of satellite bin in relation to ground radar (0, 0), parallax corrected'
    gpmset.precip_in_gr_domain.attrs['units'] = 'bool'
    gpmset.precip_in_gr_domain.attrs['description'] = 'GPM data-columns with precipitation inside the ground radar scope.'
    gpmset.range_from_gr.attrs['units'] = 'm'
    gpmset.range_from_gr.attrs['description'] = 'Range from satellite bins in relation to ground radar'
    gpmset.elev_from_gr.attrs['units'] = 'degrees'
    gpmset.elev_from_gr.attrs['description'] = 'Elevation from satellite bins in relation to ground radar'
    gpmset.strat_reflectivity_grband.attrs['units'] = 'dBZ'
    gpmset.strat_reflectivity_grband.attrs['description'] = 'Reflectivity of stratiform precipitation converted to ground radar frequency band.'
    gpmset.conv_reflectivity_grband.attrs['units'] = 'dBZ'
    gpmset.conv_reflectivity_grband.attrs['description'] = 'Reflectivity of convective precipitation converted to ground radar frequency band.'
    gpmset.attrs['nprof'] = nprof
    gpmset.attrs['earth_gaussian_radius'] = gr_gaussian_radius

    return gpmset, radar


def volume_matching(gpmfile,
                    grfile,
                    refl_name='corrected_reflectivity',
                    fname_prefix=None,
                    gr_beamwidth=1,
                    gr_refl_threshold=21,
                    gpm_refl_threshold=21,
                    output_dir=None,
                    write_output=True):
    '''
    Performs the volume matching of GPM to ground based radar.

    Parameters:
    ----------
    gpmfile: str
        GPM data file.
    grfile: str
        Ground radar input file.
    refl_name: str
        Name of the reflectivity field in the ground radar data.
    fname_prefix: str
        Name of the ground radar to use as label for the output file.
    gr_beamwidth: float
        Ground radar 3dB-beamwidth.
    gr_refl_thresold: float
        Minimum reflectivity threshold on ground radar data.
    output_dir: str
        Path to output directory.
    write_output: bool
        Does it save the data automatically or not?

    Returns:
    --------
    matchset: xarray.Dataset
        Dataset containing the matched GPM and ground radar data.
    '''
    if fname_prefix is None:
        fname_prefix = 'unknown_radar'
    if output_dir is None:
        output_dir = os.getcwd()

    bwr = gr_beamwidth
    gpmset, radar = data_load_and_checks(gpmfile, grfile, refl_name=refl_name, gpm_refl_threshold=gpm_refl_threshold)

    nprof = gpmset.precip_in_gr_domain.values.sum()
    if radar.elevation['data'].max() >= 80:
        ntilt = radar.nsweeps - 1
    else:
        ntilt = radar.nsweeps

    # Extract ground radar data.
    range_gr = radar.range['data']
    elev_gr = np.unique(radar.elevation['data'])
    xradar = radar.gate_x['data']
    yradar = radar.gate_y['data']

    rmax_gr = range_gr.max()
    dr = range_gr[1] - range_gr[0]

    R, A = np.meshgrid(radar.range['data'], radar.azimuth['data'])

    try:
        ground_radar_reflectivity = radar.fields['total_power']['data'].filled(np.NaN)
    except Exception:
        ground_radar_reflectivity = radar.fields['total_power']['data']
    ground_radar_reflectivity[ground_radar_reflectivity < gr_refl_threshold] = np.NaN
    ground_radar_reflectivity = np.ma.masked_invalid(ground_radar_reflectivity)

    # Extract GPM data.
    position_precip_domain = gpmset.precip_in_gr_domain != 0

    alpha, _ = np.meshgrid(gpmset.nray, gpmset.nscan)
    alpha = alpha[position_precip_domain]

    elev_sat = gpmset.elev_from_gr.values[position_precip_domain]
    xsat = gpmset.x.values[position_precip_domain]
    ysat = gpmset.y.values[position_precip_domain]
    zsat = gpmset.z.values[position_precip_domain]
    s_sat = np.sqrt(xsat ** 2 + ysat ** 2)

    rsat = np.zeros(gpmset.zFactorCorrected.shape)
    for i in range(rsat.shape[0]):
        rsat[i, :] = gpmset.distance_from_sr.values

    refl_gpm_raw = gpmset.zFactorCorrected.values[position_precip_domain]
    refl_gpm_stratgrband = gpmset.strat_reflectivity_grband.values[position_precip_domain]
    refl_gpm_convgrband = gpmset.conv_reflectivity_grband.values[position_precip_domain]

    volsat = 1e-9 * gpmset.dr * (rsat[position_precip_domain] * np.deg2rad(gpmset.beamwidth) / 2) ** 2  # km3
    volgr = 1e-9 * np.pi * dr * (R * np.pi / 180 * bwr / 2) ** 2  # km3

    # Initialising output data.
    datakeys = ['refl_gpm_raw', 'refl_gr_weigthed', 'refl_gpm_strat', 'refl_gpm_conv',
                'refl_gr_raw', 'std_refl_gpm', 'std_refl_gr', 'zrefl_gpm_raw', 'zrefl_gr_weigthed', 'zrefl_gpm_strat',
                'zrefl_gpm_conv', 'zrefl_gr_raw', 'std_zrefl_gpm', 'std_zrefl_gr', 'sample_gpm', 'reject_gpm',
                'sample_gr', 'reject_gr', 'volume_match_gpm', 'volume_match_gr']

    data = dict()
    for k in datakeys:
        data[k] = np.zeros((nprof, ntilt)) + np.NaN

    # For sake of simplicity, coordinates are just ndarray, they will be put in the 'data' dict after the matching process
    x = np.zeros((nprof, ntilt))  # x coordinate of sample
    y = np.zeros((nprof, ntilt))  # y coordinate of sample
    z = np.zeros((nprof, ntilt))  # z coordinate of sample
    r = np.zeros((nprof, ntilt))  # range of sample from ground radar
    dz = np.zeros((nprof, ntilt))  # depth of sample
    ds = np.zeros((nprof, ntilt))  # width of sample

    for ii, jj in itertools.product(range(nprof), range(ntilt)):
        epos = (elev_sat[ii, :] >= elev_gr[jj] - bwr / 2) & (elev_sat[ii, :] <= elev_gr[jj] + bwr / 2)
        x[ii, jj] = np.mean(xsat[ii, epos])
        y[ii, jj] = np.mean(ysat[ii, epos])
        z[ii, jj] = np.mean(zsat[ii, epos])

        data['sample_gpm'][ii, jj] = np.sum(epos)  # Nb of profiles in layer
        data['volume_match_gpm'][ii, jj] = np.sum(volsat[ii, epos])  # Total GPM volume in layer

        dz[ii, jj] = np.sum(epos) * gpmset.dr * np.cos(np.deg2rad(alpha[ii]))  # Thickness of the layer
        ds[ii, jj] = np.deg2rad(gpmset.beamwidth) * np.mean((gpmset.altitude - zsat[ii, epos])) / np.cos(np.deg2rad(alpha[ii]))  # Width of layer
        r[ii, jj] = (gpmset.earth_gaussian_radius + zsat[ii, jj]) * np.sin(s_sat[ii, jj] / gpmset.earth_gaussian_radius) / np.cos(np.deg2rad(elev_gr[jj]))

        if r[ii, jj] + ds[ii, jj] / 2 > rmax_gr:
            # More than half the sample is outside of the radar last bin.
            continue

        refl_gpm = refl_gpm_raw[ii, epos].flatten()
        refl_gpm_s = refl_gpm_stratgrband[ii, epos].flatten()
        refl_gpm_c = refl_gpm_convgrband[ii, epos].flatten()

        if np.all(np.isnan(refl_gpm)):
            continue

        data['refl_gpm_raw'][ii, jj] = np.nanmean(refl_gpm)
        data['refl_gpm_strat'][ii, jj] = np.nanmean(refl_gpm_s)
        data['refl_gpm_conv'][ii, jj] = np.nanmean(refl_gpm_c)
        data['std_refl_gpm'][ii, jj] = np.nanstd(refl_gpm)

        data['zrefl_gpm_raw'][ii, jj] = 10 * np.log10(np.nanmean(10 ** (refl_gpm / 10)))
        data['zrefl_gpm_strat'][ii, jj] = 10 * np.log10(np.nanmean(10 ** (refl_gpm_s / 10)))
        data['zrefl_gpm_conv'][ii, jj] = 10 * np.log10(np.nanmean(10 ** (refl_gpm_c / 10)))
        data['std_zrefl_gpm'][ii, jj] = 10 * np.log10(np.nanstd(10 ** (refl_gpm / 10)))

        # Number of rejected bins
        data['reject_gpm'][ii, jj] = np.sum(epos) - np.sum(np.isnan(refl_gpm))

        # Ground radar time.
        sl = radar.get_slice(jj)
        d = np.sqrt((xradar[sl] - x[ii, jj]) ** 2 + (yradar[sl] - y[ii, jj]) ** 2)
        rpos = (d <= ds[ii, jj] / 2)

        data['reject_gr'][ii, jj] = np.sum(rpos)
        data['volume_match_gr'][ii, jj] = np.sum(volgr[sl][rpos])
        if np.sum(rpos) == 0:
            continue

        refl_gr_raw = ground_radar_reflectivity[sl][rpos].flatten()
        zrefl_gr_raw = 10 ** (refl_gr_raw / 10)
        w = volgr[sl][rpos] * np.exp(-(d[rpos] / (ds[ii, jj] / 2)) ** 2)

        data['refl_gr_weigthed'][ii, jj] = np.sum(w * refl_gr_raw) / np.sum(w[~refl_gr_raw.mask])
        data['refl_gr_raw'][ii, jj] = np.mean(refl_gr_raw)

        data['zrefl_gr_weigthed'][ii, jj] = 10 * np.log10(np.sum(w * zrefl_gr_raw) / np.sum(w[~refl_gr_raw.mask]))
        data['zrefl_gr_raw'][ii, jj] = 10 * np.log10(np.mean(zrefl_gr_raw))

        data['std_refl_gr'][ii, jj] = np.std(refl_gr_raw)
        data['std_zrefl_gr'][ii, jj] = 10 * np.log10(np.std(10 ** (refl_gr_raw / 10)))

        data['sample_gr'][ii, jj] = np.sum(~refl_gr_raw.mask)

    data['x'] = x
    data['y'] = y
    data['z'] = z
    data['r'] = r
    data['nprof'] = np.arange(nprof, dtype=np.int32)
    data['ntilt'] = np.arange(ntilt, dtype=np.int32)
    data['elevation_gr'] = elev_gr[:ntilt]

    # Transform to xarray and build metadata
    match = dict()
    for k, v in data.items():
        if k in ['ntilt', 'elevation_gr']:
            match[k] = (('ntilt'), v)
        elif k == 'nprof':
            match[k] = (('nprof'), v)
        else:
            match[k] = (('nprof', 'ntilt'), np.ma.masked_invalid(v.astype(np.float32)))

    matchset = xr.Dataset(match)
    metadata = get_metadata()

    for k, v in metadata.items():
        for sk, sv in v.items():
            matchset[k].attrs[sk] = sv

    radar_start_time = netCDF4.num2date(radar.time['data'][0], radar.time['units']).isoformat()
    radar_end_time = netCDF4.num2date(radar.time['data'][-1], radar.time['units']).isoformat()

    ar = gpmset.x ** 2 + gpmset.y ** 2
    iscan, _, _ = np.where(ar == ar.min())
    gpm_overpass_time = pd.Timestamp(gpmset.nscan[iscan[0]].values).isoformat()
    gpm_mindistance = np.sqrt(gpmset.x ** 2 + gpmset.y ** 2)[:, :, 0].values[gpmset.flagPrecip > 0].min()
    offset = np.nanmean((matchset['refl_gpm_raw'] - matchset['refl_gr_weigthed']).values)

    matchset.attrs['estimated_calibration_offset'] = f'{offset:0.4} dB'
    matchset.attrs['radar_start_time'] = radar_start_time
    matchset.attrs['radar_end_time'] = radar_end_time
    matchset.attrs['gpm_overpass_time'] = gpm_overpass_time
    matchset.attrs['gpm_min_distance'] = np.round(gpm_mindistance)
    matchset.attrs['radar_longitude'] = radar.longitude['data'][0]
    matchset.attrs['radar_latitude'] = radar.latitude['data'][0]
    matchset.attrs['gpm_orbit'] = gpmset.attrs['orbit']

    matchset.attrs['country'] = 'Australia'
    matchset.attrs['creator_email'] = 'valentin.louf@bom.gov.au'
    matchset.attrs['creator_name'] = 'Valentin Louf'
    matchset.attrs['date_created'] = datetime.datetime.now().isoformat()
    matchset.attrs['uuid'] = str(uuid.uuid4())
    matchset.attrs['institution'] = 'Bureau of Meteorology'
    matchset.attrs['references'] = 'doi:10.1175/JTECH-D-18-0007.1'
    matchset.attrs['naming_authority'] = 'au.org.nci'
    matchset.attrs['summary'] = 'GPM volume matching technique.'
    matchset.attrs['field_names'] = ", ".join(sorted([k for k, v in matchset.items()]))
    matchset.attrs['history'] = f"Created by {matchset.attrs['creator_name']} on {os.uname()[1]} at {matchset.attrs['date_created']} using Py-ART."

    if write_output:
        date = netCDF4.num2date(radar.time['data'][0], radar.time['units']).strftime('%Y%m%d.%H%M')
        outfilename = f"vmatch.gpm.orbit.{gpmset.attrs['orbit']:07}.{fname_prefix}.{date}.nc"
        if not os.path.exists(os.path.join(output_dir, outfilename)):

            matchset.to_netcdf(os.path.join(output_dir, outfilename),
                               encoding={k : {'zlib': True} for k in [k for k, v in matchset.items()]})
        else:
            print('Output file already exists.')

    del radar, gpmset
    return matchset