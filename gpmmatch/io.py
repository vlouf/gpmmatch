'''
Utilities to read the input data and format them in a way to be read by
volume_matching.

@title: io
@author: Valentin Louf <valentin.louf@bom.gov.au>
@institutions: Monash University and the Australian Bureau of Meteorology
@creation: 17/02/2020
@date: 20/05/2020

    NoPrecipitationError
    get_gpm_orbit
    read_GPM
    data_load_and_checks
'''
import re
import datetime
import warnings

import h5py
import pyart
import pyproj
import cftime
import numpy as np
import xarray as xr

from . import correct


class NoPrecipitationError(Exception):
    pass


def _read_radar(infile, refl_name=None):
    """
    Read input radar file

    Parameters:
    ===========
    radar_file_list: str
        List of radar files.
    refl_name: str
        Reflectivity field name.

    Returns:
    ========
    radar: PyART.Radar
        Radar data.
    """
    try:
        if infile.lower().endswith(('.h5', '.hdf', '.hdf5')):
            radar = pyart.aux_io.read_odim_h5(infile, include_fields=[refl_name])
        else:
            radar = pyart.io.read(infile, include_fields=[refl_name])
    except Exception:
        print(f'!!!! Problem with {infile} !!!!')
        raise

    if refl_name is not None:
        try:
            radar.fields[refl_name]
        except KeyError:
            print(f'!!!! Problem with {infile} - No {refl_name} field does not exist. !!!!')
            del radar
            raise

    return radar


def get_gpm_orbit(gpmfile):
    '''
    Parameters:
    -----------
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


def read_GPM(infile, refl_min_thld=0):
    '''
    Read GPM data and organize them into a Dataset.

    Parameters:
    -----------
    gpmfile: str
        GPM data file.
    refl_min_thld: float
        Minimum threshold applied to GPM reflectivity.

    Returns:
    --------
    dset: xr.Dataset
        GPM dataset.
    '''
    if refl_min_thld != 0:
        warnings.warn('Tests have shown that no threshold should be applied to GPM reflectivity!', UserWarning)
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
                        gpm_refl = hid[f'/NS/{k}/{sk}'][:][:, :, ::-1]
                        gpm_refl[gpm_refl < 0] = np.NaN
                        data[sk] = (dims, np.ma.masked_invalid(np.ma.masked_less_equal(gpm_refl, refl_min_thld)))
                    elif sk == 'flagPrecip':
                        data[sk] = (dims, np.ma.masked_invalid(hid[f'/NS/{k}/{sk}'][:]).filled(0).astype(bool))
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

    try:
        # TRMM doesn't have a MilliSecond field.
        _ = date['MilliSecond']
    except KeyError:
        date['MilliSecond'] = date['Second']

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


def read_radar(grfile, grfile2, refl_name, gpm_time):
    '''
    Read ground radar data. If 2 files provided, then it will compute the
    displacement between these two files and then correct for advection the
    ground radar data in relation to the time of GPM exact overpass.

    Parameters:
    ===========
    grfile: str
        Ground radar input file.
    grfile2: str (optionnal)
        Second ground radar input file to compute grid displacement and
        advection.
    refl_name: str
        Name of the reflectivity field in the ground radar data.
    gpm_time: np.datetime64[s]
        Datetime of GPM overpass.

    Returns:
    ========
    radar: pyart.core.Radar
        Pyart radar dataset, corrected for advection if grfile2 provided.
    '''
    try:
        radar0 = _read_radar(grfile, refl_name)
    except Exception:
        raise

    if grfile2 is None:
        return radar0

    rtime = cftime.num2pydate(radar0.time['data'], radar0.time['units']).astype('datetime64[s]')
    timedelta = rtime - gpm_time

    # grfile2 is not None here.
    try:
        radar1 = _read_radar(grfile, refl_name)
    except Exception:
        print('!!! Could not read 2nd ground radar file, only using the first one !!!')
        return radar0

    t0 = cftime.num2pydate(radar0.time['data'][0], radar0.time['units'])
    t1 = cftime.num2pydate(radar1.time['data'][0], radar1.time['units'])
    if t1 > t0:
        dt = (t1 - t0).seconds
    else:
        # It's a datetime object, so if it's not orderly sequential,
        # it will returns days of difference.
        dt = (t0 - t1).seconds

    if dt > 1800:
        raise ValueError(f"Cannot advect the ground radar data, the 2 input files are separated by more than 30min (dt = {dt}s).")

    # TODO: Make domain a bit larger 100x100 km and then mask the part outside
    # the 80x80 km window.
    grid0 = pyart.map.grid_from_radars(radar0,
                               grid_shape=(1, 801, 801),
                               grid_limits=((2500, 25000),(-80000, 80000), (-80000, 80000)),
                               fields=[refl_name],
                               gridding_algo="map_gates_to_grid",
                               constant_roi=1000,
                               weighting_function='Barnes2')

    grid1 = pyart.map.grid_from_radars(radar1,
                                       grid_shape=(1, 801, 801),
                                       grid_limits=((2500, 25000),(-80000, 80000), (-80000, 80000)),
                                       fields=[refl_name],
                                       gridding_algo="map_gates_to_grid",
                                       constant_roi=1000,
                                       weighting_function='Barnes2')

    r0 = np.squeeze(grid0.fields[refl_name]['data'])
    r1 = np.squeeze(grid1.fields[refl_name]['data'])
    x = np.squeeze(grid0.point_x['data'])
    y = np.squeeze(grid0.point_y['data'])
    pos = np.sqrt(x ** 2 + y ** 2) < 20e3
    r0[pos] = np.NaN
    r1[pos] = np.NaN

    displacement = correct.grid_displacement(r0, r1)
    dxdt = 200 * displacement[0] / dt  # Grid resolution is 200m.
    dydt = 200 * displacement[1] / dt

    xoffset = dxdt * timedelta.astype(int)
    yoffset = dydt * timedelta.astype(int)
    xoffset = np.repeat(xoffset[:, np.newaxis], radar0.ngates, axis=1)
    yoffset = np.repeat(yoffset[:, np.newaxis], radar0.ngates, axis=1)

    radar0.gate_x['data'] = radar0.gate_x['data'] + xoffset
    radar0.gate_y['data'] = radar0.gate_y['data'] + yoffset

    del grid0, grid1, radar1
    return radar0


def get_ground_radar_attributes(grfile):
    '''
    Read the ground radar attributes, latitude/longitude, altitude, range
    min/max.

    Parameter:
    ==========
    grfile: str
        Input ground radar file.

    Returns:
    ========
    grlon: float
        Radar longitude.
    grlat: float
        Radar latitude.
    gralt: float
        Radar altitude.
    rmin : float
        Radar minimum range (cone of silence.)
    rmax: float
        Radar maximum range.
    '''
    radar = _read_radar(grfile, None)

    rmax = radar.range['data'].max()
    rmin = 20e3
    grlon = radar.longitude['data'][0]
    grlat = radar.latitude['data'][0]
    gralt = radar.altitude['data'][0]

    del radar
    return grlon, grlat, gralt, rmin, rmax


def data_load_and_checks(gpmfile,
                         grfile,
                         grfile2=None,
                         refl_name=None,
                         radar_band='C'):
    '''
    Load GPM and Ground radar files and perform some initial checks:
    domains intersect, precipitation, time difference.

    Parameters:
    -----------
    gpmfile: str
        GPM data file.
    grfile: str
        Ground radar input file.
    grfile2: str
        (Optional) Second ground radar input file to compute grid displacement
        and advection.
    refl_name: str
        Name of the reflectivity field in the ground radar data.
    radar_band: str
        Ground radar frequency band for reflectivity conversion. S, C, and X
        supported.

    Returns:
    --------
    gpmset: xarray.Dataset
        Dataset containing the input datas.
    radar: pyart.core.Radar
        Pyart radar dataset.
    '''
    if refl_name is None:
        raise ValueError('Reflectivity field name not given.')
    if grfile2 is None:
        gpmtime0 = 0

    gpmset = read_GPM(gpmfile)
    grlon, grlat, gralt, rmin, rmax = get_ground_radar_attributes(grfile)

    # Reproject satellite coordinates onto ground radar
    georef = pyproj.Proj(f"+proj=aeqd +lon_0={grlon} +lat_0={grlat} +ellps=WGS84")
    gpmlat = gpmset.Latitude.values
    gpmlon = gpmset.Longitude.values

    xgpm, ygpm = georef(gpmlon, gpmlat)
    rproj_gpm = (xgpm ** 2 + ygpm ** 2) ** 0.5

    gr_domain = (rproj_gpm <= rmax) & (rproj_gpm >= rmin)
    if gr_domain.sum() < 10:
        info = f'The closest satellite measurement is {np.min(rproj_gpm / 1e3):0.4} km away from ground radar.'
        if gr_domain.sum() == 0:
            raise NoPrecipitationError('GPM swath does not go through the radar domain. ' + info)
        else:
            raise NoPrecipitationError('Not enough GPM precipitation inside ground radar domain. ' + info)

    nprof = np.sum(gpmset.flagPrecip.values[gr_domain])
    if nprof < 10:
        raise NoPrecipitationError('No precipitation measured by GPM inside radar domain.')

    # Parallax correction
    sr_xp, sr_yp, z_sr = correct.correct_parallax(xgpm, ygpm, gpmset)

    # Compute the elevation of the satellite bins with respect to the ground radar.
    gr_gaussian_radius = correct.compute_gaussian_curvature(grlat)
    gamma = np.sqrt(sr_xp ** 2 + sr_yp ** 2) / gr_gaussian_radius
    elev_sr_grref = np.rad2deg(np.arctan2(np.cos(gamma) - (gr_gaussian_radius + gralt) / (gr_gaussian_radius + z_sr), np.sin(gamma)))

    # Convert reflectivity band correction
    reflgpm_grband = correct.convert_gpmrefl_grband_dfr(gpmset.zFactorCorrected.values, radar_band=radar_band)

    gpmset = gpmset.merge({'precip_in_gr_domain':  (('nscan', 'nray'), gpmset.flagPrecip.values & gr_domain),
                           'range_from_gr': (('nscan', 'nray'), rproj_gpm),
                           'elev_from_gr': (('nscan', 'nray', 'nbin'), elev_sr_grref),
                           'x': (('nscan', 'nray', 'nbin'), sr_xp),
                           'y': (('nscan', 'nray', 'nbin'), sr_yp),
                           'z': (('nscan', 'nray', 'nbin'), z_sr),
                           'reflectivity_grband': (('nscan', 'nray', 'nbin'), reflgpm_grband)})

    gpmtime0 = gpmset.nscan.where(gpmset.range_from_gr == gpmset.range_from_gr.min()).values.astype('datetime64[s]')
    gpmtime0 = gpmtime0[~np.isnat(gpmtime0)][0]
    gpmset = gpmset.merge({'overpass_time': (gpmtime0)})

    # Attributes
    gpmset.overpass_time.attrs['description'] = 'GPM overpass time at the closest from ground radar site'
    gpmset.x.attrs['units'] = 'm'
    gpmset.x.attrs['description'] = 'x-axis parallax corrected coordinates in relation to ground radar.'
    gpmset.y.attrs['units'] = 'm'
    gpmset.y.attrs['description'] = 'y-axis parallax corrected coordinates in relation to ground radar.'
    gpmset.z.attrs['units'] = 'm'
    gpmset.z.attrs['description'] = 'z-axis parallax corrected coordinates in relation to ground radar.'
    gpmset.precip_in_gr_domain.attrs['units'] = 'bool'
    gpmset.precip_in_gr_domain.attrs['description'] = 'Satellite data-columns with precipitation inside the ground radar scope.'
    gpmset.range_from_gr.attrs['units'] = 'm'
    gpmset.range_from_gr.attrs['description'] = 'Range from satellite bins in relation to ground radar'
    gpmset.elev_from_gr.attrs['units'] = 'degrees'
    gpmset.elev_from_gr.attrs['description'] = 'Elevation from satellite bins in relation to ground radar'
    gpmset.reflectivity_grband.attrs['units'] = 'dBZ'
    gpmset.reflectivity_grband.attrs['description'] = f'Satellite reflectivity converted to {radar_band}-band.'
    gpmset.attrs['nprof'] = nprof
    gpmset.attrs['earth_gaussian_radius'] = gr_gaussian_radius

    # Now it's turn to read the ground radar.
    radar = read_radar(grfile, grfile2, refl_name, gpm_time=gpmtime0)
    if radar_band in ['X', 'C']:
        if radar_band == 'X':
            corr_refl = correct.correct_xband_attenuation(radar.fields[refl_name]['data'])
        elif radar_band == 'C':
            corr_refl = correct.correct_cband_attenuation(radar.fields[refl_name]['data'])
        refl_dict = radar.fields.pop(refl_name)
        refl_dict['data'] = corr_refl
        radar.add_field(refl_name, refl_dict)

    return gpmset, radar
