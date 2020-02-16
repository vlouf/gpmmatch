import re
import datetime

import h5py
import pyart
import pyproj
import numpy as np
import xarray as xr

import correct


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

    # Reproject satellite coordinates onto ground radar
    georef = pyproj.Proj(f"+proj=aeqd +lon_0={grlon} +lat_0={grlat} +ellps=WGS84")
    gpmlat = gpmset.Latitude.values
    gpmlon = gpmset.Longitude.values

    xgpm, ygpm = georef(gpmlon, gpmlat)
    rproj_gpm = (xgpm ** 2 + ygpm ** 2) ** 0.5
    # Checks.
    if gpmlon.shape != rproj_gpm.shape:
        raise IndexError(f'Shape mismatch gpm and rprog {gpmlon.shape}, {rproj_gpm.shape}.')

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
    sr_xp, sr_yp, z_sr = correct.correct_parallax(xgpm, ygpm, gpmset)

    # Compute the elevation of the satellite bins with respect to the ground radar.
    gr_gaussian_radius = correct.compute_gaussian_curvature(grlat)
    gamma = np.sqrt(sr_xp ** 2 + sr_yp ** 2) / gr_gaussian_radius
    elev_sr_grref = np.rad2deg(np.arctan2(np.cos(gamma) - (gr_gaussian_radius + radar.altitude['data']) / (gr_gaussian_radius + z_sr), np.sin(gamma)))

    # Convert reflectivity band correction
    refp_strat, refp_conv = correct.convert_sat_refl_to_gr_band(gpmset.zFactorCorrected.values,
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