'''
Volume matching of ground radar and GPM satellite.

@title: gpmmatch
@author: Valentin Louf <valentin.louf@bom.gov.au>
@institutions: Monash University and the Australian Bureau of Meteorology
@creation: 17/02/2020
@date: 04/03/2020
    volume_matching
'''
import os
import uuid
import datetime
import itertools

import netCDF4
import numpy as np
import pandas as pd
import xarray as xr

from .io import data_load_and_checks
from .default import get_metadata


def volume_matching(gpmfile,
                    grfile,
                    grfile2=None,
                    radar_band='C',
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
    gpmset, radar = data_load_and_checks(gpmfile,
                                         grfile,
                                         grfile2=grfile2,
                                         refl_name=refl_name,
                                         gpm_refl_threshold=gpm_refl_threshold,
                                         radar_band=radar_band)

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
        ground_radar_reflectivity = radar.fields[refl_name]['data'].filled(np.NaN)
    except Exception:
        ground_radar_reflectivity = radar.fields[refl_name]['data']
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

    # Compute Path-integrated reflectivities
    pir_gr = 10 * np.log10(np.cumsum((10 ** (ground_radar_reflectivity / 10)).filled(0), axis=1) * dr)
    pir_gr = np.ma.masked_invalid(pir_gr)

    pir_gpm = 10 * np.log10(np.cumsum((10 ** (np.ma.masked_invalid(refl_gpm_raw) / 10)).filled(0), axis=-1) * 125)
    pir_gpm = np.ma.masked_invalid(pir_gpm)

    # Initialising output data.
    datakeys = ['refl_gpm_raw', 'refl_gr_weigthed', 'refl_gpm_strat', 'refl_gpm_conv', 'pir_gpm', 'pir_gr',
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
        data['pir_gpm'][ii, jj] = np.nanmean(pir_gpm[ii, epos].flatten())
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
        data['refl_gr_raw'][ii, jj] = np.mean(pir_gr[sl][rpos].flatten())

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
