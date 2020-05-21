'''
Volume matching of ground radar and GPM satellite. It also works with the
latest version of TRMM data.

@title: gpmmatch
@author: Valentin Louf <valentin.louf@bom.gov.au>
@institutions: Monash University and the Australian Bureau of Meteorology
@creation: 17/02/2020
@date: 20/05/2020
    _mkdir
    get_offset    
    volume_matching
    vmatch_multi_pass
'''
import os
import uuid
import datetime
import warnings
import itertools

import cftime
import numpy as np
import pandas as pd
import xarray as xr

from scipy.stats import mode

from .io import data_load_and_checks, savedata
from .default import get_metadata


class NoRainError(Exception):
    pass


def _mkdir(dir):
    """
    Make directory. Might seem redundant but you might have concurrency issue
    when dealing with multiprocessing.
    """
    if os.path.exists(dir):
        return None

    try:
        os.mkdir(dir)
    except FileExistsError:
        pass

    return None


def get_offset(matchset, loose=False) -> float:
    '''
    Compute the Offset between GR and GPM.

    Parameter:
    ==========
    matchset: xr.Dataset
        Dataset of volume matching.
    loose: bool
        Use of STD to constrain the offset calculation. Better quality if false
        but less likely to find a solution.

    Returns:
    ========
    offset: float
        Offset between GR and GPM
    '''
    refl_gpm = matchset.refl_gpm_grband.values
    refl_gr = matchset.refl_gr_weigthed.values
    std_refl_gpm = matchset.std_refl_gpm.values
    std_refl_gr = matchset.std_refl_gr.values
    r = matchset.r.values

    if loose:
        pos = ((~np.isnan(refl_gpm)) &
               (~np.isnan(refl_gr)) &
               (refl_gr >= 21) &
               (refl_gr <= 36))
    else:
        pos = ((std_refl_gpm > 0.2) & (std_refl_gpm < 5) &
               (std_refl_gr > 0) & (std_refl_gr < 5) &
               (np.abs(refl_gpm - refl_gr) < 15) &
               (~np.isnan(refl_gpm)) &
               (~np.isnan(refl_gr)) &
               (refl_gr >= 21) &
               (refl_gr <= 36))

    x1 = refl_gpm[pos]
    x2 = refl_gr[pos]
    deltax = (x2 - x1)
    deltax = deltax[~np.isnan(deltax)]

    m, _ = mode(np.round(deltax * 2) / 2, nan_policy='omit')
    npos = ((deltax < m[0] + deltax.std()) & (deltax > m[0] - deltax.std()))
    if np.sum(npos) == 0:
        return np.NaN
    offset = deltax[npos].mean()

    return offset


def volume_matching(gpmfile,
                    grfile,
                    grfile2=None,
                    gr_offset=0,
                    gr_beamwidth=1,
                    gr_refl_threshold=10,
                    radar_band='C',
                    refl_name='corrected_reflectivity',
                    fname_prefix=None,
                    is_loose_offset=False):
    '''
    Performs the volume matching of GPM satellite data to ground based radar.

    Parameters:
    ----------
    gpmfile: str
        GPM data file.
    grfile: str
        Ground radar input file.
    grfile2: str
        Second ground radar input file to compute the advection.
    gr_offset: float
        Offset to add to the reflectivity of the ground radar data.
    gr_beamwidth: float
        Ground radar 3dB-beamwidth.
    gr_refl_thresold: float
        Minimum reflectivity threshold on ground radar data.
    radar_band: str
        Ground radar frequency band.
    refl_name: str
        Name of the reflectivity field in the ground radar data.
    fname_prefix: str
        Name of the ground radar to use as label for the output file.
    is_loose_offset: bool
        Computing offset using the strict way or a less accurate way (but with
        a higher likelyhood to yield a result).

    Returns:
    --------
    matchset: xarray.Dataset
        Dataset containing the matched GPM and ground radar data.
    '''
    if gr_refl_threshold > 10:
        warnings.warn('Tests have shown that the ideal ground radar reflectivity threshold is about 10 dB.', UserWarning)
    if fname_prefix is None:
        fname_prefix = 'unknown_radar'

    gpmset, radar = data_load_and_checks(gpmfile,
                                         grfile,
                                         grfile2=grfile2,
                                         refl_name=refl_name,
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
    tradar = cftime.num2pydate(radar.time['data'], radar.time['units']).astype('datetime64')
    deltat = (tradar - gpmset.overpass_time.values)

    rmax_gr = range_gr.max()
    dr = range_gr[1] - range_gr[0]

    R, _ = np.meshgrid(radar.range['data'], radar.azimuth['data'])
    _, DT = np.meshgrid(radar.range['data'], deltat)

    # Substract offset to the ground radar reflectivity
    ground_radar_reflectivity = radar.fields[refl_name]['data'].copy().filled(np.NaN) - gr_offset
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

    refl_gpm_raw = np.ma.masked_invalid(gpmset.zFactorCorrected.values[position_precip_domain])
    reflectivity_gpm_grband = np.ma.masked_invalid(gpmset.reflectivity_grband.values[position_precip_domain])

    volsat = 1e-9 * gpmset.dr * (rsat[position_precip_domain] * np.deg2rad(gpmset.beamwidth) / 2) ** 2  # km3
    volgr = 1e-9 * np.pi * dr * (R * np.pi / 180 * gr_beamwidth / 2) ** 2  # km3

    # Compute Path-integrated reflectivities
    pir_gr = 10 * np.log10(np.cumsum((10 ** (ground_radar_reflectivity / 10)).filled(0), axis=1) * dr)
    pir_gr = np.ma.masked_invalid(pir_gr)

    pir_gpm = 10 * np.log10(np.cumsum((10 ** (np.ma.masked_invalid(refl_gpm_raw) / 10)).filled(0), axis=-1) * 125)
    pir_gpm = np.ma.masked_invalid(pir_gpm)

    # Initialising output data.
    datakeys = ['refl_gpm_raw', 'refl_gr_weigthed', 'refl_gpm_grband', 'pir_gpm', 'pir_gr',
                'refl_gr_raw', 'std_refl_gpm', 'std_refl_gr', 'zrefl_gpm_raw', 'zrefl_gr_weigthed',
                'zrefl_gpm_grband', 'zrefl_gr_raw', 'std_zrefl_gpm', 'std_zrefl_gr', 'sample_gpm', 'reject_gpm',
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
    delta_t = np.zeros((nprof, ntilt)) + np.NaN  # Timedelta of sample

    for ii, jj in itertools.product(range(nprof), range(ntilt)):
        epos = (elev_sat[ii, :] >= elev_gr[jj] - gr_beamwidth / 2) & (elev_sat[ii, :] <= elev_gr[jj] + gr_beamwidth / 2)
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

        # Ground radar side:
        sl = radar.get_slice(jj)
        roi_gr_at_vol = np.sqrt((xradar[sl] - x[ii, jj]) ** 2 + (yradar[sl] - y[ii, jj]) ** 2)
        rpos = (roi_gr_at_vol <= ds[ii, jj] / 2)
        w = volgr[sl][rpos] * np.exp(-(roi_gr_at_vol[rpos] / (ds[ii, jj] / 2)) ** 2)
        if np.sum(rpos) == 0:
            continue

        # Extract reflectivity for volume.
        refl_gpm = refl_gpm_raw[ii, epos].flatten()
        refl_gpm_grband = reflectivity_gpm_grband[ii, epos].flatten()
        refl_gr_raw = ground_radar_reflectivity[sl][rpos].flatten()
        zrefl_gr_raw = 10 ** (refl_gr_raw / 10)
        delta_t[ii, jj] = np.max(DT[sl][rpos])

        if np.all(np.isnan(refl_gpm.filled(np.NaN))):
            continue
        if np.all(np.isnan(refl_gr_raw.filled(np.NaN))):
            continue
        if np.sum(refl_gpm > 0) / np.sum(~np.isnan(refl_gpm)) < 0.7:
            # fmin parameter (Fig 5 Rob's paper).
            continue
        if np.sum(refl_gr_raw >= gr_refl_threshold) / np.sum(~np.isnan(refl_gr_raw)) < 0.7:
            continue

        # GPM
        data['refl_gpm_raw'][ii, jj] = np.mean(refl_gpm)
        data['refl_gpm_grband'][ii, jj] = np.mean(refl_gpm_grband)
        data['pir_gpm'][ii, jj] = np.mean(pir_gpm[ii, epos].flatten())
        data['std_refl_gpm'][ii, jj] = np.std(refl_gpm)
        data['zrefl_gpm_raw'][ii, jj] = 10 * np.log10(np.mean(10 ** (refl_gpm / 10)))
        data['zrefl_gpm_grband'][ii, jj] = 10 * np.log10(np.mean(10 ** (refl_gpm_grband / 10)))
        data['std_zrefl_gpm'][ii, jj] = 10 * np.log10(np.std(10 ** (refl_gpm / 10)))
        data['reject_gpm'][ii, jj] = np.sum(epos) - np.sum(refl_gpm.mask)  # Number of rejected bins

        # Ground radar.
        data['volume_match_gr'][ii, jj] = np.sum(volgr[sl][rpos])
        data['refl_gr_weigthed'][ii, jj] = np.sum(w * refl_gr_raw) / np.sum(w[~refl_gr_raw.mask])
        data['refl_gr_raw'][ii, jj] = np.mean(refl_gr_raw)
        data['pir_gr'][ii, jj] = np.mean(pir_gr[sl][rpos].flatten())
        data['zrefl_gr_weigthed'][ii, jj] = 10 * np.log10(np.sum(w * zrefl_gr_raw) / np.sum(w[~refl_gr_raw.mask]))
        data['zrefl_gr_raw'][ii, jj] = 10 * np.log10(np.mean(zrefl_gr_raw))
        data['std_refl_gr'][ii, jj] = np.std(refl_gr_raw)
        data['std_zrefl_gr'][ii, jj] = 10 * np.log10(np.std(10 ** (refl_gr_raw / 10)))
        data['reject_gr'][ii, jj] = np.sum(rpos)
        data['sample_gr'][ii, jj] = np.sum(~refl_gr_raw.mask)

    data['x'] = x
    data['y'] = y
    data['z'] = z
    data['r'] = r
    data['nprof'] = np.arange(nprof, dtype=np.int32)
    data['ntilt'] = np.arange(ntilt, dtype=np.int32)
    data['elevation_gr'] = elev_gr[:ntilt]
    data['timedelta'] = delta_t

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
            try:
                matchset[k].attrs[sk] = sv
            except KeyError:
                continue

    ar = gpmset.x ** 2 + gpmset.y ** 2
    iscan, _, _ = np.where(ar == ar.min())
    gpm_overpass_time = pd.Timestamp(gpmset.nscan[iscan[0]].values).isoformat()
    gpm_mindistance = np.sqrt(gpmset.x ** 2 + gpmset.y ** 2)[:, :, 0].values[gpmset.flagPrecip > 0].min()
    offset = get_offset(matchset, is_loose_offset)

    # if np.isnan(offset):
    #     raise NoRainError('No offset found.')

    radar_start_time = cftime.num2pydate(radar.time['data'][0], radar.time['units']).isoformat()
    radar_end_time = cftime.num2pydate(radar.time['data'][-1], radar.time['units']).isoformat()
    date = cftime.num2pydate(radar.time['data'][0], radar.time['units']).strftime('%Y%m%d.%H%M')
    outfilename = f"vmatch.gpm.orbit.{gpmset.attrs['orbit']:07}.{fname_prefix}.{date}.nc"

    matchset.attrs['offset_applied'] = gr_offset
    matchset.attrs['offset_found'] = offset
    matchset.attrs['final_offset'] = gr_offset + offset
    matchset.attrs['estimated_calibration_offset'] = f'{offset:0.4} dB'
    matchset.attrs['gpm_overpass_time'] = gpm_overpass_time
    matchset.attrs['gpm_min_distance'] = np.round(gpm_mindistance)
    matchset.attrs['gpm_orbit'] = gpmset.attrs['orbit']
    matchset.attrs['radar_start_time'] = radar_start_time
    matchset.attrs['radar_end_time'] = radar_end_time
    matchset.attrs['radar_longitude'] = radar.longitude['data'][0]
    matchset.attrs['radar_latitude'] = radar.latitude['data'][0]
    matchset.attrs['country'] = 'Australia'
    matchset.attrs['creator_email'] = 'valentin.louf@bom.gov.au'
    matchset.attrs['creator_name'] = 'Valentin Louf'
    matchset.attrs['date_created'] = datetime.datetime.now().isoformat()
    matchset.attrs['uuid'] = str(uuid.uuid4())
    matchset.attrs['institution'] = 'Bureau of Meteorology'
    matchset.attrs['references'] = 'doi:10.1175/JTECH-D-18-0007.1 ; doi:10.1175/JTECH-D-17-0128.1'
    matchset.attrs['disclaimer'] = 'If you are using this data/technique for a scientific publication, please cite the papers given in references.'
    matchset.attrs['naming_authority'] = 'au.org.nci'
    matchset.attrs['summary'] = 'GPM volume matching technique.'
    matchset.attrs['field_names'] = ", ".join(sorted([k for k, v in matchset.items()]))
    matchset.attrs['filename'] = outfilename
    try:
        history = f"Created by {matchset.attrs['creator_name']} on {os.uname()[1]} at {matchset.attrs['date_created']} using Py-ART."
    except AttributeError:  # Windows OS.
        history = f"Created by {matchset.attrs['creator_name']} at {matchset.attrs['date_created']} using Py-ART."
    matchset.attrs['history'] = history

    del radar, gpmset
    return matchset


def vmatch_multi_pass(gpmfile,
                      grfile,
                      grfile2=None,
                      gr_offset=0,
                      gr_beamwidth=1,
                      gr_refl_threshold=10,
                      radar_band='C',
                      refl_name='corrected_reflectivity',
                      fname_prefix=None,
                      output_dir=None,
                      is_loose_offset=False):
    '''
    Multi-pass volume matching with automatic offset computation.

    Parameters:
    ----------
    gpmfile: str
        GPM data file.
    grfile: str
        Ground radar input file.
    grfile2: str
        Second ground radar input file to compute the advection.
    gr_offset: float
        Offset to add to the reflectivity of the ground radar data.
    gr_beamwidth: float
        Ground radar 3dB-beamwidth.
    gr_refl_thresold: float
        Minimum reflectivity threshold on ground radar data.
    radar_band: str
        Ground radar frequency band.
    refl_name: str
        Name of the reflectivity field in the ground radar data.
    fname_prefix: str
        Name of the ground radar to use as label for the output file.
    output_dir: str
        Path to output directory.
    is_loose_offset: bool
        Computing offset using the strict way or a less accurate way (but with
        a higher likelyhood to yield a result).
    '''
    def _save(dset, output_directory):
        '''
        Generate multipass metadata and file name.
        '''
        dset.attrs['iteration_number'] = counter
        matchset.attrs['offset_history'] = ",".join([f'{float(i):0.3}' for i in offset_keeping_track])
        outfilename = dset.attrs['filename'].replace('.nc', f'.pass{counter}.nc')
        savedata(dset, output_directory, outfilename)
        return None

    counter = 0
    offset_thld = 0.5
    if fname_prefix is None:
        fname_prefix = 'unknown_radar'
    if output_dir is None:
        output_dir = os.getcwd()

    # Generate output directories.
    output_dir_first_pass = os.path.join(output_dir, 'first_pass')
    output_dir_final_pass = os.path.join(output_dir, 'final_pass')
    output_dir_inter_pass = os.path.join(output_dir, 'inter_pass')
    _mkdir(output_dir_first_pass)
    _mkdir(output_dir_final_pass)
    _mkdir(output_dir_inter_pass)

    # First pass
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        matchset = volume_matching(gpmfile,
                                   grfile,
                                   grfile2=grfile2,
                                   gr_offset=gr_offset,
                                   radar_band=radar_band,
                                   refl_name=refl_name,
                                   fname_prefix=fname_prefix,
                                   gr_beamwidth=gr_beamwidth,
                                   gr_refl_threshold=gr_refl_threshold,
                                   is_loose_offset=is_loose_offset)
    pass_offset = matchset.attrs['offset_found']
    gr_offset = pass_offset

    offset_keeping_track = [pass_offset]
    final_offset_keeping_track = [matchset.attrs['final_offset']]    
    _save(matchset, output_dir_first_pass)

    if np.isnan(pass_offset):
        print('Pass offset NAN.')
        return None

    # Multiple pass as long as the difference is more than 1dB or counter is 6
    while (np.abs(pass_offset) > offset_thld) or (counter < 6):
        offset_thld = 1
        counter += 1
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            new_matchset = volume_matching(gpmfile,
                                           grfile,
                                           grfile2=grfile2,
                                           gr_offset=gr_offset,
                                           radar_band=radar_band,
                                           refl_name=refl_name,
                                           fname_prefix=fname_prefix,
                                           gr_beamwidth=gr_beamwidth,
                                           gr_refl_threshold=gr_refl_threshold,
                                           is_loose_offset=is_loose_offset)

        # Save intermediary file.        
        _save(new_matchset, output_dir_inter_pass)

        # Check offset found.
        gr_offset = new_matchset.attrs['final_offset']
        pass_offset = new_matchset.attrs['offset_found']
        if (np.abs(pass_offset) > np.abs(offset_keeping_track[-1])) or np.isnan(pass_offset):
            # Solution converged already. Using previous iteration as final result.
            counter -= 1
            break

        matchset = new_matchset  # No error with results.
        offset_keeping_track.append(pass_offset)
        final_offset_keeping_track.append(gr_offset)
        if np.abs(pass_offset) > offset_thld:
            break

    # Save final iteration.
    _save(matchset, output_dir_final_pass)

    return None
