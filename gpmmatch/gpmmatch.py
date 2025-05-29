"""
Volume matching of ground radar and GPM satellite. It also works with the
latest version of TRMM data.

@title: gpmmatch
@author: Valentin Louf <valentin.louf@bom.gov.au>
@institutions: Monash University and the Australian Bureau of Meteorology
@creation: 17/02/2020
@date: 29/05/2025

.. autosummary::
    :toctree: generated/

    NoRainError
    get_radar_coordinates
    get_gr_reflectivity
    volume_matching
    vmatch_multi_pass
"""
import os
import uuid
import datetime
import platform
import warnings
import itertools
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
import xarray as xr

from .correct import get_offset
from .io import data_load_and_checks
from .io import savedata
from .io import _mkdir
from .default import get_metadata


class NoRainError(Exception):
    pass


def get_radar_coordinates(nradar: List[xr.Dataset], elevation_offset: Union[float, None] = None) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """
    Extracts the ground radar coordinates and elevation angles.

    Parameters:
    ----------
    nradar: list of xarray.Dataset
        List of ground radar datasets for each tilt (pyodim structure).
    elevation_offset: float, optional
        Offset to add to the elevation angles of the ground radar data.

    Returns:
    -------
    range_gr: np.ndarray
        Array of ground radar range values.
    elev_gr: np.ndarray
        Array of ground radar elevation angles.
    xradar: List[np.ndarray]
        List of x coordinates of the ground radar for each tilt.
    yradar: List[np.ndarray]
        List of y coordinates of the ground radar for each tilt.
    time_radar: List[np.ndarray]
        List of time values of the ground radar for each tilt.
    """
    range_gr = nradar[0].range.values
    elev_gr = np.array([r.elevation.values[0] for r in nradar])

    if elevation_offset is not None:
        print(f"Correcting the GR elevation by an offset of {elevation_offset}.")
        elev_gr += elevation_offset

    xradar = [r.x.values for r in nradar]
    yradar = [r.y.values for r in nradar]
    time_radar = [r.time.values for r in nradar]

    return range_gr, elev_gr, xradar, yradar, time_radar


def get_gr_reflectivity(nradar: List[xr.Dataset], refl_name: str, gr_offset: float, gr_refl_threshold: float) -> Tuple[List[np.ma.MaskedArray], List[np.ndarray]]:
    """
    Extracts the ground radar reflectivity and computes the path-integrated reflectivity.
    Parameters:
    ----------
    nradar: list of xarray.Dataset
        List of ground radar datasets.
    refl_name: str
        Name of the reflectivity field in the ground radar data.
    gr_offset: float
        Offset to add to the reflectivity of the ground radar data.
    gr_refl_threshold: float
        Minimum reflectivity threshold on ground radar data.
    Returns:
    -------
    ground_radar_reflectivity: List[np.ndarray]
        Array of ground radar reflectivity values for each radar tilt.
    pir_gr: List[np.ndarray]
        Array of path-integrated reflectivity values for ground radar for each radar tilt.
    """
    ground_radar_reflectivity = []
    pir_gr = []    
    for radar in nradar:
        refl = radar[refl_name].values - gr_offset
        refl[refl < gr_refl_threshold] = np.nan
        refl = np.ma.masked_invalid(refl)
        ground_radar_reflectivity.append(refl)

        dr = (radar.range[1] - radar.range[0]).values  # Range resolution in meters
        pir = 10 * np.log10(np.cumsum((10 ** (refl / 10)).filled(0), axis=1) * dr)
        pir_gr.append(pir)

    return ground_radar_reflectivity, pir_gr


def volume_matching(
    gpmfile: str,
    grfile: str,
    grfile2: Union[str, None] = None,
    gr_offset: float = 0,
    gr_beamwidth: float = 1,
    gr_rmax: Union[float, None] = None,
    gr_refl_threshold: float = 10,
    radar_band: str = "C",
    refl_name: str = "corrected_reflectivity",
    correct_attenuation: bool = True,
    elevation_offset: Union[float, None] = None,
    fname_prefix: Union[str, None] = None,
) -> xr.Dataset:
    """
    Performs the volume matching of GPM satellite data to ground based radar.

    Parameters:
    ----------
    gpmfile: str
        GPM data file.
    grfile: str
        Ground radar input file.
    grfile2: str, optional
        Second ground radar input file to compute the advection.
    gr_offset: float
        Offset to add to the reflectivity of the ground radar data.
    gr_beamwidth: float
        Ground radar 3dB-beamwidth.
    gr_rmax: float
        Ground radar maximum range in meters (100,000 m). Actual max range used (up to 250,000 m).
    gr_refl_thresold: float
        Minimum reflectivity threshold on ground radar data.
    radar_band: str
        Ground radar frequency band.
    refl_name: str
        Name of the reflectivity field in the ground radar data.
    correct_attenuation: bool
        Should we correct for C- or X-band ground radar attenuation
    elevation_offset: float
        Adding an offset in case the elevation angle needs to be corrected.
    fname_prefix: str
        Name of the ground radar to use as label for the output file.

    Returns:
    --------
    matchset: xarray.Dataset
        Dataset containing the matched GPM and ground radar data.
    """
    if fname_prefix is None:
        fname_prefix = "unknown_radar"

    gpmset, nradar = data_load_and_checks(
        gpmfile,
        grfile,
        refl_name=refl_name,
        correct_attenuation=correct_attenuation,
        radar_band=radar_band,
    )

    nprof = gpmset.precip_in_gr_domain.values.sum()
    ntilt = len(nradar)

    ground_radar_reflectivity, pir_gr = get_gr_reflectivity(nradar, refl_name, gr_offset, gr_refl_threshold)
    range_gr, elev_gr, xradar, yradar, tradar = get_radar_coordinates(nradar, elevation_offset)
    dr = range_gr[1] - range_gr[0]
    if gr_rmax is None:
        gr_rmax = range_gr.max() if range_gr.max() < 250e3 else 250e3

    # Extract GPM data.
    position_precip_domain = gpmset.precip_in_gr_domain.values != 0

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

    volsat = 1e-9 * gpmset.dr * (rsat[position_precip_domain] * np.deg2rad(gpmset.beamwidth)) ** 2  # km3

    refl_gpm_raw = np.ma.masked_invalid(gpmset.zFactorCorrected.values[position_precip_domain])
    reflectivity_gpm_grband = np.ma.masked_invalid(gpmset.reflectivity_grband.values[position_precip_domain])

    # Compute Path-integrated reflectivities
    pir_gpm = 10 * np.log10(np.cumsum((10 ** (np.ma.masked_invalid(refl_gpm_raw) / 10)).filled(0), axis=-1) * 125)
    pir_gpm = np.ma.masked_invalid(pir_gpm)

    # Pre-compute the ground radar coordinates and volume.
    R2d_list = []  # Indexed by tilt
    delta_t_list = []
    volgr_list = []
    for jj in range(ntilt):
        # Get the ground radar range and azimuth.
        deltat = nradar[jj] - gpmset.overpass_time.values
        R, _ = np.meshgrid(nradar[jj].range.values, nradar[jj].azimuth.values)
        _, DT = np.meshgrid(nradar[jj].range.values, deltat)
        volgr = 1e-9 * dr * (R * np.deg2rad(gr_beamwidth)) ** 2  # km3
        R2d_list.append(R)
        delta_t_list.append(DT)
        volgr_list.append(volgr)

    # Initialising output data.
    datakeys = [
        "refl_gpm_raw",
        "refl_gr_weigthed",
        "refl_gpm_grband",
        "pir_gpm",
        "pir_gr",
        "refl_gr_raw",
        "std_refl_gpm",
        "std_refl_gr",
        "sample_gpm",
        "reject_gpm",
        "fmin_gpm",
        "fmin_gr",
        "sample_gr",
        "reject_gr",
        "volume_match_gpm",
        "volume_match_gr",
    ]

    data = dict()
    for k in datakeys:
        data[k] = np.zeros((nprof, ntilt)) + np.nan

    # For sake of simplicity, coordinates are just ndarray, they will be put in the 'data' dict after the matching process
    x = np.zeros((nprof, ntilt))  # x coordinate of sample
    y = np.zeros((nprof, ntilt))  # y coordinate of sample
    z = np.zeros((nprof, ntilt))  # z coordinate of sample
    r = np.zeros((nprof, ntilt))  # range of sample from ground radar
    dz = np.zeros((nprof, ntilt))  # depth of sample
    ds = np.zeros((nprof, ntilt))  # width of sample
    delta_t = np.zeros((nprof, ntilt)) + np.nan  # Timedelta of sample

    for ii, jj in itertools.product(range(nprof), range(ntilt)):
        if elev_gr[jj] - gr_beamwidth / 2 < 0:
            # Beam partially in the ground.
            continue

        epos = (elev_sat[ii, :] >= elev_gr[jj] - gr_beamwidth / 2) & (elev_sat[ii, :] <= elev_gr[jj] + gr_beamwidth / 2)
        x[ii, jj] = np.mean(xsat[ii, epos])
        y[ii, jj] = np.mean(ysat[ii, epos])
        z[ii, jj] = np.mean(zsat[ii, epos])

        data["sample_gpm"][ii, jj] = np.sum(epos)  # Nb of profiles in layer
        data["volume_match_gpm"][ii, jj] = np.sum(volsat[ii, epos])  # Total GPM volume in layer

        dz[ii, jj] = np.sum(epos) * gpmset.dr * np.cos(np.deg2rad(alpha[ii]))  # Thickness of the layer
        ds[ii, jj] = (
            np.deg2rad(gpmset.beamwidth) * np.mean((gpmset.altitude - zsat[ii, epos])) / np.cos(np.deg2rad(alpha[ii]))
        )  # Width of layer
        r[ii, jj] = (
            (gpmset.earth_gaussian_radius + zsat[ii, jj])
            * np.sin(s_sat[ii, jj] / gpmset.earth_gaussian_radius)
            / np.cos(np.deg2rad(elev_gr[jj]))
        )

        if r[ii, jj] + ds[ii, jj] / 2 > gr_rmax:
            # More than half the sample is outside of the radar last bin.
            continue

        # Ground radar side:
        R = R2d_list[jj]
        DT = delta_t_list[jj]
        volgr = volgr_list[jj]

        roi_gr_at_vol = np.sqrt((xradar[jj] - x[ii, jj]) ** 2 + (yradar[jj] - y[ii, jj]) ** 2)
        rpos = roi_gr_at_vol <= ds[ii, jj] / 2
        if np.sum(rpos) == 0:
            continue

        w = volgr[rpos] * np.exp(-((roi_gr_at_vol[rpos] / (ds[ii, jj] / 2)) ** 2))

        # Extract reflectivity for volume.
        refl_gpm = refl_gpm_raw[ii, epos].flatten()
        refl_gpm_grband = reflectivity_gpm_grband[ii, epos].flatten()
        refl_gr_raw = ground_radar_reflectivity[jj][rpos].flatten()
        try:
            delta_t[ii, jj] = np.max(DT[rpos])
        except ValueError:
            # There's no data in the radar domain.
            continue

        if len(refl_gpm) < 5 or len(refl_gr_raw) < 5:
            continue
        if np.all(np.isnan(refl_gpm.filled(np.nan))):
            continue
        if np.all(np.isnan(refl_gr_raw.filled(np.nan))):
            continue

        # FMIN parameter.
        data["fmin_gpm"][ii, jj] = np.sum(refl_gpm > 0) / len(refl_gpm)
        data["fmin_gr"][ii, jj] = np.sum(refl_gr_raw >= gr_refl_threshold) / len(refl_gr_raw)

        # GPM
        data["refl_gpm_raw"][ii, jj] = np.mean(refl_gpm)
        data["refl_gpm_grband"][ii, jj] = np.mean(refl_gpm_grband)
        data["pir_gpm"][ii, jj] = np.mean(pir_gpm[ii, epos].flatten())
        data["std_refl_gpm"][ii, jj] = np.std(refl_gpm)
        data["reject_gpm"][ii, jj] = np.sum(epos) - np.sum(refl_gpm.mask)  # Number of rejected bins

        # Ground radar.
        data["volume_match_gr"][ii, jj] = np.sum(volgr[rpos])
        data["refl_gr_weigthed"][ii, jj] = np.sum(w * refl_gr_raw) / np.sum(w[~refl_gr_raw.mask])
        data["refl_gr_raw"][ii, jj] = np.mean(refl_gr_raw)
        data["pir_gr"][ii, jj] = np.mean(pir_gr[jj][rpos].flatten())
        data["std_refl_gr"][ii, jj] = np.std(refl_gr_raw)
        data["reject_gr"][ii, jj] = np.sum(rpos)
        data["sample_gr"][ii, jj] = np.sum(~refl_gr_raw.mask)

    data["x"] = x
    data["y"] = y
    data["z"] = z
    data["r"] = r
    data["nprof"] = np.arange(nprof, dtype=np.int32)
    data["ntilt"] = np.arange(ntilt, dtype=np.int32)
    data["elevation_gr"] = elev_gr[:ntilt]
    data["timedelta"] = delta_t

    if np.sum((~np.isnan(data["refl_gpm_raw"])) & (~np.isnan(data["refl_gr_raw"]))) < 20:
        raise NoRainError("At least 20 sample points are required.")

    # Transform to xarray and build metadata
    match = dict()
    for k, v in data.items():
        if k in ["ntilt", "elevation_gr"]:
            match[k] = (("ntilt"), v)
        elif k == "nprof":
            match[k] = (("nprof"), v)
        else:
            match[k] = (("nprof", "ntilt"), np.ma.masked_invalid(v.astype(np.float32)))

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
    offset = get_offset(matchset, dr)
    if np.abs(offset) > 15:
        raise ValueError(f"Offset of {offset} dB for {grfile} too big to mean anything.")

    matchset.attrs["offset_applied"] = gr_offset
    matchset.attrs["offset_found"] = offset
    matchset.attrs["final_offset"] = gr_offset + offset
    matchset.attrs["estimated_calibration_offset"] = f"{offset:0.4} dB"
    matchset.attrs["gpm_overpass_time"] = gpm_overpass_time
    matchset.attrs["gpm_min_distance"] = np.round(gpm_mindistance)
    matchset.attrs["gpm_orbit"] = gpmset.attrs["orbit"]
    matchset.attrs["radar_start_time"] = nradar[0].attrs["start_time"]
    matchset.attrs["radar_end_time"] = nradar[0].attrs["end_time"]
    matchset.attrs["radar_longitude"] = nradar[0].attrs["longitude"]
    matchset.attrs["radar_latitude"] = nradar[0].attrs["latitude"]
    matchset.attrs["radar_range_res"] = dr
    matchset.attrs["radar_beamwidth"] = gr_beamwidth
    matchset.attrs["country"] = "Australia"
    matchset.attrs["creator_email"] = "valentin.louf@bom.gov.au"
    matchset.attrs["creator_name"] = "Valentin Louf"
    matchset.attrs["date_created"] = datetime.datetime.now().isoformat()
    matchset.attrs["uuid"] = str(uuid.uuid4())
    matchset.attrs["institution"] = "Bureau of Meteorology"
    matchset.attrs["references"] = "doi:10.1175/JTECH-D-18-0007.1 ; doi:10.1175/JTECH-D-17-0128.1"
    matchset.attrs["disclaimer"] = "If you are using this data/technique for a scientific publication, please cite the papers given in references."
    matchset.attrs["naming_authority"] = "au.org.nci"
    matchset.attrs["summary"] = "GPM volume matching technique."
    matchset.attrs["field_names"] = ", ".join(sorted([k for k, v in matchset.items()]))
    try:
        history = f"Created by {matchset.attrs['creator_name']} on {platform.node()} at {matchset.attrs['date_created']} using Py-ART."
    except Exception:
        history = f"Created by {matchset.attrs['creator_name']} at {matchset.attrs['date_created']} using Py-ART."
    matchset.attrs["history"] = history
    matchset.attrs["history"] = history

    return matchset


def vmatch_multi_pass(
    gpmfile: str,
    grfile: str,
    grfile2: Union[str, None] = None,
    gr_offset: float = 0,
    gr_beamwidth: float = 1,
    gr_rmax: Union[float, None] = None,
    gr_refl_threshold: float = 10,
    radar_band: str = "C",
    refl_name: str = "corrected_reflectivity",
    correct_attenuation: bool = True,
    elevation_offset: Union[float, None] = None,
    fname_prefix: Union[str, None] = None,
    offset_thld: float = 0.5,
    output_dir: Union[str, None] = None,
) -> None:
    """
    Multi-pass volume matching driver function with offset computation.

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
    gr_rmax: float
        Ground radar maximum range in meters (100,000 m).
    gr_refl_thresold: float
        Minimum reflectivity threshold on ground radar data.
    radar_band: str
        Ground radar frequency band.
    refl_name: str
        Name of the reflectivity field in the ground radar data.
    fname_prefix: str
        Name of the ground radar to use as label for the output file.
    correct_attenuation: bool
        Should we correct for C- or X-band ground radar attenuation
    elevation_offset: float
        Adding an offset in case the elevation angle needs to be corrected.
    offset_thld: float
        Offset threshold (in dB) between GPM and GR to stop the iteration.
    output_dir: str
        Path to output directory.
    """

    def _save(dset: xr.Dataset, output_directory: str, debug: bool = False) -> None:
        """
        Generate multipass metadata and file name.
        """
        dset.attrs["iteration_number"] = counter
        matchset.attrs["offset_history"] = ",".join([f"{float(i):0.3}" for i in offset_keeping_track])
        outfilename = dset.attrs["filename"].replace(".nc", f".pass{counter}.nc")
        savedata(dset, output_directory, outfilename)
        if debug:
            print(f"{os.path.basename(outfilename)} written for radar {fname_prefix}")
        return None

    counter = 0
    if fname_prefix is None:
        fname_prefix = "unknown_radar"
        print(f"No 'fname_prefix' defined. The output files will be named {fname_prefix}")
    if output_dir is None:
        output_dir = os.getcwd()
        print(f"No 'output_dir' defined. The output files will be saved {output_dir}")

    # Generate output directories.
    output_dirs = {
        "first": os.path.join(output_dir, "first_pass"),
        "final": os.path.join(output_dir, "final_pass"),
    }
    [_mkdir(v) for _, v in output_dirs.items()]

    # Function arguments dictionnary.
    kwargs = {
        "gpmfile": gpmfile,
        "grfile": grfile,
        "grfile2": grfile2,
        "gr_offset": gr_offset,
        "radar_band": radar_band,
        "refl_name": refl_name,
        "fname_prefix": fname_prefix,
        "correct_attenuation": correct_attenuation,
        "gr_beamwidth": gr_beamwidth,
        "gr_rmax": gr_rmax,
        "gr_refl_threshold": gr_refl_threshold,
        "elevation_offset": elevation_offset,
    }

    # First pass
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        matchset = volume_matching(**kwargs)
    pass_offset = matchset.attrs["offset_found"]
    kwargs["gr_offset"] = pass_offset  # Update offset in kwargs for next pass
    offset_keeping_track = [pass_offset]
    final_offset_keeping_track = [matchset.attrs["final_offset"]]
    _save(matchset, output_dirs["first"])

    if np.isnan(pass_offset):
        dtime = matchset.attrs["gpm_overpass_time"]
        print(f"Offset is NAN for pass {counter} on {dtime}.")
        return None

    # Multiple pass as long as the difference is more than 1dB or counter is 6
    if np.abs(pass_offset) > offset_thld:
        for counter in range(1, 6):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                new_matchset = volume_matching(**kwargs)

            # Check offset found.
            gr_offset = new_matchset.attrs["final_offset"]
            kwargs["gr_offset"] = gr_offset  # Update offset in kwargs for next pass
            pass_offset = new_matchset.attrs["offset_found"]

            if np.isnan(pass_offset):
                # Solution converged already. Using previous iteration as final result.
                counter -= 1
                break
            if (np.abs(pass_offset) > np.abs(offset_keeping_track[-1])) and (counter > 1):
                counter -= 1
                break

            # Pass results are good enough to continue.
            matchset = new_matchset
            offset_keeping_track.append(pass_offset)
            final_offset_keeping_track.append(gr_offset)
            if np.abs(pass_offset) < offset_thld:
                break

    # Save final iteration.
    _save(matchset, output_dirs["final"], debug=True)
    return None
