"""
Volume matching of ground radar and GPM satellite. It also works with the
latest version of TRMM data.

@title: gpmmatch
@author: Valentin Louf <valentin.louf@bom.gov.au>
@institutions: Monash University and the Australian Bureau of Meteorology
@creation: 17/02/2020
@date: 12/12/2025

.. autosummary::
    :toctree: generated/

    NoRainError
    get_radar_coordinates
    get_gr_reflectivity
    volume_matching
    vmatch_multi_pass
"""

import uuid
import datetime
import warnings
import itertools
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
import xarray as xr
from scipy.spatial import cKDTree

from .correct import get_offset
from .io import data_load_and_checks
from .default import get_metadata

# Constants
MIN_SAMPLE_POINTS = 20
MIN_REFL_SAMPLES = 5
MAX_OFFSET_THRESHOLD = 15


class NoRainError(Exception):
    pass


def generate_filename(radar: xr.Dataset, gpmset: xr.Dataset, fname_prefix: str) -> str:
    """
    Generates a filename for the output dataset based on the input GPM and ground radar files.

    Parameters:
    ----------
    radar: xr.Dataset
        Ground radar dataset.
    gpmset: xr.Dataset
        GPM dataset.
    fname_prefix: str
        Name of the ground radar to use as label for the output file.

    Returns:
    -------
    str
        Generated filename for the output dataset.
    """
    date = pd.Timestamp(radar.time[0].values).strftime("%Y%m%d.%H%M")
    outfilename = f"vmatch.gpm.orbit.{gpmset.attrs['orbit']:07}.{fname_prefix}.{date}.nc"
    return outfilename


def get_radar_coordinates(
    nradar: List[xr.Dataset], elevation_offset: Union[float, None] = None
) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
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


def get_gr_reflectivity(
    nradar: List[xr.Dataset], refl_name: str, gr_offset: float, gr_refl_threshold: float
) -> Tuple[List[np.ma.MaskedArray], List[np.ndarray]]:
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
    ground_radar_reflectivity: List[np.ma.MaskedArray]
        Array of ground radar reflectivity values for each radar tilt.
    pir_gr: List[np.ndarray]
        Array of path-integrated reflectivity values for ground radar for each radar tilt.
    """
    ground_radar_reflectivity = []
    pir_gr = []

    using_corrected_field = False
    for radar in nradar:
        if "ZH_ATTEN_CORR" in radar.data_vars:
            refl = radar["ZH_ATTEN_CORR"].values.copy() - gr_offset
            using_corrected_field = True
        else:
            refl = radar[refl_name].values.copy() - gr_offset
        refl[refl < gr_refl_threshold] = np.nan
        refl = np.ma.masked_invalid(refl)
        ground_radar_reflectivity.append(refl)

        dr = radar.range[1].values - radar.range[0].values  # Range resolution in meters
        pir = 10 * np.log10(np.cumsum((10 ** (refl / 10)).filled(0), axis=1) * dr)
        pir_gr.append(pir)

    if using_corrected_field:
        print("Using attenuation-corrected reflectivity field for ground radar.")

    return ground_radar_reflectivity, pir_gr


def has_valid_data(arr: np.ma.MaskedArray, min_valid: int = MIN_REFL_SAMPLES) -> bool:
    """
    Check if array has sufficient valid (non-NaN) data.

    Parameters:
    ----------
    arr: np.ma.MaskedArray
        Array to check for valid data.
    min_valid: int
        Minimum number of valid samples required.

    Returns:
    -------
    bool
        True if array has sufficient valid data.
    """
    return len(arr) >= min_valid and not np.all(np.isnan(arr.filled(np.nan)))


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
    kdp_name: Union[str, None] = "KDP",
    phase_aware_dfr: bool = True,
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
    gr_refl_threshold: float
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
    kdp_name: Optional[str]
        Name of the KDP field in the ground radar data for attenuation correction.
    phase_aware_dfr: bool
        Use phase-aware DFR conversion that accounts for ice, melting layer, and
        liquid precipitation phases using GPM bright band height. Default is True.

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
        kdp_name=kdp_name,
        phase_aware_dfr=phase_aware_dfr,
    )

    nprof = gpmset.precip_in_gr_domain.values.sum()
    ntilt = len(nradar)

    ground_radar_reflectivity, pir_gr = get_gr_reflectivity(nradar, refl_name, gr_offset, gr_refl_threshold)
    range_gr, elev_gr, xradar, yradar, tradar = get_radar_coordinates(nradar, elevation_offset)
    dr = range_gr[1] - range_gr[0]
    if gr_rmax is None:
        gr_rmax = range_gr.max() if range_gr.max() < 250e3 else 250e3

    # Cache nradar attributes
    nradar_range = [nr.range.values for nr in nradar]
    nradar_azimuth = [nr.azimuth.values for nr in nradar]

    # Extract and cache GPM data
    position_precip_domain = gpmset.precip_in_gr_domain.values != 0
    gpm_nray = gpmset.nray.values
    gpm_nscan = gpmset.nscan.values
    gpm_elev_from_gr = gpmset.elev_from_gr.values
    gpm_x = gpmset.x.values
    gpm_y = gpmset.y.values
    gpm_z = gpmset.z.values
    gpm_dr = gpmset.dr
    gpm_beamwidth = gpmset.beamwidth
    gpm_altitude = gpmset.altitude
    gpm_earth_radius = gpmset.earth_gaussian_radius
    gpm_distance_from_sr = gpmset.distance_from_sr.values
    gpm_zfactor = gpmset.zFactorCorrected.values
    gpm_refl_grband = gpmset.reflectivity_grband.values
    gpm_overpass_time = gpmset.overpass_time.values

    alpha, _ = np.meshgrid(gpm_nray, gpm_nscan)
    alpha = alpha[position_precip_domain]

    elev_sat = gpm_elev_from_gr[position_precip_domain]
    xsat = gpm_x[position_precip_domain]
    ysat = gpm_y[position_precip_domain]
    zsat = gpm_z[position_precip_domain]

    rsat = np.zeros(gpm_zfactor.shape)
    for i in range(rsat.shape[0]):
        rsat[i, :] = gpm_distance_from_sr

    volsat = 1e-9 * gpm_dr * (rsat[position_precip_domain] * np.deg2rad(gpm_beamwidth)) ** 2

    refl_gpm_raw = np.ma.masked_invalid(gpm_zfactor[position_precip_domain])
    reflectivity_gpm_grband = np.ma.masked_invalid(gpm_refl_grband[position_precip_domain])

    # Compute Path-integrated reflectivities
    pir_gpm = 10 * np.log10(np.cumsum((10 ** (np.ma.masked_invalid(refl_gpm_raw) / 10)).filled(0), axis=-1) * dr)
    pir_gpm = np.ma.masked_invalid(pir_gpm)

    # Pre-compute and cache everything
    beamwidth_rad = np.deg2rad(gr_beamwidth)
    half_beamwidth = gr_beamwidth / 2

    # Pre-compute for each tilt
    R_list = []
    DT_list = []
    volgr_list = []
    kdtree_list = []
    xradar_flat_list = []
    yradar_flat_list = []
    gr_refl_flat_list = []
    pir_gr_flat_list = []
    orig_shapes = []

    for jj in range(ntilt):
        deltat = tradar[jj] - gpm_overpass_time
        R, _ = np.meshgrid(nradar_range[jj], nradar_azimuth[jj])
        _, DT = np.meshgrid(nradar_range[jj], deltat)
        volgr = 1e-9 * dr * (R * beamwidth_rad) ** 2

        R_list.append(R)
        DT_list.append(DT)
        volgr_list.append(volgr)

        # Flatten and cache all arrays
        x_flat = xradar[jj].ravel()
        y_flat = yradar[jj].ravel()
        xradar_flat_list.append(x_flat)
        yradar_flat_list.append(y_flat)

        orig_shapes.append(xradar[jj].shape)

        # Build KDTree
        xy_coords = np.column_stack([x_flat, y_flat])
        kdtree_list.append(cKDTree(xy_coords))

        # Pre-flatten data arrays
        gr_refl_flat_list.append(ground_radar_reflectivity[jj].ravel())
        pir_gr_flat_list.append(pir_gr[jj].ravel())

    # Initialize output
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
    data = {k: np.full((nprof, ntilt), np.nan) for k in datakeys}

    x = np.zeros((nprof, ntilt))
    y = np.zeros((nprof, ntilt))
    z = np.zeros((nprof, ntilt))
    r = np.zeros((nprof, ntilt))
    dz = np.zeros((nprof, ntilt))
    ds = np.zeros((nprof, ntilt))
    delta_t = np.full((nprof, ntilt), np.nan)

    # Optimized main loop
    for ii, jj in itertools.product(range(nprof), range(ntilt)):
        if elev_gr[jj] - half_beamwidth < 0:
            continue

        epos = (elev_sat[ii, :] >= elev_gr[jj] - half_beamwidth) & (elev_sat[ii, :] <= elev_gr[jj] + half_beamwidth)
        if not np.any(epos):
            continue

        x[ii, jj] = x_mean = np.mean(xsat[ii, epos])
        y[ii, jj] = y_mean = np.mean(ysat[ii, epos])
        z[ii, jj] = z_mean = np.mean(zsat[ii, epos])

        data["sample_gpm"][ii, jj] = n_epos = np.sum(epos)
        data["volume_match_gpm"][ii, jj] = np.sum(volsat[ii, epos])

        alpha_ii = alpha[ii]
        cos_alpha = np.cos(np.deg2rad(alpha_ii))

        dz[ii, jj] = n_epos * gpm_dr * cos_alpha
        ds[ii, jj] = np.deg2rad(gpm_beamwidth) * np.mean(gpm_altitude - zsat[ii, epos]) / cos_alpha

        # Calculate s_sat for this specific profile
        s_sat_ii = np.sqrt(x_mean**2 + y_mean**2)
        r[ii, jj] = (gpm_earth_radius + z_mean) * np.sin(s_sat_ii / gpm_earth_radius) / np.cos(np.deg2rad(elev_gr[jj]))

        if r[ii, jj] + ds[ii, jj] / 2 > gr_rmax:
            continue
        if np.isnan(x_mean) or np.isnan(y_mean) or np.isnan(ds[ii, jj]):
            continue

        # KDTree query with pre-allocated arrays
        search_radius = ds[ii, jj] / 2
        indices_flat = kdtree_list[jj].query_ball_point([x_mean, y_mean], search_radius)

        if not indices_flat:
            continue

        # Use flat indices directly - faster than unravel
        xradar_subset = xradar_flat_list[jj][indices_flat]
        yradar_subset = yradar_flat_list[jj][indices_flat]

        # Vectorized distance calculation
        dx = xradar_subset - x_mean
        dy = yradar_subset - y_mean
        roi_gr_at_vol = np.sqrt(dx * dx + dy * dy)

        # Get data using flat indices
        volgr_subset = volgr_list[jj].ravel()[indices_flat]
        R_subset = R_list[jj].ravel()[indices_flat]
        DT_subset = DT_list[jj].ravel()[indices_flat]

        # Weight calculation
        normalized_distance = roi_gr_at_vol / search_radius
        w = volgr_subset * np.exp(-(normalized_distance**2))

        # Extract reflectivity
        refl_gpm = refl_gpm_raw[ii, epos]
        refl_gpm_grband = reflectivity_gpm_grband[ii, epos]
        refl_gr_raw = gr_refl_flat_list[jj][indices_flat]
        pir_gr_flat = pir_gr_flat_list[jj][indices_flat]

        try:
            delta_t[ii, jj] = np.max(DT_subset)
        except ValueError:
            continue

        if not (has_valid_data(refl_gpm) and has_valid_data(refl_gr_raw)):
            continue

        # Statistics - use compressed() for masked arrays to avoid overhead
        refl_gpm_compressed = refl_gpm.compressed()
        refl_gr_compressed = refl_gr_raw.compressed()

        data["fmin_gpm"][ii, jj] = (
            np.sum(refl_gpm_compressed > 0) / len(refl_gpm_compressed) if len(refl_gpm_compressed) > 0 else np.nan
        )
        data["fmin_gr"][ii, jj] = (
            np.sum(refl_gr_compressed >= gr_refl_threshold) / len(refl_gr_compressed)
            if len(refl_gr_compressed) > 0
            else np.nan
        )

        data["refl_gpm_raw"][ii, jj] = np.mean(refl_gpm_compressed) if len(refl_gpm_compressed) > 0 else np.nan
        data["refl_gpm_grband"][ii, jj] = np.ma.mean(refl_gpm_grband)
        data["pir_gpm"][ii, jj] = np.ma.mean(pir_gpm[ii, epos])
        data["std_refl_gpm"][ii, jj] = np.std(refl_gpm_compressed) if len(refl_gpm_compressed) > 0 else np.nan
        data["reject_gpm"][ii, jj] = n_epos - len(refl_gpm_compressed)

        data["volume_match_gr"][ii, jj] = np.sum(volgr_subset)

        # Weighted mean - avoid creating new masked arrays
        valid_mask = ~refl_gr_raw.mask if np.ma.is_masked(refl_gr_raw) else np.ones(len(refl_gr_raw), dtype=bool)
        if np.any(valid_mask):
            data["refl_gr_weigthed"][ii, jj] = np.sum(w[valid_mask] * refl_gr_raw.data[valid_mask]) / np.sum(
                w[valid_mask]
            )

        data["refl_gr_raw"][ii, jj] = np.mean(refl_gr_compressed) if len(refl_gr_compressed) > 0 else np.nan
        data["pir_gr"][ii, jj] = np.ma.mean(pir_gr_flat)
        data["std_refl_gr"][ii, jj] = np.std(refl_gr_compressed) if len(refl_gr_compressed) > 0 else np.nan
        data["reject_gr"][ii, jj] = len(indices_flat)
        data["sample_gr"][ii, jj] = len(refl_gr_compressed)

    data["x"] = x
    data["y"] = y
    data["z"] = z
    data["r"] = r
    data["nprof"] = np.arange(nprof, dtype=np.int32)
    data["ntilt"] = np.arange(ntilt, dtype=np.int32)
    data["elevation_gr"] = elev_gr[:ntilt]
    data["timedelta"] = delta_t

    if np.sum((~np.isnan(data["refl_gpm_raw"])) & (~np.isnan(data["refl_gr_raw"]))) < MIN_SAMPLE_POINTS:
        raise NoRainError(f"At least {MIN_SAMPLE_POINTS} sample points are required.")

    # Transform to xarray
    match = {}
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

    ar = gpm_x**2 + gpm_y**2
    iscan, _, _ = np.where(ar == ar.min())
    gpm_overpass_time_iso = pd.Timestamp(gpm_nscan[iscan[0]]).isoformat()
    gpm_mindistance = np.sqrt(gpm_x**2 + gpm_y**2)[:, :, 0][gpmset.flagPrecip.values > 0].min()
    offset = get_offset(matchset, dr)
    if np.abs(offset) > MAX_OFFSET_THRESHOLD:
        raise ValueError(f"Offset of {offset} dB for {grfile} too big to mean anything.")

    matchset.attrs["offset_applied"] = gr_offset
    matchset.attrs["offset_found"] = offset
    matchset.attrs["final_offset"] = gr_offset + offset
    matchset.attrs["estimated_calibration_offset"] = f"{offset:0.4} dB"
    matchset.attrs["gpm_overpass_time"] = gpm_overpass_time_iso
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
    matchset.attrs["disclaimer"] = (
        "If you are using this data/technique for a scientific publication, please cite the papers given in references."
    )
    matchset.attrs["naming_authority"] = "au.org.nci"
    matchset.attrs["summary"] = "GPM volume matching technique."
    matchset.attrs["filename"] = generate_filename(nradar[0], gpmset, fname_prefix)

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
    kdp_name: Union[str, None] = "KDP",
    phase_aware_dfr: bool = True,
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
    gr_refl_threshold: float
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
    kdp_name: Optional[str]
        Name of the KDP field in the ground radar data for attenuation correction.
    phase_aware_dfr: bool
        Use phase-aware DFR conversion that accounts for ice, melting layer, and
        liquid precipitation phases using GPM bright band height. Default is True.
    """

    def _save(dset: xr.Dataset, output_directory: str) -> None:
        """
        Generate multipass metadata and file name.
        """
        dset.attrs["iteration_number"] = counter
        dset.attrs["offset_history"] = ",".join([f"{float(i):0.3}" for i in offset_keeping_track])
        filename = dset.attrs["filename"].replace(".nc", f".pass{counter}.nc")

        outfilename = Path(output_directory) / filename
        print(f"Saving {outfilename.name} to {output_directory}.")
        dset.to_netcdf(str(outfilename), encoding={k: {"zlib": True} for k in dset.data_vars})

        # Check if the file was created successfully.
        if outfilename.exists():
            print(f"{outfilename.name} written to {output_directory}.")
        else:
            print(f"Error writing {outfilename.name} to {output_directory}.")
            raise FileNotFoundError(f"File {outfilename} could not be created.")

        return None

    counter = 0
    if fname_prefix is None:
        fname_prefix = "unknown_radar"
        print(f"No 'fname_prefix' defined. The output files will be named {fname_prefix}")
    if output_dir is None:
        output_dir = str(Path.cwd())
        print(f"No 'output_dir' defined. The output files will be saved {output_dir}")
    if correct_attenuation and radar_band not in ["C", "X"]:
        print(
            f"Attenuation correction is only available for C- and X-band radars. Setting 'correct_attenuation' to False."
        )
        correct_attenuation = False

    # Generate output directories.
    output_dirs = {
        "first": Path(output_dir) / "first_pass",
        "final": Path(output_dir) / "final_pass",
    }
    for dir_path in output_dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)

    # Function arguments dictionary.
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
        "kdp_name": kdp_name,
        "phase_aware_dfr": phase_aware_dfr,
    }

    # First pass
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        matchset = volume_matching(**kwargs)
    pass_offset = matchset.attrs["offset_found"]
    kwargs["gr_offset"] = pass_offset  # Update offset in kwargs for next pass
    offset_keeping_track = [pass_offset]
    final_offset_keeping_track = [matchset.attrs["final_offset"]]
    _save(matchset, str(output_dirs["first"]))

    if np.isnan(pass_offset):
        dtime = matchset.attrs["gpm_overpass_time"]
        print(f"Offset is NAN for pass {counter} on {dtime}.")
        return None

    # Multiple pass as long as the difference is more than threshold or counter is 6
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
    _save(matchset, str(output_dirs["final"]))
    return None
