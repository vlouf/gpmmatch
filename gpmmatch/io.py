"""
Utilities to read the input data and format them in a way to be read by
volume_matching.

@title: io
@author: Valentin Louf <valentin.louf@bom.gov.au>
@institutions: Monash University and the Australian Bureau of Meteorology
@creation: 17/02/2020
@date: 14/04/2026

.. autosummary::
    :toctree: generated/

    NoPrecipitationError
    _read_radar
    check_precip_in_domain
    data_load_and_checks
    get_ground_radar_attributes
    get_gpm_orbit
    read_GPM
    read_GPM_v07
    read_GPM_v08
    read_radars
"""

import re
import copy
import datetime
import warnings

from typing import Tuple, List, Union, Optional
from collections import OrderedDict

import h5py
import pyodim
import pyproj
import numpy as np
import pandas as pd
import xarray as xr

from . import correct
from . import default

# Type alias for bounding box: (lat_min, lat_max, lon_min, lon_max)
BBox = Tuple[float, float, float, float]


class NoPrecipitationError(Exception):
    pass


def check_precip_in_domain(
    gpmset: xr.Dataset, grlon: float, grlat: float, rmax: float = 150e3, rmin: float = 20e3
) -> Tuple[pd.Timestamp, int]:
    """
    Check if there's GPM precipitation in the radar domain.

    Parameters:
    ===========
    gpmset: xarray
        Dataset containing the GPM data
    grlon: float
        Ground radar longitude
    grlat: float
        Ground radar latitude
    rmax: float
        Ground radar maximum range
    rmin: float
        Ground radar minimum range.

    Returns:
    ========
    gpmtime0: datetime
        Time of the closest measurement from ground radar site.
    nprof: int
        Number of GPM precipitation profiles in ground radar domain.
    """
    georef = pyproj.Proj(f"+proj=aeqd +lon_0={grlon} +lat_0={grlat} +ellps=WGS84")
    gpmlat = gpmset.Latitude.values
    gpmlon = gpmset.Longitude.values

    xgpm, ygpm = georef(gpmlon, gpmlat)
    rproj_gpm = (xgpm**2 + ygpm**2) ** 0.5

    gr_domain = (rproj_gpm <= rmax) & (rproj_gpm >= rmin)
    if gr_domain.sum() < 10:
        raise NoPrecipitationError("GPM swath does not go through the radar domain.")

    nprof = np.sum(gpmset.flagPrecip.values[gr_domain])
    if nprof < 10:
        raise NoPrecipitationError("No precipitation measured by GPM inside radar domain.")

    newset = gpmset.merge({"range_from_gr": (("nscan", "nray"), rproj_gpm)})
    gpmtime0 = newset.nscan.where(newset.range_from_gr == newset.range_from_gr.min()).values.astype("datetime64[s]")
    gpmtime0 = gpmtime0[~np.isnat(gpmtime0)][0]
    gpmtime = pd.Timestamp(gpmtime0)
    del newset
    return gpmtime, nprof


def data_load_and_checks(
    gpmfile: str,
    grfile: str,
    refl_name: Union[str, None] = None,
    correct_attenuation: bool = True,
    radar_band: str = "C",
    kdp_name: Union[str, None] = "KDP",
    phase_aware_dfr: bool = True,
    bbox: Optional[BBox] = None,
) -> Tuple[xr.Dataset, List[xr.Dataset]]:
    """
    Load GPM and Ground radar files and perform some initial checks:
    domains intersect, precipitation, time difference.

    Parameters:
    -----------
    gpmfile: str
        GPM data file.
    grfile: str
        Ground radar input file.
    refl_name: str
        Name of the reflectivity field in the ground radar data.
    correct_attenuation: bool
        Should we correct for C- or X-band ground radar attenuation.
    radar_band: str
        Ground radar frequency band for reflectivity conversion. S, C, and X
        supported.
    kdp_name: Optional[str]
        Name of the KDP field in the ground radar data for attenuation correction.
    phase_aware_dfr: bool
        Use phase-aware DFR conversion that accounts for ice, melting layer, and
        liquid precipitation phases using GPM bright band height. Default is True.
        Set to False to use the simple polynomial DFR relationship.
    bbox: tuple of (lat_min, lat_max, lon_min, lon_max), optional
        Bounding box to subset full-orbit GPM V08 files. Required for V08
        files to avoid loading the entire orbit into memory. Ignored for
        V07 (HDF5) files which are already regional subsets.

    Returns:
    --------
    gpmset: xarray.Dataset
        Dataset containing the input datas.
    radar: pyart.core.Radar
        Pyart radar dataset.
    """
    if refl_name is None:
        raise ValueError("Reflectivity field name not given.")

    gpmset = read_GPM(gpmfile, bbox=bbox)
    grlon, grlat, gralt, rmin, rmax = get_ground_radar_attributes(grfile)

    # Reproject satellite coordinates onto ground radar
    georef = pyproj.Proj(f"+proj=aeqd +lon_0={grlon} +lat_0={grlat} +ellps=WGS84")
    gpmlat = gpmset.Latitude.values
    gpmlon = gpmset.Longitude.values

    xgpm, ygpm = georef(gpmlon, gpmlat)
    rproj_gpm = (xgpm**2 + ygpm**2) ** 0.5

    gr_domain = (rproj_gpm <= rmax) & (rproj_gpm >= rmin)
    if gr_domain.sum() < 10:
        info = f"The closest satellite measurement is {np.min(rproj_gpm / 1e3):0.4} km away from ground radar."
        if gr_domain.sum() == 0:
            raise NoPrecipitationError("GPM swath does not go through the radar domain. " + info)
        else:
            raise NoPrecipitationError("Not enough GPM precipitation inside ground radar domain. " + info)

    nprof = np.sum(gpmset.flagPrecip.values[gr_domain])
    if nprof < 10:
        raise NoPrecipitationError("No precipitation measured by GPM inside radar domain.")

    # Parallax correction
    sr_xp, sr_yp, z_sr = correct.correct_parallax(xgpm, ygpm, gpmset)

    # Compute the elevation of the satellite bins with respect to the ground radar.
    gr_gaussian_radius = correct.compute_gaussian_curvature(grlat)
    gamma = np.sqrt(sr_xp**2 + sr_yp**2) / gr_gaussian_radius
    elev_sr_grref = np.rad2deg(
        np.arctan2(np.cos(gamma) - (gr_gaussian_radius + gralt) / (gr_gaussian_radius + z_sr), np.sin(gamma))
    )

    # Convert reflectivity band correction
    if phase_aware_dfr:
        # Use phase-aware DFR that accounts for ice, melting layer, and liquid phases
        reflgpm_grband = correct.convert_gpmrefl_grband_phase_aware(
            refl_gpm=gpmset.zFactorCorrected.values,
            height_gpm=z_sr,
            height_bb=gpmset.heightBB.values,
            radar_band=radar_band,
        )
    else:
        # Use simple polynomial DFR relationship (original method)
        reflgpm_grband = correct.convert_gpmrefl_grband_dfr(gpmset.zFactorCorrected.values, radar_band=radar_band)

    gpmset = gpmset.merge(
        {
            "precip_in_gr_domain": (("nscan", "nray"), gpmset.flagPrecip.values & gr_domain),
            "range_from_gr": (("nscan", "nray"), rproj_gpm),
            "elev_from_gr": (("nscan", "nray", "nbin"), elev_sr_grref),
            "x": (("nscan", "nray", "nbin"), sr_xp),
            "y": (("nscan", "nray", "nbin"), sr_yp),
            "z": (("nscan", "nray", "nbin"), z_sr),
            "reflectivity_grband": (("nscan", "nray", "nbin"), reflgpm_grband),
        }
    )

    # Get time of the overpass (closest point from ground radar).
    gpmtime0 = gpmset.nscan.where(gpmset.range_from_gr == gpmset.range_from_gr.min()).values.astype("datetime64[s]")
    gpmtime0 = gpmtime0[~np.isnat(gpmtime0)][0]
    gpmset = gpmset.merge({"overpass_time": (gpmtime0)})

    # Attributes
    metadata = default.gpmset_metadata()
    for k, v in metadata.items():
        for sk, sv in v.items():
            try:
                gpmset[k].attrs[sk] = sv
            except KeyError:
                continue
    gpmset.reflectivity_grband.attrs["description"] = f"Satellite reflectivity converted to {radar_band}-band."
    gpmset.attrs["nprof"] = nprof
    gpmset.attrs["earth_gaussian_radius"] = gr_gaussian_radius

    # Time to read the ground radar data.
    radar = read_radar(
        grfile, refl_name, radar_band=radar_band, correct_attenuation=correct_attenuation, kdp_name=kdp_name
    )

    return gpmset, radar


def get_gpm_orbit(gpmfile: str) -> int:
    """
    Extract the GPM Granule Number from a GPM file (V07 HDF5 or V08 netCDF).

    Parameters:
    -----------
    gpmfile: str
        GPM data file.

    Returns:
    --------
    orbit: int
        GPM Granule Number.
    """
    try:
        if gpmfile.endswith(".nc"):
            import netCDF4

            with netCDF4.Dataset(gpmfile) as nc:
                file_header = nc.getncattr("FileHeader")
            grannb = [s for s in file_header.split("\n") if "GranuleNumber" in s][0]
            orbit = re.findall("[0-9]{3,}", grannb)[0]
        else:
            with h5py.File(gpmfile) as hid:
                grannb = [s for s in hid.attrs["FileHeader"].split() if b"GranuleNumber" in s][0].decode("utf-8")
                orbit = re.findall("[0-9]{3,}", grannb)[0]
    except Exception:
        return 0

    return int(orbit)


def get_ground_radar_attributes(grfile: str) -> Tuple[float, float, float, float, float]:
    """
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
    """
    nradar = pyodim.read_odim(grfile)
    radar = nradar[0].compute()
    rmin = radar.range.values.min()
    if rmin < 15e3:
        rmin = 15e3
    rmax = radar.range.values.max()
    grlon = radar.attrs["longitude"]
    grlat = radar.attrs["latitude"]
    gralt = radar.attrs["height"]

    return grlon, grlat, gralt, rmin, rmax


# =============================================================================
# GPM readers
# =============================================================================


def read_GPM(infile: str, refl_min_thld: float = 0, bbox: Optional[BBox] = None) -> xr.Dataset:
    """
    Read GPM data file (V07 HDF5 or V08 netCDF) and return a standardised
    xarray Dataset.

    Dispatches to the appropriate format-specific reader based on file
    extension.

    Parameters:
    -----------
    infile: str
        GPM data file (.HDF5 for V07, .nc for V08).
    refl_min_thld: float
        Minimum threshold applied to GPM reflectivity.
    bbox: tuple of (lat_min, lat_max, lon_min, lon_max), optional
        Bounding box to subset full-orbit V08 files. Ignored for V07.

    Returns:
    --------
    dset: xr.Dataset
        GPM dataset with standardised variable names and dimensions.
    """
    if infile.endswith(".nc"):
        return read_GPM_v08(infile, refl_min_thld=refl_min_thld, bbox=bbox)
    else:
        return read_GPM_v07(infile, refl_min_thld=refl_min_thld)


def read_GPM_v07(infile: str, refl_min_thld: float = 0) -> xr.Dataset:
    """
    Read GPM V07 (HDF5) data and organize them into a Dataset.

    Parameters:
    -----------
    infile: str
        GPM data file (HDF5).
    refl_min_thld: float
        Minimum threshold applied to GPM reflectivity.

    Returns:
    --------
    dset: xr.Dataset
        GPM dataset.
    """
    if refl_min_thld != 0:
        warnings.warn("Tests have shown that no threshold should be applied to GPM reflectivity!", UserWarning)
    data = dict()
    date = dict()
    with h5py.File(infile, "r") as hid:
        try:
            master_key = "NS"
            keys = hid[f"/{master_key}"].keys()
        except KeyError:
            master_key = "FS"
            keys = hid[f"/{master_key}"].keys()

        for k in keys:
            if k == "Latitude" or k == "Longitude":
                dims = tuple(hid[f"/{master_key}/{k}"].attrs["DimensionNames"].decode("UTF-8").split(","))
                fv = hid[f"/{master_key}/{k}"].attrs["_FillValue"]
                data[k] = (dims, np.ma.masked_equal(hid[f"/{master_key}/{k}"][:], fv))
            else:
                try:
                    subkeys = hid[f"/{master_key}/{k}"].keys()
                except Exception:
                    continue
                for sk in subkeys:
                    dims = tuple(hid[f"/{master_key}/{k}/{sk}"].attrs["DimensionNames"].decode("UTF-8").split(","))
                    fv = hid[f"/{master_key}/{k}/{sk}"].attrs["_FillValue"]

                    if sk in ["Year", "Month", "DayOfMonth", "Hour", "Minute", "Second", "MilliSecond"]:
                        date[sk] = np.ma.masked_equal(hid[f"/{master_key}/{k}/{sk}"][:], fv)
                    elif sk in ["DayOfYear", "SecondOfDay"]:
                        continue
                    elif sk == "typePrecip":
                        # Simplify precipitation type
                        data[sk] = (dims, hid[f"/{master_key}/{k}/{sk}"][:] / 10000000)
                    elif sk in ["zFactorCorrected", "zFactorFinal", "zFactorMeasured"]:
                        # Reverse direction along the beam.
                        gpm_refl = hid[f"/{master_key}/{k}/{sk}"][:][:, :, ::-1]
                        gpm_refl[gpm_refl < 0] = np.nan
                        data[sk] = (dims, np.ma.masked_invalid(np.ma.masked_less_equal(gpm_refl, refl_min_thld)))
                    elif sk == "flagPrecip":
                        data[sk] = (
                            dims,
                            np.ma.masked_invalid(hid[f"/{master_key}/{k}/{sk}"][:]).filled(0).astype(bool),
                        )
                    else:
                        data[sk] = (dims, np.ma.masked_equal(hid[f"/{master_key}/{k}/{sk}"][:], fv))

    try:
        _ = data["zFactorCorrected"]
    except KeyError:
        data["zFactorCorrected"] = copy.deepcopy(data["zFactorFinal"])

    # Create Quality indicator.
    quality = np.zeros(data["heightBB"][-1].shape, dtype=np.int32)
    quality[((data["qualityBB"][-1] == 0) | (data["qualityBB"][-1] == 1)) & (data["qualityTypePrecip"][-1] == 1)] = 1
    quality[(data["qualityBB"][-1] > 1) | (data["qualityTypePrecip"][-1] > 1)] = 2
    data["quality"] = (data["heightBB"][0], quality)

    # Generate dimensions.
    nray = np.linspace(-17.04, 17.04, 49)
    nbin = np.arange(0, 125 * 176, 125)

    R, A = np.meshgrid(nbin, nray)
    distance_from_sr = 407000 / np.cos(np.deg2rad(A)) - R  # called rt in IDL code.
    data["distance_from_sr"] = (("nray", "nbin"), distance_from_sr)

    try:
        # TRMM doesn't have a MilliSecond field.
        _ = date["MilliSecond"]
    except KeyError:
        date["MilliSecond"] = date["Second"]

    dtime = np.array(
        [
            datetime.datetime(*d)
            for d in zip(
                date["Year"],
                date["Month"],
                date["DayOfMonth"],
                date["Hour"],
                date["Minute"],
                date["Second"],
                date["MilliSecond"],
            )
        ],
        dtype="datetime64",
    )

    data["nscan"] = (("nscan"), dtime)
    data["nray"] = (("nray"), nray)
    data["nbin"] = (("nbin"), nbin)

    dset = xr.Dataset(OrderedDict(sorted(data.items())))

    dset.nray.attrs = {"units": "degree", "description": "Deviation from Nadir"}
    dset.nbin.attrs = {"units": "m", "description": "Downward from 0: TOA to Earth ellipsoid."}
    dset.attrs["altitude"] = 407000
    dset.attrs["altitude_units"] = "m"
    dset.attrs["altitude_description"] = "GPM orbit"
    dset.attrs["beamwidth"] = 0.71
    dset.attrs["beamwidth_units"] = "degree"
    dset.attrs["beamwidth_description"] = "GPM beamwidth"
    dset.attrs["dr"] = 125
    dset.attrs["dr_units"] = "m"
    dset.attrs["dr_description"] = "GPM gate spacing"
    dset.attrs["orbit"] = get_gpm_orbit(infile)

    return dset


def read_GPM_v08(infile: str, refl_min_thld: float = 0, bbox: Optional[BBox] = None) -> xr.Dataset:
    """
    Read GPM V08 (netCDF-4) data and organize into a Dataset matching the
    structure produced by read_GPM_v07.

    V08 files contain the full orbit (~8000 scans). A bounding box is used
    to subset the data to the region of interest, keeping memory usage
    manageable.

    Parameters:
    -----------
    infile: str
        GPM V08 netCDF data file.
    refl_min_thld: float
        Minimum threshold applied to GPM reflectivity.
    bbox: tuple of (lat_min, lat_max, lon_min, lon_max), optional
        Geographic bounding box to subset the orbit. If None, the full
        orbit is loaded (warning: high memory usage).

    Returns:
    --------
    dset: xr.Dataset
        GPM dataset with the same structure as read_GPM_v07 output.
    """
    import netCDF4

    if refl_min_thld != 0:
        warnings.warn("Tests have shown that no threshold should be applied to GPM reflectivity!", UserWarning)

    master_key = "FS"

    # ------------------------------------------------------------------
    # Step 1: Read lat/lon and apply bounding box to find relevant scans.
    # This is lightweight — only 2D arrays (nscan, nray).
    # ------------------------------------------------------------------
    with netCDF4.Dataset(infile) as nc:
        grp = nc[master_key]
        lat = grp["Latitude"][:]
        lon = grp["Longitude"][:]

    if bbox is not None:
        lat_min, lat_max, lon_min, lon_max = bbox
        # A scan is relevant if ANY ray falls within the bounding box.
        in_bbox = (lat >= lat_min) & (lat <= lat_max) & (lon >= lon_min) & (lon <= lon_max)
        scan_mask = np.any(in_bbox, axis=1)

        if not np.any(scan_mask):
            raise NoPrecipitationError(
                f"GPM orbit does not cross the bounding box "
                f"(lat: {lat_min} to {lat_max}, lon: {lon_min} to {lon_max})."
            )

        # Find contiguous range of scans (with a small margin for parallax).
        scan_indices = np.where(scan_mask)[0]
        idx_start = max(0, scan_indices[0] - 10)
        idx_end = min(lat.shape[0], scan_indices[-1] + 11)
        scan_slice = slice(idx_start, idx_end)

        lat = lat[scan_slice]
        lon = lon[scan_slice]
        print(f"V08 bbox subset: scans {idx_start}–{idx_end} of {scan_mask.shape[0]} "
              f"({idx_end - idx_start} scans retained).")
    else:
        scan_slice = slice(None)
        warnings.warn(
            "Loading full-orbit V08 GPM file without bounding box — "
            "this will use significant memory.",
            UserWarning,
        )

    # ------------------------------------------------------------------
    # Step 2: Read all required fields from netCDF groups, sliced by scan.
    # ------------------------------------------------------------------
    data = dict()
    date_fields = dict()

    with netCDF4.Dataset(infile) as nc:
        grp = nc[master_key]

        # Lat/Lon (already loaded and sliced above)
        data["Latitude"] = (("nscan", "nray"), np.ma.masked_equal(lat, -9999.9))
        data["Longitude"] = (("nscan", "nray"), np.ma.masked_equal(lon, -9999.9))

        # ScanTime
        scan_time = grp["ScanTime"]
        for field in ["Year", "Month", "DayOfMonth", "Hour", "Minute", "Second", "MilliSecond"]:
            date_fields[field] = scan_time[field][scan_slice]

        # PRE group: flagPrecip, zFactorMeasured, etc.
        pre = grp["PRE"]
        flagPrecip = pre["flagPrecip"][scan_slice]
        data["flagPrecip"] = (
            ("nscan", "nray"),
            np.ma.masked_invalid(flagPrecip).filled(0).astype(bool),
        )
        zFactorMeasured = pre["zFactorMeasured"][scan_slice][:, :, ::-1]
        zFactorMeasured[zFactorMeasured < 0] = np.nan
        data["zFactorMeasured"] = (
            ("nscan", "nray", "nbin"),
            np.ma.masked_invalid(np.ma.masked_less_equal(zFactorMeasured, refl_min_thld)),
        )

        # CSF group: heightBB, qualityBB, typePrecip, qualityTypePrecip, flagShallowRain
        csf = grp["CSF"]
        for field in ["heightBB", "widthBB", "qualityBB", "qualityTypePrecip", "flagShallowRain", "flagBB"]:
            raw = csf[field][scan_slice]
            fv = csf[field]._FillValue
            data[field] = (("nscan", "nray"), np.ma.masked_equal(raw, fv))

        typePrecip = csf["typePrecip"][scan_slice]
        data["typePrecip"] = (("nscan", "nray"), typePrecip / 10000000)

        # SLV group: zFactorFinal (replaces zFactorCorrected in V08)
        slv = grp["SLV"]
        zFactorFinal = slv["zFactorFinal"][scan_slice][:, :, ::-1]
        zFactorFinal[zFactorFinal < 0] = np.nan
        data["zFactorCorrected"] = (
            ("nscan", "nray", "nbin"),
            np.ma.masked_invalid(np.ma.masked_less_equal(zFactorFinal, refl_min_thld)),
        )

        # VER group
        ver = grp["VER"]
        for field in ["heightZeroDeg"]:
            raw = ver[field][scan_slice]
            fv = ver[field]._FillValue
            data[field] = (("nscan", "nray"), np.ma.masked_equal(raw, fv))

        # PRE group: additional fields
        for field in ["elevation", "landSurfaceType", "binRealSurface", "binStormTop", "heightStormTop"]:
            raw = pre[field][scan_slice]
            fv = pre[field]._FillValue
            data[field] = (("nscan", "nray"), np.ma.masked_equal(raw, fv))

    # ------------------------------------------------------------------
    # Step 3: Build quality indicator (same logic as V07)
    # ------------------------------------------------------------------
    quality = np.zeros(data["heightBB"][-1].shape, dtype=np.int32)
    quality[((data["qualityBB"][-1] == 0) | (data["qualityBB"][-1] == 1)) & (data["qualityTypePrecip"][-1] == 1)] = 1
    quality[(data["qualityBB"][-1] > 1) | (data["qualityTypePrecip"][-1] > 1)] = 2
    data["quality"] = (data["heightBB"][0], quality)

    # ------------------------------------------------------------------
    # Step 4: Generate dimensions and derived fields (same as V07)
    # ------------------------------------------------------------------
    nray = np.linspace(-17.04, 17.04, 49)
    nbin = np.arange(0, 125 * 176, 125)

    R, A = np.meshgrid(nbin, nray)
    distance_from_sr = 407000 / np.cos(np.deg2rad(A)) - R
    data["distance_from_sr"] = (("nray", "nbin"), distance_from_sr)

    dtime = np.array(
        [
            datetime.datetime(*d)
            for d in zip(
                date_fields["Year"],
                date_fields["Month"],
                date_fields["DayOfMonth"],
                date_fields["Hour"],
                date_fields["Minute"],
                date_fields["Second"],
                date_fields["MilliSecond"],
            )
        ],
        dtype="datetime64",
    )

    data["nscan"] = (("nscan"), dtime)
    data["nray"] = (("nray"), nray)
    data["nbin"] = (("nbin"), nbin)

    dset = xr.Dataset(OrderedDict(sorted(data.items())))

    dset.nray.attrs = {"units": "degree", "description": "Deviation from Nadir"}
    dset.nbin.attrs = {"units": "m", "description": "Downward from 0: TOA to Earth ellipsoid."}
    dset.attrs["altitude"] = 407000
    dset.attrs["altitude_units"] = "m"
    dset.attrs["altitude_description"] = "GPM orbit"
    dset.attrs["beamwidth"] = 0.71
    dset.attrs["beamwidth_units"] = "degree"
    dset.attrs["beamwidth_description"] = "GPM beamwidth"
    dset.attrs["dr"] = 125
    dset.attrs["dr_units"] = "m"
    dset.attrs["dr_description"] = "GPM gate spacing"
    dset.attrs["orbit"] = get_gpm_orbit(infile)

    return dset


def read_radar(
    grfile: str,
    refl_name: str,
    radar_band: str = "C",
    correct_attenuation: bool = False,
    kdp_name: Union[str, None] = "KDP",
) -> List[xr.Dataset]:
    """
    Read ground radar data. If 2 files provided, then it will compute the
    displacement between these two files and then correct for advection the
    ground radar data in relation to the time of GPM exact overpass.

    Parameters:
    ===========
    grfile: str
        Ground radar input file.
    refl_name: str
        Name of the reflectivity field in the ground radar data.
    radar_band: str
        Ground radar frequency band for reflectivity conversion. S, C, and X
        supported.
    correct_attenuation: bool
        Should we correct for C- or X-band ground radar attenuation.
    kdp_name: Optional[str]
        Name of the KDP field in the ground radar data for attenuation correction.

    Returns:
    ========
    radar: List[xr.Dataset]
        Radar dataset
    """
    nradar = pyodim.read_odim(grfile, lazy_load=False)
    if nradar[-1].elevation.max() > 80:
        nradar.pop(-1)

    method = None
    if correct_attenuation and radar_band in ["X", "C"]:  # Correct attenuation of X or C bands.
        for idx, radar in enumerate(nradar):
            zh = radar[refl_name].values.copy()
            dr = 1e-3 * (radar.range.values[1] - radar.range.values[0])  # in km
            if kdp_name in radar:
                kdp = radar[kdp_name].values.copy()
                kdp[np.isnan(kdp)] = 0
                zh_corrected = correct.attenuation_correction_zphi(zh, kdp, dr=dr, wavelength=radar_band)
                method = "ZPHI"
            else:
                zh_corrected = correct.attenuation_correction_gunn_east(zh, dr=dr, wavelength=radar_band)
                method = "Gunn-East"

            nradar[idx] = nradar[idx].merge({"ZH_ATTEN_CORR": (("azimuth", "range"), zh_corrected)})

    if method is not None:
        print(f"GR reflectivity corrected for attenuation using {method} in band {radar_band}")

    return nradar