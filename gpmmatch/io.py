"""
Utilities to read the input data and format them in a way to be read by
volume_matching.

@title: io
@author: Valentin Louf <valentin.louf@bom.gov.au>
@institutions: Monash University and the Australian Bureau of Meteorology
@creation: 17/02/2020
@date: 29/05/2025

.. autosummary::
    :toctree: generated/

    NoPrecipitationError
    _read_radar
    check_precip_in_domain
    data_load_and_checks
    get_ground_radar_attributes
    get_gpm_orbit
    read_GPM
    read_radars
"""

import re
import copy
import datetime
import warnings

from typing import Tuple, List, Union
from collections import OrderedDict

import h5py
import pyodim
import pyproj
import numpy as np
import pandas as pd
import xarray as xr

from . import correct
from . import default


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
    grfile2: str
        (Optional) Second ground radar input file to compute grid displacement
        and advection.
    refl_name: str
        Name of the reflectivity field in the ground radar data.
    correct_attenuation: bool
        Should we correct for C- or X-band ground radar attenuation.
    radar_band: str
        Ground radar frequency band for reflectivity conversion. S, C, and X
        supported.

    Returns:
    --------
    gpmset: xarray.Dataset
        Dataset containing the input datas.
    radar: pyart.core.Radar
        Pyart radar dataset.
    """
    if refl_name is None:
        raise ValueError("Reflectivity field name not given.")

    gpmset = read_GPM(gpmfile)
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
    radar = read_radar(grfile, refl_name, radar_band=radar_band, correct_attenuation=correct_attenuation)

    return gpmset, radar


def get_gpm_orbit(gpmfile: str) -> int:
    """
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


def read_GPM(infile: str, refl_min_thld: float = 0) -> xr.Dataset:
    """
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


def read_radar(
    grfile: str, refl_name: str, radar_band: str = "C", correct_attenuation: bool = False
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

    Returns:
    ========
    radar: List[xr.Dataset]
        Radar dataset
    """
    nradar = pyodim.read_odim(grfile, lazy_load=False)
    if nradar[-1].elevation.max() > 80:
        nradar.pop(-1)

    for idx in range(len(nradar)):
        if correct_attenuation:
            if radar_band in ["X", "C"]:  # Correct attenuation of X or C bands.
                corr_refl = correct.correct_attenuation(nradar[idx][refl_name].values, radar_band)
                nradar[idx][refl_name].values = corr_refl

    return nradar
