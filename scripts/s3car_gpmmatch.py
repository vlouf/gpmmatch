"""
Quality control of Radar calibration monitoring using ground clutter
@creator: Valentin Louf <valentin.louf@bom.gov.au>
@project: s3car-server
@institution: Bureau of Meteorology
@date: 21/08/2020

.. autosummary::
    :toctree: generated/

    NoPrecipitationError
    precip_in_domain
    get_overpass_with_precip
    main
"""
import os
import re
import glob
import argparse
import datetime
import warnings
import traceback

import pyart
import gpmmatch
import numpy as np
import xarray as xr
import pandas as pd

import dask
import dask.bag as db
from gpmmatch import NoRainError


class NoPrecipitationError(Exception):
    pass


def _mkdir(dir: str):
    """
    Make directory.
    """
    if os.path.exists(dir):
        return None

    try:
        os.mkdir(dir)
    except FileExistsError:
        pass

    return None


def check_reflectivity_field_name(infile: str) -> str:
    """
    Check reflectivity field name in the input radar file.

    Parameter:
    ==========
    infile: str
        Input radar file.

    Returns:
    ========
    field_name: str
        Name of the reflectivity field in the input radar file.
    """
    radar = pyart.aux_io.read_odim_h5(infile, file_field_names=True)

    for field_name in ["DBZH_CLEAN", "DBZH", "TH", "FIELD_ERROR"]:
        try:
            _ = radar.fields[field_name]["data"]
            break
        except Exception:
            continue

    if field_name == "FIELD_ERROR":
        raise KeyError(f"Reflectivity field name not found in {infile}.")

    del radar
    return field_name


def get_ground_radar_file(date, rid):
    """
    Return the archive containing the radar file for a given radar ID and a
    given date.

    Parameters:
    ===========
    date: datetime
        Date.
    rid: int
        Radar ID number.

    Returns:
    ========
    grfile: str
        Radar archive if it exists at the given date.
    """
    datestr = date.strftime("%Y%m%d")
    # Input directory checks.
    input_dir = os.path.join(VOLS_ROOT_PATH, str(rid), datestr)
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Directory {input_dir} not found/does not exist for radar {rid} at {datestr}.")

    namelist = sorted(glob.glob(os.path.join(input_dir, "*.h5")))
    datestr = [re.findall("[0-9]{8}_[0-9]{6}", n)[0] for n in namelist]
    timestamps = np.array([datetime.datetime.strptime(dt, "%Y%m%d_%H%M%S") for dt in datestr], dtype="datetime64")
    pos = np.argmin(np.abs(timestamps - date.to_numpy()))
    delta = np.abs(pd.Timestamp(timestamps[pos]) - date).seconds
    grfile = namelist[pos]

    if delta >= 600:
        raise FileNotFoundError(f"Time difference ({delta}s) too big for radar {rid} at {datestr}.")

    return grfile


def load_radar_info():

    return None


def find_cases_and_generate_args(gpmfile):
    """
    Look for if there's GPM precipitation measurements within the domain of any
    radar of the australian network. If there is a potential match, then we
    look for all the necessary information related to that case (i.e. frequency,
    bandwidth) and the closest (time-wise) ground radar file corresponding to
    the GPM overpass. The list returned is formatted in a way that the
    elements can immediatly be sent to gpmmatch for processing.

    Parameters:
    ===========
    gpmfile: str
        Input GPM file.

    Returns:
    ========
    processing_list: List[Dict]
        List containing the arguments necessary to call
    """
    radar_infoset = load_radar_info()
    gpmset = gpmmatch.io.read_GPM(gpmfile)
    data = dict()
    processing_list = []
    for n in range(len(radar_infoset)):
        rid = radar_infoset.id[n]
        rname = radar_infoset.short_name[n]
        grlat = radar_infoset.site_lat[n]
        grlon = radar_infoset.site_lon[n]
        beamwidth = radar_infoset.beamwidth[n]
        band = radar_infoset.band[n]
        try:
            nprof, gpmtime = gpmmatch.io.check_precip_in_domain(gpmset, grlon=grlon, grlat=grlat)
        except NoPrecipitationError:
            print(f"No GPM precipitation in radar {rid} - {rname} domain.")
            continue

        if nprof < 10:
            print(f"Not enough precipitation in domain for {rid} - {rname}")
            continue

        try:
            grfile = get_ground_radar_file(gpmtime, rid)
        except FileNotFoundError:
            traceback.print_exc()
            continue

        try:
            refl_name = check_reflectivity_field_name(grfile)
        except KeyError:
            traceback.print_exc()
            continue

        outpath = os.path.join(OUTPATH, f"{rid}")
        _mkdir(outpath)

        arguments = {
            "gpmfile": gpmfile,
            "grfile": grfile,
            "gr_beamwidth": beamwidth,
            "radar_band": band,
            "refl_name": refl_name,
            "fname_prefix": f"{rid}",
            "output_dir": outpath,
        }
        processing_list.append(arguments)

    del gpmset
    return processing_list


def main():
    processing_list = find_cases_and_generate_args(INFILE)
    if len(processing_list) == 0:
        print(f"Nothing found for {INFILE}. Doing nothing.")
        return None

    for kwargs in processing_list:
        try:
            _ = gpmmatch.vmatch_multi_pass(**kwargs)
        except NoRainError:
            pass
        except Exception:
            print(f"ERROR: {kwargs}.")
            traceback.print_exc()
            continue

    return None


if __name__ == "__main__":
    VOLS_ROOT_PATH = "/srv/data/s3car-server/vols"
    # Parse arguments
    parser_description = """GPM volume matching for s3car-server."""
    parser = argparse.ArgumentParser(description=parser_description)
    parser.add_argument(
        "-i",
        "--input",
        dest="infile",
        type=str,
        help="GPM file to process.",
        required=True,
    )
    parser.add_argument(
        "-o",
        "--output",
        dest="outdir",
        type=str,
        help="Output directory.",
        default="/srv/data/s3car-server/gpmmatch",
    )

    args = parser.parse_args()
    INFILE = args.infile
    OUTPATH = args.outdir

    if not os.path.exists(OUTPATH):
        raise FileNotFoundError(f"Output directory {OUTPATH} does not exists.")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()
