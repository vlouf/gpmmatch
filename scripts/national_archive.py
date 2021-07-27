"""
GADI driver script for the volume matching of ground radar and GPM satellite.

@title: national_archive
@author: Valentin Louf <valentin.louf@bom.gov.au>
@institutions: Monash University and the Australian Bureau of Meteorology
@date: 08/12/2020

.. autosummary::
    :toctree: generated/

    _mkdir
    load_national_archive_info
    check_reflectivity_field_name
    remove
    get_radar_archive_file
    get_radar_band
    get_radar_beamwidth
    extract_zip
    buffer
    main
"""
import os
import re
import glob
import zipfile
import argparse
import datetime
import traceback
from typing import List

import pyart
import numpy as np
import pandas as pd
import dask.bag as db

import gpmmatch
from gpmmatch import NoRainError


def _mkdir(dir: str) -> None:
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


def load_national_archive_info() -> pd.DataFrame:
    """
    Load Australian national archive informations as a Dataframe.

    Returns:
    ========
    df: pandas.Dataframe
        Dataframe containing general information about the Australian radar
        Network (lat/lon, site name, frequency band and bandwith).
    """
    file = NATION_ARCHIVE_CONFIG
    df = pd.read_csv(file).drop_duplicates("id", keep="last").reset_index()

    return df


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


def remove(flist: List) -> None:
    """
    Remove file if it exists.
    """
    flist = [f for f in flist if f is not None]
    for f in flist:
        try:
            os.remove(f)
        except FileNotFoundError:
            pass
    return None


def get_radar_archive_file(date: pd.Timestamp, rid: int) -> str:
    """
    Return the archive containing the radar file for a given radar ID and a
    given date.
    Parameters:
    ===========
    date: datetime
        Date.
    Returns:
    ========
    file: str
        Radar archive if it exists at the given date.
    """
    try:
        date = pd.Timestamp(date)
    except Exception:
        traceback.print_exc()
        pass
    datestr = date.strftime("%Y%m%d")
    file = f"/g/data/rq0/level_1/odim_pvol/{rid:02}/{date.year}/vol/{rid:02}_{datestr}.pvol.zip"
    if not os.path.exists(file):
        return None

    return file


def get_radar_band(rid: int) -> str:
    """
    Get radar frequency-band information from the Australian radar network.

    Parameter:
    ==========
    rid: int
        Radar rapic identification number.

    Returns:
    ========
    band: str
        Radar frequency band ('S' or 'C')
    """
    df = load_national_archive_info()
    pos = df.id == int(rid)
    band = df.band[pos].values[0]
    if type(band) is not str:
        raise TypeError(f"Frequency band should be a str, not a {type(band)}.")

    return band


def get_radar_beamwidth(rid: int) -> float:
    """
    Get radar beamwidth information from the Australian radar network.

    Parameter:
    ==========
    rid: int
        Radar rapic identification number.

    Returns:
    ========
    beamwidth: float
        Radar beamwidth.
    """
    df = load_national_archive_info()
    pos = df.id == int(rid)
    beamwidth = df.beamwidth[pos].values[0]

    return beamwidth


def extract_zip(inzip: str, date: pd.Timestamp, path: str = "/scratch/kl02/vhl548/unzipdir") -> str:
    """
    Extract file in a daily archive zipfile for a specific datetime.

    Parameters:
    ===========
    inzip: str
        Input zipfile
    date: pd.Timestamp
        Which datetime we want to extract.
    path: str
        Path where we want to temporarly store the output file.

    Returns:
    ========
    grfile: str
        Output ground radar file.
    """

    def get_zipfile_name(namelist, date):
        datestr = [re.findall("[0-9]{8}_[0-9]{6}", n)[0] for n in namelist]
        timestamps = np.array([datetime.datetime.strptime(dt, "%Y%m%d_%H%M%S") for dt in datestr], dtype="datetime64")
        pos = np.argmin(np.abs(timestamps - date.to_numpy()))
        delta = np.abs(pd.Timestamp(timestamps[pos]) - date).seconds
        grfile = namelist[pos]

        if delta >= 600:
            raise FileNotFoundError("No file")

        return grfile

    with zipfile.ZipFile(inzip) as zid:
        namelist = zid.namelist()
        file = get_zipfile_name(namelist, date)
        zid.extract(file, path=path)

    grfile = os.path.join(path, file)

    return grfile


def buffer(gpmfile: str, date: pd.Timestamp, rid: str) -> None:
    """
    Driver function that extract the ground radar file from the national
    archive and then calls the volume matching function. Handles errors.

    Parameters:
    ===========
    gpmfile: str
        GPM hdf5 file to match.
    date: pd.Timestamp
        Timestamp of the closest overpass of GPM from the ground radar
    rid: str
        Groud radar identification
    """
    band = get_radar_band(rid)
    if band not in ["S", "C", "X"]:
        raise ValueError(f"Improper radar band, should be S, C or X not {band}.")
    beamwidth = get_radar_beamwidth(rid)

    inzip = get_radar_archive_file(date, rid)
    if inzip is None:
        print(f"Couldn't get zip archive for {date} and radar {rid}.")
        return None

    try:
        grfile = extract_zip(inzip, date)
    except FileNotFoundError:
        print(f"No ground {rid} radar file for {date}.")
        return None

    try:
        refl_name = check_reflectivity_field_name(grfile)
    except KeyError:
        traceback.print_exc()
        return None

    try:
        gpmmatch.vmatch_multi_pass(
            gpmfile,
            grfile,
            gr_beamwidth=beamwidth,
            radar_band=band,
            refl_name=refl_name,
            fname_prefix=rid,
            gr_offset=GR_OFFSET,
            gr_refl_threshold=GR_THLD,
            output_dir=OUTPATH,
            elevation_offset=ELEV_OFFSET,
#             gr_rmax=150e3,
        )
    except NoRainError:
        pass
    except Exception:
        print(f"ERROR: {gpmfile}.")
        traceback.print_exc()

    return None


def main() -> None:
    """
    Read the overpass csv file for the given radar ID and generates all the
    necessary arguments to call gpmmatch.
    """
    ovpass_list = glob.glob(os.path.join(ROOT_DIR, "overpass", "*.csv"))
    if len(ovpass_list) == 0:
        FileNotFoundError(f"No overpass configuration file found in {ROOT_DIR}.")

    for overpass_file in ovpass_list:
        rid = int(os.path.basename(overpass_file)[-6:-4])
        if rid != RID:
            continue
        df = pd.read_csv(
            overpass_file, parse_dates=["date"], header=0, names=["date", "name", "lon", "lat", "nprof", "source"]
        )

        argslist = []
        for n in range(len(df)):
            g = df.source[n]
            try:
                d = pd.Timestamp(df.date[n])
            except Exception:
                continue
            if d < SDATE or d > EDATE:
                continue
            argslist.append((g, d, RID))

        print(len(argslist))
        bag = db.from_sequence(argslist).starmap(buffer)
        _ = bag.compute()

    return None


if __name__ == "__main__":
    ROOT_DIR = "/scratch/kl02/vhl548/s3car-server/gpmmatch"
    NATION_ARCHIVE_CONFIG = os.path.expanduser("~/radar_site_list.csv")
    if not os.path.isfile(NATION_ARCHIVE_CONFIG):
        FileNotFoundError(f"National archive configuration file not found: {NATION_ARCHIVE_CONFIG}")

    # Parse arguments
    parser_description = """GPM volume matching on the National archive data."""
    parser = argparse.ArgumentParser(description=parser_description)
    parser.add_argument(
        "-o",
        "--output",
        dest="outdir",
        type=str,
        help="Output directory for the volume matching technique.",
        default=None,
    )
    parser.add_argument(
        "-r",
        "--rid",
        dest="rid",
        type=int,
        help="Radar standard RAPIC Volume ID for the Australian National Archive.",
        required=True,
    )
    parser.add_argument(
        "-s",
        "--sdate",
        dest="sdate",
        type=str,
        help="Start date (format YYYY-MM-DD) for processing the volume matching",
        default="2017-01-01",
    )
    parser.add_argument(
        "-e",
        "--edate",
        dest="edate",
        type=str,
        help="End date (format YYYY-MM-DD) for processing the volume matching",
        default="2020-01-01",
    )
    parser.add_argument(
        "-g",
        "--gr-thld",
        dest="grthld",
        type=float,
        help="Ground radar minimum reflectivity threshold for the volume matching technique.",
        default=10,
    )
    parser.add_argument(
        "-f",
        "--offset",
        dest="offset",
        type=float,
        help="Ground radar reflectivity offset to substract Z1=Z0-o.",
        default=0,
    )
    parser.add_argument(
        "-w",
        "--elev-offset",
        dest="elev_offset",
        type=float,
        help="Elevation offset to apply to ground radar (e_new = e_old + offset).",
        default=0,
    )

    args = parser.parse_args()
    RID = args.rid
    SDATE = pd.Timestamp(args.sdate)
    EDATE = pd.Timestamp(args.edate)
    GR_THLD = args.grthld
    GR_OFFSET = args.offset
    ELEV_OFFSET = args.elev_offset
    if ELEV_OFFSET == 0:
        ELEV_OFFSET = None

    if args.outdir is None:
        OUTPATH = os.path.join(ROOT_DIR, f"{RID}")
    else:
        OUTPATH = os.path.join(args.outdir, f"{RID}")
    _mkdir(OUTPATH)

    main()
