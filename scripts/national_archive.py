'''
GADI driver script for the volume matching of ground radar and GPM satellite.

@title: job_gadi
@author: Valentin Louf <valentin.louf@bom.gov.au>
@institutions: Monash University and the Australian Bureau of Meteorology
@date: 06/05/2020
    _mkdir
    remove
    get_radar_archive_file
    extract_zip
    buffer
    main
'''
import os
import re
import glob
import zipfile
import argparse
import datetime
import warnings
import traceback

import pyart
import numpy as np
import pandas as pd
import dask
import dask.bag as db

import gpmmatch
from gpmmatch import NoRainError


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


def remove(flist):
    '''
    Remove file if it exists.
    '''
    flist = [f for f in flist if f is not None]
    for f in flist:
        try:
            os.remove(f)
        except FileNotFoundError:
            pass
    return None


def get_radar_archive_file(date, rid):
    '''
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
    '''
    datestr = date.strftime('%Y%m%d')
    file = f"/g/data/rq0/level_1/odim_pvol/{rid}/{date.year}/vol/{rid}_{datestr}.pvol.zip"
    if not os.path.exists(file):
        return None

    return file


def extract_zip(inzip, date, path='/scratch/kl02/vhl548/unzipdir'):
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
        datestr = [re.findall('[0-9]{8}_[0-9]{6}', n)[0] for n in namelist]
        timestamps = np.array([datetime.datetime.strptime(dt, '%Y%m%d_%H%M%S') for dt in datestr], dtype='datetime64')
        pos = np.argmin(np.abs(timestamps - date.to_numpy()))
        delta = np.abs(pd.Timestamp(timestamps[pos]) - date).seconds
        grfile = namelist[pos]

        if delta >= 600:
            raise FileNotFoundError('No file')

        return grfile

    with zipfile.ZipFile(inzip) as zid:
        namelist = zid.namelist()
        file = get_zipfile_name(namelist, date)
        zid.extract(file, path=path)

    grfile = os.path.join(path, file)

    return grfile


def buffer(gpmfile, date, rid):
    '''
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
    '''
    inzip = get_radar_archive_file(date, rid)
    if inzip is None:
        return None

    try:
        grfile = extract_zip(inzip, date)
    except FileNotFoundError:
        print(f'No ground {rid} radar file for {date}.')
        return None

    try:
        matchset = gpmmatch.vmatch_multi_pass(gpmfile,
                                              grfile,
                                              radar_band='C',
                                              refl_name='reflectivity',
                                              fname_prefix=rid,
                                              gr_refl_threshold=GR_THLD,
                                              gpm_refl_threshold=0,
                                              output_dir=OUTPATH)
    except NoRainError:
        pass
    except Exception:
        print('!!! ERROR !!!')
        print(gpmfile)
        traceback.print_exc()

    remove([grfile])

    return None


def main():
    for config in CONFIG_FILES:
        rid = os.path.basename(config)[-6:-4]
        if rid != RID:
            continue
        df = pd.read_csv(config, parse_dates=['date'], header=None, names=['date', 'name', 'lon', 'lat', 'nprof', 'source'])

        argslist = []
        for n in range(len(df)):
            if rid == '02' or rid == '01':
                if 'Tasmania' in df.source[n]:
                    continue
            g = df.source[n]
            d = df.date[n]
            argslist.append((g, d, RID))

        bag = db.from_sequence(argslist).starmap(buffer)
        rslt = bag.compute()
        break


if __name__ == "__main__":
     # Parse arguments
    parser_description = """GPM volume matching on the National archive data."""
    parser = argparse.ArgumentParser(description=parser_description)
    parser.add_argument(
        '-o',
        '--output',
        dest='outdir',
        type=str,
        help='Output directory.',
        default=None)
    parser.add_argument(
        '-r',
        '--rid',
        dest='rid',
        type=str,
        help='Radar ID.',
        default='02')
    parser.add_argument(
        '-g',
        '--gr-thld',
        dest='grthld',
        type=float,
        help='Ground radar reflectivity threshold.',
        required=True)

    parser.add_argument('--unravel', dest='unravel', action='store_true')
    parser.add_argument('--no-unravel', dest='unravel', action='store_false')
    parser.set_defaults(unravel=True)

    args = parser.parse_args()
    RID = args.rid
    GR_THLD = args.grthld
    if args.outdir is None:
        OUTPATH = os.path.join(os.getcwd(), f'gr_{int(GR_THLD)}dB')
    else:
        OUTPATH = os.path.join(args.outdir, f'gr_{int(GR_THLD)}dB')
    _mkdir(OUTPATH)

    CONFIG_FILES = sorted(glob.glob('/scratch/kl02/vhl548/gpm_output/overpass/*.csv'))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()
    pass
