'''
Radar calibration monitoring using ground clutter. Processing the Australian
National archive.
@creator: Valentin Louf <valentin.louf@bom.gov.au>
@institution: Monash University and Bureau of Meteorology
@date: 02/04/2020
    buffer
    check_rid
    extract_zip    
    mkdir
    remove
    savedata
    main
'''
import gc
import os
import sys
import glob
import time
import zipfile
import argparse
import datetime
import warnings
import traceback

import crayons
import numpy as np
import pandas as pd
import dask.bag as db


def buffer(infile, cmask):
    '''
    Buffer function to catch and kill errors.
    Parameters:
    ===========
    infile: str
        Input radar file.
    Returns:
    ========
    dtime: np.datetime64
        Datetime of infile
    rca: float
        95th percentile of the clutter reflectivity.
    '''
    try:
        dtime, rca = cluttercal.extract_clutter(infile, cmask, refl_name='total_power')
    except ValueError:
        return None
    except Exception:
        print(infile)
        traceback.print_exc()
        return None

    return dtime, rca


def check_rid():
    '''
    Check if the Radar ID provided exists.
    '''
    indir = f'/g/data/rq0/odim_archive/odim_pvol/{RID}'
    return os.path.exists(indir)


def check_reflectivity(infile):
    '''
    Check if the Radar file contains the uncorrected reflectivity field.
    '''
    is_good = True
    try:
        radar = cluttercal.cluttercal._read_radar(infile, refl_name=REFL_NAME)
    except Exception:
        traceback.print_exc()
        return False

    try:
        radar.fields[REFL_NAME]
    except KeyError:
        print(crayons.red(f"{os.path.basename(infile)} does not contain {REFL_NAME} field."))
        is_good = False

    del radar
    return is_good


def extract_zip(inzip, path):
    '''
    Extract content of a zipfile inside a given directory.
    Parameters:
    ===========
    inzip: str
        Input zip file.
    path: str
        Output path.
    Returns:
    ========
    namelist: List
        List of files extracted from  the zip.
    '''
    with zipfile.ZipFile(inzip) as zid:
        zid.extractall(path=path)
        namelist = [os.path.join(path, f) for f in zid.namelist()]
    return namelist


def get_radar_archive_file(date):
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
    file = f"/g/data/rq0/odim_archive/odim_pvol/{RID}/{date.year}/vol/{RID}_{datestr}.pvol.zip"
    if not os.path.exists(file):
        return None

    return file


def mkdir(path):
    '''
    Create the DIRECTORY(ies), if they do not already exist
    '''
    try:
        os.mkdir(path)
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
