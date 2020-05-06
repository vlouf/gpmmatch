import os
import re
import glob
import zipfile
import datetime
import traceback

import pyart
import numpy as np
import pandas as pd
import dask
import dask.bag as db

import gpmmatch
from gpmmatch import NoRainError


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
#         raise FileNotFoundError(f'{file}')

    return file


def extract_zip(inzip, date, path):
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
    inzip = get_radar_archive_file(date, rid)
    if inzip is None:
        return None
    
    try:
        grfile = extract_zip(inzip, date, path)
    except FileNotFoundError:
        print(f'No ground {rid} radar file for {date}.')
        return None
    
    try:
        matchset = gpmmatch.vmatch_multi_pass(gpmfile,
                                              grfile,    
                                              radar_band='S',
                                              refl_name='reflectivity',
                                              fname_prefix=rid,   
                                              gr_refl_threshold=10,
                                              gpm_refl_threshold=0,
                                              write_output=False)
    except NoRainError:
        matchset = None
        pass
#     except OSError:
#         matchset = None
#         pass
    except Exception:
        print('!!! ERROR !!!')
        print(gpmfile)
        print(date)
        traceback.print_exc()
        matchset = None
    
    remove([grfile])
    
    return matchset


def savedata(datafile, rid):    
    dtime = pd.Timestamp(datafile.gpm_overpass_time)    
    datestr = dtime.strftime('%Y%m%d')
    timestr = dtime.strftime('%H%M%S')
    
    outpath = os.path.join(".", rid)
    try:
        os.mkdir(outpath)
    except FileExistsError:
        pass
    
    outfile = os.path.join(outpath, f"vmatch.{rid}.{datestr}.{timestr}.nc"   )
    if not os.path.exists(outfile):
        datafile.to_netcdf(outfile)
    
    return None


config_files = sorted(glob.glob('/scratch/kl02/vhl548/gpm_output/overpass/*.csv'))
path = '/scratch/kl02/vhl548/unzipdir'

for config in tqdm.tqdm_notebook(config_files, total=len(config_files)):
    rid = os.path.basename(config)[-6:-4]
    if rid != '02':
        continue
    df = pd.read_csv(config, parse_dates=['date'], header=None, names=['date', 'name', 'lon', 'lat', 'nprof', 'source'])

    argslist = []
    for n in range(len(df)):
        if rid == '02' or rid == '01':
            if 'Tasmania' in df.source[n]:
                continue
        g = df.source[n]
        d = df.date[n]
        argslist.append((g, d, rid))

    bag = db.from_sequence(argslist).starmap(buffer)
    with ProgressBar():
        rslt = bag.compute()

    rslt = [r for r in rslt if r is not None]       
    break