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
