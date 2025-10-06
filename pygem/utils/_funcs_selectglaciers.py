"""
Python Glacier Evolution Model (PyGEM)

copyright © 2018 David Rounce <drounce@cmu.edu>

Distributed under the MIT license

Functions of different ways to select glaciers
"""

# Built-in libraries
import os
import pickle

# External libraries
import numpy as np
import pandas as pd


# %% ----- Functions to select specific glacier numbers -----
def get_same_glaciers(glac_fp, ending):
    """
    Get same glaciers for testing of priors

    Parameters
    ----------
    glac_fp : str
        filepath to where netcdf files of individual glaciers are held
    ending : str
        ending of the string that you want to get (ex. '.nc')

    Returns
    -------
    glac_list : list
        list of rgi glacier numbers
    """
    glac_list = []
    for i in os.listdir(glac_fp):
        if i.endswith(ending):
            glac_list.append(i.split(ending)[0])
    glac_list = sorted(glac_list)

    return glac_list


def glac_num_fromrange(int_low, int_high):
    """
    Generate list of glaciers for all numbers between two integers.

    Parameters
    ----------
    int_low : int64
        low value of range
    int_high : int64
        high value of range

    Returns
    -------
    y : list
        list of rgi glacier numbers
    """
    x = (np.arange(int_low, int_high + 1)).tolist()
    y = [str(i).zfill(5) for i in x]
    return y


def glac_fromcsv(csv_fullfn, cn='RGIId'):
    """
    Generate list of glaciers from csv file

    Parameters
    ----------
    csv_fp, csv_fn : str
        csv filepath and filename

    Returns
    -------
    y : list
        list of glacier numbers, e.g., ['14.00001', 15.00001']
    """
    df = pd.read_csv(csv_fullfn)
    return [x.split('-')[1] for x in df[cn].values]


def glac_wo_cal(regions, prms_fp_sub=None, cal_option='MCMC'):
    """
    Glacier list of glaciers that still need to be calibrated
    """
    todo_list = []
    for reg in regions:
        prms_fns = []
        prms_fp = prms_fp_sub + str(reg).zfill(2) + '/'
        for i in os.listdir(prms_fp):
            if i.endswith('-modelprms_dict.pkl'):
                prms_fns.append(i)

        prms_fns = sorted(prms_fns)

        for nfn, prms_fn in enumerate(prms_fns):
            glac_str = prms_fn.split('-')[0]

            if nfn % 500 == 0:
                print(glac_str)

            # Load model parameters
            with open(prms_fp + prms_fn, 'rb') as f:
                modelprms_dict = pickle.load(f)

            # Check if 'MCMC' is in the modelprms_dict
            if cal_option not in modelprms_dict.keys():
                todo_list.append(glac_str)

    return todo_list
