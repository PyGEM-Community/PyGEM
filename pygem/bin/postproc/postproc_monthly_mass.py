"""
Python Glacier Evolution Model (PyGEM)

copyright © 2024 Brandon Tober <btober@cmu.edu> David Rounce <drounce@cmu.edu>

Distrubted under the MIT lisence

derive monthly glacierwide mass for PyGEM simulation using annual glacier mass and monthly total mass balance
"""
# Built-in libraries
import argparse
import collections
import copy
import inspect
import multiprocessing
import os
import glob
import sys
import time
import json
# External libraries
import pandas as pd
import pickle
import numpy as np
import xarray as xr
# pygem imports
import pygem
from pygem.setup.config import ConfigManager
# instantiate ConfigManager
config_manager = ConfigManager()
# read the config
pygem_prms = config_manager.read_config()
import pygem.pygem_modelsetup as modelsetup


# ----- FUNCTIONS -----
def getparser():
    """
    Use argparse to add arguments from the command line
    """
    parser = argparse.ArgumentParser(description="process monthly glacierwide mass from annual mass and total monthly mass balance")
    # add arguments
    parser.add_argument('-simpath', action='store', type=str, nargs='+',
                        help='path to PyGEM simulation (can take multiple)')
    parser.add_argument('-simdir', action='store', type=str, default=None,
                        help='directory with glacierwide simulation outputs for which to process monthly mass')
    parser.add_argument('-ncores', action='store', type=int, default=1,
                        help='number of simultaneous processes (cores) to use')

    return parser


def get_monthly_mass(glac_mass_annual, glac_massbaltotal_monthly):
    """
    funciton to calculate the monthly glacier mass
    from annual glacier mass and monthly total mass balance

    Parameters
    ----------
    glac_mass_annual : float
        ndarray containing the annual glacier mass for each year computed by PyGEM
        shape: [#glac, #years]
        unit: kg
    glac_massbaltotal_monthly : float
        ndarray containing the monthly total mass balance computed by PyGEM
        shape: [#glac, #months]
        unit: kg

    Returns
    -------
    glac_mass_monthly: float
        ndarray containing the monthly glacier mass
        shape : [#glac, #months]
        unit: kg

    """
    # get running total monthly mass balance - reshape into subarrays of all values for a given year, then take cumulative sum
    oshape = glac_massbaltotal_monthly.shape
    running_glac_massbaltotal_monthly = np.reshape(glac_massbaltotal_monthly, (-1,12), order='C').cumsum(axis=-1).reshape(oshape)

    # tile annual mass to then superimpose atop running glacier mass balance (trim off final year from annual mass)
    glac_mass_monthly = np.repeat(glac_mass_annual[:,:-1], 12, axis=-1)

    # add annual mass values to running glacier mass balance
    glac_mass_monthly += running_glac_massbaltotal_monthly

    return glac_mass_monthly


def update_xrdataset(input_ds, glac_mass_monthly):
    """
    update xarray dataset to add new fields

    Parameters
    ----------
    xrdataset : xarray Dataset
        existing xarray dataset
    newdata : ndarray 
        new data array
    description: str
        describing new data field

    output_ds : xarray Dataset
        empty xarray dataset that contains variables and attributes to be filled in by simulation runs
    encoding : dictionary
        encoding used with exporting xarray dataset to netcdf
    """
    # coordinates
    glac_values = input_ds.glac.values
    time_values = input_ds.time.values

    output_coords_dict = collections.OrderedDict()
    output_coords_dict['glac_mass_monthly'] = (
            collections.OrderedDict([('glac', glac_values), ('time', time_values)]))

    # Attributes dictionary
    output_attrs_dict = {}
    output_attrs_dict['glac_mass_monthly'] = {
            'long_name': 'glacier mass',
            'units': 'kg',
            'temporal_resolution': 'monthly',
            'comment': 'monthly glacier mass'}


    # Add variables to empty dataset and merge together
    count_vn = 0
    encoding = {}
    for vn in output_coords_dict.keys():
        empty_holder = np.zeros([len(output_coords_dict[vn][i]) for i in list(output_coords_dict[vn].keys())])
        output_ds = xr.Dataset({vn: (list(output_coords_dict[vn].keys()), empty_holder)},
                               coords=output_coords_dict[vn])
        count_vn += 1
        # Merge datasets of stats into one output
        if count_vn == 1:
            output_ds_all = output_ds
        else:
            output_ds_all = xr.merge((output_ds_all, output_ds))
    # Add attributes
    for vn in output_ds_all.variables:
        try:
            output_ds_all[vn].attrs = output_attrs_dict[vn]
        except:
            pass
        # Encoding (specify _FillValue, offsets, etc.)
        encoding[vn] = {'_FillValue': None,
                        'zlib':True,
                        'complevel':9
                        }    

    output_ds_all['glac_mass_monthly'].values = (
            glac_mass_monthly
            )

    return output_ds_all, encoding


def run(simpath):
    """
    create monthly mass data product
    Parameters
    ----------
    simpath : str
        patht to PyGEM simulation
    """
    if os.path.exists(simpath):
        try:
            # open dataset
            statsds = xr.open_dataset(simpath)

            # calculate monthly mass - pygem glac_massbaltotal_monthly is in units of m3, so convert to mass using density of ice
            glac_mass_monthly = get_monthly_mass(
                                                statsds.glac_mass_annual.values, 
                                                statsds.glac_massbaltotal_monthly.values * pygem_prms['constants']['density_ice'], 
                                                )
            statsds.close()

            # update dataset to add monthly mass change
            output_ds_stats, encoding = update_xrdataset(statsds, glac_mass_monthly)

            # close input ds before write
            statsds.close()

            # append to existing stats netcdf
            output_ds_stats.to_netcdf(simpath, mode='a', encoding=encoding, engine='netcdf4')

            # close datasets
            output_ds_stats.close()
        
        except:
            pass
    else:
        print('Simulation not found: ',simpath)

    return


def main():
    time_start = time.time()
    args = getparser().parse_args()

    simpath = None
    if args.simdir:
        # get list of sims
        simpath = glob.glob(args.simdir+'*.nc')
    else:
        if args.simpath:
            simpath = args.simpath

    if simpath:
        # number of cores for parallel processing
        if args.ncores > 1:
            ncores = int(np.min([len(simpath), args.ncores]))
        else:
            ncores = 1

        # Parallel processing
        print('Processing with ' + str(args.ncores) + ' cores...')
        with multiprocessing.Pool(args.ncores) as p:
            p.map(run,simpath)

    print('Total processing time:', time.time()-time_start, 's')
    
if __name__ == "__main__":
    main()