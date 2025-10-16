"""
Python Glacier Evolution Model (PyGEM)

copyright © 2024 Brandon Tober <btober@cmu.edu> David Rounce <drounce@cmu.edu>

Distributed under the MIT license

derive sub-annual glacierwide mass for PyGEM simulation using annual glacier mass and sub-annual total mass balance
"""

# Built-in libraries
import argparse
import collections
import glob
import multiprocessing
import os
import time
import json
import sys
from functools import partial

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# External libraries
import xarray as xr

# pygem imports
from pygem.setup.config import ConfigManager

# instantiate ConfigManager
config_manager = ConfigManager()
# read the config
pygem_prms = config_manager.read_config()


# ----- FUNCTIONS -----
def getparser():
    """
    Use argparse to add arguments from the command line
    """
    parser = argparse.ArgumentParser(
        description='process sub-annual glacierwide mass from annual mass and total sub-annual mass balance'
    )
    # add arguments
    parser.add_argument(
        '-simpath',
        action='store',
        type=str,
        nargs='+',
        help='path to PyGEM simulation (can take multiple)',
    )
    parser.add_argument(
        '-simdir',
        action='store',
        type=str,
        default=None,
        help='directory with glacierwide simulation outputs for which to process sub-annual mass',
    )
    parser.add_argument(
        '-ncores',
        action='store',
        type=int,
        default=1,
        help='number of simultaneous processes (cores) to use',
    )
    parser.add_argument('-v', '--debug', action='store_true', help='Flag for debugging')
    return parser


def get_subannual_mass(df_annual, df_sub, debug=False):
    """
    funciton to calculate the sub-annual glacier mass
    from annual glacier mass and sub-annual total mass balance

    Parameters
    ----------
    glac_mass_annual : float
        ndarray containing the annual glacier mass for each year computed by PyGEM
        shape: [#glac, #years]
        unit: kg
    glac_massbaltotal : float
        ndarray containing the total mass balance computed by PyGEM
        shape: [#glac, #steps]
        unit: kg

    Returns
    -------
    glac_mass: float
        ndarray containing the running glacier mass
        shape : [#glac, #steps]
        unit: kg

    """

    # ensure datetime and sorted
    df_annual['time'] = pd.to_datetime(df_annual['time'])
    df_sub['time'] = pd.to_datetime(df_sub['time'])
    df_annual = df_annual.sort_values('time').reset_index(drop=True)
    df_sub = df_sub.sort_values('time').reset_index(drop=True)

    # year columns
    df_annual['year'] = df_annual['time'].dt.year
    df_sub['year'] = df_sub['time'].dt.year

    # map annual starting mass to sub rows
    annual_by_year = df_annual.set_index('year')['mass']
    df_sub['annual_mass'] = df_sub['year'].map(annual_by_year)

    # shift massbaltotal within each year so the Jan value doesn't affect Jan mass itself
    # i.e., massbaltotal at Jan-01 contributes to Feb-01 mass
    df_sub['mb_shifted'] = df_sub.groupby('year')['massbaltotal'].shift(1).fillna(0.0)

    # cumulative sum of shifted values within each year
    df_sub['cum_mb_since_year_start'] = df_sub.groupby('year')['mb_shifted'].cumsum()

    # compute sub-annual mass
    df_sub['mass'] = df_sub['annual_mass'] + df_sub['cum_mb_since_year_start']

    if debug:
        # --- Quick plot of Jan start points (sub vs annual) ---
        # Plot all sub-annual masses as a line
        plt.figure(figsize=(12,5))
        plt.plot(df_sub['time'], df_sub['mass'], label='Sub-annual mass', color='blue')

        # Overlay annual masses as points/line
        plt.plot(df_annual['time'], df_annual['mass'], 'o--', label='Annual mass', color='orange', markersize=6)

        # Labels and legend
        plt.xlabel('Time')
        plt.ylabel('Glacier Mass')
        plt.title('Sub-annual Glacier Mass vs Annual Mass')
        plt.legend()
        plt.tight_layout()
        plt.show()

    return df_sub['mass'].values


def update_xrdataset(input_ds, glac_mass, timestep):
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
    output_coords_dict['glac_mass'] = collections.OrderedDict([('glac', glac_values), ('time', time_values)])

    # Attributes dictionary
    output_attrs_dict = {}
    output_attrs_dict['glac_mass'] = {
        'long_name': 'glacier mass',
        'units': 'kg',
        'temporal_resolution': timestep,
        'comment': 'glacier mass',
    }

    # Add variables to empty dataset and merge together
    count_vn = 0
    encoding = {}
    for vn in output_coords_dict.keys():
        empty_holder = np.zeros([len(output_coords_dict[vn][i]) for i in list(output_coords_dict[vn].keys())])
        output_ds = xr.Dataset(
            {vn: (list(output_coords_dict[vn].keys()), empty_holder)},
            coords=output_coords_dict[vn],
        )
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
        encoding[vn] = {'_FillValue': None, 'zlib': True, 'complevel': 9}
    output_ds_all['glac_mass'].values = glac_mass[np.newaxis,:]

    return output_ds_all, encoding


def run(simpath, debug=False):
    """
    create sub-annual mass data product
    Parameters
    ----------
    simpath : str
        patht to PyGEM simulation
    """
    if os.path.exists(simpath):
        try:
            # open dataset
            statsds = xr.open_dataset(simpath)
            timestep = json.loads(statsds.attrs['model_parameters'])['timestep']
            yvals = statsds.year.values
            # convert to pandas dataframe with annual mass
            annual_df = pd.DataFrame({
                                            "time": pd.to_datetime([f'{y}-01-01' for y in yvals]),
                                            "mass": statsds.glac_mass_annual[0].values
                                            })
            tvals = statsds.time.values
            # convert to pandas dataframe with sub-annual mass balance       
            steps_df = pd.DataFrame({
                                            "time": pd.to_datetime([t.strftime("%Y-%m-%d") for t in tvals]),
                                            "massbaltotal": statsds.glac_massbaltotal[0].values * pygem_prms['constants']['density_ice']
                                            })

            # calculate sub-annual mass - pygem glac_massbaltotal is in units of m3, so convert to mass using density of ice
            glac_mass = get_subannual_mass(
                annual_df, steps_df, debug=debug
            )
            statsds.close()

            # update dataset to add sub-annual mass change
            output_ds_stats, encoding = update_xrdataset(statsds, glac_mass, timestep)

            # close input ds before write
            statsds.close()

            # append to existing stats netcdf
            output_ds_stats.to_netcdf(simpath, mode='a', encoding=encoding, engine='netcdf4')

            # close datasets
            output_ds_stats.close()

        except:
            pass
    else:
        print('Simulation not found: ', simpath)

    return


def main():
    time_start = time.time()
    args = getparser().parse_args()

    simpath = None
    if args.simdir:
        # get list of sims
        simpath = glob.glob(args.simdir + '/*.nc')
    else:
        if args.simpath:
            simpath = args.simpath

    if simpath:
        # number of cores for parallel processing
        if args.ncores > 1:
            ncores = int(np.min([len(simpath), args.ncores]))
        else:
            ncores = 1

        # set up partial function with debug argument
        run_partial = partial(
            run,
            debug=args.debug)
    
        # Parallel processing
        print('Processing with ' + str(ncores) + ' cores...')
        with multiprocessing.Pool(ncores) as p:
            p.map(run_partial, simpath)

    print('Total processing time:', time.time() - time_start, 's')


if __name__ == '__main__':
    main()