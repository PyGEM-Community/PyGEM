"""
Python Glacier Evolution Model (PyGEM)

copyright © 2024 Brandon Tober <btober@cmu.edu> David Rounce <drounce@cmu.edu>

Distrubted under the MIT lisence
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
from functools import partial
import matplotlib.pyplot as plt
# External libraries
import numpy as np
import xarray as xr
# oggm
from oggm import workflow, tasks, cfg
from oggm.sandbox import distribute_2d
# pygem imports
from pygem.setup.config import ConfigManager
# instantiate ConfigManager
config_manager = ConfigManager()
# read the config
pygem_prms = config_manager.read_config()
import pygem
import pygem.pygem_modelsetup as modelsetup
from pygem.oggm_compat import single_flowline_glacier_directory
from pygem.oggm_compat import single_flowline_glacier_directory_with_calving


def getparser():
    """
    Use argparse to add arguments from the command line
    """
    parser = argparse.ArgumentParser(description="distrube PyGEM simulated ice thickness to a 2D grid")
    # add arguments
    parser.add_argument('-simpath', action='store', type=str, nargs='+',
                        help='path to PyGEM binned simulation (can take multiple)')
    parser.add_argument('-ncores', action='store', type=int, default=1,
                        help='number of simultaneous processes (cores) to use')
    parser.add_argument('-v', '--debug', action='store_true',
                        help='Flag for debugging')
    return parser


# method to convert pygem output to oggm flowline diagnostic output, in the format expected by oggm.distribute_2d
def pygem_to_oggm(pygem_simpath, oggm_diag=None, debug=False):
    """
    take PyGEM model output and temporarily store it in a way that OGGM distribute_2d expects
    this will be a netcdf file named fl_diagnostics.nc within the glacier directory - which contains 
    the following coordinates: 
    dis_along_flowline (dis_along_flowline): float64, along-flowline distance in m
    time (time): float64, model time in years
    and the following data variables:
    volume_m3 (time, dis_along_flowline): float64
    area_m2(time, dis_along_flowline): float64
    thickness_m (time, dis_along_flowline): float64
    """
    yr0,yr1 = pygem_simpath.split('_')[-3:-1]
    pygem_ds = xr.open_dataset(pygem_simpath).sel(year=slice(yr0, yr1))
    time = pygem_ds.coords['year'].values.flatten().astype(float)
    distance_along_flowline = pygem_ds['bin_distance'].values.flatten().astype(float)
    area = pygem_ds['bin_area_annual'].values[0].astype(float).T
    thick = pygem_ds['bin_thick_annual'].values[0].astype(float).T
    vol = area * thick

    diag_ds = xr.Dataset()
    diag_ds.coords['time'] = time
    diag_ds.coords['dis_along_flowline'] = distance_along_flowline
    diag_ds['area_m2'] = (('time', 'dis_along_flowline'), area)
    diag_ds['area_m2'].attrs['description'] = 'Section area'
    diag_ds['area_m2'].attrs['unit'] = 'm 2'    
    diag_ds['thickness_m'] = (('time', 'dis_along_flowline'), thick * np.nan)
    diag_ds['thickness_m'].attrs['description'] = 'Section thickness'
    diag_ds['thickness_m'].attrs['unit'] = 'm'
    diag_ds['volume_m3'] = (('time', 'dis_along_flowline'), vol)
    diag_ds['volume_m3'].attrs['description'] = 'Section volume'
    diag_ds['volume_m3'].attrs['unit'] = 'm 3'
    # diag_ds.to_netcdf(oggm_diag, 'w', group='fl_0')
    if debug:
        # plot volume
        vol = diag_ds.sum(dim=['dis_along_flowline'])['volume_m3']
        f,ax = plt.subplots(1,figsize=(5,5))
        (vol/vol[0]).plot(ax=ax)
        plt.show()

    return diag_ds


def plot_distributed_thickness(ds):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
    vmax = round(np.nanmax(ds.simulated_thickness.sel(time=ds.coords['time'].values[0]))/25) * 25
    ds.simulated_thickness.sel(time=ds.coords['time'].values[0]).plot(ax=ax1, vmin=0, vmax=vmax,add_colorbar=False)
    ds.simulated_thickness.sel(time=ds.coords['time'].values[-1]).plot(ax=ax2, vmin=0, vmax=vmax)
    ax1.axis('equal'); ax2.axis('equal')
    plt.tight_layout()
    plt.show()


def run(simpath, debug=False):

    if os.path.isfile(simpath):
        pygem_path, pygem_fn = os.path.split(simpath)
        pygem_fn_split = pygem_fn.split('_')
        f_suffix = '_'.join(pygem_fn_split[1:])[:-3]
        glac_no = pygem_fn_split[0]
        glacier_rgi_table = modelsetup.selectglaciersrgitable(glac_no=[glac_no]).loc[0, :]
        glacier_str = '{0:0.5f}'.format(glacier_rgi_table['RGIId_float'])
        # ===== Load glacier data: area (km2), ice thickness (m), width (km) =====        
        try:
            if not glacier_rgi_table['TermType'] in [1,5] or not pygem_prms['setup']['include_calving']:
                gdir = single_flowline_glacier_directory(glacier_str)
                gdir.is_tidewater = False
            else:
                # set reset=True to overwrite non-calving directory that may already exist
                gdir = single_flowline_glacier_directory_with_calving(glacier_str)
                gdir.is_tidewater = True
        except Exception as err:
            print(err)

        # create OGGM formatted flowline diagnostic dataset from PyGEM simulation
        pygem_fl_diag = pygem_to_oggm(os.path.join(pygem_path,pygem_fn),debug=debug)

        ###
        ### OGGM preprocessing steps before redistributing ice thickness form simulation
        ###
        # This is to add a new topography to the file (smoothed differently)
        workflow.execute_entity_task(distribute_2d.add_smoothed_glacier_topo, gdir)
        # This is to get the bed map at the start of the simulation
        workflow.execute_entity_task(tasks.distribute_thickness_per_altitude, gdir)
        # This is to prepare the glacier directory for the interpolation (needs to be done only once)
        workflow.execute_entity_task(distribute_2d.assign_points_to_band, gdir)
        ###
        # distribute simulation to 2d
        ds = workflow.execute_entity_task(
        distribute_2d.distribute_thickness_from_simulation,
        gdir, 
        fl_diag=pygem_fl_diag,
        concat_input_filesuffix='_spinup_historical',  # concatenate with the historical spinup
        output_filesuffix=f'_pygem_{f_suffix}',  # filesuffix added to the output filename gridded_simulation.nc, if empty input_filesuffix is used
        )[0]
        print('2D simulated ice thickness created: ', gdir.get_filepath('gridded_simulation',filesuffix=f'_pygem_{f_suffix}'))
        if debug:
            plot_distributed_thickness(ds)

    return


def main():
    time_start = time.time()
    args = getparser().parse_args()

    # number of cores for parallel processing
    if args.ncores > 1:
        ncores = int(np.min([len(args.simpath), args.ncores]))
    else:
        ncores = 1

    # set up partial function with debug argument
    run_with_debug = partial(run, debug=args.debug)
    # parallel processing
    print('Processing with ' + str(ncores) + ' cores...')
    with multiprocessing.Pool(ncores) as p:
        p.map(run_with_debug, args.simpath)

    print('Total processing time:', time.time()-time_start, 's')
    
if __name__ == "__main__":
    main()