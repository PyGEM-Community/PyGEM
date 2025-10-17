"""
Python Glacier Evolution Model (PyGEM)

copyright © 2024 Brandon Tober <btober@cmu.edu> David Rounce <drounce@cmu.edu>

Distributed under the MIT license

derive binned subannual ice thickness and mass from PyGEM simulation
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

# External libraries
import numpy as np
import xarray as xr
import pandas as pd

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
    parser = argparse.ArgumentParser(description='process binned subannual ice thickness for PyGEM simulation')
    # add arguments
    parser.add_argument(
        '-simpath',
        action='store',
        type=str,
        nargs='+',
        default=None,
        help='path to PyGEM binned simulation (can take multiple)',
    )
    parser.add_argument(
        '-simdir',
        action='store',
        type=str,
        default=None,
        help='directory with binned simulations for which to process subannual thickness',
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


def get_binned_subannual(
    bin_massbalclim,
    bin_mass_annual,
    bin_thick_annual,
    dates_subannual,
    dates_annual,
    debug=False
    ):
    """
    funciton to calculate the subannual binned ice thickness and mass
    from subannual climatic mass balance and annual mass and ice thickness products.

    to determine subannual thickness and mass, we must account for flux divergence.
    this is not so straight-forward, as PyGEM accounts for ice dynamics at the
    end of each model year and not on a subannual timestep.
    thus, subannual thickness and mass is determined assuming
    the flux divergence is constant throughout the year.

    annual flux divergence is first estimated by combining the annual binned change in ice
    thickness and the annual binned mass balance. then, assume flux divergence is constant
    throughout the year (divide annual by the number of steps in the binned climatic mass 
    balance to get subannual flux divergence).

    subannual binned flux divergence can then be combined with
    subannual binned climatic mass balance to get subannual binned change in ice thickness and mass.


    Parameters
    ----------
    bin_massbalclim : ndarray
        climatic mass balance [m w.e. yr^-1] with subannual timesteps (monthly/daily)
        shape : [#glac, #elevbins, #steps]
    bin_mass_annual : ndarray
        annual binned ice mass computed by PyGEM [kg]
        shape : [#glac, #elevbins, #years]
    bin_thick_annual : ndarray
       annual binned glacier thickness [m ice]
        shape : [#glac, #elevbins, #years]
    dates_subannual : array-like of datetime-like
        dates associated with `bin_massbalclim` (subannual)
    dates_annual : array-like of datetime-like
        dates associated with `bin_thick_annual` and `bin_mass_annual` (annual, values correspond to start of the year)


    Returns
    -------
    h_subannual : ndarray
        subannual binned ice thickness [m ice]
        shape : [#glac, #elevbins, #steps]
    m_spec_subannual : ndarray
        subannual binned specific ice mass [kg m^-2]
        shape : [#glac, #elevbins, #steps]
    m_subannual : ndarray
        subannual binned glacier mass [kg]
        shape : [#glac, #elevbins, #steps]
    """

    n_glac, n_bins, n_steps = bin_massbalclim.shape
    dates_subannual = pd.to_datetime(dates_subannual)
    dates_annual = pd.to_datetime(dates_annual)
    years_annual = np.array([d.year for d in dates_annual])
    years_subannual = np.array([d.year for d in dates_subannual])
    yrs = np.unique(years_subannual)
    nyrs = len(yrs)
    assert nyrs > 1, "Need at least two annual steps for flux divergence estimation"

    #  --- Step 1: convert mass balance from m w.e. yr^-1 to m ice yr^-1 ---
    rho_w = pygem_prms['constants']['density_water']
    rho_i = pygem_prms['constants']['density_ice']
    bin_massbalclim_ice = bin_massbalclim * (rho_w / rho_i)

    # --- Step 2: compute annual cumulative mass balance ---
    # Initialize annual cumulative mass balance (exclude last year for flux calculation)
    bin_massbalclim_annual = np.zeros((n_glac, n_bins, nyrs))
    for i, year in enumerate(yrs):
        idx = np.where(years_subannual == year)[0]
        bin_massbalclim_annual[:, :, i] = bin_massbalclim_ice[:, :, idx].sum(axis=-1)

    # --- Step 3: compute annual thickness change ---
    bin_delta_thick_annual = np.diff(bin_thick_annual, axis=-1)  # [m ice yr^-1]

    # --- Step 4: compute annual flux divergence ---
    bin_flux_divergence_annual = bin_massbalclim_annual - bin_delta_thick_annual  # [m ice yr^-1]

    # --- Step 5: expand flux divergence to subannual steps ---
    bin_flux_divergence_subannual = np.zeros_like(bin_massbalclim_ice)
    for i, year in enumerate(yrs):
        idx = np.where(years_subannual == year)[0]
        bin_flux_divergence_subannual[:, :, idx] = bin_flux_divergence_annual[:, :, i][:, :, np.newaxis] / len(idx)

    # --- Step 6: compute subannual thickness change ---
    bin_delta_thick_subannual = bin_massbalclim_ice - bin_flux_divergence_subannual

    # --- Step 7: integrate cumulative thickness ---
    running_bin_delta_thick_annual = np.cumsum(bin_delta_thick_subannual, axis=-1)
    bin_thick_subannual = running_bin_delta_thick_annual + bin_thick_annual[:, :, 0][:, :, np.newaxis]

    # --- Step 8: Compute glacier volume and area on subannual timestep ---
    bin_volume_annual = bin_mass_annual / rho_i  # annual volume [m^3] per bin
    bin_area_annual = np.divide(
        bin_volume_annual[:, :, 1:],  # exclude first year to match flux_div_annual
        bin_thick_annual[:, :, 1:],
        out=np.full(bin_thick_annual[:, :, 1:].shape, np.nan),
        where=bin_thick_annual[:, :, 1:] > 0,
    )

    # --- Step 9 : compute subannual glacier mass ---
    # First expand area to subannual steps
    bin_area_subannual = np.full(bin_massbalclim_ice.shape, np.nan)
    for i, year in enumerate(yrs):
        idx = np.where(years_subannual == year)[0]
        bin_area_subannual[:, :, idx] = bin_area_annual[:, :, i][:, :, np.newaxis]

    # multiply by ice density to get subannual mass
    bin_mass_subannual = bin_thick_subannual * rho_i * bin_area_subannual

    # --- Step 10: debug check ---
    if debug:
        for i, year in enumerate(yrs):
            # get last subannual index of that year
            idx = np.where(years_subannual == year)[0][-1]
            diff = bin_thick_subannual[:, :, idx] - bin_thick_annual[:, :, i + 1]
            print(f"Year {year}, subannual idx: {idx}")
            print("Max diff:", np.max(np.abs(diff)))
            print("Min diff:", np.min(np.abs(diff)))
            print("Mean diff:", np.mean(diff))
            print()
            # optional assertion
            np.testing.assert_allclose(
                bin_thick_subannual[:, :, idx],
                bin_thick_annual[:, :, i + 1],
                rtol=1e-6,
                atol=1e-12,
                err_msg=f"Mismatch in thickness for year {year}"
            )
    
    return bin_thick_subannual, bin_mass_subannual


def update_xrdataset(input_ds, bin_thick, bin_mass, timestep):
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
    bin_values = input_ds.bin.values

    output_coords_dict = collections.OrderedDict()
    output_coords_dict['bin_thick'] = collections.OrderedDict(
        [('glac', glac_values), ('bin', bin_values), ('time', time_values)]
    )
    output_coords_dict['bin_mass'] = collections.OrderedDict(
        [('glac', glac_values), ('bin', bin_values), ('time', time_values)]
    )

    # Attributes dictionary
    output_attrs_dict = {}
    output_attrs_dict['bin_thick'] = {
        'long_name': 'binned ice thickness',
        'units': 'm',
        'temporal_resolution': timestep,
        'comment': 'subannual ice thickness binned by surface elevation (assuming constant flux divergence throughout a given year)',
    }
    output_attrs_dict['bin_mass'] = {
        'long_name': 'binned ice mass',
        'units': 'kg',
        'temporal_resolution': timestep,
        'comment': 'subannual ice mass binned by surface elevation (assuming constant flux divergence and area throughout a given year)',
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

    output_ds_all['bin_thick'].values = bin_thick
    output_ds_all['bin_mass'].values = bin_mass

    return output_ds_all, encoding


def run(simpath, debug):
    """
    create binned subannual mass change data product
    Parameters
    ----------
    list_packed_vars : list
        list of packed variables that enable the use of parallels
    Returns
    -------
    binned_ds : netcdf Dataset
        updated binned netcdf containing binned subannual ice thickness and mass
    """

    if os.path.isfile(simpath):
        # open dataset
        binned_ds = xr.open_dataset(simpath)

        # get model time tables
        timestep = json.loads(binned_ds.attrs['model_parameters'])['timestep']
        # get model dates
        dates_annual = pd.to_datetime([f'{y}-01-01' for y in binned_ds.year.values])
        dates_subannual = pd.to_datetime([t.strftime('%Y-%m-%d') for t in binned_ds.time.values])

        # calculate subannual thickness and mass
        bin_thick, bin_mass = get_binned_subannual(
            bin_massbalclim=binned_ds.bin_massbalclim.values,
            bin_mass_annual=binned_ds.bin_mass_annual.values,
            bin_thick_annual=binned_ds.bin_thick_annual.values,
            dates_subannual=dates_subannual,
            dates_annual=dates_annual,
            debug=debug
        )

        # update dataset to add subannual binned thickness and mass
        output_ds_binned, encoding_binned = update_xrdataset(binned_ds, bin_thick=bin_thick, bin_mass=bin_mass, timestep=timestep)

        # close input ds before write
        binned_ds.close()
        
        # append to existing binned netcdf
        output_ds_binned.to_netcdf(simpath, mode='a', encoding=encoding_binned, engine='netcdf4')
        
        # close datasets
        output_ds_binned.close()
    
    return


def main():
    time_start = time.time()
    args = getparser().parse_args()

    if args.simpath:
        # filter out non-file paths
        simpath = [p for p in args.simpath if os.path.isfile(p)]

    elif args.simdir:
        # get list of sims
        simpath = sorted(glob.glob(args.simdir + '/*.nc'))
    if simpath:
        print(simpath)
        # number of cores for parallel processing
        if args.ncores > 1:
            ncores = int(np.min([len(simpath), args.ncores]))
        else:
            ncores = 1

        # set up partial function with debug argument
        run_partial = partial(run, debug=args.debug)

        # Parallel processing
        print('Processing with ' + str(ncores) + ' cores...')
        with multiprocessing.Pool(ncores) as p:
            p.map(run_partial, simpath)

    print('Total processing time:', time.time() - time_start, 's')


if __name__ == '__main__':
    main()
