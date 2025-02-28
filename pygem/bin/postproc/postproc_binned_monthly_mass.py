"""
Python Glacier Evolution Model (PyGEM)

copyright © 2024 Brandon Tober <btober@cmu.edu> David Rounce <drounce@cmu.edu>

Distrubted under the MIT lisence

derive binned monthly ice thickness and mass from PyGEM simulation
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
# External libraries
import numpy as np
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
    parser = argparse.ArgumentParser(description="process monthly ice thickness for PyGEM simulation")
    # add arguments
    parser.add_argument('-simpath', action='store', type=str, nargs='+', default=None,
                        help='path to PyGEM binned simulation (can take multiple)')
    parser.add_argument('-binned_simdir', action='store', type=str, default=None,
                        help='directory with binned simulations for which to process monthly thickness')
    parser.add_argument('-ncores', action='store', type=int, default=1,
                        help='number of simultaneous processes (cores) to use')

    return parser


def get_binned_monthly(dotb_monthly, m_annual, h_annual):
    """
    funciton to calculate the monthly binned ice thickness and mass
    from annual climatic mass balance and annual ice thickness products

    to determine monthlyt thickness and mass, we must account for flux divergence
    this is not so straight-forward, as PyGEM accounts for ice dynamics at the 
    end of each model year and not on a monthly timestep.
    here, monthly thickness and mass is determined assuming 
    the flux divergence is constant throughout the year.
    
    annual flux divergence is first estimated by combining the annual binned change in ice 
    thickness and the annual binned mass balance. then, assume flux divergence is constant 
    throughout the year (divide annual by 12 to get monthly flux divergence).

    monthly binned flux divergence can then be combined with 
    monthly binned climatic mass balance to get monthly binned change in ice thickness

    
    Parameters
    ----------
    dotb_monthly : float
        ndarray containing the climatic mass balance for each model month computed by PyGEM
        shape : [#glac, #elevbins, #months]
    m_annual : float
        ndarray containing the average (or median) binned ice mass computed by PyGEM
        shape : [#glac, #elevbins, #years]
    h_annual : float
        ndarray containing the average (or median) binned ice thickness at computed by PyGEM
        shape : [#glac, #elevbins, #years]

    Returns
    -------
    m_monthly: float
        ndarray containing the binned monthly ice mass
        shape : [#glac, #elevbins, #years]
    h_monthly: float
        ndarray containing the binned monthly ice thickness
        shape : [#glac, #elevbins, #years]
    """
    ### get monthly ice thickness ###
    # convert mass balance from m w.e. yr^-1 to m ice yr^-1
    dotb_monthly = dotb_monthly * (pygem_prms['constants']['density_water'] / pygem_prms['constants']['density_ice'])
    assert dotb_monthly.shape[2] % 12 == 0, "Number of months is not a multiple of 12!"

    # obtain annual mass balance rate, sum monthly for each year
    dotb_annual = dotb_monthly.reshape(dotb_monthly.shape[0], dotb_monthly.shape[1], -1, 12).sum(axis=-1)  # climatic mass balance [m ice a^-1]

    # compute the thickness change per year
    delta_h_annual = np.diff(h_annual, axis=-1)  # [m ice a^-1] (nbins, nyears-1)

    # compute flux divergence for each bin
    flux_div_annual = dotb_annual - delta_h_annual  # [m ice a^-1]

    ### to get monthly thickness and mass we require monthly flux divergence ###
    # we'll assume the flux divergence is constant througohut the year (is this a good assumption?)
    # ie. take annual values and divide by 12 - use numpy repeat to repeat values across 12 months
    flux_div_monthly = np.repeat(flux_div_annual / 12, 12, axis=-1)

    # get monthly binned change in thickness
    delta_h_monthly = dotb_monthly - flux_div_monthly # [m ice per month]

    # get binned monthly thickness = running thickness change + initial thickness
    running_delta_h_monthly = np.cumsum(delta_h_monthly, axis=-1)
    h_monthly =  running_delta_h_monthly + h_annual[:,:,0][:,:,np.newaxis] 

    # convert to mass per unit area
    m_spec_monthly = h_monthly * pygem_prms['constants']['density_ice']
    
    ### get monthly mass ###
    # note, binned monthly thickness and mass is currently per unit area
    # obtaining binned monthly mass requires knowledge of binned glacier area
    # we do not have monthly binned area (as glacier dynamics are performed on an annual timestep in PyGEM),
    # so we'll resort to using the annual binned glacier mass and thickness in order to get to binned glacier area
    ########################
    # first convert m_annual to bin_voluma_annual
    v_annual = m_annual / pygem_prms['constants']['density_ice']
    # now get area: use numpy divide where denominator is greater than 0 to avoid divide error
    # note, indexing of [:,:,1:] so that annual area array has same shape as flux_div_annual
    a_annual = np.divide(
            v_annual[:,:,1:], 
            h_annual[:,:,1:], 
            out=np.full(h_annual[:,:,1:].shape, np.nan), 
            where=h_annual[:,:,1:]>0)

    # tile to get monthly area, assuming area is constant thoughout the year
    a_monthly = np.tile(a_annual, 12)

    # combine monthly thickess and area to get mass
    m_monthly = m_spec_monthly * a_monthly

    return h_monthly, m_spec_monthly, m_monthly


def update_xrdataset(input_ds, h_monthly, m_spec_monthly, m_monthly):
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
    output_coords_dict['bin_thick_monthly'] = (
            collections.OrderedDict([('glac', glac_values), ('bin',bin_values), ('time', time_values)]))
    output_coords_dict['bin_mass_spec_monthly'] = (
            collections.OrderedDict([('glac', glac_values), ('bin',bin_values), ('time', time_values)]))
    output_coords_dict['bin_mass_monthly'] = (
            collections.OrderedDict([('glac', glac_values), ('bin',bin_values), ('time', time_values)]))

    # Attributes dictionary
    output_attrs_dict = {}
    output_attrs_dict['bin_thick_monthly'] = {
            'long_name': 'binned monthly ice thickness',
            'units': 'm',
            'temporal_resolution': 'monthly',
            'comment': 'monthly ice thickness binned by surface elevation (assuming constant flux divergence throughout a given year)'}
    output_attrs_dict['bin_mass_spec_monthly'] = {
            'long_name': 'binned monthly specific ice mass',
            'units': 'kg m^-2',
            'temporal_resolution': 'monthly',
            'comment': 'monthly ice mass per unit area binned by surface elevation (assuming constant flux divergence throughout a given year)'}
    output_attrs_dict['bin_mass_monthly'] = {
            'long_name': 'binned monthly ice mass',
            'units': 'kg',
            'temporal_resolution': 'monthly',
            'comment': 'monthly ice mass binned by surface elevation (assuming constant flux divergence and area throughout a given year)'}

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

    output_ds_all['bin_thick_monthly'].values = (
            h_monthly
            )
    output_ds_all['bin_mass_spec_monthly'].values = (
            m_spec_monthly 
            )
    output_ds_all['bin_mass_monthly'].values = (
            m_monthly
            )

    return output_ds_all, encoding


def run(simpath):
    """
    create binned monthly mass change data product
    Parameters
    ----------
    list_packed_vars : list
        list of packed variables that enable the use of parallels
    Returns
    -------
    binned_ds : netcdf Dataset
        updated binned netcdf containing binned monthly ice thickness and mass
    """

    if os.path.isfile(simpath):
        # open dataset
        binned_ds = xr.open_dataset(simpath)

        # calculate monthly thickness and mass
        h_monthly, m_spec_monthly, m_monthly = get_binned_monthly(
                                                    binned_ds.bin_massbalclim_monthly.values, 
                                                    binned_ds.bin_mass_annual.values,
                                                    binned_ds.bin_thick_annual.values
                                                    )

        # update dataset to add monthly mass change
        output_ds_binned, encoding_binned = update_xrdataset(binned_ds, h_monthly, m_spec_monthly, m_monthly)

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
    
    elif args.binned_simdir:
        # get list of sims
        simpath = glob.glob(args.binned_simdir+'*.nc')
    if simpath:
        # number of cores for parallel processing
        if args.ncores > 1:
            ncores = int(np.min([len(simpath), args.ncores]))
        else:
            ncores = 1

        # Parallel processing
        print('Processing with ' + str(ncores) + ' cores...')
        with multiprocessing.Pool(ncores) as p:
            p.map(run,simpath)

    print('Total processing time:', time.time()-time_start, 's')
    
if __name__ == "__main__":
    main()