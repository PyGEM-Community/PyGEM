"""
Python Glacier Evolution Model (PyGEM)

copyright © 2018 David Rounce <drounce@cmu.edu>

Distrubted under the MIT lisence

Fetch filled Hugonnet reference mass balance data
"""
# Built-in libraries
import argparse
import os
# External libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from scipy.stats import median_abs_deviation
# oggm
from oggm import utils
# pygem imports
from pygem.setup.config import ConfigManager
# instantiate ConfigManager
config_manager = ConfigManager()
# read the config
pygem_prms = config_manager.read_config()
import pygem.pygem_modelsetup as modelsetup


def run(fp='', debug=False, overwrite=False):
    """
    pull geodetic mass balance data from OGGM
    The original 'raw' were acquired and combined from https://doi.org/10.6096/13 (time series/dh_<RGI_REGION01>_rgi60_pergla_rates)
    The combined global data have been modified in three ways (code):
    1. the glaciers in RGI region 12 (Caucasus) had to be manually linked to the product by Hugonnet because of large errors in the RGI outlines. The resulting product used by OGGM in region 12 has large uncertainties.
    2. outliers have been filtered as following: all glaciers with an error estimate larger than 3 at the RGI region level are filtered out
    3. all missing data (including outliers) are attributed with the regional average.

    See https://docs.oggm.org/en/latest/reference-mass-balance-data.html and https://nbviewer.org/urls/cluster.klima.uni-bremen.de/~oggm/geodetic_ref_mb/convert.ipynb for further information.

    dmdtda represents the average climatic mass balance (in units meters water-equivalent per year) over a given period \\frac{\\partial mass}{\\partial time \\partial area}
    """
    mbdf = utils.get_geodetic_mb_dataframe()
    if debug:
        print('MB data loaded from OGGM:')
        print(mbdf.head())

    # pull only 2000-2020 period
    mbdf_subset = mbdf[mbdf.period=='2000-01-01_2020-01-01']

    # reset the index
    mbdf_subset = mbdf_subset.reset_index()

    # sort by the rgiid column
    mbdf_subset = mbdf_subset.sort_values(by='rgiid')

    # rename some keys to work with what other scripts/functions expect
    mbdf_subset= mbdf_subset.rename(columns={'dmdtda':'mb_mwea',
                                            'err_dmdtda':'mb_mwea_err'})

    if fp[-4:] != '.csv':
        fp += '.csv'

    if os.path.isfile(fp) and not overwrite:
        raise FileExistsError(f'The filled global geodetic mass balance file already exists, pass `-o` to overwrite, or pass a different file name: {fp}')
    
    mbdf_subset.to_csv(fp, index=False)
    if debug:
        print(f'Filled global geodetic mass balance data saved to: {fp}')
        print(mbdf_subset.head())


def main():
    parser = argparse.ArgumentParser(description="grab filled Hugonnet et al. 2021 geodetic mass balance data from OGGM and converts to a format PyGEM utilizes")
    # add arguments
    parser.add_argument('-fname', action='store', type=str, default=f"{pygem_prms['calib']['data']['massbalance']['hugonnet2021_fn']}",
                        help='Reference mass balance data file name (default: df_pergla_global_20yr-filled.csv)')
    parser.add_argument('-o', '--overwrite', action='store_true',
                        help='Flag to overwrite existing geodetic mass balance data')
    parser.add_argument('-v', '--debug', action='store_true',
                        help='Flag for debugging')
    args = parser.parse_args()

    # hugonnet filepath
    fp = f"{pygem_prms['root']}/{pygem_prms['calib']['data']['massbalance']['hugonnet2021_relpath']}/{args.fname}"

    run(fp, args.debug, args.overwrite)


if __name__ == "__main__":
    main()