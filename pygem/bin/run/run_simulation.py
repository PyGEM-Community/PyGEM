"""
Python Glacier Evolution Model (PyGEM)

copyright © 2018 David Rounce <drounce@cmu.edu>

Distributed under the MIT license

Run a model simulation
"""
# Default climate data is ERA5; specify CMIP5 by specifying a filename to the argument:
#    (Command line) python run_simulation_list_multiprocess.py -gcm_list_fn=C:\...\gcm_rcpXX_filenames.txt
#      - Default is running ERA5 in parallel with five processors.
#    (Spyder) %run run_simulation_list_multiprocess.py C:\...\gcm_rcpXX_filenames.txt -option_parallels=0
#      - Spyder cannot run parallels, so always set -option_parallels=0 when testing in Spyder.
# Spyder cannot run parallels, so always set -option_parallels=0 when testing in Spyder.

# Built-in libraries
import argparse
import copy
import json
import multiprocessing
import os
import sys
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np

# External libraries
import pandas as pd
import xarray as xr
from scipy.stats import median_abs_deviation

# pygem imports
from pygem.setup.config import ConfigManager

# instantiate ConfigManager
config_manager = ConfigManager()
# read the config
pygem_prms = config_manager.read_config()
# oggm imports
from oggm import cfg, tasks, utils
from oggm.core.flowline import FluxBasedModel, SemiImplicitModel
from oggm.core.massbalance import apparent_mb_from_any_mb

import pygem.gcmbiasadj as gcmbiasadj
import pygem.pygem_modelsetup as modelsetup
from pygem import class_climate, output
from pygem.glacierdynamics import MassRedistributionCurveModel
from pygem.massbalance import PyGEMMassBalance
from pygem.oggm_compat import (
    single_flowline_glacier_directory,
    single_flowline_glacier_directory_with_calving,
)
from pygem.output import calc_stats_array
from pygem.plot import graphics
from pygem.shop import debris

cfg.PARAMS['hydro_month_nh'] = 1
cfg.PARAMS['hydro_month_sh'] = 1
cfg.PARAMS['trapezoid_lambdas'] = 1


# ----- FUNCTIONS -----
def none_or_value(value):
    """Custom type function to handle 'none' or 'null' as None."""
    if value.lower() in {'none', 'null'}:
        return None
    return value


def getparser():
    """
    Use argparse to add arguments from the command line

    Parameters
    ----------
    gcm_list_fn (optional) : str
        text file that contains the climate data to be used in the model simulation
    sim_climate_name (optional) : str
        gcm name
    sim_climate_scenario (optional) : str
        representative concentration pathway or shared socioeconomic pathway (ex. 'rcp26', 'ssp585')
    realization (optional) : str
        single realization from large ensemble (ex. '1011.001', '1301.020')
        see CESM2 Large Ensemble Community Project by NCAR for more information
    realization_list (optional) : str
        text file that contains the realizations to be used in the model simulation
    ncores (optional) : int
        number of cores to use in parallels
    rgi_glac_number_fn (optional) : str
        filepath of .json file containing a list of glacier numbers that used to run batches on the supercomputer
    batch_number (optional): int
        batch number used to differentiate output on supercomputer
    option_ordered : int
        option to keep glaciers ordered or to grab every n value for the batch
        (the latter helps make sure run times on each core are similar as it removes any timing differences caused by
         regional variations)
    debug (optional) : int
        Switch for turning debug printing on or off (default = 0 (off))

    Returns
    -------
    Object containing arguments and their respective values.
    """
    parser = argparse.ArgumentParser(description='Run PyGEM simulation')
    # add arguments
    parser.add_argument(
        '-rgi_region01',
        type=int,
        default=pygem_prms['setup']['rgi_region01'],
        help='Randoph Glacier Inventory region (can take multiple, e.g. `-run_region01 1 2 3`)',
        nargs='+',
    )
    parser.add_argument(
        '-rgi_region02',
        type=str,
        default=pygem_prms['setup']['rgi_region02'],
        nargs='+',
        help='Randoph Glacier Inventory subregion (either `all` or multiple spaced integers,  e.g. `-run_region02 1 2 3`)',
    )
    parser.add_argument(
        '-rgi_glac_number',
        action='store',
        type=float,
        default=pygem_prms['setup']['glac_no'],
        nargs='+',
        help='Randoph Glacier Inventory glacier number (can take multiple)',
    )
    parser.add_argument(
        '-ref_climate_name',
        action='store',
        type=str,
        default=pygem_prms['climate']['ref_climate_name'],
        help='reference gcm name',
    )
    parser.add_argument(
        '-ref_startyear',
        action='store',
        type=int,
        default=pygem_prms['climate']['ref_startyear'],
        help='reference period starting year for calibration (typically 2000)',
    )
    parser.add_argument(
        '-ref_endyear',
        action='store',
        type=int,
        default=pygem_prms['climate']['ref_endyear'],
        help='reference period ending year for calibration (typically 2019)',
    )
    parser.add_argument(
        '-rgi_glac_number_fn',
        action='store',
        type=str,
        default=None,
        help='filepath containing list of rgi_glac_number, helpful for running batches on spc',
    )
    parser.add_argument(
        '-gcm_list_fn',
        action='store',
        type=str,
        default=pygem_prms['climate']['ref_climate_name'],
        help='text file full of commands to run (ex. CanESM2 or CESM2)',
    )
    parser.add_argument(
        '-sim_climate_name',
        action='store',
        type=str,
        default=pygem_prms['climate']['sim_climate_name'],
        help='GCM name used for model run',
    )
    parser.add_argument(
        '-sim_climate_scenario',
        action='store',
        type=none_or_value,
        default=pygem_prms['climate']['sim_climate_scenario'],
        help='rcp or ssp sim_climate_scenario used for model run (ex. rcp26 or ssp585)',
    )
    parser.add_argument(
        '-realization',
        action='store',
        type=str,
        default=None,
        help='realization from large ensemble used for model run (ex. 1011.001 or 1301.020)',
    )
    parser.add_argument(
        '-realization_list',
        action='store',
        type=str,
        default=None,
        help='text file full of realizations to run',
    )
    parser.add_argument(
        '-sim_startyear',
        action='store',
        type=int,
        default=pygem_prms['climate']['sim_startyear'],
        help='start year for the model run',
    )
    parser.add_argument(
        '-sim_endyear',
        action='store',
        type=int,
        default=pygem_prms['climate']['sim_endyear'],
        help='start year for the model run',
    )
    parser.add_argument(
        '-mcmc_burn_pct',
        action='store',
        type=int,
        default=0,
        help='percent of MCMC chain to burn off from beginning (defaults to 0, assuming that burn in was performed in calibration)',
    )
    parser.add_argument(
        '-ncores',
        action='store',
        type=int,
        default=1,
        help='number of simultaneous processes (cores) to use',
    )
    parser.add_argument(
        '-batch_number',
        action='store',
        type=int,
        default=None,
        help='Batch number used to differentiate output on supercomputer',
    )
    parser.add_argument(
        '-kp',
        action='store',
        type=float,
        default=pygem_prms['sim']['params']['kp'],
        help='Precipitation bias',
    )
    parser.add_argument(
        '-tbias',
        action='store',
        type=float,
        default=pygem_prms['sim']['params']['tbias'],
        help='Temperature bias',
    )
    parser.add_argument(
        '-ddfsnow',
        action='store',
        type=float,
        default=pygem_prms['sim']['params']['ddfsnow'],
        help='Degree-day factor of snow',
    )
    parser.add_argument(
        '-oggm_working_dir',
        action='store',
        type=str,
        default=f'{pygem_prms["root"]}/{pygem_prms["oggm"]["oggm_gdir_relpath"]}',
        help='Specify OGGM working dir - useful if performing a grid search and have duplicated glacier directories',
    )
    parser.add_argument(
        '-option_calibration',
        action='store',
        type=none_or_value,
        default=pygem_prms['calib']['option_calibration'],
        help='calibration option ("emulator", "MCMC", "HH2015", "HH2015mod", "None")',
    )
    parser.add_argument(
        '-option_dynamics',
        action='store',
        type=none_or_value,
        default=pygem_prms['sim']['option_dynamics'],
        help='glacier dynamics scheme (options: ``OGGM`, `MassRedistributionCurves`, `None`)',
    )
    parser.add_argument(
        '-use_regional_glen_a',
        action='store',
        type=bool,
        default=pygem_prms['sim']['oggm_dynamics']['use_regional_glen_a'],
        help='Take the glacier flow parameterization from regionally calibrated priors (boolean: `0` or `1`, `True` or `False`)',
    )
    parser.add_argument(
        '-option_bias_adjustment',
        action='store',
        type=int,
        default=pygem_prms['sim']['option_bias_adjustment'],
        help='Bias adjustment option (options: `0`, `1`, `2`, `3`. 0: no adjustment, \
                                    1: new prec scheme and temp building on HH2015, \
                                    2: HH2015 methods, 3: quantile delta mapping)',
    )
    parser.add_argument(
        '-nsims',
        action='store',
        type=int,
        default=pygem_prms['sim']['nsims'],
        help='number of simulations (note, defaults to 1 if `option_calibration` != `MCMC`)',
    )
    parser.add_argument(
        '-modelprms_fp',
        action='store',
        type=str,
        default=None,
        help='model parameters filepath',
    )
    parser.add_argument(
        '-outputfn_sfix',
        action='store',
        type=str,
        default='',
        help='append custom filename suffix to simulation output',
    )
    # flags
    parser.add_argument(
        '-export_all_simiters',
        action='store_true',
        help='Flag to export data from all simulations',
        default=pygem_prms['sim']['out']['export_all_simiters'],
    )
    parser.add_argument(
        '-export_extra_vars',
        action='store_true',
        help='Flag to export extra variables (temp, prec, melt, acc, etc.)',
        default=pygem_prms['sim']['out']['export_extra_vars'],
    )
    parser.add_argument(
        '-export_binned_data',
        action='store_true',
        help='Flag to export binned data',
        default=pygem_prms['sim']['out']['export_binned_data'],
    )
    parser.add_argument(
        '-export_binned_components',
        action='store_true',
        help='Flag to export binned mass balance components (melt, accumulation, refreeze)',
        default=pygem_prms['sim']['out']['export_binned_components'],
    )
    parser.add_argument(
        '-option_ordered',
        action='store_true',
        help='Flag to keep glacier lists ordered (default is off)',
    )
    parser.add_argument(
        '-spinup',
        action='store_true',
        default=False,
        help='Flag to perform dynamical spinup before calibration',
    )
    parser.add_argument('-v', '--debug', action='store_true', help='Flag for debugging')

    return parser


def run(list_packed_vars):
    """
    Model simulation
    Parameters
    ----------
    list_packed_vars : list
        list of packed variables that enable the use of parallels

    Returns
    -------
    netcdf files of the simulation output (specific output is dependent on the output option)
    """
    # Unpack variables
    parser = getparser()
    args = parser.parse_args()
    count = list_packed_vars[0]
    glac_no = list_packed_vars[1]
    sim_climate_name = list_packed_vars[2]
    realization = list_packed_vars[3]
    if (sim_climate_name != args.ref_climate_name) and (args.sim_climate_scenario is None):
        sim_climate_scenario = os.path.basename(args.gcm_list_fn).split('_')[1]
    else:
        sim_climate_scenario = args.sim_climate_scenario
    debug = args.debug
    if debug:
        print(f'sim_climate_scenario:{sim_climate_scenario}')

    # ===== LOAD GLACIERS =====
    main_glac_rgi = modelsetup.selectglaciersrgitable(glac_no=glac_no)

    # ===== TIME PERIOD =====
    # Reference Calibration Period
    # adjust end year in event that gcm_end year precedes ref_startyear
    ref_endyear = min([args.ref_endyear, args.sim_endyear])
    dates_table_ref = modelsetup.datesmodelrun(
        startyear=args.ref_startyear,
        endyear=ref_endyear,
        option_wateryear=pygem_prms['climate']['ref_wateryear'],
    )

    # GCM Full Period (includes reference and simulation periods)
    dates_table_full = modelsetup.datesmodelrun(
        startyear=min([args.ref_startyear, args.sim_startyear]),
        endyear=args.sim_endyear,
        option_wateryear=pygem_prms['climate']['sim_wateryear'],
    )

    # GCM Simulation Period
    dates_table = modelsetup.datesmodelrun(
        startyear=args.sim_startyear,
        endyear=args.sim_endyear,
        option_wateryear=pygem_prms['climate']['sim_wateryear'],
    )

    if debug:
        print('ref years:', args.ref_startyear, ref_endyear)
        print('sim years:', args.sim_startyear, args.sim_endyear)

    # ===== LOAD CLIMATE DATA =====
    # Climate class
    if sim_climate_name in ['ERA5', 'COAWST']:
        gcm = class_climate.GCM(name=sim_climate_name)
        ref_gcm = gcm
        dates_table_ref = dates_table_full
    else:
        # GCM object
        if realization is None:
            gcm = class_climate.GCM(name=sim_climate_name, sim_climate_scenario=sim_climate_scenario)
        else:
            gcm = class_climate.GCM(
                name=sim_climate_name,
                sim_climate_scenario=sim_climate_scenario,
                realization=realization,
            )
        # Reference GCM
        ref_gcm = class_climate.GCM(name=args.ref_climate_name)

    # ----- Select Temperature and Precipitation Data -----
    # Air temperature [degC]
    gcm_temp, gcm_dates = gcm.importGCMvarnearestneighbor_xarray(
        gcm.temp_fn, gcm.temp_vn, main_glac_rgi, dates_table_full, verbose=debug
    )
    ref_temp, ref_dates = ref_gcm.importGCMvarnearestneighbor_xarray(
        ref_gcm.temp_fn, ref_gcm.temp_vn, main_glac_rgi, dates_table_ref, verbose=debug
    )
    # Precipitation [m]
    gcm_prec, gcm_dates = gcm.importGCMvarnearestneighbor_xarray(
        gcm.prec_fn, gcm.prec_vn, main_glac_rgi, dates_table_full, verbose=debug
    )
    ref_prec, ref_dates = ref_gcm.importGCMvarnearestneighbor_xarray(
        ref_gcm.prec_fn, ref_gcm.prec_vn, main_glac_rgi, dates_table_ref, verbose=debug
    )
    # Elevation [m asl]
    try:
        gcm_elev = gcm.importGCMfxnearestneighbor_xarray(gcm.elev_fn, gcm.elev_vn, main_glac_rgi)
    except:
        gcm_elev = None
    ref_elev = ref_gcm.importGCMfxnearestneighbor_xarray(ref_gcm.elev_fn, ref_gcm.elev_vn, main_glac_rgi)

    # ----- Temperature and Precipitation Bias Adjustments -----
    # No adjustments
    if args.option_bias_adjustment == 0 or sim_climate_name == args.ref_climate_name:
        if pygem_prms['climate']['sim_wateryear'] == 'hydro':
            dates_cn = 'wateryear'
        else:
            dates_cn = 'year'
        sim_idx_start = dates_table_full[dates_cn].to_list().index(args.sim_startyear)
        gcm_elev_adj = gcm_elev
        gcm_temp_adj = gcm_temp[:, sim_idx_start:]
        gcm_prec_adj = gcm_prec[:, sim_idx_start:]
        gcm_prec_biasadj_frac = np.ones(gcm_prec_adj.shape[0])
    # Bias correct based on reference climate data
    else:
        # OPTION 1: Adjust temp using Huss and Hock (2015), prec similar but addresses for variance and outliers
        if args.option_bias_adjustment == 1:
            # Temperature bias correction
            gcm_temp_adj, gcm_elev_adj = gcmbiasadj.temp_biasadj_HH2015(
                ref_temp,
                ref_elev,
                gcm_temp,
                dates_table_ref,
                dates_table_full,
                args.sim_startyear,
                args.ref_startyear,
            )
            # Precipitation bias correction
            gcm_prec_adj, gcm_elev_adj, gcm_prec_biasadj_frac = gcmbiasadj.prec_biasadj_opt1(
                ref_prec,
                ref_elev,
                gcm_prec,
                dates_table_ref,
                dates_table_full,
                args.sim_startyear,
                args.ref_startyear,
            )
        # OPTION 2: Adjust temp and prec using Huss and Hock (2015)
        elif args.option_bias_adjustment == 2:
            # Temperature bias correction
            gcm_temp_adj, gcm_elev_adj = gcmbiasadj.temp_biasadj_HH2015(
                ref_temp,
                ref_elev,
                gcm_temp,
                dates_table_ref,
                dates_table_full,
                args.sim_startyear,
                args.ref_startyear,
            )
            # Precipitation bias correction
            gcm_prec_adj, gcm_elev_adj, gcm_prec_biasadj_frac = gcmbiasadj.prec_biasadj_HH2015(
                ref_prec,
                ref_elev,
                gcm_prec,
                dates_table_ref,
                dates_table_full,
            )
        # OPTION 3: Adjust temp and prec using quantile delta mapping, Cannon et al. (2015)
        elif args.option_bias_adjustment == 3:
            # Temperature bias correction
            gcm_temp_adj, gcm_elev_adj = gcmbiasadj.temp_biasadj_QDM(
                ref_temp,
                ref_elev,
                gcm_temp,
                dates_table_ref,
                dates_table_full,
                args.sim_startyear,
                args.ref_startyear,
            )

            # Precipitation bias correction
            gcm_prec_adj, gcm_elev_adj = gcmbiasadj.prec_biasadj_QDM(
                ref_prec,
                ref_elev,
                gcm_prec,
                dates_table_ref,
                dates_table_full,
                args.sim_startyear,
                args.ref_startyear,
            )
            gcm_prec_biasadj_frac = np.ones(gcm_prec_adj.shape[0])

    # assert that the gcm_elev_adj is not None
    assert gcm_elev_adj is not None, 'No GCM elevation data'

    # ----- Other Climate Datasets (Air temperature variability [degC] and Lapse rate [K m-1])
    # Air temperature variability [degC]
    if pygem_prms['mb']['option_ablation'] != 2:
        gcm_tempstd = np.zeros((main_glac_rgi.shape[0], dates_table.shape[0]))
        ref_tempstd = np.zeros((main_glac_rgi.shape[0], dates_table_ref.shape[0]))
    elif pygem_prms['mb']['option_ablation'] == 2 and sim_climate_name in ['ERA5']:
        assert pygem_prms['time']['timestep'] != 'daily', 'Option 2 for ablation should not be used with daily data'
        gcm_tempstd, gcm_dates = gcm.importGCMvarnearestneighbor_xarray(
            gcm.tempstd_fn, gcm.tempstd_vn, main_glac_rgi, dates_table, verbose=debug
        )
        ref_tempstd = gcm_tempstd
    elif pygem_prms['mb']['option_ablation'] == 2 and args.ref_climate_name in ['ERA5']:
        assert pygem_prms['time']['timestep'] != 'daily', 'Option 2 for ablation should not be used with daily data'
        # Compute temp std based on reference climate data
        ref_tempstd, ref_dates = ref_gcm.importGCMvarnearestneighbor_xarray(
            ref_gcm.tempstd_fn,
            ref_gcm.tempstd_vn,
            main_glac_rgi,
            dates_table_ref,
            verbose=debug,
        )
        # Monthly average from reference climate data
        gcm_tempstd = gcmbiasadj.monthly_avg_array_rolled(ref_tempstd, dates_table_ref, dates_table_full)
    else:
        gcm_tempstd = np.zeros((main_glac_rgi.shape[0], dates_table.shape[0]))
        ref_tempstd = np.zeros((main_glac_rgi.shape[0], dates_table_ref.shape[0]))

    # Lapse rate
    if pygem_prms['sim']['params']['use_constant_lapserate']:
        gcm_lr = np.zeros((main_glac_rgi.shape[0], dates_table.shape[0])) + pygem_prms['sim']['params']['lapserate']
        ref_lr = np.zeros((main_glac_rgi.shape[0], dates_table_ref.shape[0])) + pygem_prms['sim']['params']['lapserate']
    else:
        if sim_climate_name in ['ERA-Interim', 'ERA5']:
            gcm_lr, gcm_dates = gcm.importGCMvarnearestneighbor_xarray(gcm.lr_fn, gcm.lr_vn, main_glac_rgi, dates_table)
            ref_lr = gcm_lr
        else:
            # Compute lapse rates based on reference climate data
            ref_lr, ref_dates = ref_gcm.importGCMvarnearestneighbor_xarray(
                ref_gcm.lr_fn, ref_gcm.lr_vn, main_glac_rgi, dates_table_ref
            )
            # Monthly average from reference climate data
            gcm_lr = gcmbiasadj.monthly_avg_array_rolled(
                ref_lr,
                dates_table_ref,
                dates_table_full,
                args.sim_startyear,
                args.ref_startyear,
            )

    # ===== RUN MASS BALANCE =====
    # Number of simulations
    if args.option_calibration == 'MCMC':
        nsims = args.nsims
    else:
        nsims = 1

    # Number of years (for OGGM's run_until_and_store)
    nyears = len(dates_table.year.unique())
    nyears_ref = len(dates_table_ref.year.unique())

    for glac in range(main_glac_rgi.shape[0]):
        if glac == 0:
            print(
                sim_climate_name,
                ':',
                main_glac_rgi.loc[main_glac_rgi.index.values[glac], 'RGIId'],
            )
        # Select subsets of data
        glacier_rgi_table = main_glac_rgi.loc[main_glac_rgi.index.values[glac], :]
        glacier_str = '{0:0.5f}'.format(glacier_rgi_table['RGIId_float'])
        reg_str = str(glacier_rgi_table.O1Region).zfill(2)
        rgiid = main_glac_rgi.loc[main_glac_rgi.index.values[glac], 'RGIId']

        try:
            # ===== Load glacier data: area (km2), ice thickness (m), width (km) =====
            if glacier_rgi_table['TermType'] not in [1, 5] or not pygem_prms['setup']['include_frontalablation']:
                gdir = single_flowline_glacier_directory(glacier_str, working_dir=args.oggm_working_dir)
                gdir.is_tidewater = False
                calving_k = None
            else:
                gdir = single_flowline_glacier_directory_with_calving(glacier_str, working_dir=args.oggm_working_dir)
                gdir.is_tidewater = True
                cfg.PARAMS['use_kcalving_for_inversion'] = True
                cfg.PARAMS['use_kcalving_for_run'] = True

            # Flowlines
            fls = gdir.read_pickle('inversion_flowlines')

            # Reference gdir for ice thickness inversion
            gdir_ref = copy.deepcopy(gdir)
            gdir_ref.historical_climate = {
                'elev': ref_elev[glac],
                'temp': ref_temp[glac, :],
                'tempstd': ref_tempstd[glac, :],
                'prec': ref_prec[glac, :],
                'lr': ref_lr[glac, :],
            }
            gdir_ref.dates_table = dates_table_ref

            gdir.historical_climate = {
                'elev': gcm_elev_adj[glac],
                'temp': gcm_temp_adj[glac, :],
                'tempstd': gcm_tempstd[glac, :],
                'prec': gcm_prec_adj[glac, :],
                'lr': gcm_lr[glac, :],
            }
            # Warn if precipitation bias adjustment is greater than 2x
            if gcm_prec_biasadj_frac[glac] < 0.5 or gcm_prec_biasadj_frac[glac] > 2:
                warnings.warn(
                    f'Bias-adjusted GCM precipitation for {glacier_str} differs from that of the refernce climate data by a factor greater than 2x ({round(gcm_prec_biasadj_frac[glac], 2)})',
                    Warning,
                    stacklevel=2,
                )
            gdir.dates_table = dates_table

            glacier_area_km2 = fls[0].widths_m * fls[0].dx_meter / 1e6
            if (fls is not None) and (glacier_area_km2.sum() > 0):
                # Load model parameters
                if args.option_calibration:
                    modelprms_fp = args.modelprms_fp
                    if not modelprms_fp:
                        modelprms_fn = glacier_str + '-modelprms_dict.json'
                        modelprms_fp = (
                            pygem_prms['root'] + '/Output/calibration/' + glacier_str.split('.')[0].zfill(2) + '/'
                        ) + modelprms_fn

                    assert os.path.exists(modelprms_fp), 'Calibrated parameters do not exist.'
                    with open(modelprms_fp, 'r') as f:
                        modelprms_dict = json.load(f)

                    assert args.option_calibration in modelprms_dict, (
                        'Error: ' + args.option_calibration + ' not in modelprms_dict'
                    )
                    modelprms_all = modelprms_dict[args.option_calibration]
                    # MCMC needs model parameters to be selected
                    if args.option_calibration == 'MCMC':
                        if nsims == 1:
                            modelprms_all = {
                                'kp': [np.median(modelprms_all['kp']['chain_0'])],
                                'tbias': [np.median(modelprms_all['tbias']['chain_0'])],
                                'ddfsnow': [np.median(modelprms_all['ddfsnow']['chain_0'])],
                                'ddfice': [np.median(modelprms_all['ddfice']['chain_0'])],
                                'tsnow_threshold': modelprms_all['tsnow_threshold'],
                                'precgrad': modelprms_all['precgrad'],
                            }
                        else:
                            # Select every kth iteration to use for the ensemble
                            mcmc_sample_no = len(modelprms_all['kp']['chain_0'])
                            sims_burn = int(args.mcmc_burn_pct / 100 * mcmc_sample_no)
                            mp_spacing = int((mcmc_sample_no - sims_burn) / nsims)
                            mp_idx_start = np.arange(sims_burn, sims_burn + mp_spacing)
                            np.random.shuffle(mp_idx_start)
                            mp_idx_start = mp_idx_start[0]
                            mp_idx_all = np.arange(mp_idx_start, mcmc_sample_no, mp_spacing)
                            modelprms_all = {
                                'kp': [modelprms_all['kp']['chain_0'][mp_idx] for mp_idx in mp_idx_all],
                                'tbias': [modelprms_all['tbias']['chain_0'][mp_idx] for mp_idx in mp_idx_all],
                                'ddfsnow': [modelprms_all['ddfsnow']['chain_0'][mp_idx] for mp_idx in mp_idx_all],
                                'ddfice': [modelprms_all['ddfice']['chain_0'][mp_idx] for mp_idx in mp_idx_all],
                                'tsnow_threshold': modelprms_all['tsnow_threshold'] * nsims,
                                'precgrad': modelprms_all['precgrad'] * nsims,
                            }
                    else:
                        nsims = 1

                    # Calving parameter
                    if (
                        glacier_rgi_table['TermType'] not in [1, 5]
                        or not pygem_prms['setup']['include_frontalablation']
                    ):
                        calving_k = None
                    else:
                        # Load quality controlled frontal ablation data
                        fp = f'{pygem_prms["root"]}/{pygem_prms["calib"]["data"]["frontalablation"]["frontalablation_relpath"]}/analysis/{pygem_prms["calib"]["data"]["frontalablation"]["frontalablation_cal_fn"]}'
                        assert os.path.exists(fp), 'Calibrated calving dataset does not exist'
                        calving_df = pd.read_csv(fp)
                        calving_rgiids = list(calving_df.RGIId)

                        # Use calibrated value if individual data available
                        if rgiid in calving_rgiids:
                            calving_idx = calving_rgiids.index(rgiid)
                            calving_k = calving_df.loc[calving_idx, 'calving_k']
                            calving_k_nmad = calving_df.loc[calving_idx, 'calving_k_nmad']
                        # Otherwise, use region's median value
                        else:
                            calving_df['O1Region'] = [
                                int(x.split('-')[1].split('.')[0]) for x in calving_df.RGIId.values
                            ]
                            calving_df_reg = calving_df.loc[calving_df['O1Region'] == int(reg_str), :]
                            calving_k = np.median(calving_df_reg.calving_k)
                            calving_k_nmad = 0

                        if nsims == 1:
                            calving_k_values = np.array([calving_k])
                        else:
                            calving_k_values = calving_k + np.random.normal(loc=0, scale=calving_k_nmad, size=nsims)
                            calving_k_values[calving_k_values < 0.001] = 0.001
                            calving_k_values[calving_k_values > 5] = 5

                            #                            calving_k_values[:] = calving_k

                            while not abs(np.median(calving_k_values) - calving_k) < 0.001:
                                calving_k_values = calving_k + np.random.normal(loc=0, scale=calving_k_nmad, size=nsims)
                                calving_k_values[calving_k_values < 0.001] = 0.001
                                calving_k_values[calving_k_values > 5] = 5

                            #                                print(calving_k, np.median(calving_k_values))

                            assert abs(np.median(calving_k_values) - calving_k) < 0.001, (
                                'calving_k distribution too far off'
                            )

                        if debug:
                            print(
                                'calving_k_values:',
                                np.mean(calving_k_values),
                                np.std(calving_k_values),
                                '\n',
                                calving_k_values,
                            )

                else:
                    modelprms_all = {
                        'kp': [args.kp],
                        'tbias': [args.tbias],
                        'ddfsnow': [args.ddfsnow],
                        'ddfice': [args.ddfsnow / pygem_prms['sim']['params']['ddfsnow_iceratio']],
                        'tsnow_threshold': [pygem_prms['sim']['params']['tsnow_threshold']],
                        'precgrad': [pygem_prms['sim']['params']['precgrad']],
                    }
                    calving_k = np.zeros(nsims) + pygem_prms['sim']['params']['calving_k']
                    calving_k_values = calving_k

                if debug and gdir.is_tidewater:
                    print('calving_k:', calving_k)

                # Load OGGM glacier dynamics parameters (if necessary)
                if args.option_dynamics in ['OGGM', 'MassRedistributionCurves']:
                    # CFL number (may use different values for calving to prevent errors)
                    if (
                        glacier_rgi_table['TermType'] not in [1, 5]
                        or not pygem_prms['setup']['include_frontalablation']
                    ):
                        cfg.PARAMS['cfl_number'] = pygem_prms['sim']['oggm_dynamics']['cfl_number']
                    else:
                        cfg.PARAMS['cfl_number'] = pygem_prms['sim']['oggm_dynamics']['cfl_number_calving']

                    if debug:
                        print('cfl number:', cfg.PARAMS['cfl_number'])

                    if args.use_regional_glen_a:
                        glena_df = pd.read_csv(
                            f'{pygem_prms["root"]}/{pygem_prms["sim"]["oggm_dynamics"]["glen_a_regional_relpath"]}'
                        )
                        glena_O1regions = [int(x) for x in glena_df.O1Region.values]
                        assert glacier_rgi_table.O1Region in glena_O1regions, glacier_str + ' O1 region not in glena_df'
                        glena_idx = np.where(glena_O1regions == glacier_rgi_table.O1Region)[0][0]
                        glen_a_multiplier = glena_df.loc[glena_idx, 'glens_a_multiplier']
                        fs = glena_df.loc[glena_idx, 'fs']
                    else:
                        args.option_dynamics = None
                        fs = pygem_prms['sim']['oggm_dynamics']['fs']
                        glen_a_multiplier = pygem_prms['sim']['oggm_dynamics']['glen_a_multiplier']
                    glen_a = cfg.PARAMS['glen_a'] * glen_a_multiplier

                    # spinup
                    if args.spinup:
                        try:
                            # see if model_flowlines from spinup exist
                            nfls = gdir.read_pickle(
                                'model_flowlines',
                                filesuffix=f'_dynamic_spinup_pygem_mb_{args.sim_startyear}',
                            )
                        except:
                            raise
                        glen_a = gdir.get_diagnostics()['inversion_glen_a']
                        fs = gdir.get_diagnostics()['inversion_fs']

                # Time attributes and values
                if pygem_prms['climate']['sim_wateryear'] == 'hydro':
                    annual_columns = np.unique(dates_table['wateryear'].values)
                else:
                    annual_columns = np.unique(dates_table['year'].values)

                # append additional year to year_values to account for mass and area at end of period
                year_values = annual_columns
                year_values = np.concatenate((year_values, np.array([annual_columns[-1] + 1])))
                output_glac_temp_steps = np.zeros((dates_table.shape[0], nsims)) * np.nan
                output_glac_prec_steps = np.zeros((dates_table.shape[0], nsims)) * np.nan
                output_glac_acc_steps = np.zeros((dates_table.shape[0], nsims)) * np.nan
                output_glac_refreeze_steps = np.zeros((dates_table.shape[0], nsims)) * np.nan
                output_glac_melt_steps = np.zeros((dates_table.shape[0], nsims)) * np.nan
                output_glac_frontalablation_steps = np.zeros((dates_table.shape[0], nsims)) * np.nan
                output_glac_massbaltotal_steps = np.zeros((dates_table.shape[0], nsims)) * np.nan
                output_glac_runoff_steps = np.zeros((dates_table.shape[0], nsims)) * np.nan
                output_glac_snowline_steps = np.zeros((dates_table.shape[0], nsims)) * np.nan
                output_glac_area_annual = np.zeros((year_values.shape[0], nsims)) * np.nan
                output_glac_mass_annual = np.zeros((year_values.shape[0], nsims)) * np.nan
                output_glac_mass_bsl_annual = np.zeros((year_values.shape[0], nsims)) * np.nan
                output_glac_mass_change_ignored_annual = np.zeros((year_values.shape[0], nsims))
                output_glac_ELA_annual = np.zeros((year_values.shape[0], nsims)) * np.nan
                output_offglac_prec_steps = np.zeros((dates_table.shape[0], nsims)) * np.nan
                output_offglac_refreeze_steps = np.zeros((dates_table.shape[0], nsims)) * np.nan
                output_offglac_melt_steps = np.zeros((dates_table.shape[0], nsims)) * np.nan
                output_offglac_snowpack_steps = np.zeros((dates_table.shape[0], nsims)) * np.nan
                output_offglac_runoff_steps = np.zeros((dates_table.shape[0], nsims)) * np.nan
                output_glac_bin_icethickness_annual = None

                # Loop through model parameters
                count_exceed_boundary_errors = 0
                mb_em_sims = []
                for n_iter in range(nsims):
                    if debug:
                        print('n_iter:', n_iter)

                    if calving_k is not None:
                        calving_k = calving_k_values[n_iter]
                        cfg.PARAMS['calving_k'] = calving_k
                        cfg.PARAMS['inversion_calving_k'] = calving_k

                    # successful_run used to continue runs when catching specific errors
                    successful_run = True

                    modelprms = {
                        'kp': modelprms_all['kp'][n_iter],
                        'tbias': modelprms_all['tbias'][n_iter],
                        'ddfsnow': modelprms_all['ddfsnow'][n_iter],
                        'ddfice': modelprms_all['ddfice'][n_iter],
                        'tsnow_threshold': modelprms_all['tsnow_threshold'][n_iter],
                        'precgrad': modelprms_all['precgrad'][n_iter],
                    }

                    if debug:
                        print(
                            glacier_str
                            + '  kp: '
                            + str(np.round(modelprms['kp'], 2))
                            + ' ddfsnow: '
                            + str(np.round(modelprms['ddfsnow'], 4))
                            + ' tbias: '
                            + str(np.round(modelprms['tbias'], 2))
                        )

                    # ----- ICE THICKNESS INVERSION using OGGM -----
                    if args.option_dynamics is not None:
                        # Apply inversion_filter on mass balance with debris to avoid negative flux
                        if pygem_prms['mb']['include_debris']:
                            inversion_filter = True
                        else:
                            inversion_filter = False

                        # run inversion if spinup was not run previously
                        #   note that if `args.spinup`, spinup flowlines have already been imported above as `nfls` and thus inversion is not redone here
                        if not args.spinup:
                            # Perform inversion based on PyGEM MB using reference directory
                            mbmod_inv = PyGEMMassBalance(
                                gdir_ref,
                                modelprms,
                                glacier_rgi_table,
                                fls=fls,
                                option_areaconstant=True,
                                inversion_filter=inversion_filter,
                            )

                            # Non-tidewater glaciers
                            if not gdir.is_tidewater or not pygem_prms['setup']['include_frontalablation']:
                                # Arbitrariliy shift the MB profile up (or down) until mass balance is zero (equilibrium for inversion)
                                apparent_mb_from_any_mb(gdir, mb_model=mbmod_inv)
                                tasks.prepare_for_inversion(gdir)
                                tasks.mass_conservation_inversion(
                                    gdir,
                                    glen_a=cfg.PARAMS['glen_a'] * glen_a_multiplier,
                                    fs=fs,
                                )

                            # Tidewater glaciers
                            else:
                                cfg.PARAMS['use_kcalving_for_inversion'] = True
                                cfg.PARAMS['use_kcalving_for_run'] = True
                                tasks.find_inversion_calving_from_any_mb(
                                    gdir,
                                    mb_model=mbmod_inv,
                                    glen_a=cfg.PARAMS['glen_a'] * glen_a_multiplier,
                                    fs=fs,
                                )

                            # ----- INDENTED TO BE JUST WITH DYNAMICS -----
                            tasks.init_present_time_glacier(gdir)  # adds bins below

                            if not os.path.isfile(gdir.get_filepath('model_flowlines')):
                                tasks.compute_downstream_line(gdir)
                                tasks.compute_downstream_bedshape(gdir)
                                tasks.init_present_time_glacier(gdir)  # adds bins below

                            try:
                                if pygem_prms['mb']['include_debris']:
                                    debris.debris_binned(
                                        gdir, fl_str='model_flowlines'
                                    )  # add debris enhancement factors to flowlines
                                nfls = gdir.read_pickle('model_flowlines')
                            except:
                                raise

                        # Water Level
                        # Check that water level is within given bounds
                        cls = gdir.read_pickle('inversion_input')[-1]
                        th = cls['hgt'][-1]
                        vmin, vmax = cfg.PARAMS['free_board_marine_terminating']
                        water_level = utils.clip_scalar(0, th - vmax, th - vmin)

                    # No ice dynamics options
                    else:
                        nfls = fls

                    # Record initial surface h for overdeepening calculations
                    surface_h_initial = nfls[0].surface_h

                    # ------ MODEL WITH EVOLVING AREA ------
                    # Mass balance model
                    mbmod = PyGEMMassBalance(
                        gdir,
                        modelprms,
                        glacier_rgi_table,
                        fls=nfls,
                        option_areaconstant=False,
                    )

                    ######################################
                    ### OGGM dynamical evolution model ###
                    ######################################
                    if args.option_dynamics == 'OGGM':
                        if debug:
                            print('OGGM GLACIER DYNAMICS!')

                        # FluxBasedModel is older numerical scheme but includes frontal ablation
                        if gdir.is_tidewater:
                            ev_model = FluxBasedModel(
                                nfls,
                                y0=args.sim_startyear,
                                mb_model=mbmod,
                                glen_a=cfg.PARAMS['glen_a'] * glen_a_multiplier,
                                fs=fs,
                                is_tidewater=gdir.is_tidewater,
                                water_level=water_level,
                            )
                        # SemiImplicitModel is newer numerical solver, but does not yet include frontal ablation
                        else:
                            ev_model = SemiImplicitModel(
                                nfls,
                                y0=args.sim_startyear,
                                mb_model=mbmod,
                                glen_a=cfg.PARAMS['glen_a'] * glen_a_multiplier,
                                fs=fs,
                            )

                        if debug:
                            fig, ax = plt.subplots(1)
                            graphics.plot_modeloutput_section(ev_model, ax=ax)

                        diag = ev_model.run_until_and_store(args.sim_endyear + 1)
                        ev_model.mb_model.glac_wide_volume_annual[-1] = diag.volume_m3[-1]
                        ev_model.mb_model.glac_wide_area_annual[-1] = diag.area_m2[-1]

                        # Record frontal ablation for tidewater glaciers and update total mass balance
                        if gdir.is_tidewater:
                            # Glacier-wide frontal ablation (m3 w.e.)
                            # - note: diag.calving_m3 is cumulative calving
                            if debug:
                                print('\n\ndiag.calving_m3:', diag.calving_m3.values)
                                print(
                                    'calving_m3_since_y0:',
                                    ev_model.calving_m3_since_y0,
                                )
                            calving_m3_annual = (
                                (diag.calving_m3.values[1:] - diag.calving_m3.values[0:-1])
                                * pygem_prms['constants']['density_ice']
                                / pygem_prms['constants']['density_water']
                            )
                            for n in np.arange(calving_m3_annual.shape[0]):
                                ev_model.mb_model.glac_wide_frontalablation[12 * n + 11] = calving_m3_annual[n]

                            # Glacier-wide total mass balance (m3 w.e.)
                            ev_model.mb_model.glac_wide_massbaltotal = (
                                ev_model.mb_model.glac_wide_massbaltotal - ev_model.mb_model.glac_wide_frontalablation
                            )

                            if debug:
                                print(
                                    'avg calving_m3:',
                                    calving_m3_annual.sum() / nyears,
                                )
                                print(
                                    'avg frontal ablation [Gta]:',
                                    np.round(
                                        ev_model.mb_model.glac_wide_frontalablation.sum() / 1e9 / nyears,
                                        4,
                                    ),
                                )
                                print(
                                    'avg frontal ablation [Gta]:',
                                    np.round(
                                        ev_model.calving_m3_since_y0
                                        * pygem_prms['constants']['density_ice']
                                        / 1e12
                                        / nyears,
                                        4,
                                    ),
                                )

                    ######################################
                    ##### mass redistrubution model  #####
                    ######################################
                    elif args.option_dynamics == 'MassRedistributionCurves':
                        if debug:
                            print('MASS REDISTRIBUTION CURVES!')
                        ev_model = MassRedistributionCurveModel(
                            nfls,
                            mb_model=mbmod,
                            y0=args.sim_startyear,
                            glen_a=cfg.PARAMS['glen_a'] * glen_a_multiplier,
                            fs=fs,
                            is_tidewater=gdir.is_tidewater,
                            # water_level=gdir.get_diagnostics().get('calving_water_level', None)
                            water_level=water_level,
                        )

                        if debug:
                            print('New glacier vol', ev_model.volume_m3)
                            graphics.plot_modeloutput_section(ev_model)
                            plt.show()

                        _, diag = ev_model.run_until_and_store(args.sim_endyear + 1)
                        #    print('shape of volume:', ev_model.mb_model.glac_wide_volume_annual.shape, diag.volume_m3.shape)
                        ev_model.mb_model.glac_wide_volume_annual = diag.volume_m3.values
                        ev_model.mb_model.glac_wide_area_annual = diag.area_m2.values

                        # Record frontal ablation for tidewater glaciers and update total mass balance
                        if gdir.is_tidewater:
                            # Update glacier-wide frontal ablation (m3 w.e.)
                            ev_model.mb_model.glac_wide_frontalablation = (
                                ev_model.mb_model.glac_bin_frontalablation.sum(0)
                            )
                            # Update glacier-wide total mass balance (m3 w.e.)
                            ev_model.mb_model.glac_wide_massbaltotal = (
                                ev_model.mb_model.glac_wide_massbaltotal - ev_model.mb_model.glac_wide_frontalablation
                            )

                            if debug:
                                print(
                                    'avg frontal ablation [Gta]:',
                                    np.round(
                                        ev_model.mb_model.glac_wide_frontalablation.sum() / 1e9 / nyears,
                                        4,
                                    ),
                                )
                                print(
                                    'avg frontal ablation [Gta]:',
                                    np.round(
                                        ev_model.calving_m3_since_y0
                                        * pygem_prms['constants']['density_ice']
                                        / 1e12
                                        / nyears,
                                        4,
                                    ),
                                )

                    ######################################
                    ######### no dynamical model #########
                    ######################################
                    elif args.option_dynamics is None:
                        # Mass balance model
                        ev_model = None
                        diag = xr.Dataset()
                        mbmod = PyGEMMassBalance(
                            gdir,
                            modelprms,
                            glacier_rgi_table,
                            fls=fls,
                            option_areaconstant=True,
                        )
                        # ----- MODEL RUN WITH CONSTANT GLACIER AREA -----
                        years = np.arange(args.sim_startyear, args.sim_endyear + 1)
                        mb_all = []
                        for year in years:
                            mb_annual = mbmod.get_annual_mb(
                                nfls[0].surface_h,
                                fls=nfls,
                                fl_id=0,
                                year=year,
                                debug=True,
                            )
                            mb_mwea = (
                                mb_annual
                                * 365
                                * 24
                                * 3600
                                * pygem_prms['constants']['density_ice']
                                / pygem_prms['constants']['density_water']
                            )
                            glac_wide_mb_mwea = (
                                mb_mwea * mbmod.glacier_area_initial
                            ).sum() / mbmod.glacier_area_initial.sum()
                            mb_all.append(glac_wide_mb_mwea)
                        mbmod.glac_wide_area_annual[-1] = mbmod.glac_wide_area_annual[0]
                        mbmod.glac_wide_volume_annual[-1] = mbmod.glac_wide_volume_annual[0]
                        diag['area_m2'] = mbmod.glac_wide_area_annual
                        diag['volume_m3'] = mbmod.glac_wide_volume_annual
                        diag['volume_bsl_m3'] = 0

                        if debug:
                            print(
                                'iter:',
                                n_iter,
                                'massbal (mean, std):',
                                np.round(np.mean(mb_all), 3),
                                np.round(np.std(mb_all), 3),
                                'massbal (med):',
                                np.round(np.median(mb_all), 3),
                            )

                    #                            mb_em_mwea = run_emulator_mb(modelprms)
                    #                            print('  emulator mb:', np.round(mb_em_mwea,3))
                    #                            mb_em_sims.append(mb_em_mwea)

                    # Record output for successful runs
                    if successful_run:
                        if args.option_dynamics is not None:
                            if debug:
                                graphics.plot_modeloutput_section(ev_model, ax=ax, srfls='--')
                                plt.figure()
                                diag.volume_m3.plot()
                                plt.show()

                            # Post-process data to ensure mass is conserved and update accordingly for ignored mass losses
                            #  ignored mass losses occur because mass balance model does not know ice thickness and flux divergence
                            area_initial = mbmod.glac_bin_area_annual[:, 0].sum()
                            mb_mwea_diag = (
                                (diag.volume_m3.values[-1] - diag.volume_m3.values[0])
                                / area_initial
                                / nyears
                                * pygem_prms['constants']['density_ice']
                                / pygem_prms['constants']['density_water']
                            )
                            mb_mwea_mbmod = mbmod.glac_wide_massbaltotal.sum() / area_initial / nyears

                            if debug:
                                vol_change_diag = diag.volume_m3.values[-1] - diag.volume_m3.values[0]
                                print(
                                    '  vol init  [Gt]:',
                                    np.round(diag.volume_m3.values[0] * 0.9 / 1e9, 5),
                                )
                                print(
                                    '  vol final [Gt]:',
                                    np.round(diag.volume_m3.values[-1] * 0.9 / 1e9, 5),
                                )
                                print(
                                    '  vol change[Gt]:',
                                    np.round(vol_change_diag * 0.9 / 1e9, 5),
                                )
                                print('  mb [mwea]:', np.round(mb_mwea_diag, 2))
                                print('  mb_mbmod [mwea]:', np.round(mb_mwea_mbmod, 2))

                            if np.abs(mb_mwea_diag - mb_mwea_mbmod) > 1e-6:
                                ev_model.mb_model.ensure_mass_conservation(diag, dates_table)

                        if debug:
                            print(
                                'mass loss [Gt]:',
                                mbmod.glac_wide_massbaltotal.sum() / 1e9,
                            )

                        # RECORD PARAMETERS TO DATASET
                        output_glac_temp_steps[:, n_iter] = mbmod.glac_wide_temp
                        output_glac_prec_steps[:, n_iter] = mbmod.glac_wide_prec
                        output_glac_acc_steps[:, n_iter] = mbmod.glac_wide_acc
                        output_glac_refreeze_steps[:, n_iter] = mbmod.glac_wide_refreeze
                        output_glac_melt_steps[:, n_iter] = mbmod.glac_wide_melt
                        output_glac_frontalablation_steps[:, n_iter] = mbmod.glac_wide_frontalablation
                        output_glac_massbaltotal_steps[:, n_iter] = mbmod.glac_wide_massbaltotal
                        output_glac_runoff_steps[:, n_iter] = mbmod.glac_wide_runoff
                        output_glac_snowline_steps[:, n_iter] = mbmod.glac_wide_snowline
                        output_glac_area_annual[:, n_iter] = diag.area_m2.values
                        output_glac_mass_annual[:, n_iter] = (
                            diag.volume_m3.values * pygem_prms['constants']['density_ice']
                        )
                        output_glac_mass_bsl_annual[:, n_iter] = (
                            diag.volume_bsl_m3.values * pygem_prms['constants']['density_ice']
                        )
                        output_glac_mass_change_ignored_annual[:-1, n_iter] = (
                            mbmod.glac_wide_volume_change_ignored_annual * pygem_prms['constants']['density_ice']
                        )
                        output_glac_ELA_annual[:, n_iter] = mbmod.glac_wide_ELA_annual
                        output_offglac_prec_steps[:, n_iter] = mbmod.offglac_wide_prec

                        output_offglac_refreeze_steps[:, n_iter] = mbmod.offglac_wide_refreeze
                        output_offglac_melt_steps[:, n_iter] = mbmod.offglac_wide_melt
                        output_offglac_snowpack_steps[:, n_iter] = mbmod.offglac_wide_snowpack
                        output_offglac_runoff_steps[:, n_iter] = mbmod.offglac_wide_runoff

                        if output_glac_bin_icethickness_annual is None:
                            output_glac_bin_area_annual_sim = mbmod.glac_bin_area_annual[:, :, np.newaxis]
                            output_glac_bin_mass_annual_sim = (
                                mbmod.glac_bin_area_annual
                                * mbmod.glac_bin_icethickness_annual
                                * pygem_prms['constants']['density_ice']
                            )[:, :, np.newaxis]
                            output_glac_bin_icethickness_annual_sim = (mbmod.glac_bin_icethickness_annual)[
                                :, :, np.newaxis
                            ]
                            # Update the latest thickness and volume
                            if ev_model is not None:
                                fl_dx_meter = getattr(ev_model.fls[0], 'dx_meter', None)
                                fl_widths_m = getattr(ev_model.fls[0], 'widths_m', None)
                                fl_section = getattr(ev_model.fls[0], 'section', None)
                            else:
                                fl_dx_meter = getattr(nfls[0], 'dx_meter', None)
                                fl_widths_m = getattr(nfls[0], 'widths_m', None)
                                fl_section = getattr(nfls[0], 'section', None)
                            if fl_section is not None and fl_widths_m is not None:
                                # thickness
                                icethickness_t0 = np.zeros(fl_section.shape)
                                icethickness_t0[fl_widths_m > 0] = (
                                    fl_section[fl_widths_m > 0] / fl_widths_m[fl_widths_m > 0]
                                )
                                output_glac_bin_icethickness_annual_sim[:, -1, 0] = icethickness_t0
                                # mass
                                glacier_vol_t0 = fl_widths_m * fl_dx_meter * icethickness_t0
                                output_glac_bin_mass_annual_sim[:, -1, 0] = (
                                    glacier_vol_t0 * pygem_prms['constants']['density_ice']
                                )
                            output_glac_bin_area_annual = output_glac_bin_area_annual_sim
                            output_glac_bin_mass_annual = output_glac_bin_mass_annual_sim
                            output_glac_bin_icethickness_annual = output_glac_bin_icethickness_annual_sim
                            output_glac_bin_massbalclim_annual_sim = np.zeros(mbmod.glac_bin_icethickness_annual.shape)
                            output_glac_bin_massbalclim_annual_sim[:, :-1] = mbmod.glac_bin_massbalclim_annual
                            output_glac_bin_massbalclim_annual = output_glac_bin_massbalclim_annual_sim[
                                :, :, np.newaxis
                            ]
                            output_glac_bin_massbalclim_steps_sim = np.zeros(mbmod.glac_bin_massbalclim.shape)
                            output_glac_bin_massbalclim_steps_sim = mbmod.glac_bin_massbalclim
                            output_glac_bin_massbalclim_steps = output_glac_bin_massbalclim_steps_sim[:, :, np.newaxis]
                            # accumulation
                            output_glac_bin_acc_steps_sim = np.zeros(mbmod.bin_acc.shape)
                            output_glac_bin_acc_steps_sim = mbmod.bin_acc
                            output_glac_bin_acc_steps = output_glac_bin_acc_steps_sim[:, :, np.newaxis]
                            # refreeze
                            output_glac_bin_refreeze_steps_sim = np.zeros(mbmod.glac_bin_refreeze.shape)
                            output_glac_bin_refreeze_steps_sim = mbmod.glac_bin_refreeze
                            output_glac_bin_refreeze_steps = output_glac_bin_refreeze_steps_sim[:, :, np.newaxis]
                            # melt
                            output_glac_bin_melt_steps_sim = np.zeros(mbmod.glac_bin_melt.shape)
                            output_glac_bin_melt_steps_sim = mbmod.glac_bin_melt
                            output_glac_bin_melt_steps = output_glac_bin_melt_steps_sim[:, :, np.newaxis]

                        else:
                            # Update the latest thickness and volume
                            output_glac_bin_area_annual_sim = mbmod.glac_bin_area_annual[:, :, np.newaxis]
                            output_glac_bin_mass_annual_sim = (
                                mbmod.glac_bin_area_annual
                                * mbmod.glac_bin_icethickness_annual
                                * pygem_prms['constants']['density_ice']
                            )[:, :, np.newaxis]
                            output_glac_bin_icethickness_annual_sim = (mbmod.glac_bin_icethickness_annual)[
                                :, :, np.newaxis
                            ]
                            if ev_model is not None:
                                fl_dx_meter = getattr(ev_model.fls[0], 'dx_meter', None)
                                fl_widths_m = getattr(ev_model.fls[0], 'widths_m', None)
                                fl_section = getattr(ev_model.fls[0], 'section', None)
                            else:
                                fl_dx_meter = getattr(nfls[0], 'dx_meter', None)
                                fl_widths_m = getattr(nfls[0], 'widths_m', None)
                                fl_section = getattr(nfls[0], 'section', None)
                            if fl_section is not None and fl_widths_m is not None:
                                # thickness
                                icethickness_t0 = np.zeros(fl_section.shape)
                                icethickness_t0[fl_widths_m > 0] = (
                                    fl_section[fl_widths_m > 0] / fl_widths_m[fl_widths_m > 0]
                                )
                                output_glac_bin_icethickness_annual_sim[:, -1, 0] = icethickness_t0
                                # mass
                                glacier_vol_t0 = fl_widths_m * fl_dx_meter * icethickness_t0
                                output_glac_bin_mass_annual_sim[:, -1, 0] = (
                                    glacier_vol_t0 * pygem_prms['constants']['density_ice']
                                )
                            output_glac_bin_area_annual = np.append(
                                output_glac_bin_area_annual,
                                output_glac_bin_area_annual_sim,
                                axis=2,
                            )
                            output_glac_bin_mass_annual = np.append(
                                output_glac_bin_mass_annual,
                                output_glac_bin_mass_annual_sim,
                                axis=2,
                            )
                            output_glac_bin_icethickness_annual = np.append(
                                output_glac_bin_icethickness_annual,
                                output_glac_bin_icethickness_annual_sim,
                                axis=2,
                            )
                            output_glac_bin_massbalclim_annual_sim = np.zeros(mbmod.glac_bin_icethickness_annual.shape)
                            output_glac_bin_massbalclim_annual_sim[:, :-1] = mbmod.glac_bin_massbalclim_annual
                            output_glac_bin_massbalclim_annual = np.append(
                                output_glac_bin_massbalclim_annual,
                                output_glac_bin_massbalclim_annual_sim[:, :, np.newaxis],
                                axis=2,
                            )
                            output_glac_bin_massbalclim_steps_sim = np.zeros(mbmod.glac_bin_massbalclim.shape)
                            output_glac_bin_massbalclim_steps_sim = mbmod.glac_bin_massbalclim
                            output_glac_bin_massbalclim_steps = np.append(
                                output_glac_bin_massbalclim_steps,
                                output_glac_bin_massbalclim_steps_sim[:, :, np.newaxis],
                                axis=2,
                            )
                            # accumulation
                            output_glac_bin_acc_steps_sim = np.zeros(mbmod.bin_acc.shape)
                            output_glac_bin_acc_steps_sim = mbmod.bin_acc
                            output_glac_bin_acc_steps = np.append(
                                output_glac_bin_acc_steps,
                                output_glac_bin_acc_steps_sim[:, :, np.newaxis],
                                axis=2,
                            )
                            # melt
                            output_glac_bin_melt_steps_sim = np.zeros(mbmod.glac_bin_melt.shape)
                            output_glac_bin_melt_steps_sim = mbmod.glac_bin_melt
                            output_glac_bin_melt_steps = np.append(
                                output_glac_bin_melt_steps,
                                output_glac_bin_melt_steps_sim[:, :, np.newaxis],
                                axis=2,
                            )
                            # refreeze
                            output_glac_bin_refreeze_steps_sim = np.zeros(mbmod.glac_bin_refreeze.shape)
                            output_glac_bin_refreeze_steps_sim = mbmod.glac_bin_refreeze
                            output_glac_bin_refreeze_steps = np.append(
                                output_glac_bin_refreeze_steps,
                                output_glac_bin_refreeze_steps_sim[:, :, np.newaxis],
                                axis=2,
                            )

                # ===== Export Results =====
                if count_exceed_boundary_errors < nsims:
                    # ----- STATS OF ALL VARIABLES -----
                    # Output statistics
                    if args.export_all_simiters and nsims > 1:
                        # Instantiate dataset
                        output_stats = output.glacierwide_stats(
                            glacier_rgi_table=glacier_rgi_table,
                            dates_table=dates_table,
                            time_step=pygem_prms['time']['timestep'],
                            nsims=1,
                            sim_climate_name=sim_climate_name,
                            sim_climate_scenario=sim_climate_scenario,
                            realization=realization,
                            modelprms=modelprms,
                            ref_startyear=args.ref_startyear,
                            ref_endyear=ref_endyear,
                            sim_startyear=args.sim_startyear,
                            sim_endyear=args.sim_endyear,
                            option_calibration=args.option_calibration,
                            option_bias_adjustment=args.option_bias_adjustment,
                            option_dynamics=args.option_dynamics,
                            extra_vars=args.export_extra_vars,
                        )
                        for n_iter in range(nsims):
                            # pass model params for iteration and update output dataset model params
                            output_stats.set_modelprms({key: modelprms_all[key][n_iter] for key in modelprms_all})
                            # create and return xarray dataset
                            output_stats.create_xr_ds()
                            output_ds_all_stats = output_stats.get_xr_ds()
                            # fill values
                            output_ds_all_stats['glac_runoff'].values[0, :] = output_glac_runoff_steps[:, n_iter]
                            output_ds_all_stats['glac_area_annual'].values[0, :] = output_glac_area_annual[:, n_iter]
                            output_ds_all_stats['glac_mass_annual'].values[0, :] = output_glac_mass_annual[:, n_iter]
                            output_ds_all_stats['glac_mass_bsl_annual'].values[0, :] = output_glac_mass_bsl_annual[
                                :, n_iter
                            ]
                            output_ds_all_stats['glac_ELA_annual'].values[0, :] = output_glac_ELA_annual[:, n_iter]
                            output_ds_all_stats['offglac_runoff'].values[0, :] = output_offglac_runoff_steps[:, n_iter]
                            if args.export_extra_vars:
                                output_ds_all_stats['glac_temp'].values[0, :] = (
                                    output_glac_temp_steps[:, n_iter] + 273.15
                                )
                                output_ds_all_stats['glac_prec'].values[0, :] = output_glac_prec_steps[:, n_iter]
                                output_ds_all_stats['glac_acc'].values[0, :] = output_glac_acc_steps[:, n_iter]
                                output_ds_all_stats['glac_refreeze'].values[0, :] = output_glac_refreeze_steps[
                                    :, n_iter
                                ]
                                output_ds_all_stats['glac_melt'].values[0, :] = output_glac_melt_steps[:, n_iter]
                                output_ds_all_stats['glac_frontalablation'].values[0, :] = (
                                    output_glac_frontalablation_steps[:, n_iter]
                                )
                                output_ds_all_stats['glac_massbaltotal'].values[0, :] = output_glac_massbaltotal_steps[
                                    :, n_iter
                                ]
                                output_ds_all_stats['glac_snowline'].values[0, :] = output_glac_snowline_steps[
                                    :, n_iter
                                ]
                                output_ds_all_stats['glac_mass_change_ignored_annual'].values[0, :] = (
                                    output_glac_mass_change_ignored_annual[:, n_iter]
                                )
                                output_ds_all_stats['offglac_prec'].values[0, :] = output_offglac_prec_steps[:, n_iter]
                                output_ds_all_stats['offglac_melt'].values[0, :] = output_offglac_melt_steps[:, n_iter]
                                output_ds_all_stats['offglac_refreeze'].values[0, :] = output_offglac_refreeze_steps[
                                    :, n_iter
                                ]
                                output_ds_all_stats['offglac_snowpack'].values[0, :] = output_offglac_snowpack_steps[
                                    :, n_iter
                                ]

                            # export glacierwide stats for iteration
                            output_stats.set_fn(
                                output_stats.get_fn().replace('SETS', f'set{n_iter}') + args.outputfn_sfix + 'all.nc'
                            )
                            output_stats.save_xr_ds()

                    # instantiate dataset for merged simulations
                    output_stats = output.glacierwide_stats(
                        glacier_rgi_table=glacier_rgi_table,
                        dates_table=dates_table,
                        time_step=pygem_prms['time']['timestep'],
                        nsims=nsims,
                        sim_climate_name=sim_climate_name,
                        sim_climate_scenario=sim_climate_scenario,
                        realization=realization,
                        modelprms=modelprms,
                        ref_startyear=args.ref_startyear,
                        ref_endyear=ref_endyear,
                        sim_startyear=args.sim_startyear,
                        sim_endyear=args.sim_endyear,
                        option_calibration=args.option_calibration,
                        option_bias_adjustment=args.option_bias_adjustment,
                        option_dynamics=args.option_dynamics,
                        extra_vars=args.export_extra_vars,
                    )
                    # create and return xarray dataset
                    output_stats.create_xr_ds()
                    output_ds_all_stats = output_stats.get_xr_ds()

                    # get stats from all simulations which will be stored
                    output_glac_runoff_steps_stats = calc_stats_array(output_glac_runoff_steps)
                    output_glac_area_annual_stats = calc_stats_array(output_glac_area_annual)
                    output_glac_mass_annual_stats = calc_stats_array(output_glac_mass_annual)
                    output_glac_mass_bsl_annual_stats = calc_stats_array(output_glac_mass_bsl_annual)
                    output_glac_ELA_annual_stats = calc_stats_array(output_glac_ELA_annual)
                    output_offglac_runoff_steps_stats = calc_stats_array(output_offglac_runoff_steps)
                    if args.export_extra_vars:
                        output_glac_temp_steps_stats = calc_stats_array(output_glac_temp_steps)
                        output_glac_prec_steps_stats = calc_stats_array(output_glac_prec_steps)
                        output_glac_acc_steps_stats = calc_stats_array(output_glac_acc_steps)
                        output_glac_refreeze_steps_stats = calc_stats_array(output_glac_refreeze_steps)
                        output_glac_melt_steps_stats = calc_stats_array(output_glac_melt_steps)
                        output_glac_frontalablation_steps_stats = calc_stats_array(output_glac_frontalablation_steps)
                        output_glac_massbaltotal_steps_stats = calc_stats_array(output_glac_massbaltotal_steps)
                        output_glac_snowline_steps_stats = calc_stats_array(output_glac_snowline_steps)
                        output_glac_mass_change_ignored_annual_stats = calc_stats_array(
                            output_glac_mass_change_ignored_annual
                        )
                        output_offglac_prec_steps_stats = calc_stats_array(output_offglac_prec_steps)
                        output_offglac_melt_steps_stats = calc_stats_array(output_offglac_melt_steps)
                        output_offglac_refreeze_steps_stats = calc_stats_array(output_offglac_refreeze_steps)
                        output_offglac_snowpack_steps_stats = calc_stats_array(output_offglac_snowpack_steps)

                    # output mean/median from all simulations
                    output_ds_all_stats['glac_runoff'].values[0, :] = output_glac_runoff_steps_stats[:, 0]
                    output_ds_all_stats['glac_area_annual'].values[0, :] = output_glac_area_annual_stats[:, 0]
                    output_ds_all_stats['glac_mass_annual'].values[0, :] = output_glac_mass_annual_stats[:, 0]
                    output_ds_all_stats['glac_mass_bsl_annual'].values[0, :] = output_glac_mass_bsl_annual_stats[:, 0]
                    output_ds_all_stats['glac_ELA_annual'].values[0, :] = output_glac_ELA_annual_stats[:, 0]
                    output_ds_all_stats['offglac_runoff'].values[0, :] = output_offglac_runoff_steps_stats[:, 0]
                    if args.export_extra_vars:
                        output_ds_all_stats['glac_temp'].values[0, :] = output_glac_temp_steps_stats[:, 0] + 273.15
                        output_ds_all_stats['glac_prec'].values[0, :] = output_glac_prec_steps_stats[:, 0]
                        output_ds_all_stats['glac_acc'].values[0, :] = output_glac_acc_steps_stats[:, 0]
                        output_ds_all_stats['glac_refreeze'].values[0, :] = output_glac_refreeze_steps_stats[:, 0]
                        output_ds_all_stats['glac_melt'].values[0, :] = output_glac_melt_steps_stats[:, 0]
                        output_ds_all_stats['glac_frontalablation'].values[0, :] = (
                            output_glac_frontalablation_steps_stats[:, 0]
                        )
                        output_ds_all_stats['glac_massbaltotal'].values[0, :] = output_glac_massbaltotal_steps_stats[
                            :, 0
                        ]
                        output_ds_all_stats['glac_snowline'].values[0, :] = output_glac_snowline_steps_stats[:, 0]
                        output_ds_all_stats['glac_mass_change_ignored_annual'].values[0, :] = (
                            output_glac_mass_change_ignored_annual_stats[:, 0]
                        )
                        output_ds_all_stats['offglac_prec'].values[0, :] = output_offglac_prec_steps_stats[:, 0]
                        output_ds_all_stats['offglac_melt'].values[0, :] = output_offglac_melt_steps_stats[:, 0]
                        output_ds_all_stats['offglac_refreeze'].values[0, :] = output_offglac_refreeze_steps_stats[:, 0]
                        output_ds_all_stats['offglac_snowpack'].values[0, :] = output_offglac_snowpack_steps_stats[:, 0]

                    # output median absolute deviation
                    if nsims > 1:
                        output_ds_all_stats['glac_runoff_mad'].values[0, :] = output_glac_runoff_steps_stats[:, 1]
                        output_ds_all_stats['glac_area_annual_mad'].values[0, :] = output_glac_area_annual_stats[:, 1]
                        output_ds_all_stats['glac_mass_annual_mad'].values[0, :] = output_glac_mass_annual_stats[:, 1]
                        output_ds_all_stats['glac_mass_bsl_annual_mad'].values[0, :] = (
                            output_glac_mass_bsl_annual_stats[:, 1]
                        )
                        output_ds_all_stats['glac_ELA_annual_mad'].values[0, :] = output_glac_ELA_annual_stats[:, 1]
                        output_ds_all_stats['offglac_runoff_mad'].values[0, :] = output_offglac_runoff_steps_stats[:, 1]
                        output_ds_all_stats['offglac_runoff_mad'].values[0, :] = output_offglac_runoff_steps_stats[:, 1]
                        if args.export_extra_vars:
                            output_ds_all_stats['glac_temp_mad'].values[0, :] = output_glac_temp_steps_stats[:, 1]
                            output_ds_all_stats['glac_prec_mad'].values[0, :] = output_glac_prec_steps_stats[:, 1]
                            output_ds_all_stats['glac_acc_mad'].values[0, :] = output_glac_acc_steps_stats[:, 1]
                            output_ds_all_stats['glac_refreeze_mad'].values[0, :] = output_glac_refreeze_steps_stats[
                                :, 1
                            ]
                            output_ds_all_stats['glac_melt_mad'].values[0, :] = output_glac_melt_steps_stats[:, 1]
                            output_ds_all_stats['glac_frontalablation_mad'].values[0, :] = (
                                output_glac_frontalablation_steps_stats[:, 1]
                            )
                            output_ds_all_stats['glac_massbaltotal_mad'].values[0, :] = (
                                output_glac_massbaltotal_steps_stats[:, 1]
                            )
                            output_ds_all_stats['glac_snowline_mad'].values[0, :] = output_glac_snowline_steps_stats[
                                :, 1
                            ]
                            output_ds_all_stats['glac_mass_change_ignored_annual_mad'].values[0, :] = (
                                output_glac_mass_change_ignored_annual_stats[:, 1]
                            )
                            output_ds_all_stats['offglac_prec_mad'].values[0, :] = output_offglac_prec_steps_stats[:, 1]
                            output_ds_all_stats['offglac_melt_mad'].values[0, :] = output_offglac_melt_steps_stats[:, 1]
                            output_ds_all_stats['offglac_refreeze_mad'].values[0, :] = (
                                output_offglac_refreeze_steps_stats[:, 1]
                            )
                            output_ds_all_stats['offglac_snowpack_mad'].values[0, :] = (
                                output_offglac_snowpack_steps_stats[:, 1]
                            )

                    # export merged netcdf glacierwide stats
                    output_stats.set_fn(
                        output_stats.get_fn().replace('SETS', f'{nsims}sets') + args.outputfn_sfix + 'all.nc'
                    )
                    output_stats.save_xr_ds()

                    # ----- DECADAL ICE THICKNESS STATS FOR OVERDEEPENINGS -----
                    if (
                        args.export_binned_data
                        and glacier_rgi_table.Area > pygem_prms['sim']['out']['export_binned_area_threshold']
                    ):
                        # Distance from top of glacier downglacier
                        output_glac_bin_dist = np.arange(nfls[0].nx) * nfls[0].dx_meter

                        if args.export_all_simiters and nsims > 1:
                            # Instantiate dataset
                            output_binned = output.binned_stats(
                                glacier_rgi_table=glacier_rgi_table,
                                dates_table=dates_table,
                                time_step=pygem_prms['time']['timestep'],
                                nsims=1,
                                nbins=surface_h_initial.shape[0],
                                binned_components=args.export_binned_components,
                                sim_climate_name=sim_climate_name,
                                sim_climate_scenario=sim_climate_scenario,
                                realization=realization,
                                modelprms=modelprms,
                                ref_startyear=args.ref_startyear,
                                ref_endyear=ref_endyear,
                                sim_startyear=args.sim_startyear,
                                sim_endyear=args.sim_endyear,
                                option_calibration=args.option_calibration,
                                option_bias_adjustment=args.option_bias_adjustment,
                                option_dynamics=args.option_dynamics,
                            )
                            for n_iter in range(nsims):
                                # pass model params for iteration and update output dataset model params
                                output_binned.set_modelprms({key: modelprms_all[key][n_iter] for key in modelprms_all})
                                # create and return xarray dataset
                                output_binned.create_xr_ds()
                                output_ds_binned_stats = output_binned.get_xr_ds()
                                # fill values
                                output_ds_binned_stats['bin_distance'].values[0, :] = output_glac_bin_dist
                                output_ds_binned_stats['bin_surface_h_initial'].values[0, :] = surface_h_initial
                                output_ds_binned_stats['bin_area_annual'].values[0, :, :] = output_glac_bin_area_annual[
                                    :, :, n_iter
                                ]
                                output_ds_binned_stats['bin_mass_annual'].values[0, :, :] = output_glac_bin_mass_annual[
                                    :, :, n_iter
                                ]
                                output_ds_binned_stats['bin_thick_annual'].values[0, :, :] = (
                                    output_glac_bin_icethickness_annual[:, :, n_iter]
                                )
                                output_ds_binned_stats['bin_massbalclim_annual'].values[0, :, :] = (
                                    output_glac_bin_massbalclim_annual[:, :, n_iter]
                                )
                                output_ds_binned_stats['bin_massbalclim'].values[0, :, :] = (
                                    output_glac_bin_massbalclim_steps[:, :, n_iter]
                                )
                                if args.export_binned_components:
                                    output_ds_binned_stats['bin_accumulation'].values[0, :, :] = (
                                        output_glac_bin_acc_steps[:, :, n_iter]
                                    )
                                    output_ds_binned_stats['bin_melt'].values[0, :, :] = output_glac_bin_melt_steps[
                                        :, :, n_iter
                                    ]
                                    output_ds_binned_stats['bin_refreeze'].values[0, :, :] = (
                                        output_glac_bin_refreeze_steps[:, :, n_iter]
                                    )

                                # export binned stats for iteration
                                output_binned.set_fn(
                                    output_binned.get_fn().replace('SETS', f'set{n_iter}')
                                    + args.outputfn_sfix
                                    + 'binned.nc'
                                )
                                output_binned.save_xr_ds()

                        # instantiate dataset for merged simulations
                        output_binned = output.binned_stats(
                            glacier_rgi_table=glacier_rgi_table,
                            dates_table=dates_table,
                            time_step=pygem_prms['time']['timestep'],
                            nsims=nsims,
                            nbins=surface_h_initial.shape[0],
                            binned_components=args.export_binned_components,
                            sim_climate_name=sim_climate_name,
                            sim_climate_scenario=sim_climate_scenario,
                            realization=realization,
                            modelprms=modelprms,
                            ref_startyear=args.ref_startyear,
                            ref_endyear=ref_endyear,
                            sim_startyear=args.sim_startyear,
                            sim_endyear=args.sim_endyear,
                            option_calibration=args.option_calibration,
                            option_bias_adjustment=args.option_bias_adjustment,
                            option_dynamics=args.option_dynamics,
                        )
                        # create and return xarray dataset
                        output_binned.create_xr_ds()
                        output_ds_binned_stats = output_binned.get_xr_ds()

                        # populate dataset with stats from each variable of interest
                        output_ds_binned_stats['bin_distance'].values = output_glac_bin_dist[np.newaxis, :]
                        output_ds_binned_stats['bin_surface_h_initial'].values = surface_h_initial[np.newaxis, :]
                        output_ds_binned_stats['bin_area_annual'].values = np.median(
                            output_glac_bin_area_annual, axis=2
                        )[np.newaxis, :, :]
                        output_ds_binned_stats['bin_mass_annual'].values = np.median(
                            output_glac_bin_mass_annual, axis=2
                        )[np.newaxis, :, :]
                        output_ds_binned_stats['bin_thick_annual'].values = np.median(
                            output_glac_bin_icethickness_annual, axis=2
                        )[np.newaxis, :, :]
                        output_ds_binned_stats['bin_massbalclim_annual'].values = np.median(
                            output_glac_bin_massbalclim_annual, axis=2
                        )[np.newaxis, :, :]
                        output_ds_binned_stats['bin_massbalclim'].values = np.median(
                            output_glac_bin_massbalclim_steps, axis=2
                        )[np.newaxis, :, :]
                        if args.export_binned_components:
                            output_ds_binned_stats['bin_accumulation'].values = np.median(
                                output_glac_bin_acc_steps, axis=2
                            )[np.newaxis, :, :]
                            output_ds_binned_stats['bin_melt'].values = np.median(output_glac_bin_melt_steps, axis=2)[
                                np.newaxis, :, :
                            ]
                            output_ds_binned_stats['bin_refreeze'].values = np.median(
                                output_glac_bin_refreeze_steps, axis=2
                            )[np.newaxis, :, :]
                        if nsims > 1:
                            output_ds_binned_stats['bin_mass_annual_mad'].values = median_abs_deviation(
                                output_glac_bin_mass_annual, axis=2
                            )[np.newaxis, :, :]
                            output_ds_binned_stats['bin_thick_annual_mad'].values = median_abs_deviation(
                                output_glac_bin_icethickness_annual, axis=2
                            )[np.newaxis, :, :]
                            output_ds_binned_stats['bin_massbalclim_annual_mad'].values = median_abs_deviation(
                                output_glac_bin_massbalclim_annual, axis=2
                            )[np.newaxis, :, :]

                        # export merged netcdf glacierwide stats
                        output_binned.set_fn(
                            output_binned.get_fn().replace('SETS', f'{nsims}sets') + args.outputfn_sfix + 'binned.nc'
                        )
                        output_binned.save_xr_ds()

        except Exception as err:
            # LOG FAILURE
            fail_fp = pygem_prms['root'] + '/Output/simulations/failed/' + reg_str + '/' + sim_climate_name + '/'
            if sim_climate_name not in ['ERA5', 'COAWST']:
                fail_fp += sim_climate_scenario + '/'
            if not os.path.exists(fail_fp):
                os.makedirs(fail_fp, exist_ok=True)
            txt_fn_fail = glacier_str + '-sim_failed.txt'
            with open(fail_fp + txt_fn_fail, 'w') as text_file:
                text_file.write(glacier_str + f' failed to complete simulation: {err}')


# %% PARALLEL PROCESSING
def main():
    time_start = time.time()
    parser = getparser()
    args = parser.parse_args()
    # date range check
    try:
        assert args.ref_startyear < args.ref_endyear, (
            f'ref_startyear [{args.ref_startyear}] must be less than ref_endyear [{args.ref_endyear}]'
        )
        assert args.sim_startyear < args.sim_endyear, (
            f'sim_startyear [{args.sim_startyear}] must be less than sim_endyear [{args.sim_endyear}]'
        )
    except AssertionError as err:
        print('error: ', err)
        sys.exit(1)
    # RGI glacier number
    if args.rgi_glac_number:
        glac_no = args.rgi_glac_number
        # format appropriately
        glac_no = [float(g) for g in glac_no]
        glac_no = [f'{g:.5f}' if g >= 10 else f'0{g:.5f}' for g in glac_no]
    elif args.rgi_glac_number_fn is not None:
        with open(args.rgi_glac_number_fn, 'r') as f:
            glac_no = json.load(f)
    else:
        main_glac_rgi_all = modelsetup.selectglaciersrgitable(
            rgi_regionsO1=args.rgi_region01,
            rgi_regionsO2=args.rgi_region02,
            include_landterm=pygem_prms['setup']['include_landterm'],
            include_laketerm=pygem_prms['setup']['include_laketerm'],
            include_tidewater=pygem_prms['setup']['include_tidewater'],
            min_glac_area_km2=pygem_prms['setup']['min_glac_area_km2'],
        )
        glac_no = list(main_glac_rgi_all['rgino_str'].values)

    # Number of cores for parallel processing
    if args.ncores > 1:
        num_cores = int(np.min([len(glac_no), args.ncores]))
    else:
        num_cores = 1

    # Glacier number lists to pass for parallel processing
    glac_no_lsts = modelsetup.split_list(glac_no, n=num_cores, option_ordered=args.option_ordered)

    # Read GCM names from argument parser
    sim_climate_name = args.gcm_list_fn
    if args.sim_climate_name is not None:
        gcm_list = [args.sim_climate_name]
        sim_climate_scenario = args.sim_climate_scenario
    elif args.gcm_list_fn == args.ref_climate_name:
        gcm_list = [args.ref_climate_name]
        sim_climate_scenario = args.sim_climate_scenario
    else:
        with open(args.gcm_list_fn, 'r') as gcm_fn:
            gcm_list = gcm_fn.read().splitlines()
            sim_climate_scenario = os.path.basename(args.gcm_list_fn).split('_')[1]
            print('Found %d gcms to process' % (len(gcm_list)))

    # Read realizations from argument parser
    if args.realization is not None:
        realizations = [args.realization]
    elif args.realization_list is not None:
        with open(args.realization_list, 'r') as real_fn:
            realizations = list(real_fn.read().splitlines())
            print('Found %d realizations to process' % (len(realizations)))
    else:
        realizations = None

    # Producing realization or realization list. Best to convert them into the same format!
    # Then pass this as a list or None.
    # If passing this through the list_packed_vars, then don't go back and get from arg parser again!

    # Loop through all GCMs
    for sim_climate_name in gcm_list:
        if args.sim_climate_scenario is None:
            print('Processing:', sim_climate_name)
        elif args.sim_climate_scenario is not None:
            print('Processing:', sim_climate_name, sim_climate_scenario)
        # Pack variables for multiprocessing
        list_packed_vars = []
        if realizations is not None:
            for realization in realizations:
                for count, glac_no_lst in enumerate(glac_no_lsts):
                    list_packed_vars.append([count, glac_no_lst, sim_climate_name, realization])
        else:
            for count, glac_no_lst in enumerate(glac_no_lsts):
                list_packed_vars.append([count, glac_no_lst, sim_climate_name, realizations])

        print('Processing with ' + str(num_cores) + ' cores...')
        # Parallel processing
        if num_cores > 1:
            with multiprocessing.Pool(num_cores) as p:
                p.map(run, list_packed_vars)
        # If not in parallel, then only should be one loop
        else:
            # Loop through the chunks and export bias adjustments
            for n in range(len(list_packed_vars)):
                run(list_packed_vars[n])

    print('Total processing time:', time.time() - time_start, 's')


if __name__ == '__main__':
    main()
