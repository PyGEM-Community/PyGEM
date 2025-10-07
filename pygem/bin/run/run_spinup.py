import argparse
import json
import multiprocessing
from functools import partial

import numpy as np
import pandas as pd

# pygem imports
from pygem.setup.config import ConfigManager

# instantiate ConfigManager
config_manager = ConfigManager()
# read the config
pygem_prms = config_manager.read_config()
from oggm import tasks, workflow

import pygem.pygem_modelsetup as modelsetup
from pygem import class_climate
from pygem.massbalance import PyGEMMassBalance_wrapper

# from pygem.glacierdynamics import MassRedistributionCurveModel
from pygem.oggm_compat import (
    single_flowline_glacier_directory,
    single_flowline_glacier_directory_with_calving,
    update_cfg,
)


def run(glacno_list, mb_model_params, debug=False, **kwargs):
    # remove None kwargs
    kwargs = {k: v for k, v in kwargs.items() if v is not None}

    main_glac_rgi = modelsetup.selectglaciersrgitable(glac_no=glacno_list)
    # model dates
    dt = modelsetup.datesmodelrun(startyear=1979, endyear=2019)
    # load climate data
    ref_clim = class_climate.GCM(name='ERA5')

    # Air temperature [degC]
    temp, _ = ref_clim.importGCMvarnearestneighbor_xarray(
        ref_clim.temp_fn, ref_clim.temp_vn, main_glac_rgi, dt
    )
    # Precipitation [m]
    prec, _ = ref_clim.importGCMvarnearestneighbor_xarray(
        ref_clim.prec_fn, ref_clim.prec_vn, main_glac_rgi, dt
    )
    # Elevation [m asl]
    elev = ref_clim.importGCMfxnearestneighbor_xarray(
        ref_clim.elev_fn, ref_clim.elev_vn, main_glac_rgi
    )
    # Lapse rate [degC m-1]
    lr, _ = ref_clim.importGCMvarnearestneighbor_xarray(
        ref_clim.lr_fn, ref_clim.lr_vn, main_glac_rgi, dt
    )

    # load prior regionally averaged modelprms (from Rounce et al. 2023)
    priors_df = pd.read_csv(
        pygem_prms['root']
        + '/Output/calibration/'
        + pygem_prms['calib']['priors_reg_fn']
    )

    # loop through gdirs and add `glacier_rgi_table`, `historical_climate`, `dates_table` and `modelprms` attributes to each glacier directory
    for i, glaco in enumerate(glacno_list):
        try:
            glacier_rgi_table = main_glac_rgi.loc[main_glac_rgi.index.values[i], :]
            glacier_str = '{0:0.5f}'.format(glacier_rgi_table['RGIId_float'])
            # instantiate glacier directory
            if (
                glacier_rgi_table['TermType'] not in [1, 5]
                or not pygem_prms['setup']['include_frontalablation']
            ):
                gd = single_flowline_glacier_directory(glacier_str, reset=False)
                gd.is_tidewater = False
            else:
                gd = single_flowline_glacier_directory_with_calving(
                    glacier_str, reset=False
                )
                gd.is_tidewater = True

            # Select subsets of data
            gd.glacier_rgi_table = glacier_rgi_table
            gd.glacier_rgi_table = main_glac_rgi.loc[main_glac_rgi.index.values[i], :]
            # Add climate data to glacier directory (first inversion data)
            gd.historical_climate = {
                'elev': elev[i],
                'temp': temp[i, :],
                'tempstd': np.zeros(temp[i, :].shape),
                'prec': prec[i, :],
                'lr': lr[i, :],
            }
            gd.dates_table = dt

            if mb_model_params == 'regional_priors':
                # get modelprms from regional priors
                priors_idx = np.where(
                    (priors_df.O1Region == gd.glacier_rgi_table['O1Region'])
                    & (priors_df.O2Region == gd.glacier_rgi_table['O2Region'])
                )[0][0]
                tbias_mu = float(priors_df.loc[priors_idx, 'tbias_mean'])
                kp_mu = float(priors_df.loc[priors_idx, 'kp_mean'])
                gd.modelprms = {
                    'kp': kp_mu,
                    'tbias': tbias_mu,
                    'ddfsnow': pygem_prms['calib']['MCMC_params']['ddfsnow_mu'],
                    'ddfice': pygem_prms['calib']['MCMC_params']['ddfsnow_mu']
                    / pygem_prms['sim']['params']['ddfsnow_iceratio'],
                    'precgrad': pygem_prms['sim']['params']['precgrad'],
                    'tsnow_threshold': pygem_prms['sim']['params']['tsnow_threshold'],
                }
            elif mb_model_params == 'emulator':
                # get modelprms from emulator mass balance calibration
                modelprms_fn = glacier_str + '-modelprms_dict.json'
                modelprms_fp = (
                    pygem_prms['root']
                    + '/Output/calibration/'
                    + glacier_str.split('.')[0].zfill(2)
                    + '/'
                ) + modelprms_fn
                with open(modelprms_fp, 'r') as f:
                    modelprms_dict = json.load(f)

                modelprms_all = modelprms_dict['emulator']
                gd.modelprms = {
                    'kp': modelprms_all['kp'][0],
                    'tbias': modelprms_all['tbias'][0],
                    'ddfsnow': modelprms_all['ddfsnow'][0],
                    'ddfice': modelprms_all['ddfice'][0],
                    'tsnow_threshold': modelprms_all['tsnow_threshold'][0],
                    'precgrad': modelprms_all['precgrad'][0],
                }

            # update cfg.PARAMS
            update_cfg({'continue_on_error': True}, 'PARAMS')
            update_cfg({'store_model_geometry': True}, 'PARAMS')

            # perform OGGM dynamic spinup
            workflow.execute_entity_task(
                tasks.run_dynamic_spinup,
                gd,
                # spinup_start_yr=spinup_start_yr,  # When to start the spinup
                minimise_for='area',  # what target to match at the RGI date
                # target_yr=target_yr, # The year at which we want to match area or volume. If None, gdir.rgi_date + 1 is used (the default)
                # ye=ye,  # When the simulation should stop
                output_filesuffix='_dynamic_spinup_pygem_mb',
                store_fl_diagnostics=True,
                store_model_geometry=True,
                # first_guess_t_spinup = , could be passed as input argument for each step in the sampler based on prior tbias, current default first guess is -2
                mb_model_historical=PyGEMMassBalance_wrapper(
                    gd, fl_str='model_flowlines'
                ),
                ignore_errors=False,
                **kwargs,
            )

        except Exception as e:
            print(f'Error processing glacier {glaco}: {e}')
            # continue to next glacier
            continue


def main():
    # define ArgumentParser
    parser = argparse.ArgumentParser(description='perform dynamical spinup')
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
    (
        parser.add_argument(
            '-rgi_glac_number_fn',
            action='store',
            type=str,
            default=None,
            help='Filepath containing list of rgi_glac_number, helpful for running batches on spc',
        ),
    )
    parser.add_argument(
        '-spinup_period',
        type=int,
        default=None,
        help='Fixed spinup period (years). If not provided, OGGM default is used.',
    )
    parser.add_argument('-target_yr', type=int, default=None)
    parser.add_argument('-ye', type=int, default=2020)
    parser.add_argument(
        '-ncores',
        action='store',
        type=int,
        default=1,
        help='Number of simultaneous processes (cores) to use',
    )
    parser.add_argument(
        '-mb_model_params',
        type=str,
        default='regional_priors',
        choices=['regional_priors', 'emulator'],
        help='Which mass balance model parameters to use ("regional_priors" or "emulator")',
    )
    args = parser.parse_args()

    # RGI glacier number
    glac_no = None
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

    if glac_no is None:
        raise ValueError(
            'Need to specify either -rgi_glac_number or -rgi_glac_number_fn'
        )

    # number of cores for parallel processing
    if args.ncores > 1:
        ncores = int(np.min([len(glac_no), args.ncores]))
    else:
        ncores = 1

    # Glacier number lists to pass for parallel processing
    glac_no_lsts = modelsetup.split_list(glac_no, n=ncores)

    # set up partial function with debug argument
    run_partial = partial(
        run,
        mb_model_params=args.mb_model_params,
        target_yr=args.target_yr,
        spinup_period=args.spinup_period,
    )
    # parallel processing
    print(f'Processing with {ncores} cores... \n{glac_no_lsts}')
    with multiprocessing.Pool(ncores) as p:
        p.map(run_partial, glac_no_lsts)


if __name__ == '__main__':
    main()
