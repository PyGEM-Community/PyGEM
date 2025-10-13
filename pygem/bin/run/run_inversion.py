import argparse
import json
import os
from functools import partial

import numpy as np
import pandas as pd

# pygem imports
from pygem.setup.config import ConfigManager

# instantiate ConfigManager
config_manager = ConfigManager()
# read the config
pygem_prms = config_manager.read_config()
from oggm import cfg, tasks, workflow

import pygem.pygem_modelsetup as modelsetup
from pygem import class_climate
from pygem.massbalance import PyGEMMassBalance_wrapper

# from pygem.glacierdynamics import MassRedistributionCurveModel
from pygem.oggm_compat import update_cfg
from pygem.shop import debris, mbdata
from pygem.utils._funcs import str2bool

cfg.initialize()
cfg.PATHS['working_dir'] = f'{pygem_prms["root"]}/{pygem_prms["oggm"]["oggm_gdir_relpath"]}'


def run(glac_no, ncores=1, calibrate_regional_glen_a=False, glen_a=None, fs=None, reset_gdirs=False, debug=False):
    """
    Run OGGM's bed inversion for a list of RGI glacier IDs using PyGEM's mass balance model.
    """

    update_cfg({'continue_on_error': True}, 'PARAMS')
    if ncores > 1:
        update_cfg({'use_multiprocessing': True}, 'PARAMS')
        update_cfg({'mp_processes': ncores}, 'PARAMS')

    if not isinstance(glac_no, list):
        glac_no = [glac_no]
    main_glac_rgi = modelsetup.selectglaciersrgitable(glac_no=glac_no)
    # get list of RGIId's for each rgitable being run
    rgiids = main_glac_rgi['RGIId'].tolist()

    # initialize glacier directories
    if reset_gdirs:
        gdirs = workflow.init_glacier_directories(
            rgiids,
            from_prepro_level=2,
            prepro_border=cfg.PARAMS['border'],
            prepro_base_url=pygem_prms['oggm']['base_url'],
            prepro_rgi_version='62',
        )
    else:
        gdirs = workflow.init_glacier_directories(rgiids)

    # PyGEM setup - model datestable, climate data import, prior model parameters
    # model dates
    dt = modelsetup.datesmodelrun(
        startyear=2000, endyear=2019
    )  # will have to cover the time period of inversion (2000-2019) and spinup (1979-~2010 by default). Note, 1940 startyear just in case we want to run spinup starting earlier.
    # load climate data
    ref_clim = class_climate.GCM(name='ERA5')

    # Air temperature [degC]
    temp, _ = ref_clim.importGCMvarnearestneighbor_xarray(
        ref_clim.temp_fn, ref_clim.temp_vn, main_glac_rgi, dt, verbose=debug
    )
    # Precipitation [m]
    prec, _ = ref_clim.importGCMvarnearestneighbor_xarray(
        ref_clim.prec_fn, ref_clim.prec_vn, main_glac_rgi, dt, verbose=debug
    )
    # Elevation [m asl]
    elev = ref_clim.importGCMfxnearestneighbor_xarray(ref_clim.elev_fn, ref_clim.elev_vn, main_glac_rgi)
    # Lapse rate [degC m-1]
    lr, _ = ref_clim.importGCMvarnearestneighbor_xarray(
        ref_clim.lr_fn, ref_clim.lr_vn, main_glac_rgi, dt, verbose=debug
    )

    # load prior regionally averaged modelprms (from Rounce et al. 2023)
    priors_df = pd.read_csv(pygem_prms['root'] + '/Output/calibration/' + pygem_prms['calib']['priors_reg_fn'])

    # loop through gdirs and add `glacier_rgi_table`, `historical_climate`, `dates_table` and `modelprms` attributes to each glacier directory
    for i, gd in enumerate(gdirs):
        # Select subsets of data
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

    #####################
    ### PREPROCESSING ###
    #####################
    task_list = [
        tasks.process_climate_data,  # process climate_hisotrical data to gdir
        mbdata.mb_df_to_gdir,  # process mass balance calibration data to gdir
        debris.debris_to_gdir,  # process debris data to gdir
        debris.debris_binned,  # add debris to inversion flowlines
    ]
    for task in task_list:
        workflow.execute_entity_task(task, gdirs)

    # process mb_calib data from geodetic mass balance
    workflow.execute_entity_task(
        tasks.mb_calibration_from_geodetic_mb,
        gdirs,
        informed_threestep=True,
        overwrite_gdir=True,
    )

    #######################################
    ### CALCULATE APPARENT MASS BALANCE ###
    #######################################
    # note, PyGEMMassBalance_wrapper is passed to `tasks.apparent_mb_from_any_mb` as the `mb_model_class` so that PyGEMs mb model is used for apparent mb
    workflow.execute_entity_task(
        tasks.apparent_mb_from_any_mb,
        gdirs,
        mb_model_class=partial(
            PyGEMMassBalance_wrapper,
            fl_str='inversion_flowlines',
            option_areaconstant=True,
        ),
    )
    # add debris data to flowlines
    workflow.execute_entity_task(debris.debris_binned, gdirs, fl_str='inversion_flowlines')

    ##########################
    ### CALIBRATE GLEN'S A ###
    ##########################
    # fit ice thickness to consensus estimates to find "best" Glen's A
    # note, this runs task.mass_conservation_inversion internally
    if calibrate_regional_glen_a:
        if debug:
            print("Calibrating Glen's A")
        workflow.calibrate_inversion_from_consensus(
            gdirs,
            apply_fs_on_mismatch=True,
            error_on_mismatch=False,  # if you running many glaciers some might not work
            filter_inversion_output=True,  # this partly filters the overdeepening due to
            # the equilibrium assumption for retreating glaciers (see. Figure 5 of Maussion et al. 2019)
            volume_m3_reference=None,  # here you could provide your own total volume estimate in m3
        )

    for gdir in gdirs:
        if calibrate_regional_glen_a:
            glen_a = gdir.get_diagnostics()['inversion_glen_a']
            fs = gdir.get_diagnostics()['inversion_fs']
        else:
            if glen_a is None and fs is None:
                # get glen_a and fs values from prior calibration or manual entry
                if pygem_prms['sim']['oggm_dynamics']['use_regional_glen_a']:
                    glen_a_df = pd.read_csv(
                        f'{pygem_prms["root"]}/{pygem_prms["sim"]["oggm_dynamics"]["glen_a_regional_relpath"]}'
                    )
                    glen_a_O1regions = [int(x) for x in glen_a_df.O1Region.values]
                    assert gdir.glacier_rgi_table.O1Region in glen_a_O1regions, (
                        '{0:0.5f}'.format(gd.glacier_rgi_table['RGIId_float']) + ' O1 region not in glen_a_df'
                    )
                    glen_a_idx = np.where(glen_a_O1regions == gdir.glacier_rgi_table.O1Region)[0][0]
                    glen_a_multiplier = glen_a_df.loc[glen_a_idx, 'glens_a_multiplier']
                    fs = glen_a_df.loc[glen_a_idx, 'fs']
                else:
                    glen_a_multiplier = pygem_prms['sim']['oggm_dynamics']['glen_a_multiplier']
                    fs = pygem_prms['sim']['oggm_dynamics']['fs']
                glen_a = cfg.PARAMS['glen_a'] * glen_a_multiplier

        # non-tidewater
        if gdir.glacier_rgi_table['TermType'] not in [1, 5] or not pygem_prms['setup']['include_frontalablation']:
            if calibrate_regional_glen_a:
                # nothing else to do here - already ran inversion when calibrating Glen's A
                continue

            # run inversion using regionally calibrated Glen's A values
            cfg.PARAMS['use_kcalving_for_inversion'] = False

            ###############################
            ### INVERSION - no calving ###
            ################################
            tasks.prepare_for_inversion(gdir)
            tasks.mass_conservation_inversion(
                gdir,
                glen_a=glen_a,
                fs=fs,
            )

        # tidewater
        else:
            cfg.PARAMS['use_kcalving_for_inversion'] = True

            # Load quality controlled frontal ablation data
            fp = f'{pygem_prms["root"]}/{pygem_prms["calib"]["data"]["frontalablation"]["frontalablation_relpath"]}/analysis/{pygem_prms["calib"]["data"]["frontalablation"]["frontalablation_cal_fn"]}'
            assert os.path.exists(fp), 'Calibrated calving dataset does not exist'
            calving_df = pd.read_csv(fp)
            calving_rgiids = list(calving_df.RGIId)

            # Use calibrated value if individual data available
            if gdir.rgi_id in calving_rgiids:
                calving_idx = calving_rgiids.index(gdir.rgi_id)
                calving_k = calving_df.loc[calving_idx, 'calving_k']
            # Otherwise, use region's median value
            else:
                calving_df['O1Region'] = [int(x.split('-')[1].split('.')[0]) for x in calving_df.RGIId.values]
                calving_df_reg = calving_df.loc[calving_df['O1Region'] == int(gdir.rgi_id[6:8]), :]
                calving_k = np.median(calving_df_reg.calving_k)

            # increase calving line for inversion so that later spinup will work
            cfg.PARAMS['calving_line_extension'] = 120
            # set inversion_calving_k
            cfg.PARAMS['inversion_calving_k'] = calving_k
            if debug:
                print(f'inversion_calving_k = {calving_k}')

            ################################
            ### INVERSION - with calving ###
            ################################
            tasks.find_inversion_calving_from_any_mb(
                gdir,
                mb_model=PyGEMMassBalance_wrapper(
                    gdir,
                    fl_str='inversion_flowlines',
                    option_areaconstant=True,
                ),
                glen_a=glen_a,
                fs=fs,
            )

    ######################
    ### POSTPROCESSING ###
    ######################
    # finally create the dynamic flowlines
    workflow.execute_entity_task(tasks.init_present_time_glacier, gdirs)

    # add debris to model_flowlines
    workflow.execute_entity_task(debris.debris_binned, gdirs, fl_str='model_flowlines')


def main():
    # define ArgumentParser
    parser = argparse.ArgumentParser(
        description="Perform glacier bed inversion (defaults to find best Glen's A for each RGI order 01 region)"
    )
    # add arguments
    parser.add_argument(
        '-rgi_region01',
        type=int,
        default=pygem_prms['setup']['rgi_region01'],
        help='Randoph Glacier Inventory region (can take multiple, e.g. `-run_region01 1 2 3`)',
        nargs='+',
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
        '-calibrate_regional_glen_a',
        type=str2bool,
        default=True,
        help="If True (False) run ice thickness inversion and regionally calibrate (use previously calibrated or user-input) Glen's A values. Default is True",
    )
    parser.add_argument(
        '-glen_a',
        type=float,
        default=None,
        help="User-selected inversion Glen's creep parameter value",
    )
    parser.add_argument(
        '-fs',
        type=float,
        default=None,
        help="User-selected inversion Orleam's sliding factor value",
    )
    parser.add_argument(
        '-ncores',
        action='store',
        type=int,
        default=1,
        help='Number of simultaneous processes (cores) to use',
    )
    parser.add_argument(
        '-reset_gdirs',
        action='store_true',
        help='If True (False) reset OGGM glacier directories. Default is False',
    )
    parser.add_argument('-v', '--debug', action='store_true', help='Flag for debugging')
    args = parser.parse_args()

    # --- Validation logic ---
    if args.calibrate_regional_glen_a:
        if args.glen_a is not None or args.fs is not None:
            parser.error("When '-calibrate_regional_glen_a' is True, '-glen_a' and '-fs' must both be None.")

    # RGI glacier batches
    if args.rgi_region01:
        batches = [
            modelsetup.selectglaciersrgitable(
                rgi_regionsO1=[r01],
                rgi_regionsO2='all',
                include_landterm=pygem_prms['setup']['include_landterm'],
                include_laketerm=pygem_prms['setup']['include_laketerm'],
                include_tidewater=pygem_prms['setup']['include_tidewater'],
                min_glac_area_km2=pygem_prms['setup']['min_glac_area_km2'],
            )['rgino_str'].values.tolist()
            for r01 in args.rgi_region01
        ]
    else:
        batches = None
        if args.rgi_glac_number:
            glac_no = args.rgi_glac_number
            # format appropriately
            glac_no = [float(g) for g in glac_no]
            batches = [f'{g:.5f}' if g >= 10 else f'0{g:.5f}' for g in glac_no]
        elif args.rgi_glac_number_fn is not None:
            with open(args.rgi_glac_number_fn, 'r') as f:
                batches = json.load(f)

    # set up partial function with common arguments
    run_partial = partial(
        run,
        ncores=args.ncores,
        calibrate_regional_glen_a=args.calibrate_regional_glen_a,
        glen_a=args.glen_a,
        fs=args.fs,
        reset_gdirs=args.reset_gdirs,
        debug=args.debug,
    )

    for i, batch in enumerate(batches):
        run_partial(batch)


if __name__ == '__main__':
    main()
