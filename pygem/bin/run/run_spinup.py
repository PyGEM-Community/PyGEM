import argparse
import json
import multiprocessing
import os
import warnings
from datetime import datetime
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import binned_statistic

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
from pygem.utils._funcs import interp1d_fill_gaps


# get model monthly deltah
def get_elev_change_1d_hat(gdir):
    # load flowline_diagnostics from spinup
    f = gdir.get_filepath('fl_diagnostics', filesuffix='_dynamic_spinup_pygem_mb')
    with xr.open_dataset(f, group='fl_0') as ds_spn:
        ds_spn = ds_spn.load()

    # get binned surface area at spinup target year
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        area = np.where(
            ds_spn.sel(time=gdir.rgi_date + 1).thickness_m.values > 0,
            ds_spn.sel(time=gdir.rgi_date + 1).volume_m3.values / ds_spn.sel(time=gdir.rgi_date + 1).thickness_m.values,
            0,
        )

    thickness_m = ds_spn.thickness_m.values.T  # glacier thickness [m ice], (nbins, nyears)

    # set any < 0 thickness to nan
    thickness_m[thickness_m <= 0] = np.nan

    # climatic mass balance
    dotb_monthly = np.repeat(ds_spn.climatic_mb_myr.values.T[:, 1:] / 12, 12, axis=-1)

    # convert to m ice
    dotb_monthly = dotb_monthly * (pygem_prms['constants']['density_water'] / pygem_prms['constants']['density_ice'])
    ### to get monthly thickness and mass we require monthly flux divergence ###
    # we'll assume the flux divergence is constant througohut the year (is this a good assumption?)
    # ie. take annual values and divide by 12 - use numpy repeat to repeat values across 12 months
    flux_div_monthly_mmo = np.repeat(-ds_spn.flux_divergence_myr.values.T[:, 1:] / 12, 12, axis=-1)
    # get monthly binned change in thickness
    delta_h_monthly = dotb_monthly - flux_div_monthly_mmo  # [m ice per month]

    # get binned monthly thickness = running thickness change + initial thickness
    running_delta_h_monthly = np.cumsum(delta_h_monthly, axis=-1)
    h_monthly = running_delta_h_monthly + thickness_m[:, 0][:, np.newaxis]

    # get surface height at the specified reference year
    ref_surface_h = ds_spn.bed_h.values + ds_spn.thickness_m.sel(time=gdir.elev_change_1d['ref_dem_year']).values

    # aggregate model bin thicknesses as desired
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        h_monthly = np.column_stack(
            [
                binned_statistic(
                    x=ref_surface_h, values=x, statistic=np.nanmean, bins=gdir.elev_change_1d['bin_edges']
                )[0]
                for x in h_monthly.T
            ]
        )

    # interpolate over any empty bins
    h_monthly_ = np.column_stack([interp1d_fill_gaps(x.copy()) for x in h_monthly.T])

    # difference desired time steps, return np.nan array for any missing inds
    dh = np.column_stack(
        [
            h_monthly_[:, j] - h_monthly_[:, i]
            if (i is not None and j is not None and 0 <= i < h_monthly_.shape[1] and 0 <= j < h_monthly_.shape[1])
            else np.full(h_monthly_.shape[0], np.nan)
            for i, j in gdir.elev_change_1d['model2obs_inds_map']
        ]
    )
    return dh, ds_spn.dis_along_flowline.values, area


def loss_with_penalty(x, obs, mod, threshold=100, weight=1.0):
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        # MAE where observations exist
        mismatch = np.nanmean(np.abs(mod - obs))

        # Penalty: positive modeled values below threshold
        mask = x < threshold
        mod_sub = mod[mask]

        # keep only positives
        positives = np.clip(mod_sub, a_min=0, a_max=None)

        # add to loss (scales with mean positive magnitude)
        penalty = weight * np.nanmean(positives)

    return mismatch + penalty


# run spinup function
def run_spinup(gd, **kwargs):
    out = workflow.execute_entity_task(
        tasks.run_dynamic_spinup,
        gd,
        minimise_for='area',
        output_filesuffix='_dynamic_spinup_pygem_mb',
        store_fl_diagnostics=True,
        store_model_geometry=True,
        mb_model_historical=PyGEMMassBalance_wrapper(gd, fl_str='model_flowlines'),
        ignore_errors=False,
        **kwargs,
    )
    return out


def run(glacno_list, mb_model_params, optimize=False, periods2try=[20], outdir=None, debug=False, ncores=1, **kwargs):
    # remove any None kwargs
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    main_glac_rgi = modelsetup.selectglaciersrgitable(glac_no=glacno_list)

    # model dates - define model dates table that covers time span of interest
    if optimize:
        sy = 1940
    else:
        sy = 2000 - kwargs.get('spinup_period', 20)
    dt = modelsetup.datesmodelrun(startyear=sy, endyear=kwargs.get('ye', pygem_prms['climate']['ref_endyear']))

    # Load climate data
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
    for i, glac_no in enumerate(glacno_list):
        try:
            glacier_rgi_table = main_glac_rgi.loc[main_glac_rgi.index.values[i], :]
            glacier_str = '{0:0.5f}'.format(glacier_rgi_table['RGIId_float'])
            # instantiate glacier directory
            if glacier_rgi_table['TermType'] not in [1, 5] or not pygem_prms['setup']['include_frontalablation']:
                gd = single_flowline_glacier_directory(glacier_str, reset=False)
                gd.is_tidewater = False
            else:
                gd = single_flowline_glacier_directory_with_calving(glacier_str, reset=False)
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

            # ensure `ye` >= `target_year`
            ty = kwargs.get('target_year', gd.rgi_date + 1)
            ye = kwargs.get('ye', ty)
            if ye < ty:
                raise ValueError(f'Spinup end year (ye={ye}) must be greater than target year (target_yr={ty}):')

            if optimize:
                # model ela - take median val
                gd.ela = tasks.compute_ela(gd, years=[v for v in gd.dates_table.year.unique() if v <= 2019]).median()
                # load elevation change data
                if os.path.isfile(gd.get_filepath('elev_change_1d')):
                    gd.elev_change_1d = gd.read_json('elev_change_1d')
                else:
                    gd.elev_change_1d = None

            ############################
            ####### model params #######
            ############################
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
                    pygem_prms['root'] + '/Output/calibration/' + glacier_str.split('.')[0].zfill(2) + '/'
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
            ############################

            # update cfg.PARAMS
            update_cfg({'continue_on_error': True}, 'PARAMS')
            update_cfg({'store_model_geometry': True}, 'PARAMS')

            # optimize against binned dhdt data
            spinup_period = None
            if (optimize) and (gd.elev_change_1d is not None):
                # get number of years between surveys to normalize by time
                gd.elev_change_1d['nyrs'] = []
                for start, end in gd.elev_change_1d['dates']:
                    start_dt = datetime.strptime(start, '%Y-%m-%d')
                    end_dt = datetime.strptime(end, '%Y-%m-%d')
                    gd.elev_change_1d['nyrs'].append((end_dt - start_dt).days / 365.25)
                gd.elev_change_1d['dhdt'] = np.column_stack(gd.elev_change_1d['dh']) / gd.elev_change_1d['nyrs']
                # define minimum spinup start year
                min_start_yr = min(2000, *(int(date[:4]) for pair in gd.elev_change_1d['dates'] for date in pair))

                results = {}  # instantiate output dictionary
                fig, ax = plt.subplots(1)  # instantiate figure

                # objective function to evaluate
                def _objective(**kwargs):
                    fls = run_spinup(gd, **kwargs)
                    if fls[0] is None:
                        return kwargs['spinup_period'], float('inf'), None

                    # get true spinup period (note, if initial fails, oggm tries period/2)
                    spinup_period_ = gd.rgi_date + 1 - fls[0].y0

                    # create lookup dict (timestamp → index)
                    dtable = modelsetup.datesmodelrun(startyear=fls[0].y0, endyear=kwargs['ye'])
                    date_to_index = {d: i for i, d in enumerate(dtable['date'])}
                    gd.elev_change_1d['model2obs_inds_map'] = [
                        (
                            date_to_index.get(pd.to_datetime(start)),
                            date_to_index.get(pd.to_datetime(end)),
                        )
                        for start, end in gd.elev_change_1d['dates']
                    ]

                    dh_hat, dist, bin_area = get_elev_change_1d_hat(gd)
                    dhdt_hat = dh_hat / gd.elev_change_1d['nyrs']

                    # plot binned surface area
                    ax.plot(dist, bin_area, label=f'{spinup_period} years: {round(1e-6 * np.sum(bin_area), 1)} km$^2$')

                    # penalize positive values below specified elevation threshold
                    loss = loss_with_penalty(
                        gd.elev_change_1d['bin_centers'], gd.elev_change_1d['dhdt'], dhdt_hat, gd.ela
                    )

                    return spinup_period_, loss, dhdt_hat

                # evaluate candidate spinup periods
                for p in periods2try:
                    kwargs['spinup_period'] = p
                    p_, mismatch, model = _objective(**kwargs)
                    results[p_] = (mismatch, model)

                # find best
                best_period = min(results, key=lambda k: results[k][0])
                best_value, best_model = results[best_period]
                # update kwarg
                kwargs['spinup_period'] = best_period
                # ensure spinup start year <= min_start_yr
                if gd.rgi_date + 1 - best_period > min_start_yr:
                    kwargs['spinup_start_yr'] = min_start_yr
                    kwargs.pop('spinup_period')
                    p_, best_value, best_model = _objective(**kwargs)
                    results[p_] = (mismatch, model)
                    best_period = gd.rgi_date + 1 - min_start_yr

                if debug:
                    print('All results:', {k: v[0] for k, v in results.items()})
                    print(f'Best spinup_period = {best_period}, mismatch = {best_value}')

                # find worst - ignore failed runs
                worst_period = max(
                    (k for k in results if results[k][0] != float('inf')),
                    key=lambda k: results[k][0]
                )
                worst_value, worst_model = results[worst_period]

                ############################
                ### diagnostics plotting ###
                ############################
                # binned area
                ax.legend()
                ax.set_title(gd.rgi_id)
                ax.set_xlabel('distance along flowline (m)')
                ax.set_ylabel('surface area (m$^2$)')
                ax.set_xlim([0, ax.get_xlim()[1]])
                ax.set_ylim([0, ax.get_ylim()[1]])
                fig.tight_layout()
                if debug and ncores == 1:
                    plt.show()
                if outdir:
                    fig.savefig(f'{outdir}/{glac_no}-spinup_binned_area.png', dpi=300)
                plt.close()

                # 1d elevation change
                labels = [
                    (f'{start[:-2].replace("-", "")}:{end[:-3].replace("-", "")}')
                    for start, end in gd.elev_change_1d['dates']
                ]
                fig, ax = plt.subplots(figsize=(8, 5))

                for t in range(gd.elev_change_1d['dhdt'].shape[1]):
                    # plot Obs first, grab the color
                    (line,) = ax.plot(
                        gd.elev_change_1d['bin_centers'],
                        gd.elev_change_1d['dhdt'][:, t],
                        linestyle='-',
                        marker='.',
                        label=labels[t],
                    )
                    color = line.get_color()

                    # plot Best model with same color
                    ax.plot(
                        gd.elev_change_1d['bin_centers'],
                        best_model[:, t],
                        linestyle='--',
                        marker='.',
                        color=color,
                    )

                    # plot Worst model with same color
                    ax.plot(
                        gd.elev_change_1d['bin_centers'],
                        worst_model[:, t],
                        linestyle=':',
                        marker='.',
                        color=color,
                    )
                ax.axvline(gd.ela, c='grey', ls=':')
                ax.axhline(0, c='grey', ls='-')
                ax.plot([], [], 'k--', label=r'$\hat{best}$')
                ax.plot([], [], 'k:', label=r'$\hat{worst}$')
                ax.set_xlabel('elevation (m)')
                ax.set_ylabel(r'elevation change (m yr$^{-1}$)')
                ax.set_title(
                    f'{glac_no}\nBest={best_period} (mismatch={best_value:.3f}), '
                    f'Worst={worst_period} (mismatch={worst_value:.3f})'
                )
                ax.legend(handlelength=1, borderaxespad=0, fancybox=False)
                # plot area
                if 'bin_area' in gd.elev_change_1d:
                    area = np.array(gd.elev_change_1d['bin_area'])
                    area_mask = area > 0
                    ax2 = ax.twinx()  # shares x-axis
                    ax2.fill_between(
                        np.array(gd.elev_change_1d['bin_centers'])[area_mask],
                        0,
                        area[area_mask],
                        color='gray',
                        alpha=0.1,
                    )
                    ax2.set_ylim([0, ax2.get_ylim()[1]])
                    ax2.set_ylabel(r'area (m $^{2}$)', color='gray')
                    ax2.tick_params(axis='y', colors='gray')
                    ax2.spines['right'].set_color('gray')
                    ax2.yaxis.label.set_color('gray')
                fig.tight_layout()
                if debug and ncores == 1:
                    plt.show()
                if outdir:
                    fig.savefig(f'{outdir}/{glac_no}-spinup_optimization.png', dpi=300)
                plt.close()
                ############################

            # update spinup_period if optimized or specified as CLI argument, else remove kwarg and use OGGM default
            run_spinup(gd, **kwargs)

        except Exception as e:
            print(f'Error processing glacier {glac_no}: {e}')
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
    parser.add_argument('-target_yr', type=int, default=None)
    parser.add_argument('-ye', type=int, default=None)
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
        default='emulator',
        choices=['regional_priors', 'emulator'],
        help='Which mass balance model parameters to use ("regional_priors" or "emulator")',
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        '-spinup_period',
        type=int,
        default=None,
        help='Fixed spinup period (years). If not provided, OGGM default is used.',
    )
    group.add_argument(
        '-optimize',
        action='store_true',
        help=(
            'Optimize the spinup_period by minimizing against elevation change data. '
            'This goes through spinup periods of [20,30,40,50,60] years and finds the one '
            'that gives the best fit to any available 1d elevation change data.'
        ),
    )
    parser.add_argument(
        '-periods2try',
        type=int,
        nargs='+',
        default=[20, 30, 40, 50, 60],
        help=(
            'Optional list of spinup periods (years) to test if -optimize is used. '
            'Ignored otherwise. Example: -periods2try 20 30 40 50 60'
        ),
    )
    parser.add_argument(
        '-outdir', type=str, default=None, help='Directory to store any ouputs (diagnostic figures, etc.)'
    )
    parser.add_argument('-v', '--debug', action='store_true', help='Flag for debugging')
    args = parser.parse_args()

    # --- Validation logic ---
    if args.optimize and args.ye is None:
        parser.error("When '-optimize' is True, must specify `ye` (spinup end year).")

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
        raise ValueError('Need to specify either -rgi_glac_number or -rgi_glac_number_fn')

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
        optimize=args.optimize,
        periods2try=args.periods2try,
        outdir=args.outdir,
        debug=args.debug,
        ncores=ncores,
        mb_model_params=args.mb_model_params,
        target_yr=args.target_yr,
        spinup_period=args.spinup_period,
        ye=args.ye,
    )

    # parallel processing
    print(f'Processing with {ncores} cores... \n{glac_no_lsts}')
    with multiprocessing.Pool(ncores) as p:
        p.map(run_partial, glac_no_lsts)


if __name__ == '__main__':
    main()
