"""
Python Glacier Evolution Model (PyGEM)

copyright © 2018 David Rounce <drounce@cmu.edu>

Distributed under the MIT license
"""

# Built-in libaries
import logging
import os

# External libraries
from datetime import timedelta

import numpy as np
import pandas as pd
from oggm import cfg
from oggm.utils import entity_task

# pygem imports
from pygem.setup.config import ConfigManager

# instantiate ConfigManager
config_manager = ConfigManager()
# read the config
pygem_prms = config_manager.read_config()


# Module logger
log = logging.getLogger(__name__)

# Add the new name "mb_calib_pygem" to the list of things that the GlacierDirectory understands
if 'mb_calib_pygem' not in cfg.BASENAMES:
    cfg.BASENAMES['mb_calib_pygem'] = (
        'mb_calib_pygem.json',
        'Mass balance observations for model calibration',
    )


@entity_task(log, writes=['mb_calib_pygem'])
def mb_df_to_gdir(
    gdir,
    facorrected=pygem_prms['setup']['include_frontalablation'],
):
    """Select specific mass balance and add observations to the given glacier directory

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        where to write the data
    """
    # get dataset filepath
    mbdata_fp = f'{pygem_prms["root"]}/{pygem_prms["calib"]["data"]["massbalance"]["massbalance_relpath"]}'
    mbdata_fp_fa = mbdata_fp + pygem_prms['calib']['data']['massbalance']['massbalance_fn_facorrected_fn']
    if facorrected and os.path.exists(mbdata_fp_fa):
        mbdata_fp = mbdata_fp_fa
    else:
        mbdata_fp = mbdata_fp + pygem_prms['calib']['data']['massbalance']['massbalance_fn']

    assert os.path.exists(mbdata_fp), 'Error, mass balance dataset does not exist: {mbdata_fp}'

    # get column names and formats
    rgiid_cn = pygem_prms['calib']['data']['massbalance']['massbalance_rgiid_colname']
    mb_cn = pygem_prms['calib']['data']['massbalance']['massbalance_mb_colname']
    mberr_cn = pygem_prms['calib']['data']['massbalance']['massbalance_mb_error_colname']
    mb_clim_cn = pygem_prms['calib']['data']['massbalance']['massbalance_mb_clim_colname']
    mberr_clim_cn = pygem_prms['calib']['data']['massbalance']['massbalance_mb_clim_error_colname']
    massbalance_period_colname = pygem_prms['calib']['data']['massbalance']['massbalance_period_colname']
    massbalance_period_date_format = pygem_prms['calib']['data']['massbalance']['massbalance_period_date_format']
    massbalance_period_delimiter = pygem_prms['calib']['data']['massbalance']['massbalance_period_delimiter']

    # read reference mass balance dataset and pull data of interest
    mb_df = pd.read_csv(mbdata_fp)
    mb_df_rgiids = list(mb_df[rgiid_cn])

    if gdir.rgi_id in mb_df_rgiids:
        # RGIId index
        rgiid_idx = np.where(gdir.rgi_id == mb_df[rgiid_cn])[0][0]

        # Glacier-wide mass balance
        mb_mwea = mb_df.loc[rgiid_idx, mb_cn]
        mb_mwea_err = mb_df.loc[rgiid_idx, mberr_cn]

        if (
            mb_clim_cn is not None
            and mberr_clim_cn is not None
            and all(col in mb_df.columns for col in [mb_clim_cn, mberr_clim_cn])
        ):
            mb_clim_mwea = mb_df.loc[rgiid_idx, mb_clim_cn]
            mb_clim_mwea_err = mb_df.loc[rgiid_idx, mberr_clim_cn]
        else:
            mb_clim_mwea = None
            mb_clim_mwea_err = None

        t1_datetime, t2_datetime = parse_period(
            mb_df.loc[rgiid_idx, massbalance_period_colname],
            date_format=massbalance_period_date_format,
            delimiter=massbalance_period_delimiter,
        )

        # remove one day from t2 datetime for proper indexing (ex. 2001-01-01 want to run through 2000-12-31)
        t2_datetime = t2_datetime - timedelta(days=1)
        # Number of years
        nyears = (t2_datetime + timedelta(days=1) - t1_datetime).days / 365.25

        # Record data
        mbdata = {
            key: value
            for key, value in {
                'mb_mwea': float(mb_mwea),
                'mb_mwea_err': float(mb_mwea_err),
                'mb_clim_mwea': float(mb_clim_mwea) if mb_clim_mwea is not None else None,
                'mb_clim_mwea_err': float(mb_clim_mwea_err) if mb_clim_mwea_err is not None else None,
                't1_str': t1_datetime.strftime('%Y-%m-%d'),
                't2_str': t2_datetime.strftime('%Y-%m-%d'),
                'nyears': nyears,
            }.items()
            if value is not None
        }

        gdir.write_json(mbdata, 'mb_calib_pygem')


def parse_period(period_str, date_format=None, delimiter=None):
    """
    parse a period string (e.g. '2000-01-01_2001-01-01') into two datetimes.
    requires a user-specified date_format (e.g. 'YYYY-MM-DD').

    Parameters
    ----------
    period_str : str
        period string to parse
    date_format : str, optional
        the date format to use for parsing (default: None, i.e., try to infer automatically)
    delimiter : str, optional
        the delimiter to use for splitting the period string (default: None, i.e., try common delimiters)
    Returns
    -------
    t1, t2 : pd.Timestamp
        the two parsed datetimes
    """

    if not date_format:
        raise ValueError("Period date_format must be provided (e.g. 'YYYY-MM-DD').")
    if not delimiter:
        raise ValueError("Period delimiter must be provided (e.g. '_').")

    # notmalize user-input formats like YYYY-MM-DD -> %Y-%m-%d
    date_format = date_format.replace('YYYY', '%Y').replace('YY', '%y').replace('MM', '%m').replace('DD', '%d')

    # split and validate
    parts = [p.strip() for p in period_str.split(delimiter)]
    if len(parts) != 2:
        raise ValueError(f"Could not split '{period_str}' into two valid dates using '{delimiter}'.")

    # parse both parts
    try:
        t1 = pd.to_datetime(parts[0], format=date_format)
        t2 = pd.to_datetime(parts[1], format=date_format)
    except Exception as e:
        raise ValueError(f"Failed to parse '{period_str}' with format '{date_format}'") from e

    # ensure t2 > t1
    if t2 <= t1:
        raise ValueError(f"Invalid period '{period_str}': t2 ({t2.date()}) must be later than t1 ({t1.date()}).")

    return t1, t2


# @entity_task(log, writes=['mb_obs'])
# def mb_bins_to_glacierwide(gdir, mb_binned_fp=pygem_prms.mb_binned_fp):
#    """Convert binned mass balance data to glacier-wide and add observations to the given glacier directory
#
#    Parameters
#    ----------
#    gdir : :py:class:`oggm.GlacierDirectory`
#        where to write the data
#    """
#
#    assert os.path.exists(mb_binned_fp), "Error: mb_binned_fp does not exist."
#
#    glac_str_nolead = str(int(gdir.rgi_region)) + '.' + gdir.rgi_id.split('-')[1].split('.')[1]
#
#    # If binned mb data exists, then write to glacier directory
#    if os.path.exists(mb_binned_fp + gdir.rgi_region + '/' + glac_str_nolead + '_mb_bins.csv'):
#        mb_binned_fn = mb_binned_fp + gdir.rgi_region + '/' + glac_str_nolead + '_mb_bins.csv'
#    else:
#        mb_binned_fn = None
#
#    if mb_binned_fn is not None:
#        mbdata_fn = gdir.get_filepath('mb_obs')
#
#        # Glacier-wide mass balance
#        mb_binned_df = pd.read_csv(mb_binned_fn)
#        area_km2_valid = mb_binned_df['z1_bin_area_valid_km2'].sum()
#        mb_mwea = (mb_binned_df['z1_bin_area_valid_km2'] * mb_binned_df['mb_bin_mean_mwea']).sum() / area_km2_valid
#        mb_mwea_err = 0.3
#        t1 = 2000
#        t2 = 2018
#
#        # Record data
#        mbdata = {'mb_mwea': mb_mwea,
#                  'mb_mwea_err': mb_mwea_err,
#                  't1': t1,
#                  't2': t2,
#                  'area_km2_valid': area_km2_valid}
#        with open(mbdata_fn, 'wb') as f:
#            pickle.dump(mbdata, f)
#
#
##%%
# def mb_bins_to_reg_glacierwide(mb_binned_fp=pygem_prms.mb_binned_fp, O1Regions=['01']):
#    # Delete these import
#    mb_binned_fp=pygem_prms.mb_binned_fp
#    O1Regions=['19']
#
#    print('\n\n SPECIFYING UNCERTAINTY AS 0.3 mwea for model development - needs to be updated from mb providers!\n\n')
#    reg_mb_mwea_err = 0.3
#
#    mb_yrfrac_dict = {'01': [2000.419, 2018.419],
#                      '02': [2000.128, 2012],
#                      '03': [2000.419, 2018.419],
#                      '04': [2000.419, 2018.419],
#                      '05': [2000.419, 2018.419],
#                      '06': [2000.419, 2018.419],
#                      '07': [2000.419, 2018.419],
#                      '08': [2000.419, 2018.419],
#                      '09': [2000.419, 2018.419],
#                      '10': [2000.128, 2012],
#                      '11': [2000.128, 2013],
#                      '12': [2000.128, 2012],
#                      'HMA': [2000.419, 2018.419],
#                      '16': [2000.128, 2013.128],
#                      '17': [2000.128, 2013.128],
#                      '18': [2000.128, 2013]}
#
#    for reg in O1Regions:
#        reg_fp = mb_binned_fp + reg + '/'
#
#        main_glac_rgi = modelsetup.selectglaciersrgitable(rgi_regionsO1=[reg], rgi_regionsO2='all', rgi_glac_number='all')
#
#        reg_binned_fns = []
#        for i in os.listdir(reg_fp):
#            if i.endswith('_mb_bins.csv'):
#                reg_binned_fns.append(i)
#        reg_binned_fns = sorted(reg_binned_fns)
#
#        print('Region ' + reg + ' has binned data for ' + str(len(reg_binned_fns)) + ' glaciers.')
#
#        reg_mb_df_cns = ['RGIId', 'O1Region', 'O2Region', 'area_km2', 'mb_mwea', 'mb_mwea_err', 't1', 't2', 'perc_valid']
#        reg_mb_df = pd.DataFrame(np.zeros((main_glac_rgi.shape[0], len(reg_mb_df_cns))), columns=reg_mb_df_cns)
#        reg_mb_df.loc[:,:] = np.nan
#        reg_mb_df.loc[:, 'RGIId'] = main_glac_rgi['RGIId']
#        reg_mb_df.loc[:, 'O1Region'] = main_glac_rgi['O1Region']
#        reg_mb_df.loc[:, 'O2Region'] = main_glac_rgi['O2Region']
#        reg_mb_df.loc[:, 'area_km2'] = main_glac_rgi['Area']
#
#        # Process binned files
#        for nfn, reg_binned_fn in enumerate(reg_binned_fns):
#
#            if nfn%500 == 0:
#                print('  ', nfn, reg_binned_fn)
#
#            mb_binned_df = pd.read_csv(reg_fp + reg_binned_fn)
#            glac_str = reg_binned_fn.split('_')[0]
#            glac_rgiid = 'RGI60-' + glac_str.split('.')[0].zfill(2) + '.' + glac_str.split('.')[1]
#            rgi_idx = np.where(main_glac_rgi['RGIId'] == glac_rgiid)[0][0]
#            area_km2_valid = mb_binned_df['z1_bin_area_valid_km2'].sum()
#            mb_mwea = (mb_binned_df['z1_bin_area_valid_km2'] * mb_binned_df['mb_bin_mean_mwea']).sum() / area_km2_valid
#            mb_mwea_err = reg_mb_mwea_err
#            t1 = mb_yrfrac_dict[reg][0]
#            t2 = mb_yrfrac_dict[reg][1]
#            perc_valid = area_km2_valid / reg_mb_df.loc[rgi_idx,'area_km2'] * 100
#
#            reg_mb_df.loc[rgi_idx,'mb_mwea'] = mb_mwea
#            reg_mb_df.loc[rgi_idx,'mb_mwea_err'] = mb_mwea_err
#            reg_mb_df.loc[rgi_idx,'t1'] = t1
#            reg_mb_df.loc[rgi_idx,'t2'] = t2
#            reg_mb_df.loc[rgi_idx,'perc_valid'] = perc_valid
#
#        #%%
#        # Quality control
#        O2Regions = list(set(list(main_glac_rgi['O2Region'].values)))
#        O2Regions_mb_mwea_dict = {}
#        rgiid_outliers = []
#        for O2Region in O2Regions:
#            reg_mb_df_subset = reg_mb_df[reg_mb_df['O2Region'] == O2Region]
#            reg_mb_df_subset = reg_mb_df_subset.dropna(subset=['mb_mwea'])
#
#            # Use 1.5*IQR to remove outliers
#            reg_mb_mwea_25 = np.percentile(reg_mb_df_subset['mb_mwea'], 25)
#            reg_mb_mwea_50 = np.percentile(reg_mb_df_subset['mb_mwea'], 50)
#            reg_mb_mwea_75 = np.percentile(reg_mb_df_subset['mb_mwea'], 75)
#            reg_mb_mwea_iqr = reg_mb_mwea_75 - reg_mb_mwea_25
#
#            print(np.round(reg_mb_mwea_25,2), np.round(reg_mb_mwea_50,2), np.round(reg_mb_mwea_75,2),
#                  np.round(reg_mb_mwea_iqr,2))
#
#            reg_mb_mwea_bndlow = reg_mb_mwea_25 - 1.5 * reg_mb_mwea_iqr
#            reg_mb_mwea_bndhigh = reg_mb_mwea_75 + 1.5 * reg_mb_mwea_iqr
#
#            # Record RGIIds that are outliers
#            rgiid_outliers.extend(reg_mb_df_subset[(reg_mb_df_subset['mb_mwea'] < reg_mb_mwea_bndlow) |
#                                                   (reg_mb_df_subset['mb_mwea'] > reg_mb_mwea_bndhigh)]['RGIId'].values)
#            # Select non-outliers and record mean
#            reg_mb_df_subset_qc = reg_mb_df_subset[(reg_mb_df_subset['mb_mwea'] >= reg_mb_mwea_bndlow) &
#                                                (reg_mb_df_subset['mb_mwea'] <= reg_mb_mwea_bndhigh)]
#
#            reg_mb_mwea_qc_mean = reg_mb_df_subset_qc['mb_mwea'].mean()
#            O2Regions_mb_mwea_dict[O2Region] = reg_mb_mwea_qc_mean
#
#        #%%
#        print('CREATE DICTIONARY FOR RGIIDs with nan values or those that are outliers')
#    #        print(A['mb_mwea'].mean(), A['mb_mwea'].std(), A['mb_mwea'].min(), A['mb_mwea'].max())
#    #        print(reg_mb_mwea, reg_mb_mwea_std)
#
#
#        #%%
#        reg_mb_fn = reg + '_mb_glacwide_all.csv'
#        reg_mb_df.to_csv(mb_binned_fp + reg_mb_fn, index=False)
#
#        print('TO-DO LIST:')
#        print(' - quality control based on 3-sigma filter like Shean')
#        print(' - extrapolate for missing or outlier glaciers by region')
