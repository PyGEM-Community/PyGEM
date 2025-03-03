"""
Python Glacier Evolution Model (PyGEM)

copyright © 2018 David Rounce <drounce@cmu.edu>

Distrubted under the MIT lisence
"""
# Built-in libaries
import argparse
import os
import json
import logging
# External libraries
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import pickle
#import rasterio
#import xarray as xr
# Local libraries
from oggm import cfg
from oggm.utils import entity_task
#from oggm.core.gis import rasterio_to_gdir
#from oggm.utils import ncDataset
# pygem imports
from pygem.setup.config import ConfigManager
# instantiate ConfigManager
config_manager = ConfigManager()
# read the config
pygem_prms = config_manager.read_config()
import pygem.pygem_modelsetup as modelsetup


# Module logger
log = logging.getLogger(__name__)

# Add the new name "mb_calib_pygem" to the list of things that the GlacierDirectory understands
if not 'mb_calib_pygem' in cfg.BASENAMES:
    cfg.BASENAMES['mb_calib_pygem'] = ('mb_calib_pygem.json', 'Mass balance observations for model calibration')

    
@entity_task(log, writes=['mb_calib_pygem'])
def mb_df_to_gdir(gdir, mb_dataset='Hugonnet2021', facorrected=pygem_prms['setup']['include_frontalablation']):
    """Select specific mass balance and add observations to the given glacier directory
    
    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        where to write the data
    """
    # get dataset name (could potentially be swapped with others besides Hugonnet21)
    mbdata_fp = f"{pygem_prms['root']}/{pygem_prms['calib']['data']['massbalance']['hugonnet2021_relpath']}"
    mbdata_fp_fa = mbdata_fp + pygem_prms['calib']['data']['massbalance']['hugonnet2021_facorrected_fn']
    if facorrected and os.path.exists(mbdata_fp_fa):
        mbdata_fp = mbdata_fp_fa
    else:
        mbdata_fp = mbdata_fp + pygem_prms['calib']['data']['massbalance']['hugonnet2021_fn']

    assert os.path.exists(mbdata_fp), "Error, mass balance dataset does not exist: {mbdata_fp}"
    assert 'hugonnet2021' in mbdata_fp.lower(), "Error, mass balance dataset not yet supported: {mbdata_fp}"
    rgiid_cn = 'rgiid'
    mb_cn = 'mb_mwea'
    mberr_cn = 'mb_mwea_err'
    mb_clim_cn = 'mb_clim_mwea'
    mberr_clim_cn = 'mb_clim_mwea_err'
    
    # read reference mass balance dataset and pull data of interest
    mb_df = pd.read_csv(mbdata_fp)
    mb_df_rgiids = list(mb_df[rgiid_cn])

    if gdir.rgi_id in mb_df_rgiids:
        # RGIId index
        rgiid_idx = np.where(gdir.rgi_id == mb_df[rgiid_cn])[0][0]
        
        # Glacier-wide mass balance
        mb_mwea = mb_df.loc[rgiid_idx, mb_cn]
        mb_mwea_err = mb_df.loc[rgiid_idx, mberr_cn]
        
        if mb_clim_cn in mb_df.columns:
            mb_clim_mwea = mb_df.loc[rgiid_idx, mb_clim_cn]
            mb_clim_mwea_err = mb_df.loc[rgiid_idx, mberr_clim_cn]
        else:
            mb_clim_mwea = None
            mb_clim_mwea_err = None
        
        t1_str, t2_str = mb_df.loc[rgiid_idx, 'period'].split('_')        
        t1_datetime = pd.to_datetime(t1_str)
        t2_datetime = pd.to_datetime(t2_str)

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
                't1_str': t1_str,
                't2_str': t2_str,
                'nyears': nyears,
            }.items()
            if value is not None
        }
        mb_fn = gdir.get_filepath('mb_calib_pygem')
        with open(mb_fn, 'w') as f:
            json.dump(mbdata, f)


#@entity_task(log, writes=['mb_obs'])
#def mb_bins_to_glacierwide(gdir, mb_binned_fp=pygem_prms.mb_binned_fp):
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
#def mb_bins_to_reg_glacierwide(mb_binned_fp=pygem_prms.mb_binned_fp, O1Regions=['01']):
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
        
        
