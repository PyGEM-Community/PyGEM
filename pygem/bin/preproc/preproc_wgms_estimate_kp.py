"""
Python Glacier Evolution Model (PyGEM)

copyright © 2018 David Rounce <drounce@cmu.edu>

Distrubted under the MIT lisence

Process the WGMS data to connect with RGIIds and evaluate potential precipitation biases

This is somewhat of a legacy script, since it is hardcoded and relies on outdated RGI and WGMS data

"""
# Built-in libraries
import argparse
import os
import sys
# External libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from scipy.stats import median_abs_deviation
# pygem imports
from pygem import class_climate
from pygem.setup.config import ConfigManager
# instantiate ConfigManager
config_manager = ConfigManager()
# read the config
pygem_prms = config_manager.read_config()
import pygem.pygem_modelsetup as modelsetup


def subset_winter(wgms_eee_fp='', wgms_ee_fp='', wgms_e_fp='',  wgms_id_fp='', wgms_ee_winter_fp='', wgms_ee_winter_fp_subset='', subset_time_value=20000000):
    """
    subset winter mass balance data from WGMS
    """
    # Load data 
    wgms_e_df = pd.read_csv(wgms_e_fp, encoding='unicode_escape')
    wgms_ee_df_raw = pd.read_csv(wgms_ee_fp, encoding='unicode_escape')
    wgms_eee_df_raw = pd.read_csv(wgms_eee_fp, encoding='unicode_escape')
    wgms_id_df = pd.read_csv(wgms_id_fp, encoding='unicode_escape')
    
    # Map dictionary
    wgms_id_dict = dict(zip(wgms_id_df.WGMS_ID, wgms_id_df.RGI_ID))
    wgms_ee_df_raw['rgiid_raw'] = wgms_ee_df_raw.WGMS_ID.map(wgms_id_dict)
    wgms_ee_df_raw = wgms_ee_df_raw.dropna(subset=['rgiid_raw'])
    wgms_eee_df_raw['rgiid_raw'] = wgms_eee_df_raw.WGMS_ID.map(wgms_id_dict)
    wgms_eee_df_raw = wgms_eee_df_raw.dropna(subset=['rgiid_raw'])
    
    # Link RGIv5.0 with RGIv6.0
    rgi60_fp = pygem_prms['root'] +  '/RGI/rgi60/00_rgi60_attribs/'
    rgi50_fp = pygem_prms['root'] +  '/RGI/00_rgi50_attribs/'
    
    # Process each region
    regions_str = ['01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18']
    rgi60_df = None
    rgi50_df = None
    for reg_str in regions_str:
        # RGI60 data
        for i in os.listdir(rgi60_fp):
            if i.startswith(reg_str) and i.endswith('.csv'):
                rgi60_df_reg = pd.read_csv(rgi60_fp + i, encoding='unicode_escape')
        # append datasets
        if rgi60_df is None:
            rgi60_df = rgi60_df_reg
        else:
            rgi60_df = pd.concat([rgi60_df, rgi60_df_reg], axis=0)
    
        # RGI50 data
        for i in os.listdir(rgi50_fp):
            if i.startswith(reg_str) and i.endswith('.csv'):
                rgi50_df_reg = pd.read_csv(rgi50_fp + i, encoding='unicode_escape')
        # append datasets
        if rgi50_df is None:
            rgi50_df = rgi50_df_reg
        else:
            rgi50_df = pd.concat([rgi50_df, rgi50_df_reg], axis=0)
        
    # Merge based on GLIMSID
    glims_rgi50_dict = dict(zip(rgi50_df.GLIMSId, rgi50_df.RGIId))
    rgi60_df['RGIId_50'] = rgi60_df.GLIMSId.map(glims_rgi50_dict)
    rgi60_df_4dict = rgi60_df.dropna(subset=['RGIId_50'])
    rgi50_rgi60_dict = dict(zip(rgi60_df_4dict.RGIId_50, rgi60_df_4dict.RGIId))
    rgi60_self_dict = dict(zip(rgi60_df.RGIId, rgi60_df.RGIId))
    rgi50_rgi60_dict.update(rgi60_self_dict)
    
    # Add RGIId for version 6 to WGMS
    wgms_ee_df_raw['rgiid'] = wgms_ee_df_raw.rgiid_raw.map(rgi50_rgi60_dict)
    wgms_eee_df_raw['rgiid'] = wgms_eee_df_raw.rgiid_raw.map(rgi50_rgi60_dict)
    
    # Drop points without data
    wgms_ee_df = wgms_ee_df_raw.dropna(subset=['rgiid'])
    wgms_eee_df = wgms_eee_df_raw.dropna(subset=['rgiid'])
    
    # Winter balances only
    wgms_ee_df_winter = wgms_ee_df.dropna(subset=['WINTER_BALANCE'])
    wgms_ee_df_winter = wgms_ee_df_winter.sort_values('rgiid')
    wgms_ee_df_winter.reset_index(inplace=True, drop=True)
    
    # Add the winter time period using the E-MASS-BALANCE-OVERVIEW file
    wgms_e_cns2add = []
    for cn in wgms_e_df.columns:
        if cn not in wgms_ee_df_winter.columns:
            wgms_e_cns2add.append(cn)
            wgms_ee_df_winter[cn] = np.nan
            
    for nrow in np.arange(wgms_ee_df_winter.shape[0]):
        if nrow%500 == 0:
            print(nrow, 'of', wgms_ee_df_winter.shape[0])
        name = wgms_ee_df_winter.loc[nrow,'NAME']
        wgmsid = wgms_ee_df_winter.loc[nrow,'WGMS_ID']
        year = wgms_ee_df_winter.loc[nrow,'YEAR']
        
        try:
            e_idx = np.where((wgms_e_df['NAME'] == name) & 
                             (wgms_e_df['WGMS_ID'] == wgmsid) & 
                             (wgms_e_df['Year'] == year))[0][0]
        except:
            e_idx = None
        
        if e_idx is not None:
            wgms_ee_df_winter.loc[nrow,wgms_e_cns2add] = wgms_e_df.loc[e_idx,wgms_e_cns2add]
    
    wgms_ee_df_winter.to_csv(wgms_ee_winter_fp, index=False)
                
    # Export subset of data
    wgms_ee_df_winter_subset = wgms_ee_df_winter.loc[wgms_ee_df_winter['BEGIN_PERIOD'] > subset_time_value]
    wgms_ee_df_winter_subset = wgms_ee_df_winter_subset.dropna(subset=['END_WINTER'])
    wgms_ee_df_winter_subset.to_csv(wgms_ee_winter_fp_subset, index=False)


def est_kp(wgms_ee_winter_fp_subset='', wgms_ee_winter_fp_kp='', wgms_reg_kp_stats_fp=''):
    """
    This is used to estimate the precipitation factor for the bounds of HH2015_mod
    """
    # Load data
    assert os.path.exists(wgms_ee_winter_fp_subset), 'wgms_ee_winter_fn_subset does not exist!'
    wgms_df = pd.read_csv(wgms_ee_winter_fp_subset, encoding='unicode_escape')
    
    # Process dates
    wgms_df.loc[:,'BEGIN_PERIOD'] = wgms_df.loc[:,'BEGIN_PERIOD'].values.astype(int).astype(str)
    wgms_df['BEGIN_YEAR'] = [int(x[0:4]) for x in wgms_df.loc[:,'BEGIN_PERIOD']]
    wgms_df['BEGIN_MONTH'] = [int(x[4:6]) for x in list(wgms_df.loc[:,'BEGIN_PERIOD'])]
    wgms_df['BEGIN_DAY'] = [int(x[6:]) for x in list(wgms_df.loc[:,'BEGIN_PERIOD'])]
    wgms_df['BEGIN_YEARMONTH'] = [x[0:6] for x in list(wgms_df.loc[:,'BEGIN_PERIOD'])]
    wgms_df.loc[:,'END_WINTER'] = wgms_df.loc[:,'END_WINTER'].values.astype(int).astype(str)
    wgms_df['END_YEAR'] = [int(x[0:4]) for x in wgms_df.loc[:,'END_WINTER']]
    wgms_df['END_MONTH'] = [int(x[4:6]) for x in list(wgms_df.loc[:,'END_WINTER'])]
    wgms_df['END_DAY'] = [int(x[6:]) for x in list(wgms_df.loc[:,'END_WINTER'])]
    wgms_df['END_YEARMONTH'] = [x[0:6] for x in list(wgms_df.loc[:,'END_WINTER'])]
    
    # ===== PROCESS UNIQUE GLACIERS =====
    rgiids_unique = list(wgms_df['rgiid'].unique())
    glac_no = [x.split('-')[1] for x in rgiids_unique]
    
    main_glac_rgi = modelsetup.selectglaciersrgitable(glac_no=glac_no)
    
    # ===== TIME PERIOD =====
    dates_table = modelsetup.datesmodelrun(
            startyear=pygem_prms['climate']['ref_startyear'], endyear=pygem_prms['climate']['ref_endyear'], spinupyears=0,
            option_wateryear=pygem_prms['climate']['ref_wateryear'])
    dates_table_yearmo = [str(dates_table.loc[x,'year']) + str(dates_table.loc[x,'month']).zfill(2) 
                          for x in range(dates_table.shape[0])]
    
    # ===== LOAD CLIMATE DATA =====
    # Climate class
    gcm = class_climate.GCM(name=pygem_prms['climate']['ref_gcm_name'])
    
    # Air temperature [degC]
    gcm_temp, gcm_dates = gcm.importGCMvarnearestneighbor_xarray(gcm.temp_fn, gcm.temp_vn, main_glac_rgi,
                                                                  dates_table)
    # Precipitation [m]
    gcm_prec, gcm_dates = gcm.importGCMvarnearestneighbor_xarray(gcm.prec_fn, gcm.prec_vn, main_glac_rgi,
                                                                  dates_table)
    # Elevation [m asl]
    gcm_elev = gcm.importGCMfxnearestneighbor_xarray(gcm.elev_fn, gcm.elev_vn, main_glac_rgi)
    # Lapse rate
    gcm_lr, gcm_dates = gcm.importGCMvarnearestneighbor_xarray(gcm.lr_fn, gcm.lr_vn, main_glac_rgi, dates_table)
    
    # ===== PROCESS THE OBSERVATIONS ======
    prec_cn = pygem_prms['climate']['ref_gcm_name'] + '_prec'
    wgms_df[prec_cn] = np.nan
    wgms_df['kp'] = np.nan
    wgms_df['ndays'] = np.nan
    for glac in range(main_glac_rgi.shape[0]):
        print(glac, main_glac_rgi.loc[main_glac_rgi.index.values[glac],'RGIId'])
        # Select subsets of data
        glacier_rgi_table = main_glac_rgi.loc[main_glac_rgi.index.values[glac], :]
        glacier_str = '{0:0.5f}'.format(glacier_rgi_table['RGIId_float'])
        rgiid = glacier_rgi_table.RGIId
        
        wgms_df_single = (wgms_df.loc[wgms_df['rgiid'] == rgiid]).copy()
        glac_idx = wgms_df_single.index.values
        wgms_df_single.reset_index(inplace=True, drop=True)
        
        wgms_df_single[prec_cn] = np.nan
        for nobs in range(wgms_df_single.shape[0]):
            
            # Only process good data
            # - dates are provided and real
            # - spans more than one month
            # - positive winter balance (since we don't account for melt)
            if ((wgms_df_single.loc[nobs,'BEGIN_MONTH'] >= 1 and wgms_df_single.loc[nobs,'BEGIN_MONTH'] <= 12) and
                (wgms_df_single.loc[nobs,'BEGIN_DAY'] >= 1 and wgms_df_single.loc[nobs,'BEGIN_DAY'] <= 31) and
                (wgms_df_single.loc[nobs,'END_MONTH'] >= 1 and wgms_df_single.loc[nobs,'END_MONTH'] <= 12) and
                (wgms_df_single.loc[nobs,'END_DAY'] >= 1 and wgms_df_single.loc[nobs,'END_DAY'] <= 31) and
                (wgms_df_single.loc[nobs,'BEGIN_PERIOD'] < wgms_df_single.loc[nobs,'END_WINTER']) and
                (wgms_df_single.loc[nobs,'BEGIN_YEARMONTH'] != wgms_df_single.loc[nobs,'END_YEARMONTH']) and
                (wgms_df_single.loc[nobs,'WINTER_BALANCE'] > 0)
                ):
                # Begin index
                idx_begin = dates_table_yearmo.index(wgms_df_single.loc[nobs,'BEGIN_YEARMONTH'])
                idx_end = dates_table_yearmo.index(wgms_df_single.loc[nobs,'END_YEARMONTH'])
                
                # Fraction of the months to remove
                remove_prec_begin = (gcm_prec[glac,idx_begin] * 
                                     wgms_df_single.loc[nobs,'BEGIN_DAY'] / dates_table.loc[idx_begin,'daysinmonth'])
                remove_prec_end = (gcm_prec[glac,idx_end] * 
                                   (1 - wgms_df_single.loc[nobs,'END_DAY'] / dates_table.loc[idx_end,'daysinmonth']))
                
                # Winter Precipitation
                gcm_prec_winter = gcm_prec[glac,idx_begin:idx_end+1].sum() - remove_prec_begin - remove_prec_end
                wgms_df_single.loc[nobs,prec_cn] = gcm_prec_winter
                
                # Number of days
                ndays = (dates_table.loc[idx_begin:idx_end,'daysinmonth'].sum() - wgms_df_single.loc[nobs,'BEGIN_DAY']
                         - (dates_table.loc[idx_end,'daysinmonth'] - wgms_df_single.loc[nobs,'END_DAY']))
                wgms_df_single.loc[nobs,'ndays'] = ndays
            
        # Estimate precipitation factors
        # - assumes no melt and all snow (hence a convservative/underestimated estimate)
        wgms_df_single['kp'] = wgms_df_single['WINTER_BALANCE'] / 1000 / wgms_df_single[prec_cn]
    
        # Record precipitation, precipitation factors, and number of days in main dataframe
        wgms_df.loc[glac_idx,prec_cn] = wgms_df_single[prec_cn].values
        wgms_df.loc[glac_idx,'kp'] = wgms_df_single['kp'].values
        wgms_df.loc[glac_idx,'ndays'] = wgms_df_single['ndays'].values
        
    # Drop nan values
    wgms_df_wkp = wgms_df.dropna(subset=['kp']).copy()
    wgms_df_wkp.reset_index(inplace=True, drop=True)


    wgms_df_wkp.to_csv(wgms_ee_winter_fp_kp, index=False)

    # Calculate stats for all and each region
    wgms_df_wkp['reg'] = [x.split('-')[1].split('.')[0] for x in wgms_df_wkp['rgiid'].values]
    reg_unique = list(wgms_df_wkp['reg'].unique())
    
    # Output dataframe
    reg_kp_cns = ['region', 'count_obs', 'count_glaciers', 'kp_mean', 'kp_std', 'kp_med', 'kp_nmad', 'kp_min', 'kp_max']
    reg_kp_df = pd.DataFrame(np.zeros((len(reg_unique)+1,len(reg_kp_cns))), columns=reg_kp_cns)
    
    # Only those with at least 1 month of data
    wgms_df_wkp = wgms_df_wkp.loc[wgms_df_wkp['ndays'] >= 30]
    
    # All stats
    reg_kp_df.loc[0,'region'] = 'all'
    reg_kp_df.loc[0,'count_obs'] = wgms_df_wkp.shape[0]
    reg_kp_df.loc[0,'count_glaciers'] = len(wgms_df_wkp['rgiid'].unique())
    reg_kp_df.loc[0,'kp_mean'] = np.mean(wgms_df_wkp.kp.values)
    reg_kp_df.loc[0,'kp_std'] = np.std(wgms_df_wkp.kp.values)
    reg_kp_df.loc[0,'kp_med'] = np.median(wgms_df_wkp.kp.values)
    reg_kp_df.loc[0,'kp_nmad'] = median_abs_deviation(wgms_df_wkp.kp.values, scale='normal')
    reg_kp_df.loc[0,'kp_min'] = np.min(wgms_df_wkp.kp.values)
    reg_kp_df.loc[0,'kp_max'] = np.max(wgms_df_wkp.kp.values)
    
    # Regional stats
    for nreg, reg in enumerate(reg_unique):
        wgms_df_wkp_reg = wgms_df_wkp.loc[wgms_df_wkp['reg'] == reg]
        
        reg_kp_df.loc[nreg+1,'region'] = reg
        reg_kp_df.loc[nreg+1,'count_obs'] = wgms_df_wkp_reg.shape[0]
        reg_kp_df.loc[nreg+1,'count_glaciers'] = len(wgms_df_wkp_reg['rgiid'].unique())
        reg_kp_df.loc[nreg+1,'kp_mean'] = np.mean(wgms_df_wkp_reg.kp.values)
        reg_kp_df.loc[nreg+1,'kp_std'] = np.std(wgms_df_wkp_reg.kp.values)
        reg_kp_df.loc[nreg+1,'kp_med'] = np.median(wgms_df_wkp_reg.kp.values)
        reg_kp_df.loc[nreg+1,'kp_nmad'] = median_abs_deviation(wgms_df_wkp_reg.kp.values, scale='normal')
        reg_kp_df.loc[nreg+1,'kp_min'] = np.min(wgms_df_wkp_reg.kp.values)
        reg_kp_df.loc[nreg+1,'kp_max'] = np.max(wgms_df_wkp_reg.kp.values)
        
        
        print('region', reg)
        print('  count:', wgms_df_wkp_reg.shape[0])
        print('  glaciers:', len(wgms_df_wkp_reg['rgiid'].unique()))
        print('  mean:', np.mean(wgms_df_wkp_reg.kp.values))
        print('  std :', np.std(wgms_df_wkp_reg.kp.values))
        print('  med :', np.median(wgms_df_wkp_reg.kp.values))
        print('  nmad:', median_abs_deviation(wgms_df_wkp_reg.kp.values, scale='normal'))
        print('  min :', np.min(wgms_df_wkp_reg.kp.values))
        print('  max :', np.max(wgms_df_wkp_reg.kp.values))
        
    reg_kp_df.to_csv(wgms_reg_kp_stats_fp, index=False)


def main():
    parser = argparse.ArgumentParser(description="estimate precipitation factors from WGMS winter mass balance data")
    parser.add_argument('-o', '--overwrite', action='store_true',
                        help='Flag to overwrite existing data')
    args = parser.parse_args()
    # ===== WGMS DATA =====
    # these are hardcoded for the format downloaded from WGMS for their 2020-08 dataset, would need to be updated for newer data
    wgms_fp = f"{pygem_prms['root']}/WGMS/"
    # inputs
    wgms_dsn = 'DOI-WGMS-FoG-2020-08/'
    wgms_eee_fp = wgms_fp+wgms_dsn+ 'WGMS-FoG-2020-08-EEE-MASS-BALANCE-POINT.csv'
    wgms_ee_fp = wgms_fp+wgms_dsn+ 'WGMS-FoG-2020-08-EE-MASS-BALANCE.csv'
    wgms_e_fp = wgms_fp+wgms_dsn+ 'WGMS-FoG-2020-08-E-MASS-BALANCE-OVERVIEW.csv'
    wgms_id_fp = wgms_fp+wgms_dsn+ 'WGMS-FoG-2020-08-AA-GLACIER_ID_LUT.csv'
    in_fps = [x for x in [wgms_eee_fp, wgms_ee_fp, wgms_e_fp, wgms_id_fp]]

    # outputs
    wgms_ee_winter_fp = wgms_fp+ 'WGMS-FoG-2019-12-EE-MASS-BALANCE-winter_processed.csv'
    wgms_ee_winter_fp_subset = wgms_ee_winter_fp.replace('.csv', '-subset.csv')
    wgms_ee_winter_fp_kp = wgms_ee_winter_fp.replace('.csv', '-subset-kp.csv')
    wgms_reg_kp_stats_fp = wgms_fp+ 'WGMS-FoG-2019-12-reg_kp_summary.csv'

    out_subset_fps = [wgms_ee_winter_fp, wgms_ee_winter_fp_subset]
    output_kp_fps = [wgms_ee_winter_fp_kp,wgms_reg_kp_stats_fp]
    
    subset_time_value = 20000000

    # if not all outputs already exist, subset the input data and create the necessary outputs
    if not all(os.path.exists(filepath) for filepath in out_subset_fps):
        missing = False
        for fp in in_fps:
            if not os.path.isfile(fp):
                print(f'Missing required WGMS datafile: {fp}')
                missing = True
        if missing:
            sys.exit(1)

        subset_winter(wgms_eee_fp=wgms_eee_fp, 
                      wgms_ee_fp=wgms_ee_fp, 
                      wgms_e_fp=wgms_e_fp,  
                      wgms_id_fp=wgms_id_fp, 
                      wgms_ee_winter_fp=wgms_ee_winter_fp, 
                      wgms_ee_winter_fp_subset=wgms_ee_winter_fp_subset, 
                      subset_time_value=subset_time_value)
    
    if not all(os.path.exists(filepath) for filepath in output_kp_fps) or args.overwrite:
        est_kp(wgms_ee_winter_fp_subset=wgms_ee_winter_fp_subset, 
               wgms_ee_winter_fp_kp=wgms_ee_winter_fp_kp,
               wgms_reg_kp_stats_fp=wgms_reg_kp_stats_fp)


if __name__ == "__main__":
    main()