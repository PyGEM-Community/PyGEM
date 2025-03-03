"""
Python Glacier Evolution Model (PyGEM)

copyright © 2018 David Rounce <drounce@cmu.edu>

Distrubted under the MIT lisence

Run bias adjustments a given climate dataset
"""
# Built-in libraries
import os
import sys
import math

# External libraries
import numpy as np
from scipy.ndimage import uniform_filter
from scipy.stats import percentileofscore
from pygem.setup.config import ConfigManager
# instantiate ConfigManager
config_manager = ConfigManager()
# read the config
pygem_prms = config_manager.read_config()

#%% FUNCTIONS
def annual_avg_2darray(x):
    """
    Annual average of dataset, where columns are a monthly timeseries (temperature)
    """
    return x.reshape(-1,12).mean(1).reshape(x.shape[0],int(x.shape[1]/12))


def annual_sum_2darray(x):
    """
    Annual sum of dataset, where columns are a monthly timeseries (precipitation)
    """
    return x.reshape(-1,12).sum(1).reshape(x.shape[0],int(x.shape[1]/12))


def monthly_avg_2darray(x):
    """
    Monthly average for a given 2d dataset where columns are monthly timeseries
    """
    return x.reshape(-1,12).transpose().reshape(-1,int(x.shape[1]/12)).mean(1).reshape(12,-1).transpose()


def monthly_std_2darray(x):
    """
    Monthly standard deviation for a given 2d dataset where columns are monthly timeseries
    """
    return x.reshape(-1,12).transpose().reshape(-1,int(x.shape[1]/12)).std(1).reshape(12,-1).transpose()

    
def temp_biasadj_HH2015(ref_temp, ref_elev, gcm_temp, dates_table_ref, dates_table, gcm_startyear, ref_startyear,
                        ref_spinupyears=0, gcm_spinupyears=0, debug=False):
    """
    Huss and Hock (2015) temperature bias correction based on mean and interannual variability
    
    Note: the mean over the reference period will only equal the mean of the gcm for the same time period when the GCM
    time series is run for the same period, i.e., due to the 25-year moving average, the mean gcm temps from 2000-2019
    will differ if using a reference period of 2000-2020 to bias adjust gcm temps from 2000-2100.
    
    Parameters
    ----------
    ref_temp : np.array
        time series of reference temperature
    gcm_temp : np.array
        time series of GCM temperature
    dates_table_ref : pd.DataFrame
        dates table for reference time period
    dates_table : pd.DataFrame
        dates_table for GCM time period
    
    Returns
    -------
    gcm_temp_biasadj : np.array
        GCM temperature bias corrected to the reference climate dataset according to Huss and Hock (2015)
    gcm_elev_biasadj : float
        new gcm elevation is the elevation of the reference climate dataset
    """
    # GCM subset to agree with reference time period to calculate bias corrections
    gcm_subset_idx_start = np.where(dates_table.date.values == dates_table_ref.date.values[0])[0][0]
    gcm_subset_idx_end = np.where(dates_table.date.values == dates_table_ref.date.values[-1])[0][0]
    gcm_temp_subset = gcm_temp[:,gcm_subset_idx_start:gcm_subset_idx_end+1]

    # Remove spinup years, so adjustment performed over calibration period
    ref_temp_nospinup = ref_temp[:,ref_spinupyears*12:]
    gcm_temp_nospinup = gcm_temp_subset[:,gcm_spinupyears*12:]
    
    # Roll months so they are aligned with simulation months
    roll_amt = -1*(12 - gcm_subset_idx_start%12)
    if roll_amt == -12:
        roll_amt = 0 
    
    # Mean monthly temperature
    ref_temp_monthly_avg = np.roll(monthly_avg_2darray(ref_temp_nospinup), roll_amt, axis=1)
    gcm_temp_monthly_avg = np.roll(monthly_avg_2darray(gcm_temp_nospinup), roll_amt, axis=1)
    # Standard deviation monthly temperature
    ref_temp_monthly_std = np.roll(monthly_std_2darray(ref_temp_nospinup), roll_amt, axis=1)
    gcm_temp_monthly_std = np.roll(monthly_std_2darray(gcm_temp_nospinup), roll_amt, axis=1)

    # Monthly bias adjustment (additive)
    gcm_temp_monthly_adj = ref_temp_monthly_avg - gcm_temp_monthly_avg

    # Monthly variability
    variability_monthly_std = ref_temp_monthly_std / gcm_temp_monthly_std
    
    # if/else statement for whether or not the full GCM period is the same as the simulation period
    #   create GCM subset for applying bias-correction (e.g., 2000-2100),
    #   that does not include the earlier reference years (e.g., 1981-2000)
    if gcm_startyear == ref_startyear:
        bc_temp = gcm_temp
    else:
        if pygem_prms['climate']['gcm_wateryear'] == 'hydro':
            dates_cn = 'wateryear'
        else:
            dates_cn = 'year'
        sim_idx_start = dates_table[dates_cn].to_list().index(gcm_startyear)
        bc_temp = gcm_temp[:,sim_idx_start:]

    # Monthly temperature bias adjusted according to monthly average
    #   This is where the actual bias adjustment of temperature values occurs.
    #   All steps before this are preliminary steps (e.g., formatting,
    #   determining additive factor and std adjustment).
    t_mt = bc_temp + np.tile(gcm_temp_monthly_adj, int(bc_temp.shape[1]/12))
    
    # Mean monthly temperature bias adjusted according to monthly average
    #  t_m25avg is the avg monthly temp in a 25-year period around the given year
    N = 25
    t_m_Navg = np.zeros(t_mt.shape)
    for month in range(0,12):
        t_m_subset = t_mt[:,month::12]
        # Uniform filter computes running average and uses 'reflects' values at borders
        t_m_Navg_subset = uniform_filter(t_m_subset,size=(1,N))
        t_m_Navg[:,month::12] = t_m_Navg_subset

    gcm_temp_biasadj = t_m_Navg + (t_mt - t_m_Navg) * np.tile(variability_monthly_std, int(bc_temp.shape[1]/12))
    
    # Update elevation
    gcm_elev_biasadj = ref_elev
    
    # Assert that mean temperatures for all the glaciers must be more-or-less equal
    gcm_temp_biasadj_subset = (
            gcm_temp_biasadj[:,gcm_subset_idx_start:gcm_subset_idx_end+1][:,ref_spinupyears*12:])

    if gcm_startyear == ref_startyear:
        if debug:
            print((np.mean(gcm_temp_biasadj_subset, axis=1) - np.mean(ref_temp[:,ref_spinupyears*12:], axis=1)))
        assert np.max(np.abs(np.mean(gcm_temp_biasadj_subset, axis=1) - 
                             np.mean(ref_temp[:,ref_spinupyears*12:], axis=1))) < 1, (
                'Error with gcm temperature bias adjustment: mean ref and gcm temps differ by more than 1 degree')
    else:
        if debug:
            print((np.mean(gcm_temp_biasadj_subset, axis=1) - np.mean(ref_temp[:,ref_spinupyears*12:], axis=1))) 
        
    return gcm_temp_biasadj, gcm_elev_biasadj


def prec_biasadj_HH2015(ref_prec, ref_elev, gcm_prec, dates_table_ref, dates_table, gcm_startyear, ref_startyear,
                        ref_spinupyears=0, gcm_spinupyears=0):
    """
    Huss and Hock (2015) precipitation bias correction based on mean (multiplicative)
    
    Parameters
    ----------
    ref_prec : np.array
        time series of reference precipitation
    gcm_prec : np.array
        time series of GCM precipitation
    dates_table_ref : pd.DataFrame
        dates table for reference time period
    dates_table : pd.DataFrame
        dates_table for GCM time period
    gcm_elev_biasadj : float
        new gcm elevation is the elevation of the reference climate dataset
    
    Returns
    -------
    gcm_prec_biasadj : np.array
        GCM precipitation bias corrected to the reference climate dataset according to Huss and Hock (2015)
    """
    # GCM subset to agree with reference time period to calculate bias corrections
    gcm_subset_idx_start = np.where(dates_table.date.values == dates_table_ref.date.values[0])[0][0]
    gcm_subset_idx_end = np.where(dates_table.date.values == dates_table_ref.date.values[-1])[0][0]
    gcm_prec_subset = gcm_prec[:,gcm_subset_idx_start:gcm_subset_idx_end+1]
    
    # Remove spinup years, so adjustment performed over calibration period
    ref_prec_nospinup = ref_prec[:,ref_spinupyears*12:]
    gcm_prec_nospinup = gcm_prec_subset[:,gcm_spinupyears*12:]
    
    # Roll months so they are aligned with simulation months
    roll_amt = -1*(12 - gcm_subset_idx_start%12)
    
    # PRECIPITATION BIAS CORRECTIONS
    # Monthly mean precipitation
    ref_prec_monthly_avg = np.roll(monthly_avg_2darray(ref_prec_nospinup), roll_amt, axis=1)
    gcm_prec_monthly_avg = np.roll(monthly_avg_2darray(gcm_prec_nospinup), roll_amt, axis=1)
    bias_adj_prec_monthly = ref_prec_monthly_avg / gcm_prec_monthly_avg
    
    # if/else statement for whether or not the full GCM period is the same as the simulation period  
    #   create GCM subset for applying bias-correction (e.g., 2000-2100),
    #   that does not include the earlier reference years (e.g., 1985-2000)
    if gcm_startyear == ref_startyear:
        bc_prec = gcm_prec
    else:
        if pygem_prms['climate']['gcm_wateryear'] == 'hydro':
            dates_cn = 'wateryear'
        else:
            dates_cn = 'year'
        sim_idx_start = dates_table[dates_cn].to_list().index(gcm_startyear)
        bc_prec = gcm_prec[:,sim_idx_start:]
    
    # Bias adjusted precipitation accounting for differences in monthly mean
    gcm_prec_biasadj = bc_prec * np.tile(bias_adj_prec_monthly, int(bc_prec.shape[1]/12))
    
    # Update elevation
    gcm_elev_biasadj = ref_elev
    
    # Assertion that bias adjustment does not drastically modify the precipitation and are reasonable
    gcm_prec_biasadj_subset = (
            gcm_prec_biasadj[:,gcm_subset_idx_start:gcm_subset_idx_end+1][:,gcm_spinupyears*12:])
    gcm_prec_biasadj_frac = gcm_prec_biasadj_subset.sum(axis=1) / ref_prec_nospinup.sum(axis=1)
    assert np.min(gcm_prec_biasadj_frac) > 0.5 and np.max(gcm_prec_biasadj_frac) < 2, (
            'Error with gcm precipitation bias adjustment: total ref and gcm prec differ by more than factor of 2')
    assert gcm_prec_biasadj.max() <= 10, 'gcm_prec_adj (precipitation bias adjustment) too high, needs to be modified'
    assert gcm_prec_biasadj.min() >= 0, 'gcm_prec_adj is producing a negative precipitation value'   
    
    return gcm_prec_biasadj, gcm_elev_biasadj


def prec_biasadj_opt1(ref_prec, ref_elev, gcm_prec, dates_table_ref, dates_table, gcm_startyear, ref_startyear,
                      ref_spinupyears=0, gcm_spinupyears=0):
    """
    Precipitation bias correction based on mean with limited maximum
    
    Parameters
    ----------
    ref_prec : np.array
        time series of reference precipitation
    gcm_prec : np.array
        time series of GCM precipitation
    dates_table_ref : pd.DataFrame
        dates table for reference time period
    dates_table : pd.DataFrame
        dates_table for GCM time period
    
    Returns
    -------
    gcm_prec_biasadj : np.array
        GCM precipitation bias corrected to the reference climate dataset according to Huss and Hock (2015)
    gcm_elev_biasadj : float
        new gcm elevation is the elevation of the reference climate dataset
    """
    # GCM subset to agree with reference time period to calculate bias corrections
    gcm_subset_idx_start = np.where(dates_table.date.values == dates_table_ref.date.values[0])[0][0]
    gcm_subset_idx_end = np.where(dates_table.date.values == dates_table_ref.date.values[-1])[0][0]
    gcm_prec_subset = gcm_prec[:,gcm_subset_idx_start:gcm_subset_idx_end+1]
    
    # Remove spinup years, so adjustment performed over calibration period
    ref_prec_nospinup = ref_prec[:,ref_spinupyears*12:]
    gcm_prec_nospinup = gcm_prec_subset[:,gcm_spinupyears*12:]
    
    # Roll months so they are aligned with simulation months
    roll_amt = -1*(12 - gcm_subset_idx_start%12)
    
    # PRECIPITATION BIAS CORRECTIONS
    # Monthly mean precipitation
    ref_prec_monthly_avg = np.roll(monthly_avg_2darray(ref_prec_nospinup), roll_amt, axis=1)
    gcm_prec_monthly_avg = np.roll(monthly_avg_2darray(gcm_prec_nospinup), roll_amt, axis=1)
    bias_adj_prec_monthly = ref_prec_monthly_avg / gcm_prec_monthly_avg
    
    # if/else statement for whether or not the full GCM period is the same as the simulation period  
    #   create GCM subset for applying bias-correction (e.g., 2000-2100),
    #   that does not include the earlier reference years (e.g., 1985-2000)
    if gcm_startyear == ref_startyear:
        bc_prec = gcm_prec
    else:
        if pygem_prms['climate']['gcm_wateryear'] == 'hydro':
            dates_cn = 'wateryear'
        else:
            dates_cn = 'year'
        sim_idx_start = dates_table[dates_cn].to_list().index(gcm_startyear)
        bc_prec = gcm_prec[:,sim_idx_start:]
    
    # Bias adjusted precipitation accounting for differences in monthly mean
    gcm_prec_biasadj_raw = bc_prec * np.tile(bias_adj_prec_monthly, int(bc_prec.shape[1]/12))
    
    # Adjust variance based on zscore and reference standard deviation
    ref_prec_monthly_std = np.roll(monthly_std_2darray(ref_prec_nospinup), roll_amt, axis=1)
    gcm_prec_biasadj_raw_monthly_avg = monthly_avg_2darray(gcm_prec_biasadj_raw[:,0:ref_prec.shape[1]])
    gcm_prec_biasadj_raw_monthly_std = monthly_std_2darray(gcm_prec_biasadj_raw[:,0:ref_prec.shape[1]])
    # Calculate value compared to mean and standard deviation
    gcm_prec_biasadj_zscore = (
            (gcm_prec_biasadj_raw - np.tile(gcm_prec_biasadj_raw_monthly_avg, int(bc_prec.shape[1]/12))) / 
             np.tile(gcm_prec_biasadj_raw_monthly_std, int(bc_prec.shape[1]/12)))
    gcm_prec_biasadj = (
            np.tile(gcm_prec_biasadj_raw_monthly_avg, int(bc_prec.shape[1]/12)) +
            gcm_prec_biasadj_zscore * np.tile(ref_prec_monthly_std, int(bc_prec.shape[1]/12)))
    gcm_prec_biasadj[gcm_prec_biasadj < 0] = 0
    
    # Identify outliers using reference's monthly maximum adjusted for future increases
    ref_prec_monthly_max = np.roll((ref_prec_nospinup.reshape(-1,12).transpose()
                                    .reshape(-1,int(ref_prec_nospinup.shape[1]/12)).max(1).reshape(12,-1).transpose()), 
                                   roll_amt, axis=1)
    gcm_prec_max_check = np.tile(ref_prec_monthly_max, int(gcm_prec_biasadj.shape[1]/12))        
    # For wetter years in future, adjust monthly max by the annual increase in precipitation
    gcm_prec_annual = annual_sum_2darray(bc_prec)
    gcm_prec_annual_norm = gcm_prec_annual / gcm_prec_annual.mean(1)[:,np.newaxis]
    gcm_prec_annual_norm_repeated = np.repeat(gcm_prec_annual_norm, 12).reshape(gcm_prec_biasadj.shape)
    gcm_prec_max_check_adj = gcm_prec_max_check * gcm_prec_annual_norm_repeated
    gcm_prec_max_check_adj[gcm_prec_max_check_adj < gcm_prec_max_check] = (
            gcm_prec_max_check[gcm_prec_max_check_adj < gcm_prec_max_check])
    
    # Replace outliers with monthly mean adjusted for the normalized annual variation
    outlier_replacement = (gcm_prec_annual_norm_repeated * 
                           np.tile(ref_prec_monthly_avg, int(gcm_prec_biasadj.shape[1]/12)))
    gcm_prec_biasadj[gcm_prec_biasadj > gcm_prec_max_check_adj] = (
            outlier_replacement[gcm_prec_biasadj > gcm_prec_max_check_adj])
    
    # Update elevation
    gcm_elev_biasadj = ref_elev
    
    # Assertion that bias adjustment does not drastically modify the precipitation and are reasonable
    gcm_prec_biasadj_subset = (
            gcm_prec_biasadj[:,gcm_subset_idx_start:gcm_subset_idx_end+1][:,gcm_spinupyears*12:])
    gcm_prec_biasadj_frac = gcm_prec_biasadj_subset.sum(axis=1) / ref_prec_nospinup.sum(axis=1)
    assert np.min(gcm_prec_biasadj_frac) > 0.5 and np.max(gcm_prec_biasadj_frac) < 2, (
            'Error with gcm precipitation bias adjustment: total ref and gcm prec differ by more than factor of 2')
    assert gcm_prec_biasadj.max() <= 10, 'gcm_prec_adj (precipitation bias adjustment) too high, needs to be modified'
    assert gcm_prec_biasadj.min() >= 0, 'gcm_prec_adj is producing a negative precipitation value'   
    
    return gcm_prec_biasadj, gcm_elev_biasadj

    
def temp_biasadj_QDM(ref_temp, ref_elev, gcm_temp, dates_table_ref, dates_table, gcm_startyear, ref_startyear,
                     ref_spinupyears=0, gcm_spinupyears=0):
    """
    Cannon et al. (2015) temperature bias correction based on quantile delta mapping
        Also see Lader et al. (2017) for further documentation
    
    Perform a quantile delta mapping bias-correction procedure on temperature.
    
    This function operates by multiplying reference temperature by a ratio of
        the projected and future gcm temperature at the same percentiles 
        (e.g., ref_temp * gcm_projected/gcm_historic, with all values at same percentile).
    Quantile delta mapping is generally viewed as more capable of capturing
        climatic extemes at the lowest and highest quantiles (e.g., 0.01% and 99.9%)
        compared to standard quantile mapping (which constructs a transfer function
        using only reference and historic climate data, requiring extrapolations for
        projected values lying outside the reference and historic datasets). See
        Cannon et al. (2015) Sections 2 and 3 for further explanation.
        
    Parameters
    ----------
    ref_temp : pandas dataframe
        dataframe containing reference climate temperature data
    gcm_temp : np.array
        time series of GCM temperature
    dates_table_ref : pd.DataFrame
        dates table for reference time period
    dates_table : pd.DataFrame
        dates_table for GCM time period      

    Returns
    -------
    gcm_temp_biasadj : numpy ndarray
        ndarray that contains bias-corrected future gcm temperature data
    gcm_elev_biasadj : float
        new gcm elevation is the elevation of the reference climate dataset
    """
    # GCM historic subset to agree with reference time period to enable QDM bias correction
    gcm_subset_idx_start = np.where(dates_table.date.values == dates_table_ref.date.values[0])[0][0]
    gcm_subset_idx_end = np.where(dates_table.date.values == dates_table_ref.date.values[-1])[0][0]
    gcm_temp_historic = gcm_temp[:,gcm_subset_idx_start:gcm_subset_idx_end+1]

    # Remove spinup years, so adjustment performed over calibration period
    ref_temp_nospinup = ref_temp[:,ref_spinupyears*12:] + 273.15
    gcm_temp_nospinup = gcm_temp_historic[:,gcm_spinupyears*12:] + 273.15
    
    # if/else statement for whether or not the full GCM period is the same as the simulation period
    #   create GCM subset for applying bias-correction (e.g., 2000-2100),
    #   that does not include the earlier reference years (e.g., 1981-2000)
    if gcm_startyear == ref_startyear:
        bc_temp = gcm_temp
    else:
        if pygem_prms['climate']['gcm_wateryear'] == 'hydro':
            dates_cn = 'wateryear'
        else:
            dates_cn = 'year'
        sim_idx_start = dates_table[dates_cn].to_list().index(gcm_startyear)
        bc_temp = gcm_temp[:,sim_idx_start:]
    
    # create an empty array for the bias-corrected GCM data
    # gcm_temp_biasadj = np.zeros(bc_temp.size)
    loop_years = 20 # number of years used for each bias-correction period
    loop_months = loop_years * 12 # number of months used for each bias-correction period
    
    # convert to Kelvin to better handle Celsius values around 0)
    bc_temp = bc_temp + 273.15
    # bc_temp = bc_temp[0]
    all_gcm_temp_biasadj =[] # empty list for all glaciers
    
    for j in range(0, len(bc_temp)):
        gcm_temp_biasadj = [] # empty list for bias-corrected data
        bc_loops = len(bc_temp[j])/loop_months # determine number of loops needed for bias-correction
        
        # loop through however many times are required to bias-correct the entire time period
        #   using smaller time periods (typically 20-30 years) to better capture the
        #   quantiles and extremes at different points in the future
        for i in range(0, math.ceil(bc_loops)): 
            bc_temp_loop = bc_temp[j][i*loop_months:(i+1)*loop_months]
            bc_temp_loop_corrected = np.zeros(bc_temp_loop.size)
            
            # now loop through each individual value within the time period for bias correction        
            for ival, projected_value in enumerate(bc_temp_loop):
                percentile = percentileofscore(bc_temp_loop, projected_value) 
                bias_correction_factor = np.percentile(ref_temp_nospinup, percentile)/np.percentile(gcm_temp_nospinup, percentile)
                bc_temp_loop_corrected[ival] = projected_value * bias_correction_factor
            # append the values from each time period to a list               
            gcm_temp_biasadj.append(bc_temp_loop_corrected)

        gcm_temp_biasadj = np.concatenate(gcm_temp_biasadj, axis=0)
        # convert back to Celsius for simulation
        gcm_temp_biasadj = gcm_temp_biasadj - 273.15
        # gcm_temp_biasadj = np.array([gcm_temp_biasadj.tolist()])
        all_gcm_temp_biasadj.append(gcm_temp_biasadj)   
        # print(all_gcm_temp_biasadj)
        
    gcm_temp_biasadj = np.array(all_gcm_temp_biasadj)
    # print(gcm_temp_biasadj[0])
    # print(gcm_temp_biasadj[1])
    # print(gcm_temp_biasadj)
    
    # Update elevation
    gcm_elev_biasadj = ref_elev
    
    return gcm_temp_biasadj, gcm_elev_biasadj
    

def prec_biasadj_QDM(ref_prec, ref_elev, gcm_prec, dates_table_ref, dates_table, gcm_startyear, ref_startyear,
                     ref_spinupyears=0, gcm_spinupyears=0):
    """
    Cannon et al. (2015) precipitation bias correction based on quantile delta mapping
        Also see Lader et al. (2017) another use case
    
    Perform a quantile delta mapping bias-correction procedure on precipitation.
    
    This function operates by multiplying reference precipitation by a ratio of
        the projected and future gcm precipitations at the same percentiles 
        (e.g., ref_prec * gcm_projected/gcm_historic, with all values at same percentile).
    Quantile delta mapping is generally viewed as more capable of capturing
        climatic extemes at the lowest and highest quantiles (e.g., 0.01% and 99.9%)
        compared to standard quantile mapping (which constructs a transfer function
        using only reference and historic climate data, requiring extrapolations for
        projected values lying outside the reference and historic datasets). See
        Cannon et al. (2015) Sections 2 and 3 for further explanation.
        
    Parameters
    ----------
    ref_prec : pandas dataframe
        dataframe containing reference climate precipitation data
    gcm_prec : np.array
        time series of GCM precipitation
    dates_table_ref : pd.DataFrame
        dates table for reference time period
    dates_table : pd.DataFrame
        dates_table for GCM time period      

    Returns
    -------
    gcm_prec_biasadj : numpy ndarray
        ndarray that contains bias-corrected future gcm precipitation data
    gcm_elev_biasadj : float
        new gcm elevation is the elevation of the reference climate dataset
    """
    
    # GCM historic subset to agree with reference time period to enable QDM bias correction
    gcm_subset_idx_start = np.where(dates_table.date.values == dates_table_ref.date.values[0])[0][0]
    gcm_subset_idx_end = np.where(dates_table.date.values == dates_table_ref.date.values[-1])[0][0]
    gcm_prec_historic = gcm_prec[:,gcm_subset_idx_start:gcm_subset_idx_end+1]

    # Remove spinup years, so adjustment performed over calibration period
    ref_prec_nospinup = ref_prec[:,ref_spinupyears*12:]
    gcm_prec_nospinup = gcm_prec_historic[:,gcm_spinupyears*12:]
    
    # if/else statement for whether or not the full GCM period is the same as the simulation period
    #   create GCM subset for applying bias-correction (e.g., 2000-2100),
    #   that does not include the earlier reference years (e.g., 1981-2000)
    if gcm_startyear == ref_startyear:
        bc_prec = gcm_prec
    else:
        if pygem_prms['climate']['gcm_wateryear'] == 'hydro':
            dates_cn = 'wateryear'
        else:
            dates_cn = 'year'
        sim_idx_start = dates_table[dates_cn].to_list().index(gcm_startyear)
        bc_prec = gcm_prec[:,sim_idx_start:]
        
    # create an empty array for the bias-corrected GCM data
    # gcm_prec_biasadj = np.zeros(bc_prec.size)
    loop_years = 20 # number of years used for each bias-correction period
    loop_months = loop_years * 12 # number of months used for each bias-correction period
    
    # bc_prec = bc_prec[0]
    all_gcm_prec_biasadj =[] # empty list for all glaciers
    
    for j in range(0, len(bc_prec)):
        gcm_prec_biasadj = [] # empty list for bias-corrected data
        bc_loops = len(bc_prec[j])/loop_months # determine number of loops needed for bias-correction
        
        # loop through however many times are required to bias-correct the entire time period
        #   using smaller time periods (typically 20-30 years) to better capture the
        #   quantiles and extremes at different points in the future
        for i in range(0, math.ceil(bc_loops)):
            bc_prec_loop = bc_prec[j][i*loop_months:(i+1)*loop_months]
            bc_prec_loop_corrected = np.zeros(bc_prec_loop.size)
            
            # now loop through each individual value within the time period for bias correction
            for ival, projected_value in enumerate(bc_prec_loop):
                percentile = percentileofscore(bc_prec_loop, projected_value) 
                bias_correction_factor = np.percentile(ref_prec_nospinup, percentile)/np.percentile(gcm_prec_nospinup, percentile)
                bc_prec_loop_corrected[ival] = projected_value * bias_correction_factor   
            # append the values from each time period to a list
            gcm_prec_biasadj.append(bc_prec_loop_corrected)
            
        gcm_prec_biasadj = np.concatenate(gcm_prec_biasadj, axis=0)
        # gcm_prec_biasadj = np.array([gcm_prec_biasadj.tolist()])
        all_gcm_prec_biasadj.append(gcm_prec_biasadj)
        
    gcm_prec_biasadj = np.array(all_gcm_prec_biasadj)
    
    # Update elevation
    gcm_elev_biasadj = ref_elev
        
    return gcm_prec_biasadj, gcm_elev_biasadj 
    

def monthly_avg_array_rolled(ref_array, dates_table_ref, dates_table, gcm_startyear, ref_startyear):
    """ Monthly average array from reference data rolled to ensure proper months 
    
    Parameters
    ----------
    ref_array : np.array
        time series of reference lapse rates
    dates_table_ref : pd.DataFrame
        dates table for reference time period
    dates_table : pd.DataFrame
        dates_table for GCM time period
    
    Returns
    -------
    gcm_array : np.array
        gcm climate data based on monthly average of reference data
    """
    # GCM subset to agree with reference time period to calculate bias corrections
    gcm_subset_idx_start = np.where(dates_table.date.values == dates_table_ref.date.values[0])[0][0]
    
    # Roll months so they are aligned with simulation months
    roll_amt = -1*(12 - gcm_subset_idx_start%12)
    ref_array_monthly_avg = np.roll(monthly_avg_2darray(ref_array), roll_amt, axis=1)
    gcm_array = np.tile(ref_array_monthly_avg, int(dates_table.shape[0]/12))
    
    # if/else statement for whether or not the full GCM period is the same as the simulation period
    #   create GCM subset for applying bias-correction (e.g., 2000-2100),
    #   that does not include the earlier reference years (e.g., 1981-2000)
    if gcm_startyear != ref_startyear:
        if pygem_prms['climate']['gcm_wateryear'] == 'hydro':
            dates_cn = 'wateryear'
        else:
            dates_cn = 'year'
        sim_idx_start = dates_table[dates_cn].to_list().index(gcm_startyear)
        gcm_array = gcm_array[:,sim_idx_start:]
    
    return gcm_array