"""
Python Glacier Evolution Model (PyGEM)

copyright © 2018 David Rounce <drounce@cmu.edu

Distrubted under the MIT lisence

List of functions used to set up different aspects of the model
"""
# Built-in libaries
import os
# External libraries
import pandas as pd
import numpy as np
from datetime import datetime
from pygem.setup.config import ConfigManager
# instantiate ConfigManager
config_manager = ConfigManager()
# read the config
pygem_prms = config_manager.read_config()


def datesmodelrun(startyear=pygem_prms['climate']['ref_startyear'], endyear=pygem_prms['climate']['ref_endyear'], 
                  spinupyears=pygem_prms['climate']['ref_spinupyears'], option_wateryear=pygem_prms['climate']['ref_wateryear']):
    """
    Create table of year, month, day, water year, season and number of days in the month.

    Parameters
    ----------
    startyear : int
        starting year
    endyear : int
        ending year
    spinupyears : int
        number of spinup years

    Returns
    -------
    dates_table : pd.DataFrame
        table where each row is a timestep and each column is attributes (date, year, wateryear, etc.) of that timestep
    """
    # Include spinup time in start year
    startyear_wspinup = startyear - spinupyears
    # Convert start year into date depending on option_wateryear
    if option_wateryear == 'hydro':
        startdate = str(startyear_wspinup - 1) + '-10-01'
        enddate = str(endyear) + '-09-30'
    elif option_wateryear == 'calendar':
        startdate = str(startyear_wspinup) + '-01-01'
        enddate = str(endyear) + '-12-31'
    elif option_wateryear == 'custom':
        startdate = str(startyear_wspinup) + '-' + pygem_prms['time']['startmonthday']
        enddate = str(endyear) + '-' + pygem_prms['time']['endmonthday']
    else:
        assert True==False, "\n\nError: Select an option_wateryear that exists.\n"
    # Convert input format into proper datetime format
    startdate = datetime(*[int(item) for item in startdate.split('-')])
    enddate = datetime(*[int(item) for item in enddate.split('-')])
    if pygem_prms['time']['timestep'] == 'monthly':
        startdate = startdate.strftime('%Y-%m')
        enddate = enddate.strftime('%Y-%m')
    elif pygem_prms['time']['timestep'] == 'daily':
        startdate = startdate.strftime('%Y-%m-%d')
        enddate = enddate.strftime('%Y-%m-%d')
    # Generate dates_table using date_range function
    if pygem_prms['time']['timestep'] == 'monthly':
        # Automatically generate dates from start date to end data using a monthly frequency (MS), which generates
        # monthly data using the 1st of each month'
        dates_table = pd.DataFrame({'date' : pd.date_range(startdate, enddate, freq='MS', unit='s')})
        # Select attributes of DateTimeIndex (dt.year, dt.month, and dt.daysinmonth)
        dates_table['year'] = dates_table['date'].dt.year
        dates_table['month'] = dates_table['date'].dt.month
        dates_table['daysinmonth'] = dates_table['date'].dt.daysinmonth
        dates_table['timestep'] = np.arange(len(dates_table['date']))
        # Set date as index
        dates_table.set_index('timestep', inplace=True)
        # Remove leap year days if user selected this with option_leapyear
        if pygem_prms['time']['option_leapyear'] == 0:
            mask1 = (dates_table['daysinmonth'] == 29)
            dates_table.loc[mask1,'daysinmonth'] = 28
    elif pygem_prms['time']['timestep'] == 'daily':
        # Automatically generate daily (freq = 'D') dates
        dates_table = pd.DataFrame({'date' : pd.date_range(startdate, enddate, freq='D')})
        # Extract attributes for dates_table
        dates_table['year'] = dates_table['date'].dt.year
        dates_table['month'] = dates_table['date'].dt.month
        dates_table['day'] = dates_table['date'].dt.day
        dates_table['daysinmonth'] = dates_table['date'].dt.daysinmonth
        # Set date as index
        dates_table.set_index('date', inplace=True)
        # Remove leap year days if user selected this with option_leapyear
        if pygem_prms['time']['option_leapyear'] == 0:
            # First, change 'daysinmonth' number
            mask1 = dates_table['daysinmonth'] == 29
            dates_table.loc[mask1,'daysinmonth'] = 28
            # Next, remove the 29th days from the dates
            mask2 = ((dates_table['month'] == 2) & (dates_table['day'] == 29))
            dates_table.drop(dates_table[mask2].index, inplace=True)
    else:
        print("\n\nError: Please select 'daily' or 'monthly' for gcm_timestep. Exiting model run now.\n")
        exit()
    # Add column for water year
    # Water year for northern hemisphere using USGS definition (October 1 - September 30th),
    # e.g., water year for 2000 is from October 1, 1999 - September 30, 2000
    dates_table['wateryear'] = dates_table['year']
    for step in range(dates_table.shape[0]):
        if dates_table.loc[step, 'month'] >= 10:
            dates_table.loc[step, 'wateryear'] = dates_table.loc[step, 'year'] + 1
    # Add column for seasons
    # create a season dictionary to assist groupby functions
    seasondict = {}
    month_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    season_list = []
    for i in range(len(month_list)):
        if (month_list[i] >= pygem_prms['time']['summer_month_start'] and month_list[i] < pygem_prms['time']['winter_month_start']):
            season_list.append('summer')
            seasondict[month_list[i]] = season_list[i]
        else:
            season_list.append('winter')
            seasondict[month_list[i]] = season_list[i]
    dates_table['season'] = dates_table['month'].apply(lambda x: seasondict[x])
    return dates_table


def daysinmonth(year, month):
    """
    Return days in month based on the month and year

    Parameters
    ----------
    year : str
    month : str

    Returns
    -------
    integer of the days in the month
    """
    if year%4 == 0:
        daysinmonth_dict = {
                1:31, 2:29, 3:31, 4:30, 5:31, 6:30, 7:31, 8:31, 9:30, 10:31, 11:30, 12:31}
    else:
        daysinmonth_dict = {
                1:31, 2:28, 3:31, 4:30, 5:31, 6:30, 7:31, 8:31, 9:30, 10:31, 11:30, 12:31}
    return daysinmonth_dict[month]


def hypsometrystats(hyps_table, thickness_table):
    """Calculate the volume and mean associated with the hypsometry data.

    Output is a series of the glacier volume [km**3] and mean elevation values [m a.s.l.].
    """
    # Glacier volume [km**3]
    glac_volume = (hyps_table * thickness_table/1000).sum(axis=1).values
    # Mean glacier elevation
    glac_hyps_mean = np.zeros(glac_volume.shape)
    glac_hyps_mean[glac_volume > 0] = ((hyps_table[glac_volume > 0].values *
                                        hyps_table[glac_volume > 0].columns.values.astype(int)).sum(axis=1) /
                                       hyps_table[glac_volume > 0].values.sum(axis=1))
    # Median computations
#    main_glac_hyps_cumsum = np.cumsum(hyps_table, axis=1)
#    for glac in range(hyps_table.shape[0]):
#        # Median glacier elevation
#        # Computed as the elevation when the normalized cumulative sum of the glacier area exceeds 0.5 (50%)
#        series_glac_hyps_cumsumnorm = main_glac_hyps_cumsum.loc[glac,:].copy() / glac_area.iloc[glac]
#        series_glac_hyps_cumsumnorm_positions = (np.where(series_glac_hyps_cumsumnorm > 0.5))[0]
#        glac_hyps_median = main_glac_hyps.columns.values[series_glac_hyps_cumsumnorm_positions[0]]
#    NOTE THERE IS A 20 m (+/- 5 m) OFFSET BETWEEN THE 10 m PRODUCT FROM HUSS AND THE RGI INVENTORY """
    return glac_volume, glac_hyps_mean


def import_Husstable(rgi_table, filepath, filedict, drop_col_names, indexname=pygem_prms['rgi']['indexname'], option_shift_elevbins_20m=True):
    """Use the dictionary specified by the user to extract the desired variable.
    The files must be in the proper units (ice thickness [m], area [km2], width [km]) and should be pre-processed.

    Output is a Pandas DataFrame of the variable for all the glaciers in the model run
    (rows = GlacNo, columns = elevation bins).

    Line Profiling: Loading in the table takes the most time (~2.3 s)
    """
    rgi_regionsO1 = sorted(list(rgi_table.O1Region.unique()))
    glac_no = [x.split('-')[1] for x in rgi_table.RGIId.values]
    glac_no_byregion = {}
    for region in rgi_regionsO1:
        glac_no_byregion[region] = []
    for i in glac_no:
        region = int(i.split('.')[0])
        glac_no_only = i.split('.')[1]
        glac_no_byregion[int(region)].append(glac_no_only)

    # Load data for each region
    for count, region in enumerate(rgi_regionsO1):
        # Select regional data for indexing
        glac_no = sorted(glac_no_byregion[region])
        rgi_table_region = rgi_table.iloc[np.where(rgi_table.O1Region.values == region)[0]]

        # Load table
        ds = pd.read_csv(filepath + filedict[region])

        # Select glaciers based on 01Index value from main_glac_rgi table
        #  as long as Huss tables have all rows associated with rgi attribute table, then this shortcut works
        glac_table = ds.iloc[rgi_table_region['O1Index'].values]
        # Merge multiple regions
        if count == 0:
            glac_table_all = glac_table
        else:
            # If more columns in region, then need to expand existing dataset
            if glac_table.shape[1] > glac_table_all.shape[1]:
                all_col = list(glac_table_all.columns.values)
                reg_col = list(glac_table.columns.values)
                new_cols = [item for item in reg_col if item not in all_col]
                for new_col in new_cols:
                    glac_table_all[new_col] = 0
            elif glac_table.shape[1] < glac_table_all.shape[1]:
                all_col = list(glac_table_all.columns.values)
                reg_col = list(glac_table.columns.values)
                new_cols = [item for item in all_col if item not in reg_col]
                for new_col in new_cols:
                    glac_table[new_col] = 0
            glac_table_all = glac_table_all.append(glac_table)

    # Clean up table and re-index (make copy to avoid SettingWithCopyWarning)
    glac_table_copy = glac_table_all.copy()
    glac_table_copy.reset_index(drop=True, inplace=True)
    glac_table_copy.index.name = indexname
    # drop columns that are not elevation bins
    glac_table_copy.drop(drop_col_names, axis=1, inplace=True)
    # change NAN from -99 to 0
    glac_table_copy[glac_table_copy==-99] = 0.
    # Shift Huss bins by 20 m since the elevation bins appear to be 20 m higher than they should be
    if option_shift_elevbins_20m:
        colnames = glac_table_copy.columns.tolist()[:-2]
        glac_table_copy = glac_table_copy.iloc[:,2:]
        glac_table_copy.columns = colnames
    return glac_table_copy


# def selectcalibrationdata(main_glac_rgi):
#     """
#     Select geodetic mass balance of all glaciers in the model run that have a geodetic mass balance.  The geodetic mass
#     balances are stored in a csv file.
#     """
#     # Import .csv file
#     rgi_region = int(main_glac_rgi.loc[main_glac_rgi.index.values[0],'RGIId'].split('-')[1].split('.')[0])
#     ds = pd.read_csv(pygem_prms.cal_mb_filepath + pygem_prms.cal_mb_filedict[rgi_region])
#     main_glac_calmassbal = np.zeros((main_glac_rgi.shape[0],4))
#     ds[pygem_prms['rgi']['rgi_O1Id_colname']] = ((ds[pygem_prms.cal_rgi_colname] % 1) * 10**5).round(0).astype(int)
#     ds_subset = ds[[pygem_prms['rgi']['rgi_O1Id_colname'], pygem_prms.massbal_colname, pygem_prms.massbal_uncertainty_colname,
#                     pygem_prms.massbal_time1, pygem_prms.massbal_time2]].values
#     rgi_O1Id = main_glac_rgi[pygem_prms['rgi']['rgi_O1Id_colname']].values
#     for glac in range(rgi_O1Id.shape[0]):
#         try:
#             # Grab the mass balance based on the RGIId Order 1 glacier number
#             main_glac_calmassbal[glac,:] = ds_subset[np.where(np.in1d(ds_subset[:,0],rgi_O1Id[glac])==True)[0][0],1:]
#             #  np.in1d searches if there is a match in the first array with the second array provided and returns an
#             #   array with same length as first array and True/False values. np.where then used to identify the index
#             #   where there is a match, which is then used to select the massbalance value
#             #  Use of numpy arrays for indexing and this matching approach is much faster than looping through; however,
#             #   need the for loop because np.in1d does not order the values that match; hence, need to do it 1 at a time
#         except:
#             # If there is no mass balance data available for the glacier, then set as NaN
#             main_glac_calmassbal[glac,:] = np.empty(4)
#             main_glac_calmassbal[glac,:] = np.nan
#     main_glac_calmassbal = pd.DataFrame(main_glac_calmassbal,
#                                         columns=[pygem_prms.massbal_colname, pygem_prms.massbal_uncertainty_colname,
#                                                  pygem_prms.massbal_time1, pygem_prms.massbal_time2])
#     return main_glac_calmassbal


def selectglaciersrgitable(glac_no=None, rgi_regionsO1=None, rgi_regionsO2='all', rgi_glac_number='all',
                           rgi_fp=pygem_prms['root'] + pygem_prms['rgi']['rgi_relpath'], 
                           rgi_cols_drop=pygem_prms['rgi']['rgi_cols_drop'],
                           rgi_O1Id_colname=pygem_prms['rgi']['rgi_O1Id_colname'],
                           rgi_glacno_float_colname=pygem_prms['rgi']['rgi_glacno_float_colname'],
                           indexname=pygem_prms['rgi']['indexname'],
                           include_landterm=True,include_laketerm=True,include_tidewater=True,
                           glac_no_skip=pygem_prms['setup']['glac_no_skip'],
                           min_glac_area_km2=0,
                           debug=False):
    """
    Select all glaciers to be used in the model run according to the regions and glacier numbers defined by the RGI
    glacier inventory. This function returns the rgi table associated with all of these glaciers.

    glac_no : list of strings
        list of strings of RGI glacier numbers (e.g., ['1.00001', '13.00001'])
    rgi_regionsO1 : list of integers
        list of integers of RGI order 1 regions (e.g., [1, 13])
    rgi_regionsO2 : list of integers or 'all'
        list of integers of RGI order 2 regions or simply 'all' for all the order 2 regions
    rgi_glac_number : list of strings
        list of RGI glacier numbers without the region (e.g., ['00001', '00002'])

    Output: Pandas DataFrame of the glacier statistics for each glacier in the model run
    (rows = GlacNo, columns = glacier statistics)
    """
    if glac_no is not None:
        glac_no_byregion = {}
        rgi_regionsO1 = [int(i.split('.')[0]) for i in glac_no]
        rgi_regionsO1 = list(set(rgi_regionsO1))
        for region in rgi_regionsO1:
            glac_no_byregion[region] = []
        for i in glac_no:
            region = i.split('.')[0]
            glac_no_only = i.split('.')[1]
            glac_no_byregion[int(region)].append(glac_no_only)

        for region in rgi_regionsO1:
            glac_no_byregion[region] = sorted(glac_no_byregion[region])

    # Create an empty dataframe
    rgi_regionsO1 = sorted(rgi_regionsO1)
    glacier_table = pd.DataFrame()
    for region in rgi_regionsO1:

        if glac_no is not None:
            rgi_glac_number = glac_no_byregion[region]

#        if len(rgi_glac_number) < 50:

        for i in os.listdir(rgi_fp):
            if i.startswith(str(region).zfill(2)) and i.endswith('.csv'):
                rgi_fn = i
        try:
            csv_regionO1 = pd.read_csv(rgi_fp + rgi_fn)
        except:
            csv_regionO1 = pd.read_csv(rgi_fp + rgi_fn, encoding='latin1')
        
        # Populate glacer_table with the glaciers of interest
        if rgi_regionsO2 == 'all' and rgi_glac_number == 'all':
            if debug:
                print("All glaciers within region(s) %s are included in this model run." % (region))
            if glacier_table.empty:
                glacier_table = csv_regionO1
            else:
                glacier_table = pd.concat([glacier_table, csv_regionO1], axis=0)
        elif rgi_regionsO2 != 'all' and rgi_glac_number == 'all':
            if debug:
                print("All glaciers within subregion(s) %s in region %s are included in this model run." %
                    (rgi_regionsO2, region))
            for regionO2 in rgi_regionsO2:
                if glacier_table.empty:
                    glacier_table = csv_regionO1.loc[csv_regionO1['O2Region'] == regionO2]
                else:
                    glacier_table = (pd.concat([glacier_table, csv_regionO1.loc[csv_regionO1['O2Region'] ==
                                                                                regionO2]], axis=0))
        else:
            if len(rgi_glac_number) < 20:
                print("%s glaciers in region %s are included in this model run: %s" % (len(rgi_glac_number), region,
                                                                                       rgi_glac_number))
            else:
                print("%s glaciers in region %s are included in this model run: %s and more" %
                      (len(rgi_glac_number), region, rgi_glac_number[0:50]))
                
            rgiid_subset = ['RGI60-' + str(region).zfill(2) + '.' + x for x in rgi_glac_number] 
            rgiid_all = list(csv_regionO1.RGIId.values)
            rgi_idx = [rgiid_all.index(x) for x in rgiid_subset if x in rgiid_all]
            if glacier_table.empty:
                glacier_table = csv_regionO1.loc[rgi_idx]
            else:
                glacier_table = (pd.concat([glacier_table, csv_regionO1.loc[rgi_idx]],
                                           axis=0))
                    
    glacier_table = glacier_table.copy()
    # reset the index so that it is in sequential order (0, 1, 2, etc.)
    glacier_table.reset_index(inplace=True)
    # drop connectivity 2 for Greenland and Antarctica
    glacier_table = glacier_table.loc[glacier_table['Connect'].isin([0,1])]
    glacier_table.reset_index(drop=True, inplace=True)
    # change old index to 'O1Index' to be easier to recall what it is
    glacier_table.rename(columns={'index': 'O1Index'}, inplace=True)
    # Record the reference date
    glacier_table['RefDate'] = glacier_table['BgnDate']
    # if there is an end date, then roughly average the year
    enddate_idx = glacier_table.loc[(glacier_table['EndDate'] > 0), 'EndDate'].index.values
    glacier_table.loc[enddate_idx,'RefDate'] = (
            np.mean((glacier_table.loc[enddate_idx,['BgnDate', 'EndDate']].values / 10**4).astype(int),
                    axis=1).astype(int) * 10**4 + 9999)
    # drop columns of data that is not being used
    glacier_table.drop(rgi_cols_drop, axis=1, inplace=True)
    # add column with the O1 glacier numbers
    glacier_table[rgi_O1Id_colname] = (
            glacier_table['RGIId'].str.split('.').apply(pd.Series).loc[:,1].astype(int))
    glacier_table['rgino_str'] = [x.split('-')[1] for x in glacier_table.RGIId.values]
#    glacier_table[rgi_glacno_float_colname] = (np.array([np.str.split(glacier_table['RGIId'][x],'-')[1]
#                                                    for x in range(glacier_table.shape[0])]).astype(float))
    glacier_table[rgi_glacno_float_colname] = (np.array([x.split('-')[1] for x in glacier_table['RGIId']]
#            [np.str.split(glacier_table['RGIId'][x],'-')[1]
#                                                    for x in range(glacier_table.shape[0])]
            ).astype(float))
    # set index name
    glacier_table.index.name = indexname
    # Longitude between 0-360deg (no negative)
    glacier_table['CenLon_360'] = glacier_table['CenLon']
    glacier_table.loc[glacier_table['CenLon'] < 0, 'CenLon_360'] = (
            360 + glacier_table.loc[glacier_table['CenLon'] < 0, 'CenLon_360'])
    # Subset glaciers based on their terminus type
    termtype_values = []
    if include_landterm:
        termtype_values.append(0)
        # assume dry calving, regenerated, and not assigned are land-terminating
        termtype_values.append(3)
        termtype_values.append(4)
        termtype_values.append(9)
    if include_tidewater:
        termtype_values.append(1)
        # assume shelf-terminating glaciers are tidewater
        termtype_values.append(5)
    if include_laketerm:
        termtype_values.append(2)
    glacier_table = glacier_table.loc[glacier_table['TermType'].isin(termtype_values)]
    glacier_table.reset_index(inplace=True, drop=True)
    # Glacier number with no trailing zeros
    glacier_table['glacno'] = [str(int(x.split('-')[1].split('.')[0])) + '.' + x.split('-')[1].split('.')[1]
                               for x in glacier_table.RGIId]
    
    # Remove glaciers below threshold
    glacier_table = glacier_table.loc[glacier_table['Area'] > min_glac_area_km2,:]
    glacier_table.reset_index(inplace=True, drop=True)

    # Remove glaciers that are meant to be skipped
    if glac_no_skip is not None:
        glac_no_all = list(glacier_table['glacno'])
        glac_no_unique = [x for x in glac_no_all if x not in glac_no_skip]
        unique_idx = [glac_no_all.index(x) for x in glac_no_unique]
        glacier_table = glacier_table.loc[unique_idx,:]
        glacier_table.reset_index(inplace=True, drop=True)

    print("This study is focusing on %s glaciers in region %s" % (glacier_table.shape[0], rgi_regionsO1))

    return glacier_table

    # OPTION 2: CUSTOMIZE REGIONS USING A SHAPEFILE that specifies the
    #           various regions according to the RGI IDs, i.e., add an
    #           additional column to the RGI table.
    # [INSERT CODE FOR IMPORTING A SHAPEFILE]
    #   (1) import shapefile with custom boundaries, (2) grab the RGIIDs
    #   of glaciers that are in these boundaries, (3) perform calibration
    #   using these alternative boundaries that may (or may not) be more
    #   representative of regional processes/climate
    #   Note: this is really only important for calibration purposes and
    #         post-processing when you want to show results over specific
    #         regions.
    # Development Note: if create another method for selecting glaciers,
    #                   make sure that update way to select glacier
    #                   hypsometry as well.


def split_list(lst, n=1, option_ordered=1, group_thousands=False):
    """
    Split list into batches for the supercomputer.
    
    Parameters
    ----------
    lst : list
        List that you want to split into separate batches
    n : int
        Number of batches to split glaciers into.
    
    Returns
    -------
    lst_batches : list
        list of n lists that have sequential values in each list
    """
    # If batches is more than list, then there will be one glacier in each batch
    if option_ordered == 1:
        if n > len(lst):
            n = len(lst)
        n_perlist_low = int(len(lst)/n)
        n_perlist_high = int(np.ceil(len(lst)/n))
        lst_copy = lst.copy()
        count = 0
        lst_batches = []
        for x in np.arange(n):
            count += 1
            if count <= len(lst) % n:
                lst_subset = lst_copy[0:n_perlist_high]
                lst_batches.append(lst_subset)
                [lst_copy.remove(i) for i in lst_subset]
            else:
                lst_subset = lst_copy[0:n_perlist_low]
                lst_batches.append(lst_subset)
                [lst_copy.remove(i) for i in lst_subset]
        
    else:
        if n > len(lst):
            n = len(lst)
    
        lst_batches = [[] for x in np.arange(n)]
        nbatch = 0
        for count, x in enumerate(lst):
            if count%n == 0:
                nbatch = 0
    
            lst_batches[nbatch].append(x)
            
            nbatch += 1

    if group_thousands:
        # get unique sets of thousand glaciers (ie. RGIXX.YY)
        # this may be preferrable when running script that download glacier directories from oggm
        # otherwise if two batches contain glacier ids from the same set of thousand glaciers,
        # two downloads from oggm may run simultaneously and cause conflicts
        sets = [x[:5] for x in lst for lst in lst_batches]
        sets = list(set(sets))
        lst_batches_th = []
        # keep the number of batches, but move items around to not have sets of RGIXX.YY ids in more than one batch
        for s in sets:
            merged = [item for sublist in lst_batches for item in sublist if item[:5]==s]
            lst_batches_th.append(merged)
        # ensure that number of batches doesn't exceed original number
        while len(lst_batches_th) > n:
            # move shortest batch to next shortest batch
            lengths = np.asarray([len(batch) for batch in lst_batches_th])
            sorted = lengths.argsort()
            idx0 = sorted[0]
            idx1 = sorted[1]
            
            lst_batches_th[idx1].extend(lst_batches_th[idx0])
            del lst_batches_th[idx0]

        lst_batches = lst_batches_th

    return lst_batches