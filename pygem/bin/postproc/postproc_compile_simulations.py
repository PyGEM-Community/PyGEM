"""
Python Glacier Evolution Model (PyGEM)

copyright © 2024 Brandon Tober <btober@cmu.edu> David Rounce <drounce@cmu.edu>

Distrubted under the MIT lisence

compile individual glacier simulations to the regional level
"""
# imports
import os
import glob
import sys
import time
import argparse
import xarray as xr
import numpy as np
import pygem
from datetime import datetime
import multiprocessing

# pygem imports
import pygem
from pygem.setup.config import ConfigManager
# instantiate ConfigManager
config_manager = ConfigManager()
# read the config
pygem_prms = config_manager.read_config()
import pygem.pygem_modelsetup as modelsetup

rgi_reg_dict = {'all':'Global',
                'all_no519':'Global, excl. GRL and ANT',
                'global':'Global',
                1:'Alaska',
                2:'W Canada & US',
                3:'Arctic Canada North',
                4:'Arctic Canada South',
                5:'Greenland Periphery',
                6:'Iceland',
                7:'Svalbard',
                8:'Scandinavia',
                9:'Russian Arctic',
                10:'North Asia',
                11:'Central Europe',
                12:'Caucasus & Middle East',
                13:'Central Asia',
                14:'South Asia West',
                15:'South Asia East',
                16:'Low Latitudes',
                17:'Southern Andes',
                18:'New Zealand',
                19:'Antarctic & Subantarctic'
                }


def run(args):
    # unpack arguments
    reg, simpath, gcms, realizations, scenario, calibration, bias_adj, gcm_startyear, gcm_endyear, vars = args
    print(f'RGI region {reg}')
    # #%% ----- PROCESS DATASETS FOR INDIVIDUAL GLACIERS AND ELEVATION BINS -----
    comppath = simpath + '/compile/'

    # define base directory
    base_dir = simpath + "/" + str(reg).zfill(2) + "/"

    # get all glaciers in region to see which fraction ran successfully
    main_glac_rgi_all = modelsetup.selectglaciersrgitable(rgi_regionsO1=[reg], 
                                                        rgi_regionsO2='all', rgi_glac_number='all', 
                                                        glac_no=None,
                                                        debug=True)

    glacno_list_all = list(main_glac_rgi_all['rgino_str'].values)

    ### CREATE BATCHES ###
    # get last glacier number to define number of batches
    lastn = int(sorted(glacno_list_all)[-1].split('.')[1])
    # round up to thosand
    batch_interval = 1000
    last_thous = np.ceil(lastn / batch_interval) * batch_interval
    # get number of batches
    nbatches = last_thous // batch_interval

    # split glaciers into groups of a thousand based on all glaciers in region
    glacno_list_batches = modelsetup.split_list(glacno_list_all, n=nbatches, group_thousands=True)

    # make sure batch sublists are sorted properly and that each goes from NN001 to N(N+1)000
    glacno_list_batches = sorted(glacno_list_batches, key=lambda x:x[0])
    for i in range(len(glacno_list_batches) - 1): 
        glacno_list_batches[i].append(glacno_list_batches[i+1][0])
        glacno_list_batches[i+1].pop(0)

    ############################################################
    ### get time values - should be the same across all sims ###
    ############################################################
    if scenario:
        # ensure scenario has been run for each gcm
        for gcm in gcms:
            if scenario not in os.listdir(base_dir + '/' + gcm):
                # remove the gcm from our gcm list if the desired scenario is not contained
                gcms.remove(gcm)
                print(f'scenario {scenario} not found for {gcm}, skipping')
        fn = glob.glob(base_dir + gcm  + "/" + scenario  + "/stats/" + f'*{gcm}_{scenario}_{realizations[0]}_{calibration}_ba{bias_adj}_*_{gcm_startyear}_{gcm_endyear}_all.nc'.replace('__','_'))[0]
    else:
        fn = glob.glob(base_dir + gcm  + "/stats/" + f'*{gcm}_{calibration}_ba{bias_adj}_*_{gcm_startyear}_{gcm_endyear}_all.nc')[0]
    nsets = fn.split('/')[-1].split('_')[-4]

    ds_glac = xr.open_dataset(fn)
    year_values = ds_glac.year.values
    time_values = ds_glac.time.values
    # check if desired vars are in ds
    ds_vars = list(ds_glac.keys())
    missing_vars = list(set(vars) - set(ds_vars))
    if len(missing_vars) > 0:
        vars = list(set(vars).intersection(ds_vars))
        print(f'Warning: Requested variables are missing: {missing_vars}')
    ############################################################

    print(f'Compiling GCMS: {gcms}')
    print(f'Realizations: {realizations}')
    print(f'Variables: {vars}')

    ### LEVEL I ###
    # loop through glacier batches of 1000
    for nbatch, glacno_list in enumerate(glacno_list_batches):
        print(f'Batch {nbatch}:')

        # batch start timer
        loop_start = time.time()

        # get batch start and end numbers
        batch_start = glacno_list[0].split('.')[1]
        batch_end = glacno_list[-1].split('.')[1]

        print(nbatch, batch_start, batch_end)

        # get all glacier info for glaciers in batch
        main_glac_rgi_batch = main_glac_rgi_all.loc[main_glac_rgi_all.apply(lambda x: x.rgino_str in glacno_list, axis=1)]

        # instantiate variables that will hold all concatenated data for GCMs/realizations
        # monthly vars
        reg_glac_allgcms_runoff_monthly = None
        reg_offglac_allgcms_runoff_monthly = None
        reg_glac_allgcms_acc_monthly = None
        reg_glac_allgcms_melt_monthly = None
        reg_glac_allgcms_refreeze_monthly = None
        reg_glac_allgcms_frontalablation_monthly = None
        reg_glac_allgcms_massbaltotal_monthly = None
        reg_glac_allgcms_prec_monthly = None
        reg_glac_allgcms_mass_monthly = None

        # annual vars
        reg_glac_allgcms_area_annual = None
        reg_glac_allgcms_mass_annual = None    

        ### LEVEL II ###
        # for each batch, loop through GCM(s) and realization(s)
        for gcm in gcms:
            # get list of glacier simulation files 
            sim_dir = base_dir + gcm  + '/' + scenario + '/stats/'

            ### LEVEL III ###
            for realization in realizations:
                print(f'GCM: {gcm} {realization}')
                fps = glob.glob(sim_dir + f'*{gcm}_{scenario}_{realization}_{calibration}_ba{bias_adj}_{nsets}_{gcm_startyear}_{gcm_endyear}_all.nc'.replace('__','_'))

                # during 0th batch, print the regional stats of glaciers and area successfully simulated for all regional glaciers for given gcm scenario
                if nbatch==0:
                    # Glaciers with successful runs to process
                    glacno_ran = [x.split('/')[-1].split('_')[0] for x in fps]
                    glacno_ran = [x.split('.')[0].zfill(2) + '.' + x[-5:] for x in glacno_ran]
                    main_glac_rgi = main_glac_rgi_all.loc[main_glac_rgi_all.apply(lambda x: x.rgino_str in glacno_ran, axis=1)]
                    print(f'Glaciers successfully simulated:\n  - {main_glac_rgi.shape[0]} of {main_glac_rgi_all.shape[0]} glaciers ({np.round(main_glac_rgi.shape[0]/main_glac_rgi_all.shape[0]*100,3)}%)')
                    print(f'  - {np.round(main_glac_rgi.Area.sum(),0)} km2 of {np.round(main_glac_rgi_all.Area.sum(),0)} km2 ({np.round(main_glac_rgi.Area.sum()/main_glac_rgi_all.Area.sum()*100,3)}%)')

                # instantiate variables that will hold concatenated data for the current GCM
                # monthly vars
                reg_glac_gcm_runoff_monthly = None
                reg_offglac_gcm_runoff_monthly = None
                reg_glac_gcm_acc_monthly = None
                reg_glac_gcm_melt_monthly = None
                reg_glac_gcm_refreeze_monthly = None
                reg_glac_gcm_frontalablation_monthly = None
                reg_glac_gcm_massbaltotal_monthly = None
                reg_glac_gcm_prec_monthly = None
                reg_glac_gcm_mass_monthly = None

                # annual vars
                reg_glac_gcm_area_annual = None
                reg_glac_gcm_mass_annual = None    


                ### LEVEL IV ###
                # loop through each glacier in batch list
                for i, glacno in enumerate(glacno_list):
                    # get glacier string and file name
                    glacier_str = '{0:0.5f}'.format(float(glacno))
                    glacno_fn = f'{sim_dir}/{glacier_str}_{gcm}_{scenario}_{realization}_{calibration}_ba{bias_adj}_{nsets}_{gcm_startyear}_{gcm_endyear}_all.nc'.replace('__','_')
                    # try to load all glaciers in region
                    try:
                        # open netcdf file
                        ds_glac = xr.open_dataset(glacno_fn)
                        # get monthly vars
                        glac_runoff_monthly = ds_glac.glac_runoff_monthly.values
                        offglac_runoff_monthly = ds_glac.offglac_runoff_monthly.values
                        # try extra vars
                        try:
                            glac_acc_monthly = ds_glac.glac_acc_monthly.values
                            glac_melt_monthly = ds_glac.glac_melt_monthly.values
                            glac_refreeze_monthly = ds_glac.glac_refreeze_monthly.values
                            glac_frontalablation_monthly = ds_glac.glac_frontalablation_monthly.values
                            glac_massbaltotal_monthly = ds_glac.glac_massbaltotal_monthly.values
                            glac_prec_monthly = ds_glac.glac_prec_monthly.values
                            glac_mass_monthly = ds_glac.glac_mass_monthly.values
                        except:
                            glac_acc_monthly = np.full((1,len(time_values)), np.nan)
                            glac_melt_monthly = np.full((1,len(time_values)), np.nan)
                            glac_refreeze_monthly = np.full((1,len(time_values)), np.nan)
                            glac_frontalablation_monthly = np.full((1,len(time_values)), np.nan)
                            glac_massbaltotal_monthly = np.full((1,len(time_values)), np.nan)
                            glac_prec_monthly = np.full((1,len(time_values)), np.nan)
                            glac_mass_monthly = np.full((1,len(time_values)), np.nan)
                        # get annual vars
                        glac_area_annual = ds_glac.glac_area_annual.values
                        glac_mass_annual = ds_glac.glac_mass_annual.values


                    # if glacier output DNE in sim output file, create empty nan arrays to keep record of missing glaciers
                    except:
                        # monthly vars
                        glac_runoff_monthly = np.full((1,len(time_values)), np.nan)
                        offglac_runoff_monthly = np.full((1,len(time_values)), np.nan)
                        glac_acc_monthly = np.full((1,len(time_values)), np.nan)
                        glac_melt_monthly = np.full((1,len(time_values)), np.nan)
                        glac_refreeze_monthly = np.full((1,len(time_values)), np.nan)
                        glac_frontalablation_monthly = np.full((1,len(time_values)), np.nan)
                        glac_massbaltotal_monthly = np.full((1,len(time_values)), np.nan)
                        glac_prec_monthly = np.full((1,len(time_values)), np.nan)
                        glac_mass_monthly = np.full((1,len(time_values)), np.nan)
                        # annual vars
                        glac_area_annual = np.full((1,year_values.shape[0]), np.nan)
                        glac_mass_annual = np.full((1,year_values.shape[0]), np.nan)

                    
                    # append each glacier output to master regional set of arrays
                    if reg_glac_gcm_mass_annual is None:
                        # monthly vars
                        reg_glac_gcm_runoff_monthly = glac_runoff_monthly
                        reg_offglac_gcm_runoff_monthly = offglac_runoff_monthly
                        reg_glac_gcm_acc_monthly = glac_acc_monthly
                        reg_glac_gcm_melt_monthly = glac_melt_monthly
                        reg_glac_gcm_refreeze_monthly = glac_refreeze_monthly
                        reg_glac_gcm_frontalablation_monthly = glac_frontalablation_monthly
                        reg_glac_gcm_massbaltotal_monthly = glac_massbaltotal_monthly
                        reg_glac_gcm_prec_monthly = glac_prec_monthly
                        reg_glac_gcm_mass_monthly = glac_mass_monthly
                        # annual vars
                        reg_glac_gcm_area_annual = glac_area_annual
                        reg_glac_gcm_mass_annual = glac_mass_annual    
                    # otherwise concatenate existing arrays
                    else:
                        # monthly vars
                        reg_glac_gcm_runoff_monthly = np.concatenate((reg_glac_gcm_runoff_monthly, glac_runoff_monthly), axis=0)
                        reg_offglac_gcm_runoff_monthly = np.concatenate((reg_offglac_gcm_runoff_monthly, offglac_runoff_monthly), axis=0)
                        reg_glac_gcm_acc_monthly = np.concatenate((reg_glac_gcm_acc_monthly, glac_acc_monthly), axis=0)
                        reg_glac_gcm_melt_monthly = np.concatenate((reg_glac_gcm_melt_monthly, glac_melt_monthly), axis=0)
                        reg_glac_gcm_refreeze_monthly = np.concatenate((reg_glac_gcm_refreeze_monthly, glac_refreeze_monthly), axis=0)
                        reg_glac_gcm_frontalablation_monthly = np.concatenate((reg_glac_gcm_frontalablation_monthly, glac_frontalablation_monthly), axis=0)
                        reg_glac_gcm_massbaltotal_monthly = np.concatenate((reg_glac_gcm_massbaltotal_monthly, glac_massbaltotal_monthly), axis=0)
                        reg_glac_gcm_prec_monthly = np.concatenate((reg_glac_gcm_prec_monthly, glac_prec_monthly), axis=0)
                        reg_glac_gcm_mass_monthly = np.concatenate((reg_glac_gcm_mass_monthly, glac_mass_monthly), axis=0)
                        # annual vars
                        reg_glac_gcm_area_annual = np.concatenate((reg_glac_gcm_area_annual, glac_area_annual), axis=0)
                        reg_glac_gcm_mass_annual = np.concatenate((reg_glac_gcm_mass_annual, glac_mass_annual), axis=0)  

                # aggregate gcms
                if reg_glac_allgcms_runoff_monthly is None:
                    # monthly vars
                    reg_glac_allgcms_runoff_monthly = reg_glac_gcm_runoff_monthly[np.newaxis,:,:]
                    reg_offglac_allgcms_runoff_monthly = reg_offglac_gcm_runoff_monthly[np.newaxis,:,:]
                    reg_glac_allgcms_acc_monthly = reg_glac_gcm_acc_monthly[np.newaxis,:,:]
                    reg_glac_allgcms_melt_monthly = reg_glac_gcm_melt_monthly[np.newaxis,:,:]
                    reg_glac_allgcms_refreeze_monthly = reg_glac_gcm_refreeze_monthly[np.newaxis,:,:]
                    reg_glac_allgcms_frontalablation_monthly = reg_glac_gcm_frontalablation_monthly[np.newaxis,:,:]
                    reg_glac_allgcms_massbaltotal_monthly = reg_glac_gcm_massbaltotal_monthly[np.newaxis,:,:]
                    reg_glac_allgcms_prec_monthly = reg_glac_gcm_prec_monthly[np.newaxis,:,:]
                    reg_glac_allgcms_mass_monthly = reg_glac_gcm_mass_monthly[np.newaxis,:,:]
                    # annual vars
                    reg_glac_allgcms_area_annual = reg_glac_gcm_area_annual[np.newaxis,:,:]
                    reg_glac_allgcms_mass_annual = reg_glac_gcm_mass_annual[np.newaxis,:,:]
                else:
                    # monthly vrs
                    reg_glac_allgcms_runoff_monthly  = np.concatenate((reg_glac_allgcms_runoff_monthly, reg_glac_gcm_runoff_monthly[np.newaxis,:,:]), axis=0)
                    reg_offglac_allgcms_runoff_monthly  = np.concatenate((reg_offglac_allgcms_runoff_monthly, reg_offglac_gcm_runoff_monthly[np.newaxis,:,:]), axis=0)
                    reg_glac_allgcms_acc_monthly  = np.concatenate((reg_glac_allgcms_acc_monthly, reg_glac_gcm_acc_monthly[np.newaxis,:,:]), axis=0)
                    reg_glac_allgcms_melt_monthly  = np.concatenate((reg_glac_allgcms_melt_monthly, reg_glac_gcm_melt_monthly[np.newaxis,:,:]), axis=0)
                    reg_glac_allgcms_refreeze_monthly  = np.concatenate((reg_glac_allgcms_refreeze_monthly, reg_glac_gcm_refreeze_monthly[np.newaxis,:,:]), axis=0)
                    reg_glac_allgcms_frontalablation_monthly  = np.concatenate((reg_glac_allgcms_frontalablation_monthly, reg_glac_gcm_frontalablation_monthly[np.newaxis,:,:]), axis=0)
                    reg_glac_allgcms_massbaltotal_monthly  = np.concatenate((reg_glac_allgcms_massbaltotal_monthly, reg_glac_gcm_massbaltotal_monthly[np.newaxis,:,:]), axis=0)
                    reg_glac_allgcms_prec_monthly  = np.concatenate((reg_glac_allgcms_prec_monthly, reg_glac_gcm_prec_monthly[np.newaxis,:,:]), axis=0)
                    reg_glac_allgcms_mass_monthly  = np.concatenate((reg_glac_allgcms_mass_monthly, reg_glac_gcm_mass_monthly[np.newaxis,:,:]), axis=0)
                    # annual vars
                    reg_glac_allgcms_area_annual  = np.concatenate((reg_glac_allgcms_area_annual, reg_glac_gcm_area_annual[np.newaxis,:,:]), axis=0)
                    reg_glac_allgcms_mass_annual  = np.concatenate((reg_glac_allgcms_mass_annual, reg_glac_gcm_mass_annual[np.newaxis,:,:]), axis=0)

                            
        #===== CREATE NETCDF FILES=====

        # get common attributes
        rgiid_list = ['RGI60-' + x for x in glacno_list]
        cenlon_list = list(main_glac_rgi_batch.CenLon.values)
        cenlat_list = list(main_glac_rgi_batch.CenLat.values)
        attrs_dict = {'Region':str(reg) + ' - ' + rgi_reg_dict[reg],
                                    'source': f'PyGEMv{pygem.__version__}',
                                    'institution': pygem_prms['user']['institution'],
                                    'history': f"Created by {pygem_prms['user']['name']} ({pygem_prms['user']['email']}) on " + datetime.today().strftime('%Y-%m-%d'),
                                    'references': 'doi:10.1126/science.abo1324',
                                    'Conventions': 'CF-1.9',
                                    'featureType': 'timeSeries'}
        # loop through variables
        for var in vars:

            # get common coords
            if 'annual' in var:
                tvals = year_values
            else:
                tvals = time_values
            if realizations[0]:
                coords_dict=dict(
                        RGIId=(["glacier"], rgiid_list),
                        Climate_Model= (["realization"], realizations),
                        lon=(["glacier"], cenlon_list),
                        lat=(["glacier"], cenlat_list),
                        time=tvals,
                ) 
                coord_order = ["realization","glacier","time"]
            else:
                coords_dict=dict(
                        RGIId=(["glacier"], rgiid_list),
                        Climate_Model= (["model"], gcms),
                        lon=(["glacier"], cenlon_list),
                        lat=(["glacier"], cenlat_list),
                        time=tvals,
                ) 
                coord_order = ["model","glacier","time"]

            #glac_runoff_monthly
            if var=='glac_runoff_monthly':
                ds = xr.Dataset(
                        data_vars=dict(
                                glac_runoff_monthly=(coord_order, reg_glac_allgcms_runoff_monthly),
                                crs = np.nan
                                ),
                                coords=coords_dict,
                                attrs=attrs_dict
                                )
                ds.glac_runoff_monthly.attrs['long_name'] = 'glacier-wide runoff'
                ds.glac_runoff_monthly.attrs['units'] = 'm3'
                ds.glac_runoff_monthly.attrs['temporal_resolution'] = 'monthly'
                ds.glac_runoff_monthly.attrs['comment'] = 'runoff from the glacier terminus, which moves over time'
                ds.glac_runoff_monthly.attrs['grid_mapping'] = 'crs'

            #offglac_runoff_monthly
            elif var=='offglac_runoff_monthly':
                ds = xr.Dataset(
                        data_vars=dict(
                                offglac_runoff_monthly=(coord_order, reg_offglac_allgcms_runoff_monthly),
                                crs = np.nan
                                ),
                                coords=coords_dict,
                                attrs=attrs_dict
                                )
                ds.offglac_runoff_monthly.attrs['long_name'] = 'off-glacier-wide runoff'
                ds.offglac_runoff_monthly.attrs['units'] = 'm3'
                ds.offglac_runoff_monthly.attrs['temporal_resolution'] = 'monthly'
                ds.offglac_runoff_monthly.attrs['comment'] = 'off-glacier runoff from area where glacier no longer exists'
                ds.offglac_runoff_monthly.attrs['grid_mapping'] = 'crs'

            #glac_acc_monthly
            elif var=='glac_acc_monthly':
                ds = xr.Dataset(
                        data_vars=dict(
                                glac_acc_monthly=(coord_order, reg_glac_allgcms_acc_monthly),
                                crs = np.nan
                                ),
                                coords=coords_dict,
                                attrs=attrs_dict
                                )
                ds.glac_acc_monthly.attrs['long_name'] = 'glacier-wide accumulation, in water equivalent'
                ds.glac_acc_monthly.attrs['units'] = 'm3'
                ds.glac_acc_monthly.attrs['temporal_resolution'] = 'monthly'
                ds.glac_acc_monthly.attrs['comment'] = 'only the solid precipitation'
                ds.glac_acc_monthly.attrs['grid_mapping'] = 'crs'

            #glac_melt_monthly
            elif var=='glac_melt_monthly':
                ds = xr.Dataset(
                        data_vars=dict(
                                glac_melt_monthly=(coord_order, reg_glac_allgcms_melt_monthly),
                                crs = np.nan
                                ),
                                coords=coords_dict,
                                attrs=attrs_dict
                                )
                ds.glac_melt_monthly.attrs['long_name'] = 'glacier-wide melt, in water equivalent'
                ds.glac_melt_monthly.attrs['units'] = 'm3'
                ds.glac_melt_monthly.attrs['temporal_resolution'] = 'monthly'
                ds.glac_melt_monthly.attrs['grid_mapping'] = 'crs'

            #glac_refreeze_monthly
            elif var=='glac_refreeze_monthly':
                ds = xr.Dataset(
                        data_vars=dict(
                                glac_refreeze_monthly=(coord_order, reg_glac_allgcms_refreeze_monthly),
                                crs = np.nan
                                ),
                                coords=coords_dict,
                                attrs=attrs_dict
                                )
                ds.glac_refreeze_monthly.attrs['long_name'] = 'glacier-wide refreeze, in water equivalent'
                ds.glac_refreeze_monthly.attrs['units'] = 'm3'
                ds.glac_refreeze_monthly.attrs['temporal_resolution'] = 'monthly'
                ds.glac_refreeze_monthly.attrs['grid_mapping'] = 'crs'

            #glac_frontalablation_monthly
            elif var=='glac_frontalablation_monthly':
                ds = xr.Dataset(
                        data_vars=dict(
                                glac_frontalablation_monthly=(coord_order, reg_glac_allgcms_frontalablation_monthly),
                                crs = np.nan
                                ),
                                coords=dict(
                                        RGIId=(["glacier"], rgiid_list),
                                        Climate_Model= (["model"], gcms),
                                        lon=(["glacier"], cenlon_list),
                                        lat=(["glacier"], cenlat_list),
                                        time=time_values,
                                ),
                                attrs=attrs_dict
                                )
                ds.glac_frontalablation_monthly.attrs['long_name'] = 'glacier-wide frontal ablation, in water equivalent'
                ds.glac_frontalablation_monthly.attrs['units'] = 'm3'
                ds.glac_frontalablation_monthly.attrs['temporal_resolution'] = 'monthly'
                ds.glac_frontalablation_monthly.attrs['comment'] = 'mass losses from calving, subaerial frontal melting, \
                    sublimation above the waterline and subaqueous frontal melting below the waterline; \
                    positive values indicate mass lost like melt'
                ds.glac_frontalablation_monthly.attrs['grid_mapping'] = 'crs'

            #glac_massbaltotal_monthly
            elif var=='glac_massbaltotal_monthly':
                ds = xr.Dataset(
                        data_vars=dict(
                                glac_massbaltotal_monthly=(coord_order, reg_glac_allgcms_massbaltotal_monthly),
                                crs = np.nan
                                ),
                                coords=coords_dict,
                                attrs=attrs_dict
                                )
                ds.glac_massbaltotal_monthly.attrs['long_name'] = 'glacier-wide total mass balance, in water equivalent'
                ds.glac_massbaltotal_monthly.attrs['units'] = 'm3'
                ds.glac_massbaltotal_monthly.attrs['temporal_resolution'] = 'monthly'
                ds.glac_massbaltotal_monthly.attrs['comment'] = 'total mass balance is the sum of the climatic mass balance and frontal ablation'
                ds.glac_massbaltotal_monthly.attrs['grid_mapping'] = 'crs'


            #glac_prec_monthly
            elif var=='glac_prec_monthly':
                ds = xr.Dataset(
                        data_vars=dict(
                                glac_prec_monthly=(coord_order, reg_glac_allgcms_prec_monthly),
                                crs = np.nan
                                ),
                                coords=coords_dict,
                                attrs=attrs_dict
                                )
                ds.glac_prec_monthly.attrs['long_name'] = 'glacier-wide precipitation (liquid)'
                ds.glac_prec_monthly.attrs['units'] = 'm3'
                ds.glac_prec_monthly.attrs['temporal_resolution'] = 'monthly'
                ds.glac_prec_monthly.attrs['comment'] = 'only the liquid precipitation, solid precipitation excluded'
                ds.glac_prec_monthly.attrs['grid_mapping'] = 'crs'
    
            #glac_mass_monthly
            elif var=='glac_mass_monthly':
                ds = xr.Dataset(
                        data_vars=dict(
                                glac_mass_monthly=(coord_order, reg_glac_allgcms_mass_monthly),
                                crs = np.nan
                                ),
                                coords=coords_dict,
                                attrs=attrs_dict
                                )
                ds.glac_mass_monthly.attrs['long_name'] = 'glacier mass'
                ds.glac_mass_monthly.attrs['units'] = 'kg'
                ds.glac_mass_monthly.attrs['temporal_resolution'] = 'monthly'
                ds.glac_mass_monthly.attrs['comment'] = 'mass of ice based on area and ice thickness at start of the year and the monthly total mass balance'
                ds.glac_mass_monthly.attrs['grid_mapping'] = 'crs'

            #glac_area_annual
            elif var=='glac_area_annual':
                ds = xr.Dataset(
                        data_vars=dict(
                                glac_area_annual=(coord_order, reg_glac_allgcms_area_annual),
                                crs = np.nan
                                ),
                                coords=coords_dict,
                                attrs=attrs_dict
                                )
                ds.glac_area_annual.attrs['long_name'] = 'glacier area'
                ds.glac_area_annual.attrs['units'] = 'm2'
                ds.glac_area_annual.attrs['temporal_resolution'] = 'annual'
                ds.glac_area_annual.attrs['comment'] = 'area at start of the year'
                ds.glac_area_annual.attrs['grid_mapping'] = 'crs'

            #glac_mass_annual
            elif var=='glac_mass_annual':
                ds = xr.Dataset(
                        data_vars=dict(
                                glac_mass_annual=(coord_order, reg_glac_allgcms_mass_annual),
                                crs = np.nan
                                ),
                                coords=coords_dict,
                                attrs=attrs_dict
                                )
                ds.glac_mass_annual.attrs['long_name'] = 'glacier mass'
                ds.glac_mass_annual.attrs['units'] = 'kg'
                ds.glac_mass_annual.attrs['temporal_resolution'] = 'annual'
                ds.glac_mass_annual.attrs['comment'] = 'mass of ice based on area and ice thickness at start of the year'
                ds.glac_mass_annual.attrs['grid_mapping'] = 'crs'
        
            # crs attributes - same for all vars
            ds.crs.attrs['grid_mapping_name'] = 'latitude_longitude'
            ds.crs.attrs['longitude_of_prime_meridian'] = 0.0
            ds.crs.attrs['semi_major_axis'] = 6378137.0
            ds.crs.attrs['inverse_flattening'] = 298.257223563
            ds.crs.attrs['proj4text'] = '+proj=longlat +datum=WGS84 +no_defs'
            ds.crs.attrs['crs_wkt'] = 'GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4326"]]'
            
            # time attributes - different for monthly v annual
            ds.time.attrs['long_name'] = 'time'
            if 'annual' in var:
                ds.time.attrs['range'] = str(year_values[0]) + ' - ' + str(year_values[-1])
                ds.time.attrs['comment'] = 'years referring to the start of each year'
            elif 'monthly' in var:
                ds.time.attrs['range'] = str(time_values[0]) + ' - ' + str(time_values[-1])
                ds.time.attrs['comment'] = 'start of the month'
            
            ds.RGIId.attrs['long_name'] = 'Randolph Glacier Inventory Id'
            ds.RGIId.attrs['comment'] = 'RGIv6.0 (https://nsidc.org/data/nsidc-0770/versions/6)'
            ds.RGIId.attrs['cf_role'] = 'timeseries_id'

            if realizations[0]:
                ds.Climate_Model.attrs['long_name'] = f'{gcms[0]} realization'
            else:
                ds.Climate_Model.attrs['long_name'] = 'General Circulation Model'
            
            ds.lon.attrs['standard_name'] = 'longitude'
            ds.lon.attrs['long_name'] = 'longitude of glacier center'
            ds.lon.attrs['units'] = 'degrees_east'
            
            ds.lat.attrs['standard_name'] = 'latitude'
            ds.lat.attrs['long_name'] = 'latitude of glacier center'
            ds.lat.attrs['units'] = 'degrees_north'
            
            # save batch
            vn_fp = f'{comppath}/glacier_stats/{var}/{str(reg).zfill(2)}/'
            if not os.path.exists(vn_fp):
                os.makedirs(vn_fp, exist_ok=True)
                
            if realizations[0]:
                ds_fn = f'R{str(reg).zfill(2)}_{var}_{gcms[0]}_{scenario}_Batch-{str(batch_start)}-{str(batch_end)}_{calibration}_ba{bias_adj}_{nsets}_{gcm_startyear}_{gcm_endyear}_all.nc'.replace('__','_')
            else:
                ds_fn = f'R{str(reg).zfill(2)}_{var}_{scenario}_Batch-{str(batch_start)}-{str(batch_end)}_{calibration}_ba{bias_adj}_{nsets}_{gcm_startyear}_{gcm_endyear}_all.nc'.replace('__','_')

            ds.to_netcdf(vn_fp + ds_fn)

        loop_end = time.time()
        print(f'Batch {nbatch} runtime:\t{np.round(loop_end - loop_start,2)} seconds')
            
            
    ### MERGE BATCHES FOR ANNUAL VARS ###
    vns = ['glac_mass_annual', 'glac_area_annual']

    for vn in vns:
        if vn in vars:
            vn_fp = f'{comppath}glacier_stats/{vn}/{str(reg).zfill(2)}/'

            fn_merge_list_start = []

            if realizations[0]:
                fn_merge_list = glob.glob(f'{vn_fp}/R{str(reg).zfill(2)}_{vn}_{gcms[0]}_{scenario}_Batch-*_{calibration}_ba{bias_adj}_{nsets}_{gcm_startyear}_{gcm_endyear}_all.nc'.replace('__','_'))
            else:
                fn_merge_list = glob.glob(f'{vn_fp}/R{str(reg).zfill(2)}_{vn}_{scenario}_Batch-*_{calibration}_ba{bias_adj}_{nsets}_{gcm_startyear}_{gcm_endyear}_all.nc'.replace('__','_'))
            fn_merge_list_start = [int(f.split('-')[-2]) for f in fn_merge_list]
        
            if len(fn_merge_list) > 0:
                fn_merge_list = [x for _,x in sorted(zip(fn_merge_list_start,fn_merge_list))]
            
                ds = None
                for fn in fn_merge_list:
                    ds_batch = xr.open_dataset(fn)
                    
                    if ds is None:
                        ds = ds_batch
                    else:
                        ds = xr.concat([ds, ds_batch], dim="glacier")
                # save
                ds_fn = fn.split('Batch')[0][:-1] + f'_{calibration}_ba{bias_adj}_{nsets}_{gcm_startyear}_{gcm_endyear}_all.nc'
                ds.to_netcdf(ds_fn)
                
                ds_batch.close()
                
                for fn in fn_merge_list:
                    os.remove(fn)


def main():
    start = time.time()

    # Set up CLI
    parser = argparse.ArgumentParser(
    description="""description: program for compiling regional stats from the python glacier evolution model (PyGEM)\nnote, this script is not embarrassingly parallel\nit is currently set up to be parallelized by splitting into n jobs based on the number of regions and scenarios scecified\nfor example, the call below could be parallelized into 4 jobs (2 regions x 2 scenarios)\n\nexample call: $python compile_simulations -rgi_region 01 02 -scenario ssp345 ssp585 -gcm_startyear2000 -gcm_endyear 2100 -ncores 4 -vars glac_mass_annual glac_area_annual""",
    formatter_class=argparse.RawTextHelpFormatter)
    requiredNamed = parser.add_argument_group('required named arguments')
    requiredNamed.add_argument('-rgi_region01', type=int, default=None, required=True, nargs='+',
                        help='Randoph Glacier Inventory region (can take multiple, e.g. "1 2 3")')
    requiredNamed.add_argument('-gcm_name', type=str, default=None, required=True, nargs='+', 
                        help='GCM name for which to compile simulations (can take multiple, ex. "ERA5" or "CESM2")')
    parser.add_argument('-scenario', action='store', type=str, default=None, nargs='+',
                        help='rcp or ssp scenario used for model run (can take multiple, ex. "ssp245 ssp585")')
    parser.add_argument('-realization', action='store', type=str, default=None, nargs='+',
                        help='realization from large ensemble used for model run (cant take multiple, ex. "r1i1p1f1 r2i1p1f1 r3i1p1f1")')
    parser.add_argument('-gcm_startyear', action='store', type=int, default=pygem_prms['climate']['gcm_startyear'],
                        help='start year for the model run')
    parser.add_argument('-gcm_endyear', action='store', type=int, default=pygem_prms['climate']['gcm_endyear'],
                        help='start year for the model run')
    parser.add_argument('-sim_path', type=str, default=pygem_prms['root'] + '/Output/simulations/', 
                        help='PyGEM simulations filepath')
    parser.add_argument('-option_calibration', action='store', type=str, default=pygem_prms['calib']['option_calibration'],
                        help='calibration option ("emulator", "MCMC", "HH2015", "HH2015mod", "None")')
    parser.add_argument('-option_bias_adjustment', action='store', type=int, default=pygem_prms['sim']['option_bias_adjustment'],
                        help='Bias adjustment option (options: 0, "1", "2", "3".\n0: no adjustment\n1: new prec scheme and temp building on HH2015\n2: HH2015 methods\n3: quantile delta mapping)')
    parser.add_argument('-vars',type=str, help='comm delimited list of PyGEM variables to compile (can take multiple, ex. "monthly_mass annual_area")', 
                        choices=['glac_runoff_monthly','offglac_runoff_monthly','glac_acc_monthly','glac_melt_monthly','glac_refreeze_monthly','glac_frontalablation_monthly','glac_massbaltotal_monthly','glac_prec_monthly','glac_mass_monthly','glac_mass_annual','glac_area_annual'],
                        nargs='+')
    parser.add_argument('-ncores', action='store', type=int, default=1,
                        help='number of simultaneous processes (cores) to use, defualt is 1, ie. no parallelization')

    args = parser.parse_args()
    simpath = args.sim_path
    region = args.rgi_region01
    gcms = args.gcm_name
    scenarios = args.scenario
    realizations = args.realization
    calib = args.option_calibration
    bias_adj = args.option_bias_adjustment
    gcm_startyear = args.gcm_startyear
    gcm_endyear = args.gcm_endyear
    vars = args.vars

    if not simpath:
        simpath = pygem_prms['root'] + '/Output/simulations/'

    if not os.path.exists(simpath + 'compile/'):
        os.makedirs(simpath + 'compile/')

    if not isinstance(region, list):
        region = [region]

    if not isinstance(gcms, list):
        gcms = [gcms]

    if scenarios:
        if not isinstance(scenarios, list):
            scenarios = [scenarios]
        if set(['ERA5', 'ERA-Interim', 'COAWST']) & set(gcms):
            raise ValueError(f'Cannot compile present-day and future data simulataneously.  A scenario was specified, which does not exist for one of the specified GCMs.\nGCMs: {gcms}\nScenarios: {scenarios}')
    else:            
        scenarios = ['']
        if set(gcms) - set(['ERA5', 'ERA-Interim', 'COAWST']):
            raise ValueError(f'Must specify a scenario for future GCM runs\nGCMs: {gcms}\nscenarios: {scenarios}')

    if realizations is None:
        realizations = ['']
    else:
        if not isinstance(realizations, list):
            realizations = [realizations]
        if len(gcms) > 1:
            raise ValueError(f'Script not set up to aggregate multiple GCMs and realizations simultaneously - if aggregating multiple realizations, specify a single GCM at a time\nGCMs: {gcms}\nrealizations: {realizations}')

    if not vars:
        vars = ['glac_runoff_monthly','offglac_runoff_monthly','glac_acc_monthly','glac_melt_monthly','glac_refreeze_monthly','glac_frontalablation_monthly','glac_massbaltotal_monthly','glac_prec_monthly','glac_mass_monthly','glac_mass_annual','glac_area_annual']

    # get number of jobs and split into desired number of cores
    njobs = int(len(region) * len(scenarios))
    # number of cores for parallel processing
    if args.ncores > 1:
        num_cores = int(np.min([njobs, args.ncores]))
    else:
        num_cores = 1

    # pack variables for multiprocessing
    list_packed_vars = []
    kwargs=['region', 'simpath', 'gcms', 'realizations', 'scenario', 'calib', 'bias_adj', 'gcm_startyear', 'gcm_endyear', 'vars']
    i=0
    # if realizations specified, aggregate all realizations for each gcm and scenario by region
    for sce in scenarios:
        for reg in region:
            list_packed_vars.append([reg, simpath, gcms, realizations, sce, calib, bias_adj, gcm_startyear, gcm_endyear, vars])
            print(f'job {i}:', [f'{name}={val}' for name, val in zip(kwargs,list_packed_vars[-1])])
            i+=1

    # parallel processing
    print('Processing with ' + str(num_cores) + ' cores...')
    with multiprocessing.Pool(num_cores) as p:
        p.map(run, list_packed_vars)     

    end = time.time()
    print(f'Total runtime: {np.round(end - start,2)} seconds')

if __name__=='__main__':
    main()