"""
Python Glacier Evolution Model (PyGEM)

copyright © 2018 David Rounce <drounce@cmu.edu>

Distrubted under the MIT lisence

Find the optimal values of glens_a_multiplier to match the consensus ice thickness estimates 
"""
# Built-in libraries
import argparse
from collections import OrderedDict
import os
import sys
import time
import json
# External libraries
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq
# pygem imports
import pygem
from pygem.setup.config import ConfigManager
# instantiate ConfigManager
config_manager = ConfigManager()
# read the config
pygem_prms = config_manager.read_config()
from pygem import class_climate
from pygem.massbalance import PyGEMMassBalance
from pygem.oggm_compat import single_flowline_glacier_directory
import pygem.pygem_modelsetup as modelsetup
# oggm imports
from oggm import cfg
from oggm import tasks
from oggm.core.massbalance import apparent_mb_from_any_mb


#%% FUNCTIONS
def getparser():
    """
    Use argparse to add arguments from the command line

    Parameters
    ----------
    ref_gcm_name (optional) : str
        reference gcm name
    rgi_glac_number_fn : str
        filename of .pkl file containing a list of glacier numbers which is used to run batches on the supercomputer
    rgi_glac_number : str
        rgi glacier number to run for supercomputer
    option_ordered : bool (default: False)
        option to keep glaciers ordered or to grab every n value for the batch
        (the latter helps make sure run times on each core are similar as it removes any timing differences caused by
         regional variations)
    debug : bool (defauls: False)
        Switch for turning debug printing on or off (default = 0 (off))

    Returns
    -------
    Object containing arguments and their respective values.
    """
    parser = argparse.ArgumentParser(description="run calibration in parallel")
    # add arguments
    parser.add_argument('-rgi_region01', type=int, default=pygem_prms['setup']['rgi_region01'],
                        help='Randoph Glacier Inventory region (can take multiple, e.g. `-run_region01 1 2 3`)', nargs='+')
    parser.add_argument('-ref_gcm_name', action='store', type=str, default=pygem_prms['climate']['ref_gcm_name'],
                        help='reference gcm name')
    parser.add_argument('-ref_startyear', action='store', type=int, default=pygem_prms['climate']['ref_startyear'],
                        help='reference period starting year for calibration (typically 2000)')
    parser.add_argument('-ref_endyear', action='store', type=int, default=pygem_prms['climate']['ref_endyear'],
                        help='reference period ending year for calibration (typically 2019)')
    parser.add_argument('-rgi_glac_number_fn', action='store', type=str, default=None,
                        help='Filename containing list of rgi_glac_number, helpful for running batches on spc')
    parser.add_argument('-rgi_glac_number', action='store', type=float, default=pygem_prms['setup']['glac_no'], nargs='+',
                        help='Randoph Glacier Inventory glacier number (can take multiple)')
    parser.add_argument('-fs', action='store', type=float, default=pygem_prms['out']['fs'],
                        help='Sliding parameter')
    parser.add_argument('-a_multiplier', action='store', type=float, default=pygem_prms['out']['glen_a_multiplier'],
                        help="Glen’s creep parameter A multiplier")
    parser.add_argument('-a_multiplier_bndlow', action='store', type=float, default=0.1,
                        help="Glen’s creep parameter A multiplier, low bound (default 0.1)")
    parser.add_argument('-a_multiplier_bndhigh', action='store', type=float, default=10,
                        help="Glen’s creep parameter A multiplier, upper bound (default 10)")

    # flags
    parser.add_argument('-option_ordered', action='store_true',
                        help='Flag to keep glacier lists ordered (default is off)')
    parser.add_argument('-v', '--debug', action='store_true',
                        help='Flag for debugging')
    return parser


def plot_nfls_section(nfls):
    """
    Modification of OGGM's plot_modeloutput_section()
    """
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_axes([0.07, 0.08, 0.7, 0.84])

    # Compute area histo
    area = np.array([])
    height = np.array([])
    bed = np.array([])
    for cls in nfls:
        a = cls.widths_m * cls.dx_meter * 1e-6
        a = np.where(cls.thick > 0, a, 0)
        area = np.concatenate((area, a))
        height = np.concatenate((height, cls.surface_h))
        bed = np.concatenate((bed, cls.bed_h))
    ylim = [bed.min(), height.max()]

    # Plot histo
    posax = ax.get_position()
    posax = [posax.x0 + 2 * posax.width / 3.0,
             posax.y0,  posax.width / 3.0,
             posax.height]
    axh = fig.add_axes(posax, frameon=False)

    axh.hist(height, orientation='horizontal', range=ylim, bins=20,
             alpha=0.3, weights=area)
    axh.invert_xaxis()
    axh.xaxis.tick_top()
    axh.set_xlabel('Area incl. tributaries (km$^2$)')
    axh.xaxis.set_label_position('top')
    axh.set_ylim(ylim)
    axh.yaxis.set_ticks_position('right')
    axh.set_yticks([])
    axh.axhline(y=ylim[1], color='black', alpha=1)  # qick n dirty trick

    # plot Centerlines
    cls = nfls[-1]
    x = np.arange(cls.nx) * cls.dx * cls.map_dx

    # Plot the bed
    ax.plot(x, cls.bed_h, color='k', linewidth=2.5, label='Bed (Parab.)')

    # Where trapezoid change color
    if hasattr(cls, '_do_trapeze') and cls._do_trapeze:
        bed_t = cls.bed_h * np.nan
        pt = cls.is_trapezoid & (~cls.is_rectangular)
        bed_t[pt] = cls.bed_h[pt]
        ax.plot(x, bed_t, color='rebeccapurple', linewidth=2.5,
                label='Bed (Trap.)')
        bed_t = cls.bed_h * np.nan
        bed_t[cls.is_rectangular] = cls.bed_h[cls.is_rectangular]
        ax.plot(x, bed_t, color='crimson', linewidth=2.5, label='Bed (Rect.)')

    # Plot glacier
    def surf_to_nan(surf_h, thick):
        t1 = thick[:-2]
        t2 = thick[1:-1]
        t3 = thick[2:]
        pnan = ((t1 == 0) & (t2 == 0)) & ((t2 == 0) & (t3 == 0))
        surf_h[np.where(pnan)[0] + 1] = np.nan
        return surf_h
    
    surfh = surf_to_nan(cls.surface_h, cls.thick)
    ax.plot(x, surfh, color='#003399', linewidth=2, label='Glacier')

    ax.set_ylim(ylim)

    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xlabel('Distance along flowline (m)')
    ax.set_ylabel('Altitude (m)')

    # Legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    ax.legend(list(by_label.values()), list(by_label.keys()),
              bbox_to_anchor=(0.5, 1.0),
              frameon=False)
    plt.show()

def reg_vol_comparison(gdirs, mbmods, nyears, a_multiplier=1, fs=0, debug=False):
    """ Calculate the modeled volume [km3] and consensus volume [km3] for the given set of glaciers """
    
    reg_vol_km3_consensus = 0
    reg_vol_km3_modeled = 0
    for nglac, gdir in enumerate(gdirs):
        if nglac%2000 == 0:
            print(gdir.rgi_id)
        mbmod_inv = mbmods[nglac]
        
        # Arbitrariliy shift the MB profile up (or down) until mass balance is zero (equilibrium for inversion)
        apparent_mb_from_any_mb(gdir, mb_model=mbmod_inv, mb_years=np.arange(nyears))
    
        tasks.prepare_for_inversion(gdir)
        tasks.mass_conservation_inversion(gdir, glen_a=cfg.PARAMS['glen_a']*a_multiplier, fs=fs)
        tasks.init_present_time_glacier(gdir) # adds bins below
        nfls = gdir.read_pickle('model_flowlines')
            
        # Load consensus volume
        if os.path.exists(gdir.get_filepath('consensus_mass')):
            consensus_fn = gdir.get_filepath('consensus_mass')
            with open(consensus_fn, 'rb') as f:
                consensus_km3 = pickle.load(f) / pygem_prms['constants']['density_ice'] / 1e9
            
            reg_vol_km3_consensus += consensus_km3
            reg_vol_km3_modeled += nfls[0].volume_km3
        
            if debug:   
                plot_nfls_section(nfls)
                print('\n\n  Modeled vol [km3]:  ', nfls[0].volume_km3)
                print('  Consensus vol [km3]:', consensus_km3,'\n\n')
                
    return reg_vol_km3_modeled, reg_vol_km3_consensus

    
#%%
def main():
    parser = getparser()
    args = parser.parse_args()
    time_start = time.time()

    if args.debug == 1:
        debug = True
    else:
        debug = False

    # Calibrate each region
    for reg in args.rgi_region01:
        
        print('Region:', reg)
        
        # ===== LOAD GLACIERS =====
        main_glac_rgi_all = modelsetup.selectglaciersrgitable(
                    rgi_regionsO1=[reg], rgi_regionsO2='all', rgi_glac_number='all', 
                    include_landterm=True,include_laketerm=True, include_tidewater=True)
            
        
        main_glac_rgi_all = main_glac_rgi_all.sort_values('Area', ascending=False)
        main_glac_rgi_all.reset_index(inplace=True, drop=True)
        main_glac_rgi_all['Area_cum'] = np.cumsum(main_glac_rgi_all['Area'])
        main_glac_rgi_all['Area_cum_frac'] = main_glac_rgi_all['Area_cum'] / main_glac_rgi_all.Area.sum()
        
        glac_idx = np.where(main_glac_rgi_all.Area_cum_frac > pygem_prms['calib']['icethickness_cal_frac_byarea'])[0][0]
        main_glac_rgi_subset = main_glac_rgi_all.loc[0:glac_idx, :]
        main_glac_rgi_subset = main_glac_rgi_subset.sort_values('O1Index', ascending=True)
        main_glac_rgi_subset.reset_index(inplace=True, drop=True)
        
        print(f'But only the largest {int(100*pygem_prms['calib']['icethickness_cal_frac_byarea'])}% of the glaciers by area, which includes', main_glac_rgi_subset.shape[0], 'glaciers.')
        
        # ===== TIME PERIOD =====
        dates_table = modelsetup.datesmodelrun(
                startyear=args.ref_startyear, endyear=args.ref_endyear, spinupyears=pygem_prms['climate']['ref_spinupyears'],
                option_wateryear=pygem_prms['climate']['ref_wateryear'])
        
        # ===== LOAD CLIMATE DATA =====
        # Climate class
        gcm_name = args.ref_gcm_name
        assert gcm_name in ['ERA5', 'ERA-Interim'], 'Error: Calibration not set up for ' + gcm_name
        gcm = class_climate.GCM(name=gcm_name)
        # Air temperature [degC]
        gcm_temp, gcm_dates = gcm.importGCMvarnearestneighbor_xarray(gcm.temp_fn, gcm.temp_vn, main_glac_rgi_subset, dates_table)
        if pygem_prms['mbmod']['option_ablation'] == 2 and gcm_name in ['ERA5']:
            gcm_tempstd, gcm_dates = gcm.importGCMvarnearestneighbor_xarray(gcm.tempstd_fn, gcm.tempstd_vn,
                                                                            main_glac_rgi_subset, dates_table)
        else:
            gcm_tempstd = np.zeros(gcm_temp.shape)
        # Precipitation [m]
        gcm_prec, gcm_dates = gcm.importGCMvarnearestneighbor_xarray(gcm.prec_fn, gcm.prec_vn, main_glac_rgi_subset, dates_table)
        # Elevation [m asl]
        gcm_elev = gcm.importGCMfxnearestneighbor_xarray(gcm.elev_fn, gcm.elev_vn, main_glac_rgi_subset)
        # Lapse rate [degC m-1]
        gcm_lr, gcm_dates = gcm.importGCMvarnearestneighbor_xarray(gcm.lr_fn, gcm.lr_vn, main_glac_rgi_subset, dates_table)
        
        # ===== RUN MASS BALANCE =====
        # Number of years (for OGGM's run_until_and_store)
        if pygem_prms['time']['timestep'] == 'monthly':
            nyears = int(dates_table.shape[0]/12)
        else:
            assert True==False, 'Adjust nyears for non-monthly timestep'
        
        reg_vol_km3_consensus = 0
        reg_vol_km3_modeled = 0
        mbmods = []
        gdirs = []
        for glac in range(main_glac_rgi_subset.shape[0]):
            # Select subsets of data
            glacier_rgi_table = main_glac_rgi_subset.loc[main_glac_rgi_subset.index.values[glac], :]
            glacier_str = '{0:0.5f}'.format(glacier_rgi_table['RGIId_float'])
            
            if glac%1000 == 0:
                print(glacier_str)
        
            # ===== Load glacier data: area (km2), ice thickness (m), width (km) =====
            try:
                gdir = single_flowline_glacier_directory(glacier_str)
                
                # Flowlines
                fls = gdir.read_pickle('inversion_flowlines')
                
                # Add climate data to glacier directory
                gdir.historical_climate = {'elev': gcm_elev[glac],
                                            'temp': gcm_temp[glac,:],
                                            'tempstd': gcm_tempstd[glac,:],
                                            'prec': gcm_prec[glac,:],
                                            'lr': gcm_lr[glac,:]}
                gdir.dates_table = dates_table
            
                glacier_area_km2 = fls[0].widths_m * fls[0].dx_meter / 1e6
                if (fls is not None) and (glacier_area_km2.sum() > 0):
            
                    modelprms_fn = glacier_str + '-modelprms_dict.json'
                    modelprms_fp = pygem_prms['root'] + '/Output/calibration/' + glacier_str.split('.')[0].zfill(2) + '/'
                    modelprms_fullfn = modelprms_fp + modelprms_fn
                    assert os.path.exists(modelprms_fullfn), glacier_str + ' calibrated parameters do not exist.'            
                    with open(modelprms_fullfn, 'r') as f:
                        modelprms_dict = json.load(f)
                    
                    assert 'emulator' in modelprms_dict, ('Error: ' + glacier_str + ' emulator not in modelprms_dict')
                    modelprms_all = modelprms_dict['emulator']
            
                    # Loop through model parameters
                    modelprms = {'kp': modelprms_all['kp'][0],
                                'tbias': modelprms_all['tbias'][0],
                                'ddfsnow': modelprms_all['ddfsnow'][0],
                                'ddfice': modelprms_all['ddfice'][0],
                                'tsnow_threshold': modelprms_all['tsnow_threshold'][0],
                                'precgrad': modelprms_all['precgrad'][0]}
                    
                    # ----- ICE THICKNESS INVERSION using OGGM -----
                    # Apply inversion_filter on mass balance with debris to avoid negative flux
                    if pygem_prms['mbmod']['include_debris']:
                        inversion_filter = True
                    else:
                        inversion_filter = False
                            
                    # Perform inversion based on PyGEM MB
                    mbmod_inv = PyGEMMassBalance(gdir, modelprms, glacier_rgi_table,
                                                fls=fls, option_areaconstant=True,
                                                inversion_filter=inversion_filter)
            
            #        if debug:
            #            h, w = gdir.get_inversion_flowline_hw()
            #            mb_t0 = (mbmod_inv.get_annual_mb(h, year=0, fl_id=0, fls=fls) * cfg.SEC_IN_YEAR * 
            #                     pygem_prms['constants']['density_ice'] / pygem_prms['constants']['density_water']) 
            #            plt.plot(mb_t0, h, '.')
            #            plt.ylabel('Elevation')
            #            plt.xlabel('Mass balance (mwea)')
            #            plt.show()
                    
                    mbmods.append(mbmod_inv)
                    gdirs.append(gdir)
            except:
                print(glacier_str + ' failed - likely no gdir')
                
        print('\n\n------\nModel setup time:', time.time()-time_start, 's')
                        
        # ===== CHECK BOUNDS =====
        reg_vol_km3_mod, reg_vol_km3_con = reg_vol_comparison(gdirs, mbmods, nyears, a_multiplier=args.a_multiplier, fs=args.fs,
                                                            debug=debug)
        # Lower bound
        reg_vol_km3_mod_bndlow, reg_vol_km3_con = reg_vol_comparison(gdirs, mbmods, nyears,
                                                                    a_multiplier=args.a_multiplier_bndlow, fs=args.fs, 
                                                                    debug=debug)
        # Higher bound
        reg_vol_km3_mod_bndhigh, reg_vol_km3_con = reg_vol_comparison(gdirs, mbmods, nyears,
                                                                    a_multiplier=args.a_multiplier_bndhigh, fs=args.fs, 
                                                                    debug=debug)
        
        print('Region:', reg)
        print('Consensus [km3]    :', reg_vol_km3_con)
        print('Model [km3]        :', reg_vol_km3_mod)
        print('Model bndlow [km3] :', reg_vol_km3_mod_bndlow)
        print('Model bndhigh [km3]:', reg_vol_km3_mod_bndhigh)
        
        # ===== OPTIMIZATION =====
        # Check consensus is within bounds
        if reg_vol_km3_con < reg_vol_km3_mod_bndhigh:
            a_multiplier_opt = args.a_multiplier_bndhigh
        elif reg_vol_km3_con > reg_vol_km3_mod_bndlow:
            a_multiplier_opt = args.a_multiplier_bndhigh
        # If so, then find optimal glens_a_multiplier
        else:
            def to_minimize(a_multiplier):
                """Objective function to minimize"""
                reg_vol_km3_mod, reg_vol_km3_con = reg_vol_comparison(gdirs, mbmods, nyears, a_multiplier=a_multiplier, fs=args.fs, 
                                                                    debug=debug)
                return reg_vol_km3_mod - reg_vol_km3_con
            # Brentq minimization
            a_multiplier_opt, r = brentq(to_minimize, args.a_multiplier_bndlow, args.a_multiplier_bndhigh, rtol=1e-2,
                                            full_output=True)
            # Re-run to get estimates
            reg_vol_km3_mod, reg_vol_km3_con = reg_vol_comparison(gdirs, mbmods, nyears, a_multiplier=a_multiplier_opt, fs=args.fs, 
                                                                debug=debug)
            
            print('\n\nOptimized:\n  glens_a_multiplier:', np.round(a_multiplier_opt,3))
            print('  Consensus [km3]:', reg_vol_km3_con)
            print('  Model [km3]    :', reg_vol_km3_mod)
                
        # ===== EXPORT RESULTS =====
        glena_cns = ['O1Region', 'count', 'glens_a_multiplier', 'fs', 'reg_vol_km3_consensus', 'reg_vol_km3_modeled']
        glena_df_single = pd.DataFrame(np.zeros((1,len(glena_cns))), columns=glena_cns)
        glena_df_single.loc[0,:] = [reg, main_glac_rgi_subset.shape[0], a_multiplier_opt, args.fs, reg_vol_km3_con, reg_vol_km3_mod]

        try:
            glena_df = pd.read_csv(f"{pygem_prms['root']}/{pygem_prms['out']['glena_reg_relpath']}")
            
            # Add or overwrite existing file
            glena_idx = np.where((glena_df.O1Region == reg))[0]
            if len(glena_idx) > 0:
                glena_df.loc[glena_idx,:] = glena_df_single.values
            else:
                glena_df = pd.concat([glena_df, glena_df_single], axis=0)
                
        except FileNotFoundError:
            glena_df = glena_df_single
        
        except Exception as err:
            print(f'Error saving results: {err}')
            
        glena_df = glena_df.sort_values('O1Region', ascending=True)
        glena_df.reset_index(inplace=True, drop=True)
        glena_df.to_csv(f"{pygem_prms['root']}/{pygem_prms['out']['glena_reg_relpath']}", index=False)    
    
    print('\n\n------\nTotal processing time:', time.time()-time_start, 's')

if __name__ == "__main__":
    main()    