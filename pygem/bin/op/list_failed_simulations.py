"""
Python Glacier Evolution Model (PyGEM)

copyright © 2018 David Rounce <drounce@cmu.edu>

Distrubted under the MIT lisence

script to check for failed glaciers for a given simulation and export a pickle file containing a list of said glacier numbers to be reprocessed
"""
# imports
import os
import glob
import sys
import json
import argparse
import numpy as np
# pygem imports
from pygem.setup.config import ConfigManager
# instantiate ConfigManager
config_manager = ConfigManager()
# read the config
pygem_prms = config_manager.read_config()
import pygem.pygem_modelsetup as modelsetup

def run(reg, simpath, gcm, scenario, calib_opt, bias_adj, gcm_startyear, gcm_endyear):

    # define base directory
    base_dir = simpath + "/" + str(reg).zfill(2) + "/"


    # get all glaciers in region to see which fraction ran successfully
    main_glac_rgi_all = modelsetup.selectglaciersrgitable(rgi_regionsO1=[reg], 
                                                        rgi_regionsO2='all', rgi_glac_number='all', 
                                                        glac_no=None,
                                                        debug=True)

    glacno_list_all = list(main_glac_rgi_all['rgino_str'].values)

    # get list of glacier simulation files 
    if scenario:
        sim_dir = base_dir + gcm  + '/' + scenario + '/stats/'
    else:
        sim_dir = base_dir + gcm  + '/stats/'

    # check if gcm has given scenario
    assert os.path.isdir(sim_dir),  f'Error: simulation path not found, {sim_dir}'

    # instantiate list of galcnos that are not in sim_dir
    failed_glacnos = []

    fps = glob.glob(sim_dir + f'*_{calib_opt}_ba{bias_adj}_*_{gcm_startyear}_{gcm_endyear}_all.nc')

    # Glaciers with successful runs to process
    glacno_ran = [x.split('/')[-1].split('_')[0] for x in fps]
    glacno_ran = [x.split('.')[0].zfill(2) + '.' + x[-5:] for x in glacno_ran]

    # print stats of successfully simualated glaciers
    main_glac_rgi = main_glac_rgi_all.loc[main_glac_rgi_all.apply(lambda x: x.rgino_str in glacno_ran, axis=1)]
    print(f'{gcm} {str(scenario).replace('None','')} glaciers successfully simulated:\n  - {main_glac_rgi.shape[0]} of {main_glac_rgi_all.shape[0]} glaciers ({np.round(main_glac_rgi.shape[0]/main_glac_rgi_all.shape[0]*100,3)}%)')
    print(f'  - {np.round(main_glac_rgi.Area.sum(),0)} km2 of {np.round(main_glac_rgi_all.Area.sum(),0)} km2 ({np.round(main_glac_rgi.Area.sum()/main_glac_rgi_all.Area.sum()*100,3)}%)')

    glacno_ran = ['{0:0.5f}'.format(float(x)) for x in glacno_ran]

    # loop through each glacier in batch list
    for i, glacno in enumerate(glacno_list_all):
        # gat glacier string and file name
        glacier_str = '{0:0.5f}'.format(float(glacno))  

        if glacier_str not in glacno_ran:
            failed_glacnos.append(glacier_str)
    return failed_glacnos


def main():

    # Set up CLI
    parser = argparse.ArgumentParser(
    description="""description: script to check for failed PyGEM glacier simulations\n\nexample call: $python list_failed_simulations.py -rgi_region01=1 -gcm_name=CanESM5 -scenrio=ssp585 -outdir=/path/to/output/failed/glaciers/""",
    formatter_class=argparse.RawTextHelpFormatter)
    requiredNamed = parser.add_argument_group('required named arguments')
    requiredNamed.add_argument('-rgi_region01', type=int, default=pygem_prms['setup']['rgi_region01'],
                        help='Randoph Glacier Inventory region (can take multiple, e.g. `-run_region01 1 2 3`)', nargs='+')
    parser.add_argument('-gcm_name', type=str, default=None, 
                        help='GCM name to compile results from (ex. ERA5 or CESM2)')
    parser.add_argument('-scenario', action='store', type=str, default=None,
                        help='rcp or ssp scenario used for model run (ex. rcp26 or ssp585)')
    parser.add_argument('-gcm_startyear', action='store', type=int, default=pygem_prms['climate']['gcm_startyear'],
                        help='start year for the model run')
    parser.add_argument('-gcm_endyear', action='store', type=int, default=pygem_prms['climate']['gcm_endyear'],
                        help='start year for the model run')
    parser.add_argument('-option_calibration', action='store', type=str, default=pygem_prms['calib']['option_calibration'],
                        help='calibration option ("emulator", "MCMC", "HH2015", "HH2015mod", "None")')
    parser.add_argument('-option_bias_adjustment', action='store', type=int, default=pygem_prms['sim']['option_bias_adjustment'],
                        help='Bias adjustment option (options: `0`, `1`, `2`, `3`. 0: no adjustment, \
                                    1: new prec scheme and temp building on HH2015, \
                                    2: HH2015 methods, 3: quantile delta mapping)')
    parser.add_argument('-outdir', type=str, default=None, help='directory to output json file containing list of failed glaciers in each RGI region')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='verbose flag')
    args = parser.parse_args()

    region = args.rgi_region01
    scenario = args.scenario
    gcm_name = args.gcm_name
    bias_adj = args.option_bias_adjustment
    simpath = pygem_prms['root']+'/Output/simulations/'

    if gcm_name in ['ERA5', 'ERA-Interim', 'COAWST']:
        scenario = None
        bias_adj = 0

    if not isinstance(region, list):
        region = [region]

    if args.outdir and not os.path.isdir(args.outdir):
        print(f'Specified output path does not exist: {args.outdir}')
        sys.exit(1)

    for reg in region:
        failed_glacs = run(reg, simpath, args.gcm_name, scenario, args.option_calibration, bias_adj, args.gcm_startyear, args.gcm_endyear)
        if len(failed_glacs)>0:
            if args.outdir:
                fout = os.path.join(args.outdir, f'R{str(reg).zfill(2)}_{args.gcm_name}_{scenario}_{args.gcm_startyear}_{args.gcm_endyear}_failed_rgiids.json').replace('None_','')
                with open(fout, 'w') as f:
                    json.dump(failed_glacs, f)
                    print(f'List of failed glaciers for {gcm_name} {str(scenario).replace('None','')} exported to: {fout}')
            if args.verbose:
                print(f'Failed glaciers for RGI region R{str(reg).zfill(2)} {args.gcm_name} {str(scenario).replace('None','')} {args.gcm_startyear}-{args.gcm_endyear}:')
                print(failed_glacs)

        else: 
            print(f'No glaciers failed from R{region}, for {gcm_name} {scenario.replace('None','')}')