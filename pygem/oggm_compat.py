"""
Python Glacier Evolution Model (PyGEM)

copyright © 2018 David Rounce <drounce@cmu.edu

Distrubted under the MIT lisence

PYGEM-OGGGM COMPATIBILITY FUNCTIONS
"""
import os
# External libraries
import numpy as np
import pandas as pd
import netCDF4
from oggm import cfg, utils
from oggm import workflow, tasks
#from oggm import tasks
from oggm.cfg import SEC_IN_YEAR
from oggm.core import flowline
from oggm.core.massbalance import MassBalanceModel
#from oggm.shop import rgitopo
from pygem.shop import debris, mbdata, icethickness
from pygem.setup.config import ConfigManager
# instantiate ConfigManager
config_manager = ConfigManager()
# read the config
pygem_prms = config_manager.read_config()


class CompatGlacDir:
    def __init__(self, rgiid):
        self.rgiid = rgiid
           
def single_flowline_glacier_directory(rgi_id, reset=pygem_prms['oggm']['overwrite_gdirs'], prepro_border=pygem_prms['oggm']['border'], 
                                      logging_level= pygem_prms['oggm']['logging_level'], has_internet= pygem_prms['oggm']['has_internet'], 
                                      working_dir=f"{pygem_prms['root']}/{pygem_prms['oggm']['oggm_gdir_relpath']}"):
    """Prepare a GlacierDirectory for PyGEM (single flowline to start with)

    Parameters
    ----------
    rgi_id : str
        the rgi id of the glacier (RGIv60-)
    reset : bool
        set to true to delete any pre-existing files. If false (the default),
        the directory won't be re-downloaded if already available locally in
        order to spare time.
    prepro_border : int
        the size of the glacier map: 10, 80, 160, 240

    Returns
    -------
    a GlacierDirectory object
    """
    if type(rgi_id) != str:
        raise ValueError('We expect rgi_id to be a string')
    if rgi_id.startswith('RGI60-') == False:
        rgi_id = 'RGI60-' + rgi_id.split('.')[0].zfill(2) + '.' + rgi_id.split('.')[1]
    else:
        raise ValueError('Check RGIId is correct')
        
    # Initialize OGGM and set up the default run parameters
    cfg.initialize(logging_level=logging_level)
    # Set multiprocessing to false; otherwise, causes daemonic error due to PyGEM's multiprocessing
    #  - avoids having multiple multiprocessing going on at the same time
    cfg.PARAMS['use_multiprocessing']  = False
    
    # Avoid erroneous glaciers (e.g., Centerlines too short or other issues)
    cfg.PARAMS['continue_on_error'] = True
    
    # Has internet
    cfg.PARAMS['has_internet'] = has_internet
    
    # Set border boundary
    cfg.PARAMS['border'] = prepro_border
    # Usually we recommend to set dl_verify to True - here it is quite slow
    # because of the huge files so we just turn it off.
    # Switch it on for real cases!
    cfg.PARAMS['dl_verify'] = True
    cfg.PARAMS['use_multiple_flowlines'] = False
    # temporary directory for testing (deleted on computer restart)
    cfg.PATHS['working_dir'] = working_dir

    # check if gdir is already processed
    if not reset:
        try:
            gdir = utils.GlacierDirectory(rgi_id)
            gdir.read_pickle('inversion_flowlines')

        except:
            reset = True
        
    if reset:
        # Start after the prepro task level
        base_url = pygem_prms['oggm']['base_url']

        cfg.PARAMS['has_internet'] =  pygem_prms['oggm']['has_internet']
        gdir = workflow.init_glacier_directories([rgi_id], from_prepro_level=2, prepro_border=cfg.PARAMS['border'], 
                                                  prepro_base_url=base_url, prepro_rgi_version='62')[0]
    
    # go through shop tasks to process auxiliary datasets to gdir if necessary
    # consensus glacier mass
    if not os.path.isfile(gdir.get_filepath('consensus_mass')):
        workflow.execute_entity_task(icethickness.consensus_gridded, gdir)
    # mass balance calibration data
    if not os.path.isfile(gdir.get_filepath('mb_calib_pygem')):
        workflow.execute_entity_task(mbdata.mb_df_to_gdir, gdir)
    # debris thickness and melt enhancement factors
    if not os.path.isfile(gdir.get_filepath('debris_ed')) or os.path.isfile(gdir.get_filepath('debris_hd')):
        workflow.execute_entity_task(debris.debris_to_gdir, gdir)
        workflow.execute_entity_task(debris.debris_binned, gdir)

    return gdir
        


def single_flowline_glacier_directory_with_calving(rgi_id, reset=pygem_prms['oggm']['overwrite_gdirs'], 
                                                   prepro_border=pygem_prms['oggm']['border'], k_calving=1,
                                                   logging_level= pygem_prms['oggm']['logging_level'], 
                                                   has_internet= pygem_prms['oggm']['has_internet'],
                                                   working_dir=pygem_prms['root'] + pygem_prms['oggm']['oggm_gdir_relpath'],
                                                   facorrected=pygem_prms['setup']['include_frontalablation']):
    """Prepare a GlacierDirectory for PyGEM (single flowline to start with)

    k_calving is free variable!

    Parameters
    ----------
    rgi_id : str
        the rgi id of the glacier
    reset : bool
        set to true to delete any pre-existing files. If false (the default),
        the directory won't be re-downloaded if already available locally in
        order to spare time.
    prepro_border : int
        the size of the glacier map: 10, 80, 160, 250
    Returns
    -------
    a GlacierDirectory object
    """
    if type(rgi_id) != str:
        raise ValueError('We expect rgi_id to be a string')
    if rgi_id.startswith('RGI60-') == False:
        rgi_id = 'RGI60-' + rgi_id.split('.')[0].zfill(2) + '.' + rgi_id.split('.')[1]
    else:
        raise ValueError('Check RGIId is correct')

    # Initialize OGGM and set up the default run parameters
    cfg.initialize(logging_level=logging_level)
    # Set multiprocessing to false; otherwise, causes daemonic error due to PyGEM's multiprocessing
    #  - avoids having multiple multiprocessing going on at the same time
    cfg.PARAMS['use_multiprocessing']  = False
    
    # Avoid erroneous glaciers (e.g., Centerlines too short or other issues)
    cfg.PARAMS['continue_on_error'] = True
    
    # Has internet
    cfg.PARAMS['has_internet'] = has_internet
    
    # Set border boundary
    cfg.PARAMS['border'] = prepro_border
    # Usually we recommend to set dl_verify to True - here it is quite slow
    # because of the huge files so we just turn it off.
    # Switch it on for real cases!
    cfg.PARAMS['dl_verify'] = True
    cfg.PARAMS['use_multiple_flowlines'] = False
    # temporary directory for testing (deleted on computer restart)
    cfg.PATHS['working_dir'] = working_dir
    
    # check if gdir is already processed
    if not reset:
        try:
            gdir = utils.GlacierDirectory(rgi_id)
            gdir.read_pickle('inversion_flowlines')

        except:
            reset = True
        
    if reset:
        # Start after the prepro task level
        base_url = pygem_prms['oggm']['base_url']

        cfg.PARAMS['has_internet'] =  pygem_prms['oggm']['has_internet']
        gdir = workflow.init_glacier_directories([rgi_id], from_prepro_level=2, prepro_border=cfg.PARAMS['border'], 
                                                  prepro_base_url=base_url, prepro_rgi_version='62')[0]
        
        if not gdir.is_tidewater:
            raise ValueError(f'{rgi_id} is not tidewater!')

    # go through shop tasks to process auxiliary datasets to gdir if necessary
    # consensus glacier mass
    if not os.path.isfile(gdir.get_filepath('consensus_mass')):
        workflow.execute_entity_task(icethickness.consensus_gridded, gdir)

    # mass balance calibration data (note facorrected kwarg)
    if not os.path.isfile(gdir.get_filepath('mb_calib_pygem')):
        workflow.execute_entity_task(mbdata.mb_df_to_gdir, gdir, **{"facorrected": facorrected})

    return gdir


def l3_proc(gdir):
    """
    OGGGM L3 preprocessing steps
    """
    # process climate_hisotrical data to gdir
    workflow.execute_entity_task(tasks.process_climate_data, gdir);

    # process mb_calib data from geodetic mass balance
    workflow.execute_entity_task(tasks.mb_calibration_from_geodetic_mb,
                                gdir, informed_threestep=True, overwrite_gdir=True,
                                );

    # glacier bed inversion
    workflow.execute_entity_task(tasks.apparent_mb_from_any_mb, gdir);
    workflow.calibrate_inversion_from_consensus(
        gdir,
        apply_fs_on_mismatch=True,
        error_on_mismatch=True,  # if you running many glaciers some might not work
        filter_inversion_output=True,  # this partly filters the overdeepening due to
        # the equilibrium assumption for retreating glaciers (see. Figure 5 of Maussion et al. 2019)
        volume_m3_reference=None,  # here you could provide your own total volume estimate in m3
    );
    # after inversion, merge data from preprocessing tasks form mode_flowlines
    workflow.execute_entity_task(tasks.init_present_time_glacier, gdir);


def oggm_spinup(gdir):
    # perform OGGM dynamic spinup and return flowline model at year 2000
    # define mb_model for spinup
    workflow.execute_entity_task(tasks.run_dynamic_spinup,
                            gdir,
                            spinup_start_yr=1979,  # When to start the spinup
                            minimise_for='area',  # what target to match at the RGI date
                            output_filesuffix='_dynamic_area',  # Where to write the output
                            ye=2020,  # When the simulation should stop
                            # first_guess_t_spinup = , could be passed as input argument for each step in the sampler based on prior tbias, current default first guess is -2
    )
    fmd_dynamic = flowline.FileModel(gdir.get_filepath('model_geometry', filesuffix='_dynamic_area'))
    fmd_dynamic.run_until(2000)
    return fmd_dynamic.fls # flowlines after dynamic spinup at year 2000


def create_empty_glacier_directory(rgi_id):
    """Create empty GlacierDirectory for PyGEM's alternative ice thickness products

    Parameters
    ----------
    rgi_id : str
        the rgi id of the glacier (RGIv60-)

    Returns
    -------
    a GlacierDirectory object
    """
    # RGIId check
    if type(rgi_id) != str:
        raise ValueError('We expect rgi_id to be a string')
    assert rgi_id.startswith('RGI60-'), 'Check RGIId starts with RGI60-'

    # Create empty directory
    gdir = CompatGlacDir(rgi_id)

    return gdir


def get_glacier_zwh(gdir):
    """Computes this glaciers altitude, width and ice thickness.

    Parameters
    ----------
    gdir : GlacierDirectory
        the glacier to compute

    Returns
    -------
    a dataframe with the requested data
    """

    fls = gdir.read_pickle('model_flowlines')
    z = np.array([])
    w = np.array([])
    h = np.array([])
    for fl in fls:
        # Widths (in m)
        w = np.append(w, fl.widths_m)
        # Altitude (in m)
        z = np.append(z, fl.surface_h)
        # Ice thickness (in m)
        h = np.append(h, fl.thick)
    # Distance between two points
    dx = fl.dx_meter

    # Output
    df = pd.DataFrame()
    df['z'] = z
    df['w'] = w
    df['h'] = h
    df['dx'] = dx

    return df


class RandomLinearMassBalance(MassBalanceModel):
    """Mass-balance as a linear function of altitude with random ELA.

    This is a dummy MB model to illustrate how to program one.

    The reference ELA is taken at a percentile altitude of the glacier.
    It then varies randomly from year to year.

    This class implements the MassBalanceModel interface so that the
    dynamical model can use it. Even if you are not familiar with object
    oriented programming, I hope that the example below is simple enough.
    """

    def __init__(self, gdir, grad=3., h_perc=60, sigma_ela=100., seed=None):
        """ Initialize.

        Parameters
        ----------
        gdir : oggm.GlacierDirectory
            the working glacier directory
        grad: float
            Mass-balance gradient (unit: [mm w.e. yr-1 m-1])
        h_perc: int
            The percentile of the glacier elevation to choose the ELA
        sigma_ela: float
            The standard deviation of the ELA (unit: [m])
        seed : int, optional
            Random seed used to initialize the pseudo-random number generator.

        """
        super(RandomLinearMassBalance, self).__init__()
        self.valid_bounds = [-1e4, 2e4]  # in m
        self.grad = grad
        self.sigma_ela = sigma_ela
        self.hemisphere = 'nh'
        self.rng = np.random.RandomState(seed)

        # Decide on a reference ELA
        grids_file = gdir.get_filepath('gridded_data')
        with netCDF4.Dataset(grids_file) as nc:
            glacier_mask = nc.variables['glacier_mask'][:]
            glacier_topo = nc.variables['topo_smoothed'][:]

        self.orig_ela_h = np.percentile(glacier_topo[glacier_mask == 1],
                                        h_perc)
        self.ela_h_per_year = dict()  # empty dictionary

    def get_random_ela_h(self, year):
        """This generates a random ELA for the requested year.

        Since we do not know which years are going to be asked for we generate
        them on the go.
        """

        year = int(year)
        if year in self.ela_h_per_year:
            # If already computed, nothing to be done
            return self.ela_h_per_year[year]

        # Else we generate it for this year
        ela_h = self.orig_ela_h + self.rng.randn() * self.sigma_ela
        self.ela_h_per_year[year] = ela_h
        return ela_h

    def get_annual_mb(self, heights, year=None, fl_id=None):

        # Compute the mass-balance gradient
        ela_h = self.get_random_ela_h(year)
        mb = (np.asarray(heights) - ela_h) * self.grad

        # Convert to units of [m s-1] (meters of ice per second)
        return mb / SEC_IN_YEAR / cfg.PARAMS['ice_density']
