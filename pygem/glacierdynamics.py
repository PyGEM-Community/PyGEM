"""
Python Glacier Evolution Model (PyGEM)

copyright © 2018 David Rounce <drounce@cmu.edu

Distrubted under the MIT lisence
"""
from collections import OrderedDict
from time import gmtime, strftime

import numpy as np
#import pandas as pd
#import netCDF4
import xarray as xr

from oggm import cfg, utils
from oggm.core.flowline import FlowlineModel
from oggm.exceptions import InvalidParamsError
from oggm import __version__
from pygem.setup.config import ConfigManager
# instantiate ConfigManager
config_manager = ConfigManager()
# read the config
pygem_prms = config_manager.read_config()

cfg.initialize()


#%%
class MassRedistributionCurveModel(FlowlineModel):
    """Glacier geometry updated using mass redistribution curves; also known as the "delta-h method"

    This uses mass redistribution curves from Huss et al. (2010) to update the glacier geometry
    """

    def __init__(self, flowlines, mb_model=None, y0=0.,
                 glen_a=None, fs=0., is_tidewater=False, water_level=0, 
#                 calving_k=0,
                 inplace=False,
                 debug=True,
                 option_areaconstant=False, spinupyears=0, 
                 constantarea_years=0,
                 **kwargs):
        """ Instanciate the model.
        
        Parameters
        ----------
        flowlines : list
            the glacier flowlines
        mb_model : MassBalanceModel
            the mass-balance model
        y0 : int
            initial year of the simulation
        inplace : bool
            whether or not to make a copy of the flowline objects for the run
            setting to True implies that your objects will be modified at run
            time by the model (can help to spare memory)
        is_tidewater: bool, default: False
            use the very basic parameterization for tidewater glaciers
        mb_elev_feedback : str, default: 'annual'
            'never', 'always', 'annual', or 'monthly': how often the
            mass-balance should be recomputed from the mass balance model.
            'Never' is equivalent to 'annual' but without elevation feedback
            at all (the heights are taken from the first call).
        check_for_boundaries: bool, default: True
            raise an error when the glacier grows bigger than the domain
            boundaries
        """
        super(MassRedistributionCurveModel, self).__init__(flowlines, mb_model=mb_model, y0=y0, inplace=inplace,
                                                           mb_elev_feedback='annual', **kwargs)
        self.option_areaconstant = option_areaconstant
        self.constantarea_years = constantarea_years
        self.spinupyears = spinupyears
        self.glac_idx_initial = [fl.thick.nonzero()[0] for fl in flowlines]
        self.y0 = 0
        self.is_tidewater = is_tidewater
        self.water_level = water_level

#        widths_t0 = flowlines[0].widths_m
#        area_v1 = widths_t0 * flowlines[0].dx_meter
#        print('area v1:', area_v1.sum())
#        area_v2 = np.copy(area_v1)
#        area_v2[flowlines[0].thick == 0] = 0
#        print('area v2:', area_v2.sum())
        
        # HERE IS THE STUFF TO RECORD FOR EACH FLOWLINE!
        if self.is_tidewater:
            self.calving_k = cfg.PARAMS['calving_k']
        self.calving_m3_since_y0 = 0.  # total calving since time y0
        
        assert len(flowlines) == 1, 'MassRedistributionCurveModel is not set up for multiple flowlines'
        
        
    def run_until(self, y1, run_single_year=False):
        """Runs the model from the current year up to a given year date y1.

        This function runs the model for the time difference y1-self.y0
        If self.y0 has not been specified at some point, it is 0 and y1 will
        be the time span in years to run the model for.

        Parameters
        ----------
        y1 : float
            Upper time span for how long the model should run
        """                                   
                    
        # We force timesteps to yearly timesteps
        if run_single_year:
            self.updategeometry(y1)
        else:
            years = np.arange(self.yr, y1)
            for year in years:
                self.updategeometry(year)
            
        # Check for domain bounds
        if self.check_for_boundaries:
            if self.fls[-1].thick[-1] > 10:
                raise RuntimeError('Glacier exceeds domain boundaries, '
                                   'at year: {}'.format(self.yr))
        # Check for NaNs
        for fl in self.fls:
            if np.any(~np.isfinite(fl.thick)):
                raise FloatingPointError('NaN in numerical solution.')
        
                    

    def run_until_and_store(self, y1, run_path=None, diag_path=None,
                            store_monthly_step=None):
        """Runs the model and returns intermediate steps in xarray datasets.

        This function repeatedly calls FlowlineModel.run_until for either
        monthly or yearly time steps up till the upper time boundary y1.

        Parameters
        ----------
        y1 : int
            Upper time span for how long the model should run (needs to be
            a full year)
        run_path : str
            Path and filename where to store the model run dataset
        diag_path : str
            Path and filename where to store the model diagnostics dataset
        store_monthly_step : Bool
            If True (False)  model diagnostics will be stored monthly (yearly).
            If unspecified, we follow the update of the MB model, which
            defaults to yearly (see __init__).

        Returns
        -------
        run_ds : xarray.Dataset
            stores the entire glacier geometry. It is useful to visualize the
            glacier geometry or to restart a new run from a modelled geometry.
            The glacier state is stored at the begining of each hydrological
            year (not in between in order to spare disk space).
        diag_ds : xarray.Dataset
            stores a few diagnostic variables such as the volume, area, length
            and ELA of the glacier.
        """

        if int(y1) != y1:
            raise InvalidParamsError('run_until_and_store only accepts '
                                     'integer year dates.')

        if not self.mb_model.hemisphere:
            raise InvalidParamsError('run_until_and_store needs a '
                                     'mass-balance model with an unambiguous '
                                     'hemisphere.')
        # time
        yearly_time = np.arange(np.floor(self.yr), np.floor(y1)+1)

        if store_monthly_step is None:
            store_monthly_step = self.mb_step == 'monthly'

        if store_monthly_step:
            monthly_time = utils.monthly_timeseries(self.yr, y1)
        else:
            monthly_time = np.arange(np.floor(self.yr), np.floor(y1)+1)

        sm = cfg.PARAMS['hydro_month_' + self.mb_model.hemisphere]

        yrs, months = utils.floatyear_to_date(monthly_time)
        cyrs, cmonths = utils.hydrodate_to_calendardate(yrs, months,
                                                        start_month=sm)

        # init output
        if run_path is not None:
            self.to_netcdf(run_path)
        ny = len(yearly_time)
        if ny == 1:
            yrs = [yrs]
            cyrs = [cyrs]
            months = [months]
            cmonths = [cmonths]
        nm = len(monthly_time)
        sects = [(np.zeros((ny, fl.nx)) * np.nan) for fl in self.fls]
        widths = [(np.zeros((ny, fl.nx)) * np.nan) for fl in self.fls]
        bucket = [(np.zeros(ny) * np.nan) for _ in self.fls]
        diag_ds = xr.Dataset()

        # Global attributes
        diag_ds.attrs['description'] = 'OGGM model output'
        diag_ds.attrs['oggm_version'] = __version__
        diag_ds.attrs['calendar'] = '365-day no leap'
        diag_ds.attrs['creation_date'] = strftime("%Y-%m-%d %H:%M:%S",
                                                  gmtime())
        diag_ds.attrs['hemisphere'] = self.mb_model.hemisphere
        diag_ds.attrs['water_level'] = self.water_level

        # Coordinates
        diag_ds.coords['time'] = ('time', monthly_time)
        diag_ds.coords['hydro_year'] = ('time', yrs)
        diag_ds.coords['hydro_month'] = ('time', months)
        diag_ds.coords['calendar_year'] = ('time', cyrs)
        diag_ds.coords['calendar_month'] = ('time', cmonths)

        diag_ds['time'].attrs['description'] = 'Floating hydrological year'
        diag_ds['hydro_year'].attrs['description'] = 'Hydrological year'
        diag_ds['hydro_month'].attrs['description'] = 'Hydrological month'
        diag_ds['calendar_year'].attrs['description'] = 'Calendar year'
        diag_ds['calendar_month'].attrs['description'] = 'Calendar month'

        # Variables and attributes
        diag_ds['volume_m3'] = ('time', np.zeros(nm) * np.nan)
        diag_ds['volume_m3'].attrs['description'] = 'Total glacier volume'
        diag_ds['volume_m3'].attrs['unit'] = 'm 3'
        diag_ds['volume_bsl_m3'] = ('time', np.zeros(nm) * np.nan)
        diag_ds['volume_bsl_m3'].attrs['description'] = ('Glacier volume '
                                                            'below '
                                                            'sea-level')
        diag_ds['volume_bsl_m3'].attrs['unit'] = 'm 3'
        diag_ds['volume_bwl_m3'] = ('time', np.zeros(nm) * np.nan)
        diag_ds['volume_bwl_m3'].attrs['description'] = ('Glacier volume '
                                                            'below ')
        diag_ds['volume_bwl_m3'].attrs['unit'] = 'm 3'

        diag_ds['area_m2'] = ('time', np.zeros(nm) * np.nan)
        diag_ds['area_m2'].attrs['description'] = 'Total glacier area'
        diag_ds['area_m2'].attrs['unit'] = 'm 2'
        diag_ds['length_m'] = ('time', np.zeros(nm) * np.nan)
        diag_ds['length_m'].attrs['description'] = 'Glacier length'
        diag_ds['length_m'].attrs['unit'] = 'm 3'
        diag_ds['ela_m'] = ('time', np.zeros(nm) * np.nan)
        diag_ds['ela_m'].attrs['description'] = ('Annual Equilibrium Line '
                                                 'Altitude  (ELA)')
        diag_ds['ela_m'].attrs['unit'] = 'm a.s.l'
        if self.is_tidewater:
            diag_ds['calving_m3'] = ('time', np.zeros(nm) * np.nan)
            diag_ds['calving_m3'].attrs['description'] = ('Total accumulated '
                                                          'calving flux')
            diag_ds['calving_m3'].attrs['unit'] = 'm 3'
            diag_ds['calving_rate_myr'] = ('time', np.zeros(nm) * np.nan)
            diag_ds['calving_rate_myr'].attrs['description'] = 'Calving rate'
            diag_ds['calving_rate_myr'].attrs['unit'] = 'm yr-1'

        # Run
        j = 0
        for i, (yr, mo) in enumerate(zip(yearly_time[:-1], months[:-1])):
            
            # Record initial parameters
            if i == 0:
                diag_ds['volume_m3'].data[i] = self.volume_m3
                diag_ds['area_m2'].data[i] = self.area_m2
                diag_ds['length_m'].data[i] = self.length_m
                
                if self.is_tidewater:
                    diag_ds['volume_bsl_m3'].data[i] = self.volume_bsl_m3
                    diag_ds['volume_bwl_m3'].data[i] = self.volume_bwl_m3
            
            self.run_until(yr, run_single_year=True)
            # Model run
            if mo == 1:
                for s, w, b, fl in zip(sects, widths, bucket, self.fls):
                    s[j, :] = fl.section
                    w[j, :] = fl.widths_m
                    if self.is_tidewater:
                        try:
                            b[j] = fl.calving_bucket_m3
                        except AttributeError:
                            pass
                j += 1
            # Diagnostics
            diag_ds['volume_m3'].data[i+1] = self.volume_m3
            diag_ds['area_m2'].data[i+1] = self.area_m2
            diag_ds['length_m'].data[i+1] = self.length_m

            if self.is_tidewater:
                diag_ds['calving_m3'].data[i+1] = self.calving_m3_since_y0
                diag_ds['calving_rate_myr'].data[i+1] = self.calving_rate_myr
                diag_ds['volume_bsl_m3'].data[i+1] = self.volume_bsl_m3
                diag_ds['volume_bwl_m3'].data[i+1] = self.volume_bwl_m3

        # to datasets
        run_ds = []
        for (s, w, b) in zip(sects, widths, bucket):
            ds = xr.Dataset()
            ds.attrs['description'] = 'OGGM model output'
            ds.attrs['oggm_version'] = __version__
            ds.attrs['calendar'] = '365-day no leap'
            ds.attrs['creation_date'] = strftime("%Y-%m-%d %H:%M:%S",
                                                 gmtime())
            ds.coords['time'] = yearly_time
            ds['time'].attrs['description'] = 'Floating hydrological year'
            varcoords = OrderedDict(time=('time', yearly_time),
                                    year=('time', yearly_time))
            ds['ts_section'] = xr.DataArray(s, dims=('time', 'x'),
                                            coords=varcoords)
            ds['ts_width_m'] = xr.DataArray(w, dims=('time', 'x'),
                                            coords=varcoords)
            if self.is_tidewater:
                ds['ts_calving_bucket_m3'] = xr.DataArray(b, dims=('time', ),
                                                          coords=varcoords)
            run_ds.append(ds)

        # write output?
        if run_path is not None:
            encode = {'ts_section': {'zlib': True, 'complevel': 5},
                      'ts_width_m': {'zlib': True, 'complevel': 5},
                      }
            for i, ds in enumerate(run_ds):
                ds.to_netcdf(run_path, 'a', group='fl_{}'.format(i),
                             encoding=encode)
            # Add other diagnostics
            diag_ds.to_netcdf(run_path, 'a')

        if diag_path is not None:
            diag_ds.to_netcdf(diag_path)

        return run_ds, diag_ds
    
    
    def updategeometry(self, year, debug=False):
        """Update geometry for a given year"""
        
        if debug:
            print('year:', year)
            
        # Loop over flowlines
        for fl_id, fl in enumerate(self.fls):

            # Flowline state
            heights = self.fls[fl_id].surface_h.copy()
            section_t0 = self.fls[fl_id].section.copy()
            thick_t0 = self.fls[fl_id].thick.copy()
            width_t0 = self.fls[fl_id].widths_m.copy()
            
            # CONSTANT AREAS
            #  Mass redistribution ignored for calibration and spinup years (glacier properties constant)
            if (self.option_areaconstant) or (year < self.spinupyears) or (year < self.constantarea_years):
                # run mass balance
                glac_bin_massbalclim_annual = self.mb_model.get_annual_mb(heights, fls=self.fls, fl_id=fl_id, 
                                                                          year=year, debug=False)                                
            # MASS REDISTRIBUTION
            else:
                # FRONTAL ABLATION
                if self.is_tidewater:
                    # Frontal ablation (m3 ice)
                    fa_m3 = self._get_annual_frontalablation(heights, fls=self.fls, fl_id=fl_id, 
                                                             year=year, debug=False)
                    if debug:
                        print('fa_m3_init:', fa_m3)
                        vol_init = (self.fls[fl_id].section * fl.dx_meter).sum()
                        print('  volume init:', np.round(vol_init))
                        print('  volume final:', np.round(vol_init-fa_m3))
                    # First, remove volume lost to frontal ablation
                    #  changes to _t0 not _t1, since t1 will be done in the mass redistribution
                    
                    glac_idx_bsl = np.where((thick_t0 > 0) & (fl.bed_h < self.water_level))[0]
                    while fa_m3 > 0 and len(glac_idx_bsl) > 0:
                        if debug:
                            print('fa_m3_remaining:', fa_m3)

                        # OGGM code
#                        glac_idx_bsl = np.where((thick_t0 > 0) & (fl.bed_h < self.water_level))[0]
                        last_idx = glac_idx_bsl[-1]
                        
                        if debug:
                            print('before:', np.round(self.fls[fl_id].section[last_idx],0), 
                                  np.round(self.fls[fl_id].thick[last_idx],0), 
                                  np.round(heights[last_idx],0))
                        
                        vol_last = section_t0[last_idx] * fl.dx_meter
                        
                        # If frontal ablation more than bin volume, remove entire bin
                        if fa_m3 > vol_last:
                            # Record frontal ablation (m3 w.e.) in mass balance model for output
                            self.mb_model.glac_bin_frontalablation[last_idx,int(12*(year+1)-1)] = (
                                    vol_last * pygem_prms['constants']['density_ice'] / pygem_prms['constants']['density_water'])
                            # Update ice thickness and section area
                            section_t0[last_idx] = 0
                            self.fls[fl_id].section = section_t0 
                            # Remove volume from frontal ablation "bucket"
                            fa_m3 -= vol_last
                            
                        # Otherwise, remove ice from the section
                        else:
                            # Update section to remove frontal ablation
                            section_t0[last_idx] = section_t0[last_idx] - fa_m3 / fl.dx_meter
                            self.fls[fl_id].section = section_t0 
                            # Record frontal ablation(m3 w.e.)
                            self.mb_model.glac_bin_frontalablation[last_idx,int(12*(year+1)-1)] = (
                                    fa_m3 * pygem_prms['constants']['density_ice'] / pygem_prms['constants']['density_water'])
                            # Frontal ablation bucket now empty
                            fa_m3 = 0
                        
                        # Update flowline
                        heights = self.fls[fl_id].surface_h.copy()
                        section_t0 = self.fls[fl_id].section.copy()
                        thick_t0 = self.fls[fl_id].thick.copy()
                        width_t0 = self.fls[fl_id].widths_m.copy()
                        
                        if debug:
                            print('after:', np.round(self.fls[fl_id].section[last_idx],0), 
                                  np.round(self.fls[fl_id].thick[last_idx],0), 
                                  np.round(heights[last_idx],0))
                            print('  vol final:', (self.fls[fl_id].section * fl.dx_meter).sum())
                        
                        glac_idx_bsl = np.where((thick_t0 > 0) & (fl.bed_h < self.water_level))[0]
                
                
                # Redistribute mass if glacier was not fully removed by frontal ablation
                if len(section_t0.nonzero()[0]) > 0:
                    # Mass redistribution according to Huss empirical curves
                    # Annual glacier mass balance [m ice s-1]
                    glac_bin_massbalclim_annual = self.mb_model.get_annual_mb(heights, fls=self.fls, fl_id=fl_id, 
                                                                              year=year, debug=False)   
                    sec_in_year = (self.mb_model.dates_table.loc[12*year:12*(year+1)-1,'daysinmonth'].values.sum() 
                                   * 24 * 3600)
                    
#                    print(' volume change [m3]:', (glac_bin_massbalclim_annual * sec_in_year * 
#                                                  (width_t0 * fl.dx_meter)).sum())
#                    print(glac_bin_masssbalclim_annual)
#                    print(sec_in_year)
#                    print(width_t0.sum())
#                    print(fl.dx_meter)
#                    print(width_t0 * fl.dx_meter)
                    
#                    # Debugging block
#                    debug_years = [71]
#                    if year in debug_years:
#                        print(year, glac_bin_massbalclim_annual)
#                        print('section t0:', section_t0)
#                        print('thick_t0:', thick_t0)
#                        print('width_t0:', width_t0)
#                        print(self.glac_idx_initial[fl_id])
#                        print('heights:', heights)
                    
                    self._massredistributionHuss(section_t0, thick_t0, width_t0, glac_bin_massbalclim_annual, 
                                                 self.glac_idx_initial[fl_id], heights, sec_in_year=sec_in_year)     
                    
            # Record glacier properties (volume [m3], area [m2], thickness [m], width [km])
            #  record the next year's properties as well
            #  'year + 1' used so the glacier properties are consistent with mass balance computations
            year = int(year)  # required to ensure proper indexing with run_until_and_store (10/21/2020)
            glacier_area = fl.widths_m * fl.dx_meter
            glacier_area[fl.thick == 0] = 0
            self.mb_model.glac_bin_area_annual[:,year+1] = glacier_area
            self.mb_model.glac_bin_icethickness_annual[:,year+1] = fl.thick
            self.mb_model.glac_bin_width_annual[:,year+1] = fl.widths_m
            self.mb_model.glac_wide_area_annual[year+1] = glacier_area.sum()
            self.mb_model.glac_wide_volume_annual[year+1] = (fl.section * fl.dx_meter).sum()
            
            
    #%% ----- FRONTAL ABLATION -----
    def _get_annual_frontalablation(self, heights, year=None, fls=None, fl_id=None, calving_k=None, debug=False
                                    ):
        """Calculate frontal ablation for a given year
        
        Returns frontal ablation (m3 ice)

        Parameters
        ----------
        heights : np.array
            surface elevation
        year : int
            year starting with 0 to the number of years in the study
        """
        # Flowlines and various attributes
        fl = fls[fl_id]
        np.testing.assert_allclose(heights, fl.surface_h)
        glacier_area_t0 = fl.widths_m * fl.dx_meter
        fl_widths_m = getattr(fl, 'widths_m', None)
        fl_section = getattr(fl,'section',None)
        # Ice thickness (average)
        if fl_section is not None and fl_widths_m is not None:
            icethickness_t0 = np.zeros(fl_section.shape)
            icethickness_t0[fl_widths_m > 0] = fl_section[fl_widths_m > 0] / fl_widths_m[fl_widths_m > 0]
        else:
            icethickness_t0 = None

        # Quality control: ensure you only have glacier area where there is ice
        if icethickness_t0 is not None:
            glacier_area_t0[icethickness_t0 == 0] = 0

        # ----- FRONTAL ABLATION -----
        # -- using OGGM's parameterization -----
        # Copied from https://docs.oggm.org/en/latest/_modules/oggm/core/flowline.html (08/29/2021)
        # We do calving only if the last glacier bed pixel is below water
        # (this is to avoid calving elsewhere than at the front)
        q_calving = 0
        if glacier_area_t0.sum() > 0:
            try:
                last_above_wl = np.nonzero((fl.surface_h > self.water_level) &
                                           (fl.thick > 0))[0][-1]
            except:
                last_above_wl = np.nonzero((fl.bed_h <= self.water_level) &
                                           (fl.thick > 0))[0][-1]
            
            if not last_above_wl is None:
                if (fl.bed_h[last_above_wl] < self.water_level):
                    # Volume [m3] and bed elevation [masl] of each bin
                    if debug:
                        print('\nyear:', year, '\n  sea level:', self.water_level, 'bed elev:', np.round(fl.bed_h[last_above_wl], 2))
                        print('  estimate frontal ablation')
                        print(' min elevation:', fl.surface_h[last_above_wl])
                    
                    # --- The rest is for calving only ---
                    self.calving_rate_myr = 0.
        
                    # OK, we're really calving
                    section = fl.section
        
                    # Calving law
                    h = fl.thick[last_above_wl]
                    d = h - (fl.surface_h[last_above_wl] - self.water_level)
                    k = self.calving_k
                    q_calving = k * d * h * fl.widths_m[last_above_wl]
                    
                    # Max frontal ablation is removing all bins with bed below water level
                    glac_idx_bsl = np.where((fl.thick > 0) & (fl.bed_h < self.water_level))[0]
                    q_calving_max = np.sum(section[glac_idx_bsl]) * fl.dx_meter
                    
                    if q_calving > q_calving_max + pygem_prms['constants']['tolerance']:
                        q_calving = q_calving_max
        
                    # Add to the bucket and the diagnostics
                    self.calving_m3_since_y0 += q_calving
                    self.calving_rate_myr = q_calving / section[last_above_wl]

        return q_calving

            
    #%%%% ====== START OF MASS REDISTRIBUTION CURVE  
    def _massredistributionHuss(self, section_t0, thick_t0, width_t0, glac_bin_massbalclim_annual, 
                                glac_idx_initial, heights, debug=False, hindcast=0, sec_in_year=365*24*3600):
        """
        Mass redistribution according to empirical equations from Huss and Hock (2015) accounting for retreat/advance.
        glac_idx_initial is required to ensure that the glacier does not advance to area where glacier did not exist 
        before (e.g., retreat and advance over a vertical cliff)
        
        Note: since OGGM uses the DEM, heights along the flowline do not necessarily decrease, i.e., there can be
        overdeepenings along the flowlines that occur as the glacier retreats. This is problematic for 'adding' a bin 
        downstream in cases of glacier advance because you'd be moving new ice to a higher elevation. To avoid this 
        unrealistic case, in the event that this would occur, the overdeepening will simply fill up with ice first until
        it reaches an elevation where it would put new ice into a downstream bin.

        Parameters
        ----------
        section_t0 : np.ndarray
            Glacier cross-sectional area (m2) from previous year for each elevation bin
        thick_t0 : np.ndarray
            Glacier ice thickness [m] from previous year for each elevation bin
        width_t0 : np.ndarray
            Glacier width [m] from previous year for each elevation bin
        glac_bin_massbalclim_annual : np.ndarray
            Climatic mass balance [m ice s-1] for each elevation bin and year
        glac_idx_initial : np.ndarray
            Initial glacier indices
        debug : Boolean
            option to turn on print statements for development or debugging of code (default False)
            
        Returns
        -------
        Updates the flowlines automatically, so does not return anything
        """        
        # Glacier area [m2]
        glacier_area_t0 = width_t0 * self.fls[0].dx_meter
        glacier_area_t0[thick_t0 == 0] = 0
        
        # Annual glacier-wide volume change [m3]
        #  units: [m ice / s] * [s] * [m2] = m3 ice
        glacier_volumechange = (glac_bin_massbalclim_annual * sec_in_year * glacier_area_t0).sum()
        
        # For hindcast simulations, volume change is the opposite
        if hindcast == 1:
            glacier_volumechange = -1 * glacier_volumechange
            
        if debug:
            print('\nDebugging Mass Redistribution Huss function\n')
            print('glacier volume change:', glacier_volumechange)
            
        # If volume loss is more than the glacier volume, melt everything and stop here
        glacier_volume_total = (self.fls[0].section * self.fls[0].dx_meter).sum()
        if (glacier_volume_total + glacier_volumechange) < 0:
            # Set all to zero and return
            self.fls[0].section *= 0
            return 
            
        # Otherwise, redistribute mass loss/gains across the glacier
        # Determine where glacier exists            
        glac_idx_t0 = self.fls[0].thick.nonzero()[0]
        
        # Compute ice thickness [m ice], glacier area [m2], ice thickness change [m ice] after redistribution
        icethickness_change, glacier_volumechange_remaining = (
                self._massredistributioncurveHuss(section_t0, thick_t0, width_t0, glac_idx_t0,
                                                  glacier_volumechange, glac_bin_massbalclim_annual,
                                                  heights, debug=False))
        if debug:
            print('\nmax icethickness change:', np.round(icethickness_change.max(),3), 
                  '\nmin icethickness change:', np.round(icethickness_change.min(),3), 
                  '\nvolume remaining:', glacier_volumechange_remaining)
            nloop = 0

        # Glacier retreat
        #  if glacier retreats (ice thickness == 0), volume change needs to be redistributed over glacier again
        while glacier_volumechange_remaining < 0:
            
            if debug:
                print('\n\nGlacier retreating (loop ' + str(nloop) + '):')
            
            section_t0_retreated = self.fls[0].section.copy()
            thick_t0_retreated = self.fls[0].thick.copy()
            width_t0_retreated = self.fls[0].widths_m.copy()
            glacier_volumechange_remaining_retreated = glacier_volumechange_remaining.copy()
            glac_idx_t0_retreated = thick_t0_retreated.nonzero()[0]  
            glacier_area_t0_retreated = width_t0_retreated * self.fls[0].dx_meter
            glacier_area_t0_retreated[thick_t0 == 0] = 0
            # Set climatic mass balance for the case when there are less than 3 bins  
            #  distribute the remaining glacier volume change over the entire glacier (remaining bins)
            massbalclim_retreat = np.zeros(thick_t0_retreated.shape)
            massbalclim_retreat[glac_idx_t0_retreated] = (glacier_volumechange_remaining / 
                                                           glacier_area_t0_retreated.sum() / sec_in_year)
            # Mass redistribution 
            # apply mass redistribution using Huss' empirical geometry change equations
            icethickness_change, glacier_volumechange_remaining = (
                self._massredistributioncurveHuss(
                        section_t0_retreated, thick_t0_retreated, width_t0_retreated, glac_idx_t0_retreated, 
                        glacier_volumechange_remaining_retreated, massbalclim_retreat, heights, debug=False))
            # Avoid rounding errors that get loop stuck
            if abs(glacier_volumechange_remaining) < 1:
                glacier_volumechange_remaining = 0
            
            if debug:
                print('ice thickness change:', icethickness_change)
                print('\nmax icethickness change:', np.round(icethickness_change.max(),3), 
                      '\nmin icethickness change:', np.round(icethickness_change.min(),3), 
                      '\nvolume remaining:', glacier_volumechange_remaining)
                nloop += 1

        # Glacier advances 
        #  based on ice thickness change exceeding threshold
        #  Overview:
        #    1. Add new bin and fill it up to a maximum of terminus average ice thickness
        #    2. If additional volume after adding new bin, then redistribute mass gain across all bins again,
        #       i.e., increase the ice thickness and width
        #    3. Repeat adding a new bin and redistributing the mass until no addiitonal volume is left
        while (icethickness_change > pygem_prms['sim']['icethickness_advancethreshold']).any() == True: 
            if debug:
                print('advancing glacier')
                
            # Record glacier area and ice thickness before advance corrections applied
            section_t0_raw = self.fls[0].section.copy()
            thick_t0_raw = self.fls[0].thick.copy()
            width_t0_raw = self.fls[0].widths_m.copy()
            glacier_area_t0_raw = width_t0_raw * self.fls[0].dx_meter
            
            if debug:
                print('\n\nthickness t0:', thick_t0_raw)
                print('glacier area t0:', glacier_area_t0_raw)
                print('width_t0_raw:', width_t0_raw,'\n\n')
            
            # Index bins that are advancing
            icethickness_change[icethickness_change <= pygem_prms['sim']['icethickness_advancethreshold']] = 0
            glac_idx_advance = icethickness_change.nonzero()[0]
            
            # Update ice thickness based on maximum advance threshold [m ice]
            self.fls[0].thick[glac_idx_advance] = (self.fls[0].thick[glac_idx_advance] - 
                           (icethickness_change[glac_idx_advance] - pygem_prms['sim']['icethickness_advancethreshold']))
            glacier_area_t1 = self.fls[0].widths_m.copy() * self.fls[0].dx_meter
            
            # Advance volume [m3]
            advance_volume = ((glacier_area_t0_raw[glac_idx_advance] * thick_t0_raw[glac_idx_advance]).sum() 
                              - (glacier_area_t1[glac_idx_advance] * self.fls[0].thick[glac_idx_advance]).sum())

            if debug:
                print('advance volume [m3]:', advance_volume)

            # Set the cross sectional area of the next bin
            advance_section = advance_volume / self.fls[0].dx_meter
            
            # Index of bin to add
            glac_idx_t0 = self.fls[0].thick.nonzero()[0]
            min_elev = self.fls[0].surface_h[glac_idx_t0].min()
            
            # -------------------
            glac_idx_t0_term = np.where(self.fls[0].surface_h == min_elev)[0]
            
            # Check if last bin is below sea-level and if it is, then fill it up
            if self.fls[0].surface_h[glac_idx_t0_term] < self.water_level:
                
                # Check that not additional bin is not higher than others
                if len(glac_idx_t0) > 2:
                    elev_sorted = np.sort(self.fls[0].surface_h[glac_idx_t0])
                    elev_term = elev_sorted[1] - abs(elev_sorted[2] - elev_sorted[1])
                else:
                    elev_term = self.water_level
                
                if debug:
                    print(self.fls[0].surface_h[glac_idx_t0])
                    print(glac_idx_t0_term, 'height:', self.fls[0].surface_h[glac_idx_t0_term], 
                          'thickness:', self.fls[0].thick[glac_idx_t0_term])
                    print(np.where(self.fls[0].surface_h[glac_idx_t0] > self.fls[0].surface_h[glac_idx_t0_term])[0])
                    print('advance section:', advance_section)
                
                thick_prior = np.copy(self.fls[0].thick)
                section_updated = np.copy(self.fls[0].section)
                section_updated[glac_idx_t0_term] = section_updated[glac_idx_t0_term] + advance_section
                
                if debug:
                    print(self.fls[0].section[glac_idx_t0_term], self.fls[0].surface_h[glac_idx_t0_term], self.fls[0].thick[glac_idx_t0_term])
                
                self.fls[0].section = section_updated
                
                # Set advance volume to zero
                advance_volume = 0
                icethickness_change = self.fls[0].thick - thick_prior

                if debug:
                    print(self.fls[0].section[glac_idx_t0_term], self.fls[0].surface_h[glac_idx_t0_term], self.fls[0].thick[glac_idx_t0_term])
                
                    print('surface_h:', self.fls[0].surface_h[glac_idx_t0], 
                          '\nmax term elev:', elev_term)
                    print('icethickness_change:', icethickness_change)


                if self.fls[0].surface_h[glac_idx_t0_term] > elev_term:
                    
                    # Record parameters to calculate advance_volume if necessary
                    section_t0_raw = self.fls[0].section.copy()
                    thick_t0_raw = self.fls[0].thick.copy()
                    width_t0_raw = self.fls[0].widths_m.copy()
                    glacier_area_t0_raw = width_t0_raw * self.fls[0].dx_meter
                    
                    thick_reduction = self.fls[0].surface_h[glac_idx_t0_term] - elev_term
                    
                    if debug:
                        print('thick_reduction:', thick_reduction)
                        print('----\nprior to correction:', self.fls[0].thick[glac_idx_t0_term], self.fls[0].section[glac_idx_t0_term])
                    
                    self.fls[0].thick[glac_idx_t0_term] = (self.fls[0].thick[glac_idx_t0_term] - thick_reduction)
                    glacier_area_t1 = self.fls[0].widths_m.copy() * self.fls[0].dx_meter
                    
                    # Advance volume [m3]
                    advance_volume = ((glacier_area_t0_raw[glac_idx_t0_term] * thick_t0_raw[glac_idx_t0_term]).sum() 
                                      - (glacier_area_t1[glac_idx_t0_term] * self.fls[0].thick[glac_idx_t0_term]).sum())
                    
                    if debug:
                        print('post correction:', self.fls[0].thick[glac_idx_t0_term], self.fls[0].section[glac_idx_t0_term])
                        print('surface_h:', self.fls[0].surface_h[glac_idx_t0])
                        print('advance_volume:', advance_volume)
                        print('icethickness_change:', icethickness_change)
                    
                # Set icethickness change of terminus to 0 to avoid while loop issues
                icethickness_change[glac_idx_t0_term] = 0


            if advance_volume > 0:
                glac_idx_bin2add = (
                        np.where(self.fls[0].surface_h == 
                                 self.fls[0].surface_h[np.where(self.fls[0].surface_h < min_elev)[0]].max())[0][0])
                section_2add = self.fls[0].section.copy()
                section_2add[glac_idx_bin2add] = advance_section
                self.fls[0].section = section_2add              
    
                # Advance characteristics
                # Indices that define the glacier terminus
                glac_idx_terminus = (
                        glac_idx_t0[(heights[glac_idx_t0] - heights[glac_idx_t0].min()) / 
                                    (heights[glac_idx_t0].max() - heights[glac_idx_t0].min()) * 100 
                                    < pygem_prms['sim']['terminus_percentage']])
                # For glaciers with so few bands that the terminus is not identified (ex. <= 4 bands for 20% threshold),
                #  then use the information from all the bands
                if glac_idx_terminus.shape[0] <= 1:
                    glac_idx_terminus = glac_idx_t0.copy()
                
                if debug:
                    print('glacier index terminus:',glac_idx_terminus)
    
                # Average area of glacier terminus [m2]
                #  exclude the bin at the terminus, since this bin may need to be filled first
                try:
                    minelev_idx = np.where(heights == heights[glac_idx_terminus].min())[0][0]
                    glac_idx_terminus_removemin = list(glac_idx_terminus)
                    glac_idx_terminus_removemin.remove(minelev_idx)
                    terminus_thickness_avg = np.mean(self.fls[0].thick[glac_idx_terminus_removemin])
                except:  
                    glac_idx_terminus_initial = (
                        glac_idx_initial[(heights[glac_idx_initial] - heights[glac_idx_initial].min()) / 
                                    (heights[glac_idx_initial].max() - heights[glac_idx_initial].min()) * 100 
                                    < pygem_prms['sim']['terminus_percentage']])
                    if glac_idx_terminus_initial.shape[0] <= 1:
                        glac_idx_terminus_initial = glac_idx_initial.copy()
                        
                    minelev_idx = np.where(heights == heights[glac_idx_terminus_initial].min())[0][0]
                    glac_idx_terminus_removemin = list(glac_idx_terminus_initial)
                    glac_idx_terminus_removemin.remove(minelev_idx)
                    terminus_thickness_avg = np.mean(self.fls[0].thick[glac_idx_terminus_removemin])
                
                # If last bin exceeds terminus thickness average then fill up the bin to average and redistribute mass
                if self.fls[0].thick[glac_idx_bin2add] > terminus_thickness_avg:
                    self.fls[0].thick[glac_idx_bin2add] = terminus_thickness_avg
                    # Redistribute remaining mass
                    volume_added2bin = self.fls[0].section[glac_idx_bin2add] * self.fls[0].dx_meter
                    advance_volume -= volume_added2bin

            # With remaining advance volume, add a bin or redistribute over existing bins if no bins left
            if advance_volume > 0:
                # Indices for additional bins below the terminus
                glac_idx_t1 = np.where(glacier_area_t1 > 0)[0]
                below_glac_idx = np.where(heights < heights[glac_idx_t1].min())[0]

                # if no more bins below, then distribute volume over the glacier without further adjustments
                #  this occurs with OGGM flowlines when the terminus is in an overdeepening, so we just fill up 
                #  the overdeepening
                if len(below_glac_idx) == 0:
                    # Revert to the initial section, which also updates the thickness and width automatically
                    self.fls[0].section = section_t0_raw
                    
                    # set icethickness change and advance_volume to 0 to break the loop
                    icethickness_change[icethickness_change > 0] = 0
                    advance_volume = 0
                    
                # otherwise, redistribute mass
                else:
                    glac_idx_t0 = self.fls[0].thick.nonzero()[0]
                    glacier_area_t0 = self.fls[0].widths_m.copy() * self.fls[0].dx_meter
                    glac_bin_massbalclim_annual = np.zeros(self.fls[0].thick.shape)
                    glac_bin_massbalclim_annual[glac_idx_t0] = (glacier_volumechange_remaining / 
                                                                glacier_area_t0.sum() / sec_in_year)
                    icethickness_change, glacier_volumechange_remaining = (
                        self._massredistributioncurveHuss(
                                self.fls[0].section.copy(), self.fls[0].thick.copy(), self.fls[0].widths_m.copy(), 
                                glac_idx_t0, advance_volume, glac_bin_massbalclim_annual, heights, debug=False))
    
    
    def _massredistributioncurveHuss(self, section_t0, thick_t0, width_t0, glac_idx_t0, glacier_volumechange, 
                                     massbalclim_annual, heights, debug=False):
        """
        Apply the mass redistribution curves from Huss and Hock (2015).
        This is paired with massredistributionHuss, which takes into consideration retreat and advance.
        
        Parameters
        ----------
        section_t0 : np.ndarray
            Glacier cross-sectional area [m2] from previous year for each elevation bin
        thick_t0 : np.ndarray
            Glacier ice thickness [m] from previous year for each elevation bin
        width_t0 : np.ndarray
            Glacier width [m] from previous year for each elevation bin
        glac_idx_t0 : np.ndarray
            glacier indices for present timestep
        glacier_volumechange : float
            glacier-wide volume change [m3 ice] based on the annual climatic mass balance
        massbalclim_annual : np.ndarray
            Annual climatic mass balance [m ice s-1] for each elevation bin for a single year
        Returns
        -------
        icethickness_change : np.ndarray
            Ice thickness change [m] for each elevation bin
        glacier_volumechange_remaining : float
            Glacier volume change remaining [m3 ice]; occurs if there is less ice than melt in a bin, i.e., retreat
        """ 
          
        if debug:
            print('\nDebugging mass redistribution curve Huss\n')

        # Apply Huss redistribution if there are at least 3 elevation bands; otherwise, use the mass balance        
        # Glacier area used to select parameters
        glacier_area_t0 = width_t0 * self.fls[0].dx_meter
        glacier_area_t0[thick_t0 == 0] = 0
        
        # Apply mass redistribution curve
        if glac_idx_t0.shape[0] > 3:
            # Select the factors for the normalized ice thickness change curve based on glacier area
            if glacier_area_t0.sum() > 20 * 1e6:
                [gamma, a, b, c] = [6, -0.02, 0.12, 0]
            elif glacier_area_t0.sum() > 5 * 1e6:
                [gamma, a, b, c] = [4, -0.05, 0.19, 0.01]
            else:
                [gamma, a, b, c] = [2, -0.30, 0.60, 0.09]
            # reset variables
            elevrange_norm = np.zeros(glacier_area_t0.shape)
            icethicknesschange_norm = np.zeros(glacier_area_t0.shape)
            # Normalized elevation range [-]
            #  (max elevation - bin elevation) / (max_elevation - min_elevation)
            elevrange_norm[glacier_area_t0 > 0] = ((heights[glac_idx_t0].max() - heights[glac_idx_t0]) / 
                                                   (heights[glac_idx_t0].max() - heights[glac_idx_t0].min()))
            
            #  using indices as opposed to elevations automatically skips bins on the glacier that have no area
            #  such that the normalization is done only on bins where the glacier lies
            # Normalized ice thickness change [-]
            icethicknesschange_norm[glacier_area_t0 > 0] = ((elevrange_norm[glacier_area_t0 > 0] + a)**gamma + 
                                                            b*(elevrange_norm[glacier_area_t0 > 0] + a) + c)
            #  delta_h = (h_n + a)**gamma + b*(h_n + a) + c
            #  indexing is faster here
            # limit the icethicknesschange_norm to between 0 - 1 (ends of fxns not exactly 0 and 1)
            icethicknesschange_norm[icethicknesschange_norm > 1] = 1
            icethicknesschange_norm[icethicknesschange_norm < 0] = 0
            # Huss' ice thickness scaling factor, fs_huss [m ice]         
            #  units: m3 / (m2 * [-]) * (1000 m / 1 km) = m ice
            fs_huss = glacier_volumechange / (glacier_area_t0 * icethicknesschange_norm).sum()
            if debug:
                print('fs_huss:', fs_huss)
            # Volume change [m3 ice]
            bin_volumechange = icethicknesschange_norm * fs_huss * glacier_area_t0
            
        # Otherwise, compute volume change in each bin based on the climatic mass balance
        else:
            bin_volumechange = massbalclim_annual * glacier_area_t0
            
        if debug:
            print('-----\n')
            vol_before = section_t0 * self.fls[0].dx_meter

        # Update cross sectional area (updating thickness does not conserve mass in OGGM!) 
        #  volume change divided by length (dx); units m2
        section_change = bin_volumechange / self.fls[0].dx_meter
        self.fls[0].section = utils.clip_min(self.fls[0].section + section_change, 0)
        # Ice thickness change [m ice]
        icethickness_change = self.fls[0].thick - thick_t0
        # Glacier volume
        vol_after = self.fls[0].section * self.fls[0].dx_meter
        
        if debug:
            print('vol_chg_wanted:', bin_volumechange.sum())
            print('vol_chg:', (vol_after.sum() - vol_before.sum()))
            print('\n-----')
        
        # Compute the remaining volume change
        bin_volumechange_remaining = (bin_volumechange - (self.fls[0].section * self.fls[0].dx_meter - 
                                                          section_t0 * self.fls[0].dx_meter))
        # remove values below tolerance to avoid rounding errors
        bin_volumechange_remaining[abs(bin_volumechange_remaining) < pygem_prms['constants']['tolerance']] = 0
        # Glacier volume change remaining - if less than zero, then needed for retreat
        glacier_volumechange_remaining = bin_volumechange_remaining.sum()  
        
        if debug:
            print(glacier_volumechange_remaining)

        return icethickness_change, glacier_volumechange_remaining
