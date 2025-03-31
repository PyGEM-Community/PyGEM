"""
Python Glacier Evolution Model (PyGEM)

copyright © 2018 David Rounce <drounce@cmu.edu

Distrubted under the MIT lisence
"""
# External libraries
import numpy as np
# Local libraries
from oggm.core.massbalance import MassBalanceModel
from pygem.utils._funcs import annualweightedmean_array
from pygem.setup.config import ConfigManager
# instantiate ConfigManager
config_manager = ConfigManager()
# read the config
pygem_prms = config_manager.read_config()

#%%
class PyGEMMassBalance(MassBalanceModel):
    """Mass-balance computed from the Python Glacier Evolution Model.

    This mass balance accounts for ablation, accumulation, and refreezing.

    This class implements the MassBalanceModel interface so that the dynamical model can use it.
    """
    def __init__(self, gdir, modelprms, glacier_rgi_table,
                 option_areaconstant=False, hindcast=pygem_prms['climate']['hindcast'], frontalablation_k=None,
                 debug=pygem_prms['debug']['mb'], debug_refreeze=pygem_prms['debug']['refreeze'],
                 fls=None, fl_id=0,
                 heights=None, repeat_period=False,
                 inversion_filter=False,
                 ignore_debris=False
                       ):
        """ Initialize.

        Parameters
        ----------
        modelprms : dict
            Model parameters dictionary (lrgcm, lrglac, precfactor, precgrad, ddfsnow, ddfice, tempsnow, tempchange)
        glacier_rgi_table : pd.Series
            Table of glacier's RGI information
        option_areaconstant : Boolean
            option to keep glacier area constant (default False allows glacier area to change annually)
        frontalablation_k : float
            frontal ablation parameter
        debug : Boolean
            option to turn on print statements for development or debugging of code
        debug_refreeze : Boolean
            option to turn on print statements for development/debugging of refreezing code
        hindcast : Boolean
            switch to run the model in reverse or not (may be irrelevant after converting to OGGM's setup)
        """
        if debug:
            print('\n\nDEBUGGING MASS BALANCE FUNCTION\n\n')
        self.debug_refreeze = debug_refreeze
        self.inversion_filter = inversion_filter

        super(PyGEMMassBalance, self).__init__()
        self.valid_bounds = [-1e4, 2e4]  # in m
        self.hemisphere = gdir.hemisphere

        # Glacier data
        self.modelprms = modelprms
        self.glacier_rgi_table = glacier_rgi_table
        self.is_tidewater = gdir.is_tidewater
        self.icethickness_initial = getattr(fls[fl_id], 'thick', None)
        self.width_initial = fls[fl_id].widths_m
        self.glacier_area_initial = fls[fl_id].widths_m * fls[fl_id].dx_meter
        self.heights = fls[fl_id].surface_h
        if pygem_prms['mb']['include_debris'] and not ignore_debris and not gdir.is_tidewater:
            try:
                self.debris_ed = fls[fl_id].debris_ed
            except:
                self.debris_ed = np.ones(self.glacier_area_initial.shape[0])
        else:
            self.debris_ed = np.ones(self.glacier_area_initial.shape[0])

        self.glac_idx_initial = self.glacier_area_initial.nonzero()

        # Climate data
        self.dates_table = gdir.dates_table
        self.glacier_gcm_temp = gdir.historical_climate['temp']
        self.glacier_gcm_tempstd = gdir.historical_climate['tempstd']
        self.glacier_gcm_prec = gdir.historical_climate['prec']
        self.glacier_gcm_elev = gdir.historical_climate['elev']
        self.glacier_gcm_lrgcm = gdir.historical_climate['lr']
        self.glacier_gcm_lrglac = gdir.historical_climate['lr']

        if pygem_prms['climate']['hindcast'] == True:
            self.glacier_gcm_prec = self.glacier_gcm_prec[::-1]
            self.glacier_gcm_temp = self.glacier_gcm_temp[::-1]
            self.glacier_gcm_lrgcm = self.glacier_gcm_lrgcm[::-1]
            self.glacier_gcm_lrglac = self.glacier_gcm_lrglac[::-1]

        self.repeat_period = repeat_period

        # Variables to store (consider storing in xarray)
        nbins = self.glacier_area_initial.shape[0]
        
        self.nmonths = self.glacier_gcm_temp.shape[0]
        self.nyears = int(self.dates_table.shape[0] / 12)

        self.bin_temp = np.zeros((nbins,self.nmonths))
        self.bin_prec = np.zeros((nbins,self.nmonths))
        self.bin_acc = np.zeros((nbins,self.nmonths))
        self.bin_refreezepotential = np.zeros((nbins,self.nmonths))
        self.bin_refreeze = np.zeros((nbins,self.nmonths))
        self.bin_meltglac = np.zeros((nbins,self.nmonths))
        self.bin_meltsnow = np.zeros((nbins,self.nmonths))
        self.bin_melt = np.zeros((nbins,self.nmonths))
        self.bin_snowpack = np.zeros((nbins,self.nmonths))
        self.snowpack_remaining = np.zeros((nbins,self.nmonths))
        self.glac_bin_refreeze = np.zeros((nbins,self.nmonths))
        self.glac_bin_melt = np.zeros((nbins,self.nmonths))
        self.glac_bin_frontalablation = np.zeros((nbins,self.nmonths))
        self.glac_bin_snowpack = np.zeros((nbins,self.nmonths))
        self.glac_bin_massbalclim = np.zeros((nbins,self.nmonths))
        self.glac_bin_massbalclim_annual = np.zeros((nbins,self.nyears))
        self.glac_bin_surfacetype_annual = np.zeros((nbins,self.nyears+1))
        self.glac_bin_area_annual = np.zeros((nbins,self.nyears+1))
        self.glac_bin_icethickness_annual = np.zeros((nbins,self.nyears+1)) # Needed for MassRedistributionCurves
        self.glac_bin_width_annual = np.zeros((nbins,self.nyears+1))        # Needed for MassRedistributionCurves
        self.offglac_bin_prec = np.zeros((nbins,self.nmonths))
        self.offglac_bin_melt = np.zeros((nbins,self.nmonths))
        self.offglac_bin_refreeze = np.zeros((nbins,self.nmonths))
        self.offglac_bin_snowpack = np.zeros((nbins,self.nmonths))
        self.offglac_bin_area_annual = np.zeros((nbins,self.nyears+1))
        self.glac_wide_temp = np.zeros(self.nmonths)
        self.glac_wide_prec = np.zeros(self.nmonths)
        self.glac_wide_acc = np.zeros(self.nmonths)
        self.glac_wide_refreeze = np.zeros(self.nmonths)
        self.glac_wide_melt = np.zeros(self.nmonths)
        self.glac_wide_frontalablation = np.zeros(self.nmonths)
        self.glac_wide_massbaltotal = np.zeros(self.nmonths)
        self.glac_wide_runoff = np.zeros(self.nmonths)
        self.glac_wide_snowline = np.zeros(self.nmonths)
        self.glac_wide_area_annual = np.zeros(self.nyears+1)
        self.glac_wide_volume_annual = np.zeros(self.nyears+1)
        self.glac_wide_volume_change_ignored_annual = np.zeros(self.nyears)
        self.glac_wide_ELA_annual = np.zeros(self.nyears+1)
        self.offglac_wide_prec = np.zeros(self.nmonths)
        self.offglac_wide_refreeze = np.zeros(self.nmonths)
        self.offglac_wide_melt = np.zeros(self.nmonths)
        self.offglac_wide_snowpack = np.zeros(self.nmonths)
        self.offglac_wide_runoff = np.zeros(self.nmonths)

        self.dayspermonth = self.dates_table['daysinmonth'].values
        self.surfacetype_ddf = np.zeros((nbins))

        # Surface type DDF dictionary (manipulate this function for calibration or for each glacier)
        self.surfacetype_ddf_dict = self._surfacetypeDDFdict(self.modelprms)
        self.surfacetype, self.firnline_idx = self._surfacetypebinsinitial(self.heights)

        # Refreezing specific layers
        if pygem_prms['mb']['option_refreezing'] == 'HH2015':
            # Refreezing layers density, volumetric heat capacity, and thermal conductivity
            self.rf_dens_expb = (pygem_prms['mb']['HH2015_rf_opts']['rf_dens_bot'] / pygem_prms['mb']['HH2015_rf_opts']['rf_dens_top'])**(1/(pygem_prms['mb']['HH2015_rf_opts']['rf_layers']-1))
            self.rf_layers_dens = np.array([pygem_prms['mb']['HH2015_rf_opts']['rf_dens_top'] * self.rf_dens_expb**x
                                            for x in np.arange(0,pygem_prms['mb']['HH2015_rf_opts']['rf_layers'])])
            self.rf_layers_ch = ((1 - self.rf_layers_dens/1000) * pygem_prms['constants']['ch_air'] + self.rf_layers_dens/1000 *
                                 pygem_prms['constants']['ch_ice'])
            self.rf_layers_k = ((1 - self.rf_layers_dens/1000) * pygem_prms['constants']['k_air'] + self.rf_layers_dens/1000 *
                                pygem_prms['constants']['k_ice'])
            # refreeze in each bin
            self.refr = np.zeros(nbins)
            # refrezee cold content or "potential" refreeze
            self.rf_cold = np.zeros(nbins)
            # layer temp of each elev bin for present time step
            self.te_rf = np.zeros((pygem_prms['mb']['HH2015_rf_opts']['rf_layers'],nbins,self.nmonths)) 
            # layer temp of each elev bin for previous time step
            self.tl_rf = np.zeros((pygem_prms['mb']['HH2015_rf_opts']['rf_layers'],nbins,self.nmonths)) 

        # Sea level for marine-terminating glaciers
        self.sea_level = 0
        rgi_region = int(glacier_rgi_table.RGIId.split('-')[1].split('.')[0])


    def get_annual_mb(self, heights, year=None, fls=None, fl_id=None,
                      debug=pygem_prms['debug']['mb'], option_areaconstant=False):
        """FIXED FORMAT FOR THE FLOWLINE MODEL

        Returns annual climatic mass balance [m ice per second]

        Parameters
        ----------
        heights : np.array
            elevation bins
        year : int
            year starting with 0 to the number of years in the study
            
        Returns
        -------
        mb : np.array
            mass balance for each bin [m ice per second]
        """
        year = int(year)
        if self.repeat_period:
            year = year % (pygem_prms['climate']['gcm_endyear'] - pygem_prms['climate']['gcm_startyear'])

        fl = fls[fl_id]
        np.testing.assert_allclose(heights, fl.surface_h)
        glacier_area_t0 = fl.widths_m * fl.dx_meter
        glacier_area_initial = self.glacier_area_initial
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
            
        # Record ice thickness
        self.glac_bin_icethickness_annual[:,year] = icethickness_t0
        
        # Glacier indices
        glac_idx_t0 = glacier_area_t0.nonzero()[0]
        
        nbins = heights.shape[0]
        nmonths = self.glacier_gcm_temp.shape[0]

        # Local variables
        bin_precsnow = np.zeros((nbins,nmonths))

        # Refreezing specific layers
        if pygem_prms['mb']['option_refreezing'] == 'HH2015' and year == 0:
            self.te_rf[:,:,0] = 0     # layer temp of each elev bin for present time step
            self.tl_rf[:,:,0] = 0     # layer temp of each elev bin for previous time step
        elif pygem_prms['mb']['option_refreezing'] == 'Woodward':
            refreeze_potential = np.zeros(nbins)

        if self.glacier_area_initial.sum() > 0:
#        if len(glac_idx_t0) > 0:
    
            # Surface type [0=off-glacier, 1=ice, 2=snow, 3=firn, 4=debris]
            if year == 0:
                self.surfacetype, self.firnline_idx = self._surfacetypebinsinitial(self.heights)
            self.glac_bin_surfacetype_annual[:,year] = self.surfacetype

            # Off-glacier area and indices
            if option_areaconstant == False:
                self.offglac_bin_area_annual[:,year] = glacier_area_initial - glacier_area_t0
                offglac_idx = np.where(self.offglac_bin_area_annual[:,year] > 0)[0]

            # Functions currently set up for monthly timestep
            #  only compute mass balance while glacier exists
            if (pygem_prms['time']['timestep'] == 'monthly'):
#            if (pygem_prms['time']['timestep'] == 'monthly') and (glac_idx_t0.shape[0] != 0):

                # AIR TEMPERATURE: Downscale the gcm temperature [deg C] to each bin
                if pygem_prms['mb']['option_temp2bins'] == 1:
                    # Downscale using gcm and glacier lapse rates
                    #  T_bin = T_gcm + lr_gcm * (z_ref - z_gcm) + lr_glac * (z_bin - z_ref) + tempchange               
                    self.bin_temp[:,12*year:12*(year+1)] = (self.glacier_gcm_temp[12*year:12*(year+1)] +
                         self.glacier_gcm_lrgcm[12*year:12*(year+1)] *
                         (self.glacier_rgi_table.loc[pygem_prms['mb']['option_elev_ref_downscale']] - self.glacier_gcm_elev) +
                         self.glacier_gcm_lrglac[12*year:12*(year+1)] * (heights -
                         self.glacier_rgi_table.loc[pygem_prms['mb']['option_elev_ref_downscale']])[:, np.newaxis] +
                                                self.modelprms['tbias'])

                # PRECIPITATION/ACCUMULATION: Downscale the precipitation (liquid and solid) to each bin
                if pygem_prms['mb']['option_prec2bins'] == 1:
                    # Precipitation using precipitation factor and precipitation gradient
                    #  P_bin = P_gcm * prec_factor * (1 + prec_grad * (z_bin - z_ref))
                    bin_precsnow[:,12*year:12*(year+1)] = (self.glacier_gcm_prec[12*year:12*(year+1)] *
                            self.modelprms['kp'] * (1 + self.modelprms['precgrad'] * (heights -
                            self.glacier_rgi_table.loc[pygem_prms['mb']['option_elev_ref_downscale']]))[:,np.newaxis])
                # Option to adjust prec of uppermost 25% of glacier for wind erosion and reduced moisture content
                if pygem_prms['mb']['option_preclimit'] == 1:
                    # Elevation range based on all flowlines
                    raw_min_elev = []
                    raw_max_elev = []
                    if len(fl.surface_h[fl.widths_m > 0]):
                        raw_min_elev.append(fl.surface_h[fl.widths_m > 0].min())
                        raw_max_elev.append(fl.surface_h[fl.widths_m > 0].max())
                    elev_range = np.max(raw_max_elev) - np.min(raw_min_elev)
                    elev_75 = np.min(raw_min_elev) + 0.75 * (elev_range)

                    # If elevation range > 1000 m, apply corrections to uppermost 25% of glacier (Huss and Hock, 2015)
                    if elev_range > 1000:
                        # Indices of upper 25%
                        glac_idx_upper25 = glac_idx_t0[heights[glac_idx_t0] >= elev_75]
                        # Exponential decay according to elevation difference from the 75% elevation
                        #  prec_upper25 = prec * exp(-(elev_i - elev_75%)/(elev_max- - elev_75%))
                        # height at 75% of the elevation
                        height_75 = heights[glac_idx_upper25].min()
                        glac_idx_75 = np.where(heights == height_75)[0][0]
                        # exponential decay
                        bin_precsnow[glac_idx_upper25,12*year:12*(year+1)] = (
                                bin_precsnow[glac_idx_75,12*year:12*(year+1)] *
                                np.exp(-1*(heights[glac_idx_upper25] - height_75) /
                                       (heights[glac_idx_upper25].max() - heights[glac_idx_upper25].min()))
                                [:,np.newaxis])
                        # Precipitation cannot be less than 87.5% of the maximum accumulation elsewhere on the glacier
                        for month in range(0,12):
                            bin_precsnow[glac_idx_upper25[(bin_precsnow[glac_idx_upper25,month] < 0.875 *
                                                           bin_precsnow[glac_idx_t0,month].max()) &
                                                          (bin_precsnow[glac_idx_upper25,month] != 0)], month] = (
                                    0.875 * bin_precsnow[glac_idx_t0,month].max())
                                                                                              
                # Separate total precipitation into liquid (bin_prec) and solid (bin_acc)
                if pygem_prms['mb']['option_accumulation'] == 1:
                    # if temperature above threshold, then rain
                    (self.bin_prec[:,12*year:12*(year+1)]
                                  [self.bin_temp[:,12*year:12*(year+1)] > self.modelprms['tsnow_threshold']]) = (
                        bin_precsnow[:,12*year:12*(year+1)]
                            [self.bin_temp[:,12*year:12*(year+1)] > self.modelprms['tsnow_threshold']])
                    # if temperature below threshold, then snow
                    (self.bin_acc[:,12*year:12*(year+1)]
                                 [self.bin_temp[:,12*year:12*(year+1)] <= self.modelprms['tsnow_threshold']]) = (
                        bin_precsnow[:,12*year:12*(year+1)]
                            [self.bin_temp[:,12*year:12*(year+1)] <= self.modelprms['tsnow_threshold']])
                elif pygem_prms['mb']['option_accumulation'] == 2:
                    # if temperature between min/max, then mix of snow/rain using linear relationship between min/max
                    self.bin_prec[:,12*year:12*(year+1)] = (
                            (0.5 + (self.bin_temp[:,12*year:12*(year+1)] -
                             self.modelprms['tsnow_threshold']) / 2) * bin_precsnow[:,12*year:12*(year+1)])
                    self.bin_acc[:,12*year:12*(year+1)] = (
                            bin_precsnow[:,12*year:12*(year+1)] - self.bin_prec[:,12*year:12*(year+1)])
                    # if temperature above maximum threshold, then all rain
                    (self.bin_prec[:,12*year:12*(year+1)]
                            [self.bin_temp[:,12*year:12*(year+1)] > self.modelprms['tsnow_threshold'] + 1]) = (
                        bin_precsnow[:,12*year:12*(year+1)]
                            [self.bin_temp[:,12*year:12*(year+1)] > self.modelprms['tsnow_threshold'] + 1])
                    (self.bin_acc[:,12*year:12*(year+1)]
                        [self.bin_temp[:,12*year:12*(year+1)] > self.modelprms['tsnow_threshold'] + 1]) = 0
                    # if temperature below minimum threshold, then all snow
                    (self.bin_acc[:,12*year:12*(year+1)]
                            [self.bin_temp[:,12*year:12*(year+1)] <= self.modelprms['tsnow_threshold'] - 1]) = (
                        bin_precsnow[:,12*year:12*(year+1)]
                            [self.bin_temp[:,12*year:12*(year+1)] <= self.modelprms['tsnow_threshold'] - 1])
                    (self.bin_prec[:,12*year:12*(year+1)]
                        [self.bin_temp[:,12*year:12*(year+1)] <= self.modelprms['tsnow_threshold'] - 1]) = 0

                # ENTER MONTHLY LOOP (monthly loop required since surface type changes)
                for month in range(0,12):
                    # Step is the position as a function of year and month, which improves readability
                    step = 12*year + month

                    # ACCUMULATION, MELT, REFREEZE, AND CLIMATIC MASS BALANCE
                    # Snowpack [m w.e.] = snow remaining + new snow
                    if step == 0:
                        self.bin_snowpack[:,step] = self.bin_acc[:,step]
                    else:
                        self.bin_snowpack[:,step] = self.snowpack_remaining[:,step-1] + self.bin_acc[:,step]

                    # MELT [m w.e.]
                    # energy available for melt [degC day]
                    if pygem_prms['mb']['option_ablation'] == 1:
                        # option 1: energy based on monthly temperature
                        melt_energy_available = self.bin_temp[:,step]*self.dayspermonth[step]
                        melt_energy_available[melt_energy_available < 0] = 0
                    elif pygem_prms['mb']['option_ablation'] == 2:
                        # Seed randomness for repeatability, but base it on step to ensure the daily variability is not
                        #  the same for every single time step
                        np.random.seed(step)
                        # option 2: monthly temperature superimposed with daily temperature variability
                        # daily temperature variation in each bin for the monthly timestep
                        bin_tempstd_daily = np.repeat(
                                np.random.normal(loc=0, scale=self.glacier_gcm_tempstd[step],
                                                 size=self.dayspermonth[step])
                                .reshape(1,self.dayspermonth[step]), heights.shape[0], axis=0)
                        # daily temperature in each bin for the monthly timestep
                        bin_temp_daily = self.bin_temp[:,step][:,np.newaxis] + bin_tempstd_daily
                        # remove negative values
                        bin_temp_daily[bin_temp_daily < 0] = 0
                        # Energy available for melt [degC day] = sum of daily energy available
                        melt_energy_available = bin_temp_daily.sum(axis=1)
                    # SNOW MELT [m w.e.]
                    self.bin_meltsnow[:,step] = self.surfacetype_ddf_dict[2] * melt_energy_available
                    # snow melt cannot exceed the snow depth
                    self.bin_meltsnow[self.bin_meltsnow[:,step] > self.bin_snowpack[:,step], step] = (
                            self.bin_snowpack[self.bin_meltsnow[:,step] > self.bin_snowpack[:,step], step])
                    # GLACIER MELT (ice and firn) [m w.e.]
                    # energy remaining after snow melt [degC day]
                    melt_energy_available = (
                            melt_energy_available - self.bin_meltsnow[:,step] / self.surfacetype_ddf_dict[2])
                    # remove low values of energy available caused by rounding errors in the step above
                    melt_energy_available[abs(melt_energy_available) < pygem_prms['constants']['tolerance']] = 0
                    # DDF based on surface type [m w.e. degC-1 day-1]
                    for surfacetype_idx in self.surfacetype_ddf_dict:
                        self.surfacetype_ddf[self.surfacetype == surfacetype_idx] = (
                                self.surfacetype_ddf_dict[surfacetype_idx])
                        # Debris enhancement factors in ablation area (debris in accumulation area would submerge)
                        if surfacetype_idx == 1 and pygem_prms['mb']['include_debris']:
                            self.surfacetype_ddf[self.surfacetype == 1] = (
                                    self.surfacetype_ddf[self.surfacetype == 1] * self.debris_ed[self.surfacetype == 1])
                    self.bin_meltglac[glac_idx_t0,step] = (
                            self.surfacetype_ddf[glac_idx_t0] * melt_energy_available[glac_idx_t0])
                    # TOTAL MELT (snow + glacier)
                    #  off-glacier need to include melt of refreeze because there are no glacier dynamics,
                    #  but on-glacier do not need to account for this (simply assume refreeze has same surface type)
                    self.bin_melt[:,step] = self.bin_meltglac[:,step] + self.bin_meltsnow[:,step]

                    # REFREEZING
                    if pygem_prms['mb']['option_refreezing'] == 'HH2015':
                        if step > 0:
                            self.tl_rf[:,:,step] = self.tl_rf[:,:,step-1]
                            self.te_rf[:,:,step] = self.te_rf[:,:,step-1]

                        # Refreeze based on heat conduction approach (Huss and Hock 2015)
                        # refreeze time step (s)
                        rf_dt = 3600 * 24 * self.dayspermonth[step] / pygem_prms['mb']['HH2015_rf_opts']['rf_dsc']

                        if pygem_prms['mb']['HH2015_rf_opts']['option_rf_limit_meltsnow'] == 1:
                            bin_meltlimit = self.bin_meltsnow.copy()
                        else:
                            bin_meltlimit = self.bin_melt.copy()

                        # Debug lowest bin
                        if self.debug_refreeze:
                            gidx_debug = np.where(heights == heights[glac_idx_t0].min())[0]

                        # Loop through each elevation bin of glacier
                        for nbin, gidx in enumerate(glac_idx_t0):
                            # COMPUTE HEAT CONDUCTION - BUILD COLD RESERVOIR
                            # If no melt, then build up cold reservoir (compute heat conduction)
                            if self.bin_melt[gidx,step] < pygem_prms['mb']['HH2015_rf_opts']['rf_meltcrit']:

                                if self.debug_refreeze and gidx == gidx_debug and step < 12:
                                    print('\nMonth ' + str(self.dates_table.loc[step,'month']),
                                          'Computing heat conduction')

                                # Set refreeze equal to 0
                                self.refr[gidx] = 0
                                # Loop through multiple iterations to converge on a solution
                                #  -> this will loop through 0, 1, 2
                                for h in np.arange(0, pygem_prms['mb']['HH2015_rf_opts']['rf_dsc']):
                                    # Compute heat conduction in layers (loop through rows)
                                    #  go from 1 to rf_layers-1 to avoid indexing errors with "j-1" and "j+1"
                                    #  "j+1" is set to zero, which is fine for temperate glaciers but inaccurate for
                                    #  cold/polythermal glaciers
                                    for j in np.arange(1, pygem_prms['mb']['HH2015_rf_opts']['rf_layers']-1):
                                        # Assume temperature of first layer equals air temperature
                                        #  assumption probably wrong, but might still work at annual average
                                        # Since next line uses tl_rf for all calculations, set tl_rf[0] to present mean
                                        #  monthly air temperature to ensure the present calculations are done with the
                                        #  present time step's air temperature
                                        self.tl_rf[0, gidx,step] = self.bin_temp[gidx,step]
                                        # Temperature for each layer
                                        self.te_rf[j,gidx,step] = (self.tl_rf[j,gidx,step] +
                                             rf_dt * self.rf_layers_k[j] / self.rf_layers_ch[j] / pygem_prms['mb']['HH2015_rf_opts']['rf_dz']**2 *
                                             0.5 * ((self.tl_rf[j-1,gidx,step] - self.tl_rf[j,gidx,step]) -
                                                     (self.tl_rf[j,gidx,step] - self.tl_rf[j+1,gidx,step])))
                                        # Update previous time step
                                        self.tl_rf[:,gidx,step] = self.te_rf[:,gidx,step]

                                if self.debug_refreeze and gidx == gidx_debug and step < 12:
                                    print('tl_rf:', ["{:.2f}".format(x) for x in self.tl_rf[:,gidx,step]])

                            # COMPUTE REFREEZING - TAP INTO "COLD RESERVOIR" or potential refreezing
                            else:

                                if self.debug_refreeze and gidx == gidx_debug and step < 12:
                                    print('\nMonth ' + str(self.dates_table.loc[step,'month']), 'Computing refreeze')

                                # Refreezing over firn surface
                                if (self.surfacetype[gidx] == 2) or (self.surfacetype[gidx] == 3):
                                    nlayers = pygem_prms['mb']['HH2015_rf_opts']['rf_layers']-1
                                # Refreezing over ice surface
                                else:
                                    # Approximate number of layers of snow on top of ice
                                    smax = np.round((self.bin_snowpack[gidx,step] / (self.rf_layers_dens[0] / 1000) +
                                                     pygem_prms['mb']['HH2015_rf_opts']['pp']) / pygem_prms['mb']['HH2015_rf_opts']['rf_dz'], 0)
                                    # if there is very little snow on the ground (SWE > 0.06 m for pp=0.3),
                                    #  then still set smax (layers) to 1
                                    if self.bin_snowpack[gidx,step] > 0 and smax == 0:
                                        smax=1
                                    # if no snow on the ground, then set to rf_cold to NoData value
                                    if smax == 0:
                                        self.rf_cold[gidx] = 0
                                    # if smax greater than the number of layers, set to max number of layers minus 1
                                    if smax > pygem_prms['mb']['HH2015_rf_opts']['rf_layers'] - 1:
                                        smax = pygem_prms['mb']['HH2015_rf_opts']['rf_layers'] - 1
                                    nlayers = int(smax)
                                # Compute potential refreeze, "cold reservoir", from temperature in each layer
                                # only calculate potential refreezing first time it starts melting each year
                                if self.rf_cold[gidx] == 0 and self.tl_rf[:,gidx,step].min() < 0:

                                    if self.debug_refreeze and gidx == gidx_debug and step < 12:
                                        print('calculating potential refreeze from ' + str(nlayers) + ' layers')

                                    for j in np.arange(0,nlayers):
                                        j += 1
                                        # units: (degC) * (J K-1 m-3) * (m) * (kg J-1) * (m3 kg-1)
                                        rf_cold_layer = (self.tl_rf[j,gidx,step] * self.rf_layers_ch[j] *
                                                         pygem_prms['mb']['HH2015_rf_opts']['rf_dz'] / pygem_prms['constants']['Lh_rf'] / pygem_prms['constants']['density_water'])
                                        self.rf_cold[gidx] -= rf_cold_layer

                                        if self.debug_refreeze and gidx == gidx_debug and step < 12:
                                            print('j:', j, 'tl_rf @ j:', np.round(self.tl_rf[j,gidx,step],2),
                                                           'ch @ j:', np.round(self.rf_layers_ch[j],2),
                                                           'rf_cold_layer @ j:', np.round(rf_cold_layer,2),
                                                           'rf_cold @ j:', np.round(self.rf_cold[gidx],2))

                                    if self.debug_refreeze and gidx == gidx_debug and step < 12:
                                        print('rf_cold:', np.round(self.rf_cold[gidx],2))

                                # Compute refreezing
                                # If melt and liquid prec < potential refreeze, then refreeze all melt and liquid prec
                                if (bin_meltlimit[gidx,step] + self.bin_prec[gidx,step]) < self.rf_cold[gidx]:
                                    self.refr[gidx] = bin_meltlimit[gidx,step] + self.bin_prec[gidx,step]
                                # otherwise, refreeze equals the potential refreeze
                                elif self.rf_cold[gidx] > 0:
                                    self.refr[gidx] = self.rf_cold[gidx]
                                else:
                                    self.refr[gidx] = 0

                                # Track the remaining potential refreeze
                                self.rf_cold[gidx] -= (bin_meltlimit[gidx,step] + self.bin_prec[gidx,step])
                                # if potential refreeze consumed, set to 0 and set temperature to 0 (temperate firn)
                                if self.rf_cold[gidx] < 0:
                                    self.rf_cold[gidx] = 0
                                    self.tl_rf[:,gidx,step] = 0

                            # Record refreeze
                            self.bin_refreeze[gidx,step] = self.refr[gidx]

                            if self.debug_refreeze and step < 12 and gidx == gidx_debug:
                                print('Month ' + str(self.dates_table.loc[step,'month']),
                                      'Rf_cold remaining:', np.round(self.rf_cold[gidx],2),
                                      'Snow depth:', np.round(self.bin_snowpack[glac_idx_t0[nbin],step],2),
                                      'Snow melt:', np.round(self.bin_meltsnow[glac_idx_t0[nbin],step],2),
                                      'Rain:', np.round(self.bin_prec[glac_idx_t0[nbin],step],2),
                                      'Rfrz:', np.round(self.bin_refreeze[gidx,step],2))

                    elif pygem_prms['mb']['option_refreezing'] == 'Woodward':
                        # Refreeze based on annual air temperature (Woodward etal. 1997)
                        #  R(m) = (-0.69 * Tair + 0.0096) * 1 m / 100 cm
                        # calculate annually and place potential refreeze in user defined month
                        if step%12 == 0:
                            bin_temp_annual = annualweightedmean_array(self.bin_temp[:,12*year:12*(year+1)],
                                                                       self.dates_table.iloc[12*year:12*(year+1),:])
                            bin_refreezepotential_annual = (-0.69 * bin_temp_annual + 0.0096) / 100
                            # Remove negative refreezing values
                            bin_refreezepotential_annual[bin_refreezepotential_annual < 0] = 0
                            self.bin_refreezepotential[:,step] = bin_refreezepotential_annual
                            # Reset refreeze potential every year
                            if self.bin_refreezepotential[:,step].max() > 0:
                                refreeze_potential = self.bin_refreezepotential[:,step]

                        if self.debug_refreeze:
                            print('Year ' + str(year) + ' Month ' + str(self.dates_table.loc[step,'month']),
                                  'Refreeze potential:', np.round(refreeze_potential[glac_idx_t0[0]],3),
                                  'Snow depth:', np.round(self.bin_snowpack[glac_idx_t0[0],step],2),
                                  'Snow melt:', np.round(self.bin_meltsnow[glac_idx_t0[0],step],2),
                                  'Rain:', np.round(self.bin_prec[glac_idx_t0[0],step],2))

                        # Refreeze [m w.e.]
                        #  refreeze cannot exceed rain and melt (snow & glacier melt)
                        self.bin_refreeze[:,step] = self.bin_meltsnow[:,step] + self.bin_prec[:,step]
                        # refreeze cannot exceed snow depth
                        self.bin_refreeze[self.bin_refreeze[:,step] > self.bin_snowpack[:,step], step] = (
                                self.bin_snowpack[self.bin_refreeze[:,step] > self.bin_snowpack[:,step], step])
                        # refreeze cannot exceed refreeze potential
                        self.bin_refreeze[self.bin_refreeze[:,step] > refreeze_potential, step] = (
                                refreeze_potential[self.bin_refreeze[:,step] > refreeze_potential])
                        self.bin_refreeze[abs(self.bin_refreeze[:,step]) < pygem_prms['constants']['tolerance'], step] = 0
                        # update refreeze potential
                        refreeze_potential -= self.bin_refreeze[:,step]
                        refreeze_potential[abs(refreeze_potential) < pygem_prms['constants']['tolerance']] = 0

                    # SNOWPACK REMAINING [m w.e.]
                    self.snowpack_remaining[:,step] = self.bin_snowpack[:,step] - self.bin_meltsnow[:,step]
                    self.snowpack_remaining[abs(self.snowpack_remaining[:,step]) < pygem_prms['constants']['tolerance'], step] = 0

                    # Record values
                    self.glac_bin_melt[glac_idx_t0,step] = self.bin_melt[glac_idx_t0,step]
                    self.glac_bin_refreeze[glac_idx_t0,step] = self.bin_refreeze[glac_idx_t0,step]
                    self.glac_bin_snowpack[glac_idx_t0,step] = self.bin_snowpack[glac_idx_t0,step]
                    # CLIMATIC MASS BALANCE [m w.e.]
                    self.glac_bin_massbalclim[glac_idx_t0,step] = (
                            self.bin_acc[glac_idx_t0,step] + self.glac_bin_refreeze[glac_idx_t0,step] -
                            self.glac_bin_melt[glac_idx_t0,step])

                    # OFF-GLACIER ACCUMULATION, MELT, REFREEZE, AND SNOWPACK
                    if option_areaconstant == False:
                        # precipitation, refreeze, and snowpack are the same both on- and off-glacier
                        self.offglac_bin_prec[offglac_idx,step] = self.bin_prec[offglac_idx,step]
                        self.offglac_bin_refreeze[offglac_idx,step] = self.bin_refreeze[offglac_idx,step]
                        self.offglac_bin_snowpack[offglac_idx,step] = self.bin_snowpack[offglac_idx,step]
                        # Off-glacier melt includes both snow melt and melting of refreezing
                        #  (this is not an issue on-glacier because energy remaining melts underlying snow/ice)
                        # melt of refreezing (assumed to be snow)
                        self.offglac_meltrefreeze = self.surfacetype_ddf_dict[2] * melt_energy_available
                        # melt of refreezing cannot exceed refreezing
                        self.offglac_meltrefreeze[self.offglac_meltrefreeze > self.bin_refreeze[:,step]] = (
                                self.bin_refreeze[:,step][self.offglac_meltrefreeze > self.bin_refreeze[:,step]])
                        # off-glacier melt = snow melt + refreezing melt
                        self.offglac_bin_melt[offglac_idx,step] = (self.bin_meltsnow[offglac_idx,step] +
                                                                   self.offglac_meltrefreeze[offglac_idx])

                # ===== RETURN TO ANNUAL LOOP =====
                # SURFACE TYPE (-)
                # Annual climatic mass balance [m w.e.] used to determine the surface type
                self.glac_bin_massbalclim_annual[:,year] = self.glac_bin_massbalclim[:,12*year:12*(year+1)].sum(1)
                # Update surface type for each bin
                self.surfacetype, firnline_idx = self._surfacetypebinsannual(self.surfacetype,
                                                                             self.glac_bin_massbalclim_annual, year)
                # Record binned glacier area
                self.glac_bin_area_annual[:,year] = glacier_area_t0
                # Store glacier-wide results
                self._convert_glacwide_results(year, glacier_area_t0, heights, fls=fls, fl_id=fl_id, 
                                               option_areaconstant=option_areaconstant)

##                if debug:
#                debug_startyr = 57
#                debug_endyr = 61
#                if year > debug_startyr and year < debug_endyr:
#                    print('\n', year, 'glac_bin_massbalclim:', self.glac_bin_massbalclim[:,12*year:12*(year+1)].sum(1))
#                    print('ice thickness:', icethickness_t0)
#                    print('heights:', heights[glac_idx_t0])
##                    print('surface type present:', self.glac_bin_surfacetype_annual[12:20,year])
##                    print('surface type updated:', self.surfacetype[12:20])

        # Mass balance for each bin [m ice per second]
        seconds_in_year = self.dayspermonth[12*year:12*(year+1)].sum() * 24 * 3600
        mb = (self.glac_bin_massbalclim[:,12*year:12*(year+1)].sum(1)
              * pygem_prms['constants']['density_water'] / pygem_prms['constants']['density_ice'] / seconds_in_year)
        
        if self.inversion_filter:
            mb = np.minimum.accumulate(mb)

        # Fill in non-glaciated areas - needed for OGGM dynamics to remove small ice flux into next bin
        mb_filled = mb.copy()
        if len(glac_idx_t0) > 3:
            mb_max = np.max(mb[glac_idx_t0])
            mb_min = np.min(mb[glac_idx_t0])
            height_max = np.max(heights[glac_idx_t0])
            height_min = np.min(heights[glac_idx_t0])
            mb_grad = (mb_min - mb_max) / (height_max - height_min)
            mb_filled[(mb_filled==0) & (heights < height_max)] = (
                    mb_min + mb_grad * (height_min - heights[(mb_filled==0) & (heights < height_max)]))

        elif len(glac_idx_t0) >= 1 and len(glac_idx_t0) <= 3 and mb.max() <= 0:
            mb_min = np.min(mb[glac_idx_t0])
            height_max = np.max(heights[glac_idx_t0])
            mb_filled[(mb_filled==0) & (heights < height_max)] = mb_min
            
#            if year > debug_startyr and year < debug_endyr:
#                print('mb_min:', mb_min)
#                
#        if year > debug_startyr and year < debug_endyr:
#            import matplotlib.pyplot as plt
#            plt.plot(mb_filled, heights, '.')
#            plt.ylabel('Elevation')
#            plt.xlabel('Mass balance (mwea)')
#            plt.show()
#            
#            print('mb_filled:', mb_filled)
                
        return mb_filled


    #%%
    def _convert_glacwide_results(self, year, glacier_area, heights, 
                                  fls=None, fl_id=None, option_areaconstant=False, debug=False):
        """
        Convert raw runmassbalance function output to glacier-wide results for output package 2

        Parameters
        ----------
        year : int
            the year of the model run starting from zero
        glacier_area : np.array
            glacier area for each elevation bin (m2)
        heights : np.array
            surface elevation of each elevatio nin
        fls : object
            flowline object
        fl_id : int
            flowline id
        """
        # Glacier area
        glac_idx = glacier_area.nonzero()[0]
        glacier_area_monthly = glacier_area[:,np.newaxis].repeat(12,axis=1)
        
        # Check if need to adjust for complete removal of the glacier
        #  - needed for accurate runoff calcs and accurate mass balance components
        icethickness_t0 = getattr(fls[fl_id], 'thick', None)
        if icethickness_t0 is not None:
            # Mass loss cannot exceed glacier volume
            if glacier_area.sum() > 0:
                mb_max_loss = (-1 * (glacier_area * icethickness_t0).sum() / glacier_area.sum() *
                               pygem_prms['constants']['density_ice'] / pygem_prms['constants']['density_water'])
                # Check annual climatic mass balance (mwea)
                mb_mwea = ((glacier_area * self.glac_bin_massbalclim[:,12*year:12*(year+1)].sum(1)).sum() /
                            glacier_area.sum())    
            else:
                mb_max_loss = 0
                mb_mwea = 0

        if len(glac_idx) > 0:
            # Quality control for thickness
            if hasattr(fls[fl_id], 'thick'):
                thickness = fls[fl_id].thick
                glacier_area[thickness == 0] = 0
                section = fls[fl_id].section
                section[thickness == 0] = 0
                # Glacier-wide area (m2)
                self.glac_wide_area_annual[year] = glacier_area.sum()
                # Glacier-wide volume (m3)
                self.glac_wide_volume_annual[year] = (section * fls[fl_id].dx_meter).sum()
            else:
                # Glacier-wide area (m2)
                self.glac_wide_area_annual[year] = glacier_area.sum()
            # Glacier-wide temperature (degC)
            self.glac_wide_temp[12*year:12*(year+1)] = (
                    (self.bin_temp[:,12*year:12*(year+1)][glac_idx] * glacier_area_monthly[glac_idx]).sum(0) /
                    glacier_area.sum())
            # Glacier-wide precipitation (m3)
            self.glac_wide_prec[12*year:12*(year+1)] = (
                    (self.bin_prec[:,12*year:12*(year+1)][glac_idx] * glacier_area_monthly[glac_idx]).sum(0))
            # Glacier-wide accumulation (m3 w.e.)
            self.glac_wide_acc[12*year:12*(year+1)] = (
                    (self.bin_acc[:,12*year:12*(year+1)][glac_idx] * glacier_area_monthly[glac_idx]).sum(0))
            # Glacier-wide refreeze (m3 w.e.)
            self.glac_wide_refreeze[12*year:12*(year+1)] = (
                    (self.glac_bin_refreeze[:,12*year:12*(year+1)][glac_idx] * glacier_area_monthly[glac_idx]).sum(0))
            # Glacier-wide melt (m3 w.e.)
            self.glac_wide_melt[12*year:12*(year+1)] = (
                    (self.glac_bin_melt[:,12*year:12*(year+1)][glac_idx] * glacier_area_monthly[glac_idx]).sum(0))
            # Glacier-wide total mass balance (m3 w.e.)
            self.glac_wide_massbaltotal[12*year:12*(year+1)] = (
                    self.glac_wide_acc[12*year:12*(year+1)] + self.glac_wide_refreeze[12*year:12*(year+1)]
                    - self.glac_wide_melt[12*year:12*(year+1)] - self.glac_wide_frontalablation[12*year:12*(year+1)])

            # If mass loss more negative than glacier mass, reduce melt so glacier completely melts (no excess)
            if icethickness_t0 is not None and mb_mwea < mb_max_loss:
                melt_yr_raw = self.glac_wide_melt[12*year:12*(year+1)].sum()
                melt_yr_max = (self.glac_wide_volume_annual[year] 
                                * pygem_prms['constants']['density_ice'] / pygem_prms['constants']['density_water'] +
                               self.glac_wide_acc[12*year:12*(year+1)].sum() + 
                               self.glac_wide_refreeze[12*year:12*(year+1)].sum())
                melt_frac = melt_yr_max / melt_yr_raw
                # Update glacier-wide melt (m3 w.e.)
                self.glac_wide_melt[12*year:12*(year+1)] = self.glac_wide_melt[12*year:12*(year+1)] * melt_frac
                
            
            # Glacier-wide runoff (m3)
            self.glac_wide_runoff[12*year:12*(year+1)] = (
                        self.glac_wide_prec[12*year:12*(year+1)] + self.glac_wide_melt[12*year:12*(year+1)] -
                        self.glac_wide_refreeze[12*year:12*(year+1)])
            # Snow line altitude (m a.s.l.)
            heights_monthly = heights[:,np.newaxis].repeat(12, axis=1)
            snow_mask = np.zeros(heights_monthly.shape)
            snow_mask[self.glac_bin_snowpack[:,12*year:12*(year+1)] > 0] = 1
            heights_monthly_wsnow = heights_monthly * snow_mask
            heights_monthly_wsnow[heights_monthly_wsnow == 0] = np.nan
            heights_change = np.zeros(heights.shape)
            heights_change[0:-1] = heights[0:-1] - heights[1:]
            try:
                snowline_idx = np.nanargmin(heights_monthly_wsnow, axis=0)
                self.glac_wide_snowline[12*year:12*(year+1)] = heights[snowline_idx] - heights_change[snowline_idx] / 2
            except:
                snowline_idx = np.zeros((heights_monthly_wsnow.shape[1])).astype(int)
                snowline_idx_nan = []
                for ncol in range(heights_monthly_wsnow.shape[1]):
                    if ~np.isnan(heights_monthly_wsnow[:,ncol]).all():
                        snowline_idx[ncol] = np.nanargmin(heights_monthly_wsnow[:,ncol])
                    else:
                        snowline_idx_nan.append(ncol)
                heights_manual = heights[snowline_idx] - heights_change[snowline_idx] / 2
                heights_manual[snowline_idx_nan] = np.nan
                # this line below causes a potential All-NaN slice encountered issue at some time steps
                self.glac_wide_snowline[12*year:12*(year+1)] = heights_manual

            # Equilibrium line altitude (m a.s.l.)
            ela_mask = np.zeros(heights.shape)
            ela_mask[self.glac_bin_massbalclim_annual[:,year] > 0] = 1
            ela_onlypos = heights * ela_mask
            ela_onlypos[ela_onlypos == 0] = np.nan        
            if np.isnan(ela_onlypos).all():
                self.glac_wide_ELA_annual[year] = np.nan
            else:
                ela_idx = np.nanargmin(ela_onlypos)
                self.glac_wide_ELA_annual[year] = heights[ela_idx] - heights_change[ela_idx] / 2

        # ===== Off-glacier ====                
        offglac_idx = np.where(self.offglac_bin_area_annual[:,year] > 0)[0]
        if option_areaconstant == False and len(offglac_idx) > 0:
            offglacier_area_monthly = self.offglac_bin_area_annual[:,year][:,np.newaxis].repeat(12,axis=1)

            # Off-glacier precipitation (m3)
            self.offglac_wide_prec[12*year:12*(year+1)] = (
                    (self.bin_prec[:,12*year:12*(year+1)][offglac_idx] * offglacier_area_monthly[offglac_idx]).sum(0))
            # Off-glacier melt (m3 w.e.)
            self.offglac_wide_melt[12*year:12*(year+1)] = (
                    (self.offglac_bin_melt[:,12*year:12*(year+1)][offglac_idx] * offglacier_area_monthly[offglac_idx]
                    ).sum(0))
            # Off-glacier refreeze (m3 w.e.)
            self.offglac_wide_refreeze[12*year:12*(year+1)] = (
                    (self.offglac_bin_refreeze[:,12*year:12*(year+1)][offglac_idx] * offglacier_area_monthly[offglac_idx]
                    ).sum(0))
            # Off-glacier runoff (m3)
            self.offglac_wide_runoff[12*year:12*(year+1)] = (
                    self.offglac_wide_prec[12*year:12*(year+1)] + self.offglac_wide_melt[12*year:12*(year+1)] -
                    self.offglac_wide_refreeze[12*year:12*(year+1)])
            # Off-glacier snowpack (m3 w.e.)
            self.offglac_wide_snowpack[12*year:12*(year+1)] = (
                    (self.offglac_bin_snowpack[:,12*year:12*(year+1)][offglac_idx] * offglacier_area_monthly[offglac_idx]
                    ).sum(0))
                
                
    def ensure_mass_conservation(self, diag):
        """
        Ensure mass conservation that may result from using OGGM's glacier dynamics model. This will be resolved on an 
        annual basis, and since the glacier dynamics are updated annually, the melt and runoff will be adjusted on a
        monthly-scale based on percent changes.
        
        OGGM's dynamic model limits mass loss based on the ice thickness and flux divergence. As a result, the actual
        volume change, glacier runoff, glacier melt, etc. may be less than that recorded by the mb_model. For PyGEM
        this is important because the glacier runoff and all parameters should be mass conserving.
        
        Note: other dynamical models (e.g., mass redistribution curves, volume-length-area scaling) are based on the 
        total volume change and therefore do not impose limitations like this because they do not estimate the flux
        divergence. As a result, they may systematically overestimate mass loss compared to OGGM's dynamical model.
        """
        # Compute difference between volume change 
        vol_change_annual_mbmod = (self.glac_wide_massbaltotal.reshape(-1,12).sum(1) * 
                                   pygem_prms['constants']['density_water'] / pygem_prms['constants']['density_ice'])
        vol_change_annual_diag = np.zeros(vol_change_annual_mbmod.shape)
        vol_change_annual_diag[0:diag.volume_m3.values[1:].shape[0]] = diag.volume_m3.values[1:] - diag.volume_m3.values[:-1]
        vol_change_annual_dif = vol_change_annual_diag - vol_change_annual_mbmod

        # Reduce glacier melt by the difference
        vol_change_annual_mbmod_melt = (self.glac_wide_melt.reshape(-1,12).sum(1) * 
                                        pygem_prms['constants']['density_water'] / pygem_prms['constants']['density_ice'])
        vol_change_annual_melt_reduction = np.zeros(vol_change_annual_mbmod.shape)
        chg_idx = vol_change_annual_mbmod.nonzero()[0]
        chg_idx_posmbmod = vol_change_annual_mbmod_melt.nonzero()[0]
        chg_idx_melt = list(set(chg_idx).intersection(chg_idx_posmbmod))
        
        vol_change_annual_melt_reduction[chg_idx_melt] = (
                1 - vol_change_annual_dif[chg_idx_melt] / vol_change_annual_mbmod_melt[chg_idx_melt])      
        
        vol_change_annual_melt_reduction_monthly = np.repeat(vol_change_annual_melt_reduction, 12)
        
        # Glacier-wide melt (m3 w.e.)
        self.glac_wide_melt = self.glac_wide_melt * vol_change_annual_melt_reduction_monthly
        
        # Glacier-wide total mass balance (m3 w.e.)
        self.glac_wide_massbaltotal = (self.glac_wide_acc + self.glac_wide_refreeze - self.glac_wide_melt -
                                       self.glac_wide_frontalablation)
        
        # Glacier-wide runoff (m3)
        self.glac_wide_runoff = self.glac_wide_prec + self.glac_wide_melt - self.glac_wide_refreeze
        
        self.glac_wide_volume_change_ignored_annual = vol_change_annual_dif
        

    # ===== SURFACE TYPE FUNCTIONS =====
    def _surfacetypebinsinitial(self, elev_bins):
        """
        Define initial surface type according to median elevation such that the melt can be calculated over snow or ice.

        Convention: (0 = off-glacier, 1 = ice, 2 = snow, 3 = firn, 4 = debris).
        Function options: 1 =

        Function options specified in pygem_pygem_prms.py:
        - option_surfacetype_initial
            > 1 (default) - use median elevation to classify snow/firn above the median and ice below
            > 2 - use mean elevation instead
        - include_firn : Boolean

        To-do list
        ----------
        Add option_surfacetype_initial to specify an AAR ratio and apply this to estimate initial conditions

        Parameters
        ----------
        elev_bins : np.ndarray
            Elevation bins [masl]
        Returns
        -------
        surfacetype : np.ndarray
            Updated surface type for each elevation bin
        firnline_idx : int
            Firn line index
        """
        surfacetype = np.zeros(self.glacier_area_initial.shape)
        # Option 1 - initial surface type based on the median elevation
        if pygem_prms['mb']['option_surfacetype_initial'] == 1:
            surfacetype[(elev_bins < self.glacier_rgi_table.loc['Zmed']) & (self.glacier_area_initial > 0)] = 1
            surfacetype[(elev_bins >= self.glacier_rgi_table.loc['Zmed']) & (self.glacier_area_initial > 0)] = 2
        # Option 2 - initial surface type based on the mean elevation
        elif pygem_prms['mb']['option_surfacetype_initial'] ==2:
            surfacetype[(elev_bins < self.glacier_rgi_table['Zmean']) & (self.glacier_area_initial > 0)] = 1
            surfacetype[(elev_bins >= self.glacier_rgi_table['Zmean']) & (self.glacier_area_initial > 0)] = 2
        else:
            print("This option for 'option_surfacetype' does not exist. Please choose an option that exists. "
                  + "Exiting model run.\n")
            exit()
        # Compute firnline index
        try:
            # firn in bins >= firnline_idx
            firnline_idx = np.where(surfacetype==2)[0][0]
        except:
            # avoid errors if there is no firn, i.e., the entire glacier is melting
            firnline_idx = np.where(surfacetype!=0)[0][-1]
        # If firn is included, then specify initial firn conditions
        if pygem_prms['mb']['include_firn'] == 1:
            surfacetype[surfacetype == 2] = 3
            #  everything initially considered snow is considered firn, i.e., the model initially assumes there is no
            #  snow on the surface anywhere.
        return surfacetype, firnline_idx


    def _surfacetypebinsannual(self, surfacetype, glac_bin_massbalclim_annual, year_index):
        """
        Update surface type according to climatic mass balance over the last five years.

        If 5-year climatic balance is positive, then snow/firn.  If negative, then ice/debris.
        Convention: 0 = off-glacier, 1 = ice, 2 = snow, 3 = firn, 4 = debris

        Function Options:
          > 1 (default) - update surface type according to Huss and Hock (2015)
          > 2 - Radic and Hock (2011)
        Huss and Hock (2015): Initially, above median glacier elevation is firn, below is ice. Surface type updated for
          each elevation band and month depending on specific mass balance.  If the cumulative balance since the start
          of mass balance year is positive, then snow is assigned. If the cumulative mass balance is negative (i.e.,
          all snow of current mass balance year has melted), then bare ice or firn exposed. Surface type is assumed to
          be firn if the elevation band's average annual balance over the preceding 5 years (B_t-5_avg) is positive. If
          B_t-5_avg is negative, surface type is ice.
              > climatic mass balance calculated at each bin and used with the mass balance over the last 5 years to
                determine whether the surface is firn or ice.  Snow is separate based on each month.
        Radic and Hock (2011): "DDF_snow is used above the ELA regardless of snow cover.  Below the ELA, use DDF_ice is
          used only when snow cover is 0.  ELA is calculated from observed annual mass balance profiles averaged over
          the observational period and is kept constant in time for the calibration period.  For the future projections,
          ELA is set to the mean glacier height and is time dependent since glacier volume, area, and length are time
          dependent (volume-area-length scaling).
          Bliss et al. (2014) uses the same as Valentina's model

        Parameters
        ----------
        surfacetype : np.ndarray
            Surface type for each elevation bin
        glac_bin_massbalclim_annual : np.ndarray
            Annual climatic mass balance for each year and each elevation bin
        year_index : int
            Count of the year of model run (first year is 0)
        Returns
        -------
        surfacetype : np.ndarray
            Updated surface type for each elevation bin
        firnline_idx : int
            Firn line index
        """
        # Next year's surface type is based on the bin's average annual climatic mass balance over the last 5 years.  If
        #  less than 5 years, then use the average of the existing years.
        if year_index < 5:
            # Calculate average annual climatic mass balance since run began
            massbal_clim_mwe_runningavg = glac_bin_massbalclim_annual[:,0:year_index+1].mean(1)
        else:
            massbal_clim_mwe_runningavg = glac_bin_massbalclim_annual[:,year_index-4:year_index+1].mean(1)
        # If the average annual specific climatic mass balance is negative, then the surface type is ice (or debris)
        surfacetype[(surfacetype !=0 ) & (massbal_clim_mwe_runningavg <= 0)] = 1
        # If the average annual specific climatic mass balance is positive, then the surface type is snow (or firn)
        surfacetype[(surfacetype != 0) & (massbal_clim_mwe_runningavg > 0)] = 2
        # Compute the firnline index
        try:
            # firn in bins >= firnline_idx
            firnline_idx = np.where(surfacetype==2)[0][0]
        except:
            # avoid errors if there is no firn, i.e., the entire glacier is melting
            firnline_idx = np.where(surfacetype!=0)[0][-1]
        # Apply surface type model options
        # If firn surface type option is included, then snow is changed to firn
        if pygem_prms['mb']['include_firn'] == 1:
            surfacetype[surfacetype == 2] = 3
        return surfacetype, firnline_idx


    def _surfacetypeDDFdict(self, modelprms, include_firn=pygem_prms['mb']['include_firn'],
                            option_ddf_firn=pygem_prms['mb']['option_ddf_firn']):
        """
        Create a dictionary of surface type and its respective DDF.

        Convention: [0=off-glacier, 1=ice, 2=snow, 3=firn, 4=debris]

        To-do list
        ----------
        - Add option_surfacetype_initial to specify an AAR ratio and apply this to estimate initial conditions

        Parameters
        ----------
        modelprms : dictionary
            Model parameters may include kp (precipitation factor), precgrad (precipitation gradient), ddfsnow, ddfice,
            tsnow_threshold (temperature threshold for snow/rain), tbias (temperature bias)
        include_firn : Boolean
            Option to include or exclude firn (specified in config.yaml)
        option_ddf_firn : int
            Option for the degree day factor of firn to be the average of snow and ice or a different value

        Returns
        -------
        surfacetype_ddf_dict : dictionary
            Dictionary relating the surface types with their respective degree day factors
        """
        surfacetype_ddf_dict = {
                0: modelprms['ddfsnow'],
                1: modelprms['ddfice'],
                2: modelprms['ddfsnow']}
        if include_firn:
            if option_ddf_firn == 0:
                surfacetype_ddf_dict[3] = modelprms['ddfsnow']
            elif option_ddf_firn == 1:
                surfacetype_ddf_dict[3] = np.mean([modelprms['ddfsnow'],modelprms['ddfice']])
        return surfacetype_ddf_dict