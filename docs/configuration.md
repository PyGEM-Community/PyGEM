(pygem_config_overview_target)=
# Configuration File
PyGEM's configuration file (*~/PyGEM/config.yaml*) is where the user is able to specify the glaciers to model, choose model parameterizations and calibration options, specify relative filepaths and filenames for the model to function, and specify other model details and constants. The configuration settings are loaded to PyGEM as a dictionary object.

```{warning}
The only configuration setting a user must modify before running PyGEM is the **root** data path. Most other common parameter settings can be passed to PyGEM's operational scripts as command-line arguments.
```
The configuration is loosely organized to have the most frequently changed items at the top of the file, while also separating the file into organized chunks/keys. The general organization (by primary dictionary key) is:
* [root](input_rootpath_target): root filepath - all other input and ouput filepaths are relative to this
* [user](input_user_info_target): user information
* [setup](input_glacier_selection_target): glacier selection
* [oggm](input_oggm_target): OGGM settings
* [climate](input_climate_data_time_target): climate data and time periods
* [calib](input_cal_options_target): calibration options
* [sim](input_sim_dyn_options_target): simulation options
* [mb](input_mb_model_options_target): mass balance model options
* [rgi](input_glacier_data_target): RGI glacier data
* [time](input_model_time_details_target): model time period details
* [constants](input_model_constants_target): model constants
* [debug](input_debugging_options): debugging options

```{note}
*config.yaml* is heavily commented, so this information should hopefully be clear when modifying variables within the file itself.
```

#### General Formatting Rules

- **Boolean values:** Either `true` or `false`
- **Exponential values:** Should be formatted as `##.#e+#` (e.g., `1.2e+3` for 1200)
- **Infinity values:** Represented as `.inf` or `-.inf`
- **Null values:** Represented as `null` (imported as `None` in Python)
- **Lists:** Indicated with a `-` before each item
---

(input_rootpath_target)=
## Root Path

| Variable | Type | Comment/Note |
| :--- | :--- | :--- |
| `root` | `string` | Base path for input and output data. Must be modified to the appropriate location. |
---

(input_user_info_target)=
## User Information

| Variable | Type | Comment/Note |
| :--- | :--- | :--- |
| `user.name` | `string` | User's name |
| `user.institution` | `string` | Institution name |
| `user.email` | `string` | Contact email |

```{note}
The user information is strictly for bookkeeping purposes. This information is stored within each model output file.
```
---

(input_glacier_selection_target)=
## Glacier Selection

| Variable | Type | Comment/Note |
| :--- | :--- | :--- |
| `setup.rgi_region01` | `list` of `integers` | List of RGI order 01 region numbers to include |
| `setup.rgi_region02` | `list` of `integers` or `'all'` | List of RGI  order 02 subregion numbers to include, or `'all'` to include all subregions |
| `setup.glac_no_skip` | `null` or `list` of `strings` | Glacier numbers to skip |
| `setup.glac_no` | `null` or `list` of `strings` | List of RGI glacier numbers (e.g., `1.00570`) |
| `setup.min_glac_area_km2` | `float` | Minimum glacier area (km$^{2}$) threshold |
| `setup.include_landterm` | `boolean` | Include land-terminating glaciers |
| `setup.include_laketerm` | `boolean` | Include lake-terminating glaciers |
| `setup.include_tidewater` | `boolean` | Include marine-terminating glaciers |
| `setup.include_frontalablation` | `boolean` | Ignore calving and treat tidewater glaciers as land-terminating |
```{warning}
Set glac_no will always overwrite the rgi regions, so if you want to use the rgi region options, then set glac_no=None
```
---

(input_oggm_target)=
## OGGM Settings

| Variable | Type | Comment/Note |
| :--- | :--- | :--- |
| `oggm.base_url` | `string` | URL for OGGM glacier directories |
| `oggm.logging_level` | `string` | Log level (`'DEBUG'`, `'INFO'`, `'WARNING'`, `'ERROR'`, `'WORKFLOW'`, `'CRITICAL'`) |
| `oggm.border` | `integer` | Border size (options: `10`, `80`, `160`, `240`) |
| `oggm.oggm_gdir_relpath` | `string` | Relative path to OGGM glacier directories |
| `oggm.overwrite_gdirs` | `boolean` | Overwrite glacier directories if they exist |
| `oggm.has_internet` | `boolean` | Model has internet access to download data from OGGM's server |
---

(input_climate_data_time_target)=
## Climate Data and Time Periods

| Variable | Type | Comment/Note |
| :--- | :--- | :--- |
| `climate.ref_climate_name` | `string` | Historical reference climate dataset (e.g.`'ERA5'`) |
| `climate.ref_startyear` | `integer` | Reference period start year (inclusive) |
| `climate.ref_endyear` | `integer` | Reference period end year (inclusive) |
| `climate.ref_wateryear` | `string` | Reference period year type (`'calendar'`, `'hydro'`, `'custom'`) |
| `climate.sim_climate_name` | `string` | Climate dataset used for simulations (can be reference climate or GCM, e.g. `ERA5` or `CESM2`) |
| `climate.sim_climate_scenario` | `null` or `string` | Climate emission scenario (e.g. `'ssp245', 'rcp65'`) |
| `climate.sim_startyear` | `integer` | Start year for model simulation (inclusive, e.g. `2000`) |
| `climate.sim_endyear` | `integer` | End year for model simulation (inclusinve, e.g. `2100`) |
| `climate.sim_wateryear` | `string` | Year type (`'calendar'`, `'hydro'`, `'custom'`) |
| `climate.constantarea_years` | `integer` | Number of years from `climate.ref_startyear` to keep glacier area constant |
| `climate.paths` | `string` | Relative filepaths, filenames and variable names for climate datasets (relative to `root`) |
```{note}
`ref` (short for 'reference') refers to the calibration period (will typically vary between 1980-present). `sim` (short for 'simulation') refers to simulation period.
```
---

(input_cal_options_target)=
## Calibration Options

| Variable | Type | Comment/Note |
| :--- | :--- | :--- |
| `calib.option_calibration` | `string` | Calibration option (`'emulator'`, `'MCMC'`, `'HH2015'`, `'HH2015mod'`, `'null'`) |
| `calib.priors_reg_fn` | `string` | Prior distribution of regional model parameters (specify filename, relative to `root/Output/calibration/`, or set to `'null'`) |

### HH2015 Parameters

| Variable | Type | Comment/Note |
| :--- | :--- | :--- |
| `calib.HH2015_params.tbias_init` | `float` | Initial temperature bias  [K] |
| `calib.HH2015_params.tbias_step` | `float` | Temperature bias step size [K] |
| `calib.HH2015_params.kp_init` | `float` | Initial precipitation factor [-] |
| `calib.HH2015_params.kp_bndlow` | `float` | Lower bound for precipitation factor [-] |
| `calib.HH2015_params.kp_bndhigh` | `float` | Upper bound for precipitation factor [-] |
| `calib.HH2015_params.ddfsnow_init` | `float` | Initial degree-day factor for snow [mm w.e. d$^{-1}$ K$^{-1}$] |
| `calib.HH2015_params.ddfsnow_bndlow` | `float` | Lower bound for degree-day factor for snow [mm w.e. d$^{-1}$ K$^{-1}$] |
| `calib.HH2015_params.ddfsnow_bndhigh` | `float` | Upper bound for degree-day factor for snow [mm w.e. d$^{-1}$ K$^{-1}$] |

### HH2015mod Parameters

| Variable | Type | Comment/Note |
| :--- | :--- | :--- |
| `calib.HH2015mod_params.tbias_init` | `float` | Initial temperature bias [K] |
| `calib.HH2015mod_params.tbias_step` | `float` | Temperature bias step size [K] |
| `calib.HH2015mod_params.kp_init` | `float` | Initial precipitation factor [-] |
| `calib.HH2015mod_params.kp_bndlow` | `float` | Lower bound for precipitation factor [-] |
| `calib.HH2015mod_params.kp_bndhigh` | `float` | Upper bound for precipitation factor [-] |
| `calib.HH2015mod_params.ddfsnow_init`| `float` | Initial degree-day factor for snow [mm w.e. d$^{-1}$ K$^{-1}$] |
| `calib.HH2015mod_params.method_opt` | `string` | SciPy optimization scheme ('SLSQP' or 'L-BFGS-B') |
| `calib.HH2015mod_params.params2opt` | `list` of `strings` | Parameters to optimize (e.g., 'tbias', 'kp') |
| `calib.HH2015mod_params.ftol_opt` | `float` | Tolerance for SciPy optimization scheme |
| `calib.HH2015mod_params.eps_opt` | `float` | Epsilon for SciPy optimization scheme (e.g., 1e-6) |

### Emulator Parameters

| Variable | Type | Comment/Note |
| :--- | :--- | :--- |
| `calib.emulator_params.emulator_sims` | `integer` | Number of simulations to develop the emulator |
| `calib.emulator_params.overwrite_em_sims` | `boolean` | Whether to overwrite emulator simulations |
| `calib.emulator_params.opt_hh2015_mod` | `boolean` | Option to also perform HH2015_mod calibration using emulator |
| `calib.emulator_params.tbias_step` | `float` | Temperature bias step size [K] |
| `calib.emulator_params.tbias_init` | `float` | Initial temperature bias [K] |
| `calib.emulator_params.kp_init` | `float` | Initial precipitation factor [-] |
| `calib.emulator_params.kp_bndlow` | `float` | Lower bound for precipitation factor [-] |
| `calib.emulator_params.kp_bndhigh` | `float` | Upper bound for precipitation factor [-] |
| `calib.emulator_params.ddfsnow_init` | `float` | Initial degree-day factor for snow [mm w.e. d$^{-1}$ K$^{-1}$] |
| `calib.emulator_params.option_areaconstant` | `boolean` | Option to keep area constant or evolve |
| `calib.emulator_params.tbias_disttype` | `string` | Temperature bias distribution type ('truncnormal', 'uniform') |
| `calib.emulator_params.tbias_sigma` | `float` | Temperature bias standard deviation  [K] |
| `calib.emulator_params.kp_gamma_alpha` | `float` | Precipitation factor gamma distribution alpha [-] |
| `calib.emulator_params.kp_gamma_beta` | `float` | Precipitation factor gamma distribution beta [-] |
| `calib.emulator_params.ddfsnow_disttype` | `string` | Degree-day factor for snow distribution type ('truncnormal') |
| `calib.emulator_params.ddfsnow_mu` | `float` | Degree-day factor for snow mean [mm w.e. d$^{-1}$ K$^{-1}$] |
| `calib.emulator_params.ddfsnow_sigma` | `float` | Degree-day factor for snow standard deviation [mm w.e. d$^{-1}$ K$^{-1}$] |
| `calib.emulator_params.ddfsnow_bndlow` | `float` | Lower bound for degree-day factor for snow [mm w.e. d$^{-1}$ K$^{-1}$] |
| `calib.emulator_params.ddfsnow_bndhigh` | `float` | Upper bound for degree-day factor for snow [mm w.e. d$^{-1}$ K$^{-1}$] |
| `calib.emulator_params.method_opt` | `string` | SciPy optimization scheme ('SLSQP' or 'L-BFGS-B') |
| `calib.emulator_params.params2opt` | `list` of `strings` | Parameters to optimize (e.g., 'tbias', 'kp') |
| `calib.emulator_params.ftol_opt` | `float` | Tolerance for SciPy optimization scheme |
| `calib.emulator_params.eps_opt` | `float` | Epsilon for SciPy optimization scheme |

### MCMC Parameters

| Variable | Type | Comment/Note |
| :--- | :--- | :--- |
| `calib.MCMC_params.option_use_emulator` | `boolean` | Switch to emulator instead of full mass balance model |
| `calib.MCMC_params.emulator_sims` | `integer` | Number of emulator simulations |
| `calib.MCMC_params.tbias_step` | `float` | Temperature bias step size [K] |
| `calib.MCMC_params.tbias_stepsmall` | `float` | Small temperature bias step size [K] |
| `calib.MCMC_params.option_areaconstant` | `boolean` | Option to keep area constant or evolve |
| `calib.MCMC_params.mcmc_step` | `float` | MCMC step size (in terms of standard deviation of each parameters prior distribution) |
| `calib.MCMC_params.n_chains` | `integer` | Number of MCMC chains (min 1, max 3) |
| `calib.MCMC_params.mcmc_sample_no` | `integer` | Number of MCMC steps |
| `calib.MCMC_params.mcmc_burn_pct` | `float` | Percentage of steps to burn-in (0 records all steps) |
| `calib.MCMC_params.thin_interval` | `integer` | Thin interval for reducing file size |
| `calib.MCMC_params.ddfsnow_disttype` | `string` | Degree-day factor for snow distribution type ('truncnormal', 'uniform') |
| `calib.MCMC_params.ddfsnow_mu` | `float` | Degree-day factor for snow mean [mm w.e. d$^{-1}$ K$^{-1}$] |
| `calib.MCMC_params.ddfsnow_sigma` | `float` | Degree-day factor for snow standard deviation [mm w.e. d$^{-1}$ K$^{-1}$] |
| `calib.MCMC_params.ddfsnow_bndlow` | `float` | Lower bound for degree-day factor for snow [mm w.e. d$^{-1}$ K$^{-1}$] |
| `calib.MCMC_params.ddfsnow_bndhigh` | `float` | Upper bound for degree-day factor for snow [mm w.e. d$^{-1}$ K$^{-1}$] |
| `calib.MCMC_params.kp_disttype` | `string` | Precipitation factor distribution type ('gamma', 'lognormal', 'uniform') |
| `calib.MCMC_params.tbias_disttype` | `string` | Temperature bias distribution type ('normal', 'truncnormal', 'uniform') |
| `calib.MCMC_params.tbias_mu` | `float` | Temperature bias mean [K] |
| `calib.MCMC_params.tbias_sigma` | `float` | Temperature bias standard deviation [K] |
| `calib.MCMC_params.tbias_bndlow` | `float` | Lower bound for temperature bias [K] |
| `calib.MCMC_params.tbias_bndhigh` | `float` | Upper bound for temperature bias [K] |
| `calib.MCMC_params.kp_gamma_alpha` | `float` | Precipitation factor gamma distribution alpha [-] |
| `calib.MCMC_params.kp_gamma_beta` | `float` | Precipitation factor gamma distribution beta [-] |
| `calib.MCMC_params.kp_lognorm_mu` | `float` | Precipitation factor lognormal distribution mean [-] |
| `calib.MCMC_params.kp_lognorm_tau` | `float` | Precipitation factor lognormal distribution tau [-] |
| `calib.MCMC_params.kp_mu` | `float` | Precipitation factor normal distribution mean [-] |
| `calib.MCMC_params.kp_sigma` | `float` | Precipitation factor normal distribution standard deviation [-]|
| `calib.MCMC_params.kp_bndlow` | `float` | Lower bound for precipitation factor [-] |
| `calib.MCMC_params.kp_bndhigh` | `float` | Upper bound for precipitation factor [-] |

### Calibration Datasets

| Variable | Type | Comment/Note |
| :--- | :--- | :--- |
| `calib.data.massbalance.hugonnet2021_relpath` | `string` | Path to Hugonnet (2021) mass balance data (relative to `root`) |
| `calib.data.massbalance.hugonnet2021_fn` | `string` | Filename for Hugonnet (2021) mass balance data |
| `calib.data.massbalance.hugonnet2021_facorrected_fn` | `string` | Filename for corrected Hugonnet (2021) mass balance data |
| `calib.data.oib.oib_relpath` | `string` | Path to OIB lidar surface elevation data (relative to `root`) |
| `calib.data.oib.oib_rebin` | `integer` | Elevation rebinning interval [meters] |
| `calib.data.oib.oib_filter_pctl` | `float` | Pixel count percentile filter (filters out any data from each survey where the pixel count for a given bin is below the specified threshold) |
| `calib.data.frontalablation.frontalablation_relpath` | `string` | Path to frontal ablation data (relative to `root`) |
| `calib.data.frontalablation.frontalablation_cal_fn` | `string` | Filename for frontal ablation calibration data |
| `calib.data.icethickness.h_ref_relpath` | `string` | Path to reference ice thickness data (relative to `root`) |

### Ice Thickness Calibration

| Variable | Type | Comment/Note |
| :--- | :--- | :--- |
| `calib.icethickness_cal_frac_byarea` | `float` | Regional glacier area fraction used for ice thickness calibration (e.g., 0.9 means the largest 90% of glaciers by area) |
---

(input_sim_dyn_options_target)=
## Simulation Options

| Variable | Type | Comment/Note |
| :--- | :--- | :--- |
| `sim.option_dynamics` | `null` or `string` | Glacier dynamics scheme (`'OGGM'`, `'MassRedistributionCurves'`, `'null'`) |
| `sim.option_bias_adjustment` | `integer` | Bias adjustment option (`0`: no adjustment, `1`: precipitation factor and temp bias scaling (builds off HH2015 methods), `2`: HH2015 methods, `3`: quantile delta mapping) |
| `sim.nsims` | `integer` | Number of simulations to run. Default is 1. More than 1 simulation can be run for glaciers that have been calibrated using the `'MCMC'` option. (Running 1 simulation with `sim.option_dynamics == 'MCMC` will take the **median** parameter values) |

### Output Options

| Variable | Type | Comment/Note |
| :--- | :--- | :--- |
| `sim.out.sim_stats` | `list` of `strings` | Output statistics (`'mean'`, `'std'`, `'median'`, etc.) |
| `sim.out.export_all_simiters` | `boolean` | Export all simulation results (for `sim.nsims` > 1), otherwise calculate statistics specified by `sim.out.sim_stats` |
| `sim.out.export_extra_vars` | `boolean` | Export extra variables (temperature, precipitation, melt, etc.) |
| `sim.out.export_binned_data` | `boolean` | Export binned ice data |
| `sim.out.export_binned_components` | `boolean` | Export mass balance components |

### Dynamics
| `sim.oggm_dynamics.cfl_number` | `float` |  [seconds] |
| `sim.oggm_dynamics.cfl_number_calving` | `float` | Time step threshold for marine-terimating glaciers [seconds] |
| `sim.oggm_dynamics.use_reg_glena` | `boolean` | Use regionally averaged glen_a creep paremeter |
| `sim.oggm_dynamics.glena_reg_relpath` | `string` | Path to comma-separated value file containing regionally averaged `glen_a` and `fs` parameter values (relative to `root`) |
| `sim.oggm_dynamics.fs` | `float` | Oerlemans sliding parameter |
| `sim.oggm_dynamics.glen_a_multiplier` | `float` | glen_a multiplier if not using regionally calibrated glen_a |
| `sim.icethickness_advancethreshold` | `float` | advancing glacier ice thickness change threshold (if not running OGGM dynamics) |
| `sim.terminus_percentage` | `float` | glacier elevation range considered terminus, used to size advancing new bin (if not running OGGM dynamics) [%] |

### Model Parameters

| Variable | Type | Comment/Note |
| :--- | :--- | :--- |
| `sim.params.use_constant_lapserate` | `boolean` | Use constant or variable lapse rate |
| `sim.params.kp` | `float` | Precipitation factor [-] |
| `sim.params.tbias` | `float` | Temperature bias [K] |
| `sim.params.ddfsnow` | `float` | Snow degree-day factor [mm w.e. d$^{-1}$ K$^{-1}$] |
| `sim.params.ddfsnow_iceratio` | `float` | Ratio of snow to ice degree-day factor [-] |
---

(input_mb_model_options_target)=
## Mass Balance Model Options

| Variable | Type | Comment/Note |
| :--- | :--- | :--- |
| `mb.option_surfacetype_initial` | `integer` | 1: median elevation (default), 2: mean elevation |
| `mb.include_firn` | `boolean` | `true`: firn included, `false`: firn is modeled as snow |
| `mb.include_debris` | `boolean` | `true`: account for debris with melt factors, `false`: do not account for debris |
| `mb.debris_relpath` | `string` | Path to debris dataset (relative to `root`) |
| `mb.option_elev_ref_downscale` | `string` | `'Zmed'`, `'Zmax'`, or `'Zmin'` for median, max, or min glacier elevations |
| `mb.option_adjusttemp_surfelev` | `integer` | `1`: adjust temperatures based on surface elevation changes, `0`: no adjustment |
| `mb.option_preclimit` | `integer` | `1`: limit uppermost 25% using an exponential function (currently the only scheme in place - any value other than 1 will result in no precipitation limit to be applied) |
| `mb.option_accumulation` | `integer` | `1`: single threshold, `2`: threshold ± 1$^\circ$C using linear interpolation |
| `mb.option_ablation` | `integer` | `1`: monthly temperatures, `2`: superimpose daily temperatures to allow melt during months with negative mean temperatures but occasional positive daily values |
| `mb.option_ddf_firn` | `integer` | `0`: ddf_firn = ddf_snow, `1`: ddf_firn = mean of ddf_snow and ddf_ice |
| `mb.option_refreezing` | `string` | `'Woodward'`: annual air temp (Woodward et al. 1997), `'HH2015'`: heat conduction (Huss & Hock 2015) |
| `mb.Woodard_rf_opts.rf_month` | `integer` | Month number in which refreezing is reset to zero each year (`1`-`12`, dafult is 10) |
| `mb.HH2015_rf_opts.rf_layers` | `integer` | Number of refreezing layers (default is 8) |
| `mb.HH2015_rf_opts.rf_dz` | `integer` | Layer thickness [m] |
| `mb.HH2015_rf_opts.rf_dsc` | `integer` | Number of time steps for numerical stability |
| `mb.HH2015_rf_opts.rf_meltcrit` | `float` | Critical melt amount to initialize refreezing [m w.e.] |
| `mb.HH2015_rf_opts.pp` | `float` | Additional refreeze water fraction for bare ice |
| `mb.HH2015_rf_opts.rf_dens_top` | `integer` | Snow density at surface [kg m$^{-3}$] |
| `mb.HH2015_rf_opts.rf_dens_bot` | `integer` | Snow density at bottom refreezing layer [kg m$^{-3}$] |
| `mb.HH2015_rf_opts.option_rf_limit_meltsnow` | `integer` | Refreezing limit option |
---

(input_glacier_data_target)=
## RGI Glacier Data

| Variable | Type | Comment/Note |
| :--- | :--- | :--- |
| `rgi.rgi_relpath` | `string` | Filepath for RGI files (relative to `root`) |
| `rgi.rgi_lat_colname` | `string` | Name of latitude column |
| `rgi.rgi_lon_colname` | `string` | Name of longitude column (360-based) |
| `rgi.elev_colname` | `string` | Name of elevation column |
| `rgi.indexname` | `string` | Name of glacier index column |
| `rgi.rgi_O1Id_colname` | `string` | Name of glacier ID column |
| `rgi.rgi_glacno_float_colname` | `string` | Name of floating-point glacier ID column |
| `rgi.rgi_cols_drop` | `list` of `strings` | Columns to drop from RGI dataset |
---

(input_model_time_details_target)=
## Model Time Period Details

| Variable | Type | Comment/Note |
| :--- | :--- | :--- |
| `time.option_leapyear` | `integer` | `1`: include 29FEB in leap years, `0`: exclude 29FEB during leap years |
| `time.startmonthday` | `string` | Start date (MM-DD), used with custom calendars |
| `time.endmonthday` | `string` | End date (MM-DD), used with custom calendars |
| `time.wateryear_month_start` | `integer` | Water year start month |
| `time.winter_month_start` | `integer` | First month of winter |
| `time.summer_month_start` | `integer` | First month of summer |
| `time.timestep` | `string` | Time step (currently only `'monthly'` is supported) |
---

(input_model_constants_target)=
## Model Constants

| Variable | Type | Comment/Note |
| :--- | :--- | :--- |
| `constants.density_ice` | `integer` | Density of ice [900 kg m$^{-3}$] |
| `constants.density_water` | `integer` | Density of water [1000 kg m$^{-3}$] |
| `constants.k_ice` | `float` | Thermal conductivity of ice [2.330 J s$^{-1}$ K$^{-1}$ m$^{-1}$] |
| `constants.k_air` | `float` | Thermal conductivity of air [0.023 J s$^{-1}$ K$^{-1}$ m$^{-1}$] |
| `constants.ch_ice` | `integer` | Volumetric heat capacity of ice [1890000 J K$^{-1}$ m$^{-3}$] |
| `constants.ch_air` | `integer` | Volumetric heat capacity of air [1297 J K$^{-1}$ m$^{-3}$] |
| `constants.Lh_rf` | `integer` | Latent heat of fusion [333550 J kg$^{-1}$] |
| `constants.tolerance` | `float` | Model tolerance (used to remove rounding errors) |
---

(input_debugging_options)=
## Debugging Options

| Variable | Type | Comment/Note |
| :--- | :--- | :--- |
| `debug.refreeze` | `boolean` | Debug option for refreezing module |
| `debug.mb` | `boolean` | Debug option for mass balance calculations |