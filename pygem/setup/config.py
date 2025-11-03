"""
Python Glacier Evolution Model (PyGEM)

copyright © 2018 David Rounce <drounce@cmu.edu>

Distributed under the MIT license
"""

import os
import shutil

import ruamel.yaml

__all__ = ['ConfigManager']


class ConfigManager:
    """Manages PyGEMs configuration file, ensuring it exists, reading, updating, and validating its contents."""

    def __init__(self, config_filename='config.yaml', base_dir=None, overwrite=False):
        """
        Initialize the ConfigManager class.

        Parameters:
        config_filename (str, optional): Name of the configuration file. Defaults to 'config.yaml'.
        base_dir (str, optional): Directory where the configuration file is stored. Defaults to '~/PyGEM'.
        overwrite (bool, optional): Whether to overwrite an existing configuration file. Defaults to False.
        """
        self.config_filename = config_filename
        self.base_dir = base_dir or os.path.join(os.path.expanduser('~'), 'PyGEM')
        self.config_path = os.path.join(self.base_dir, self.config_filename)
        self.source_config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.yaml')
        self.overwrite = overwrite
        self._ensure_config()

    def _ensure_config(self):
        """Ensure the configuration file exists, creating or overwriting it if necessary"""
        if not os.path.isfile(self.config_path) or self.overwrite:
            self._copy_source_config()

    def _copy_source_config(self):
        """Copy the default configuration file to the expected location"""

        os.makedirs(self.base_dir, exist_ok=True)
        shutil.copy(self.source_config_path, self.config_path)
        print(f'Copied default configuration to {self.config_path}')

    def read_config(self, validate=True):
        """Read the configuration file and return its contents as a dictionary while preserving formatting.
        Parameters:
        validate (bool): Whether to validate the configuration file contents. Defaults to True.
        """
        ryaml = ruamel.yaml.YAML()
        with open(self.config_path, 'r') as f:
            user_config = ryaml.load(f)

        if validate:
            self._validate_config(user_config)

        return user_config

    def _write_config(self, config):
        """Write the configuration dictionary to the file while preserving quotes.

        Parameters:
        config (dict): configuration dictionary object
        """
        ryaml = ruamel.yaml.YAML()
        ryaml.preserve_quotes = True
        with open(self.config_path, 'w') as file:
            ryaml.dump(config, file)

    def update_config(self, updates):
        """Update multiple keys in the YAML configuration file while preserving quotes and original types.

        Parameters:
        updates (dict): Key-Value pairs to be updated
        """
        config = self.read_config(validate=False)

        for key, value in updates.items():
            if key not in self.EXPECTED_TYPES:
                raise KeyError(f'Unrecognized configuration key: {key}')
            keys = key.split('.')
            d = config
            for sub_key in keys[:-1]:
                d = d[sub_key]

            d[keys[-1]] = value

        self._validate_config(config)
        self._write_config(config)

    def _validate_config(self, config):
        """Validate the configuration dictionary against expected types and required keys.

        Parameters:
        config (dict): The configuration dictionary to be validated.
        """
        for key, expected_type in self.EXPECTED_TYPES.items():
            keys = key.split('.')
            sub_data = config
            for sub_key in keys:
                if isinstance(sub_data, dict) and sub_key in sub_data:
                    sub_data = sub_data[sub_key]
                else:
                    raise KeyError(f'Missing required key in configuration: {key}')

            if not isinstance(sub_data, expected_type):
                raise TypeError(f"Invalid type for '{key}': expected {expected_type}, not {type(sub_data)}")

            # Check elements inside lists (if defined)
            if key in self.LIST_ELEMENT_TYPES and isinstance(sub_data, list):
                elem_type = self.LIST_ELEMENT_TYPES[key]
                if not all(isinstance(item, elem_type) for item in sub_data):
                    raise TypeError(
                        f"Invalid type for elements in '{key}': expected all elements to be {elem_type}, but got {sub_data}"
                    )

        # check that all defined paths exist, raise error for any critical ones

    # expected config types
    EXPECTED_TYPES = {
        'root': str,
        'user': dict,
        'user.name': (str, type(None)),
        'user.institution': (str, type(None)),
        'user.email': (str, type(None)),
        'setup': dict,
        'setup.rgi_region01': (list, type(None)),
        'setup.rgi_region02': (str, type(None)),
        'setup.glac_no_skip': (list, type(None)),
        'setup.glac_no': (list, type(None)),
        'setup.min_glac_area_km2': (int, float),
        'setup.include_landterm': bool,
        'setup.include_laketerm': bool,
        'setup.include_tidewater': bool,
        'setup.include_frontalablation': bool,
        'oggm': dict,
        'oggm.base_url': str,
        'oggm.logging_level': str,
        'oggm.border': int,
        'oggm.oggm_gdir_relpath': str,
        'oggm.overwrite_gdirs': bool,
        'oggm.has_internet': bool,
        'climate': dict,
        'climate.ref_climate_name': str,
        'climate.ref_startyear': int,
        'climate.ref_endyear': int,
        'climate.ref_wateryear': str,
        'climate.sim_climate_name': str,
        'climate.sim_climate_scenario': (str, type(None)),
        'climate.sim_startyear': int,
        'climate.sim_endyear': int,
        'climate.sim_wateryear': str,
        'climate.constantarea_years': int,
        'climate.paths': dict,
        'climate.paths.era5_relpath': (str, type(None)),
        'climate.paths.era5_temp_fn': str,
        'climate.paths.era5_tempstd_fn': str,
        'climate.paths.era5_prec_fn': str,
        'climate.paths.era5_elev_fn': str,
        'climate.paths.era5_pressureleveltemp_fn': str,
        'climate.paths.era5_lr_fn': str,
        'climate.paths.cmip5_relpath': str,
        'climate.paths.cmip5_fp_var_ending': str,
        'climate.paths.cmip5_fp_fx_ending': str,
        'climate.paths.cmip6_relpath': str,
        'climate.paths.cesm2_relpath': str,
        'climate.paths.cesm2_fp_var_ending': str,
        'climate.paths.cesm2_fp_fx_ending': str,
        'climate.paths.gfdl_relpath': str,
        'climate.paths.gfdl_fp_var_ending': str,
        'climate.paths.gfdl_fp_fx_ending': str,
        'calib': dict,
        'calib.option_calibration': str,
        'calib.priors_reg_fn': str,
        'calib.HH2015_params': dict,
        'calib.HH2015_params.tbias_init': (int, float),
        'calib.HH2015_params.tbias_step': (int, float),
        'calib.HH2015_params.kp_init': (int, float),
        'calib.HH2015_params.kp_bndlow': (int, float),
        'calib.HH2015_params.kp_bndhigh': (int, float),
        'calib.HH2015_params.ddfsnow_init': (int, float),
        'calib.HH2015_params.ddfsnow_bndlow': (int, float),
        'calib.HH2015_params.ddfsnow_bndhigh': (int, float),
        'calib.HH2015mod_params': dict,
        'calib.HH2015mod_params.tbias_init': (int, float),
        'calib.HH2015mod_params.tbias_step': (int, float),
        'calib.HH2015mod_params.kp_init': (int, float),
        'calib.HH2015mod_params.kp_bndlow': (int, float),
        'calib.HH2015mod_params.kp_bndhigh': (int, float),
        'calib.HH2015mod_params.ddfsnow_init': (int, float),
        'calib.HH2015mod_params.method_opt': str,
        'calib.HH2015mod_params.params2opt': list,
        'calib.HH2015mod_params.ftol_opt': float,
        'calib.HH2015mod_params.eps_opt': float,
        'calib.emulator_params': dict,
        'calib.emulator_params.emulator_sims': int,
        'calib.emulator_params.overwrite_em_sims': bool,
        'calib.emulator_params.opt_hh2015_mod': bool,
        'calib.emulator_params.tbias_step': (int, float),
        'calib.emulator_params.tbias_init': (int, float),
        'calib.emulator_params.kp_init': (int, float),
        'calib.emulator_params.kp_bndlow': (int, float),
        'calib.emulator_params.kp_bndhigh': (int, float),
        'calib.emulator_params.ddfsnow_init': (int, float),
        'calib.emulator_params.option_areaconstant': bool,
        'calib.emulator_params.tbias_disttype': str,
        'calib.emulator_params.tbias_sigma': (int, float),
        'calib.emulator_params.kp_gamma_alpha': (int, float),
        'calib.emulator_params.kp_gamma_beta': (int, float),
        'calib.emulator_params.ddfsnow_disttype': str,
        'calib.emulator_params.ddfsnow_mu': (int, float),
        'calib.emulator_params.ddfsnow_sigma': (int, float),
        'calib.emulator_params.ddfsnow_bndlow': (int, float),
        'calib.emulator_params.ddfsnow_bndhigh': (int, float),
        'calib.emulator_params.method_opt': str,
        'calib.emulator_params.params2opt': list,
        'calib.emulator_params.ftol_opt': float,
        'calib.emulator_params.eps_opt': float,
        'calib.MCMC_params': dict,
        'calib.MCMC_params.option_use_emulator': bool,
        'calib.MCMC_params.emulator_sims': int,
        'calib.MCMC_params.tbias_step': (int, float),
        'calib.MCMC_params.tbias_stepsmall': (int, float),
        'calib.MCMC_params.option_areaconstant': bool,
        'calib.MCMC_params.mcmc_step': (int, float),
        'calib.MCMC_params.n_chains': int,
        'calib.MCMC_params.mcmc_sample_no': int,
        'calib.MCMC_params.mcmc_burn_pct': int,
        'calib.MCMC_params.thin_interval': int,
        'calib.MCMC_params.ddfsnow_disttype': str,
        'calib.MCMC_params.ddfsnow_mu': (int, float),
        'calib.MCMC_params.ddfsnow_sigma': (int, float),
        'calib.MCMC_params.ddfsnow_bndlow': (int, float),
        'calib.MCMC_params.ddfsnow_bndhigh': (int, float),
        'calib.MCMC_params.kp_disttype': str,
        'calib.MCMC_params.tbias_disttype': str,
        'calib.MCMC_params.tbias_mu': (int, float),
        'calib.MCMC_params.tbias_sigma': (int, float),
        'calib.MCMC_params.tbias_bndlow': (int, float),
        'calib.MCMC_params.tbias_bndhigh': (int, float),
        'calib.MCMC_params.kp_gamma_alpha': (int, float),
        'calib.MCMC_params.kp_gamma_beta': (int, float),
        'calib.MCMC_params.kp_lognorm_mu': (int, float),
        'calib.MCMC_params.kp_lognorm_tau': (int, float),
        'calib.MCMC_params.kp_mu': (int, float),
        'calib.MCMC_params.kp_sigma': (int, float),
        'calib.MCMC_params.kp_bndlow': (int, float),
        'calib.MCMC_params.kp_bndhigh': (int, float),
        'calib.MCMC_params.option_calib_elev_change_1d': bool,
        'calib.MCMC_params.rhoabl_disttype': str,
        'calib.MCMC_params.rhoabl_mu': (int, float),
        'calib.MCMC_params.rhoabl_sigma': (int, float),
        'calib.MCMC_params.rhoaccum_disttype': str,
        'calib.MCMC_params.rhoaccum_mu': (int, float),
        'calib.MCMC_params.rhoaccum_sigma': (int, float),
        'calib.MCMC_params.option_calib_meltextent_1d': bool,
        'calib.MCMC_params.option_calib_snowline_1d': bool,
        'calib.data': dict,
        'calib.data.massbalance': dict,
        'calib.data.massbalance.hugonnet2021_relpath': str,
        'calib.data.massbalance.hugonnet2021_fn': str,
        'calib.data.massbalance.hugonnet2021_facorrected_fn': str,
        'calib.data.frontalablation': dict,
        'calib.data.frontalablation.frontalablation_relpath': str,
        'calib.data.frontalablation.frontalablation_cal_fn': str,
        'calib.data.icethickness': dict,
        'calib.data.icethickness.h_ref_relpath': str,
        'calib.data.elev_change_1d': dict,
        'calib.data.elev_change_1d.elev_change_1d_relpath': (str, type(None)),
        'calib.data.meltextent_1d': dict,
        'calib.data.meltextent_1d.meltextent_1d_relpath': (str, type(None)),
        'calib.data.snowline_1d': dict,
        'calib.data.snowline_1d.snowline_1d_relpath': (str, type(None)),
        'calib.icethickness_cal_frac_byarea': float,
        'sim': dict,
        'sim.option_dynamics': (str, type(None)),
        'sim.option_bias_adjustment': int,
        'sim.nsims': int,
        'sim.out': dict,
        'sim.out.sim_stats': list,
        'sim.out.export_all_simiters': bool,
        'sim.out.export_extra_vars': bool,
        'sim.out.export_binned_data': bool,
        'sim.out.export_binned_components': bool,
        'sim.out.export_binned_area_threshold': (int, float),
        'sim.oggm_dynamics': dict,
        'sim.oggm_dynamics.cfl_number': float,
        'sim.oggm_dynamics.cfl_number_calving': float,
        'sim.oggm_dynamics.glen_a_regional_relpath': str,
        'sim.oggm_dynamics.use_regional_glen_a': bool,
        'sim.oggm_dynamics.fs': (int, float),
        'sim.oggm_dynamics.glen_a_multiplier': (int, float),
        'sim.icethickness_advancethreshold': (int, float),
        'sim.terminus_percentage': (int, float),
        'sim.params': dict,
        'sim.params.use_constant_lapserate': bool,
        'sim.params.kp': (int, float),
        'sim.params.tbias': (int, float),
        'sim.params.ddfsnow': (int, float),
        'sim.params.ddfsnow_iceratio': (int, float),
        'sim.params.precgrad': (int, float),
        'sim.params.lapserate': (int, float),
        'sim.params.tsnow_threshold': (int, float),
        'sim.params.calving_k': (int, float),
        'mb': dict,
        'mb.option_surfacetype_initial': int,
        'mb.include_firn': bool,
        'mb.include_debris': bool,
        'mb.debris_relpath': str,
        'mb.option_elev_ref_downscale': str,
        'mb.option_adjusttemp_surfelev': int,
        'mb.option_preclimit': int,
        'mb.option_accumulation': int,
        'mb.option_ablation': int,
        'mb.option_ddf_firn': int,
        'mb.option_refreezing': str,
        'mb.Woodard_rf_opts': dict,
        'mb.Woodard_rf_opts.rf_month': int,
        'mb.HH2015_rf_opts': dict,
        'mb.HH2015_rf_opts.rf_layers': int,
        'mb.HH2015_rf_opts.rf_dz': (int, float),
        'mb.HH2015_rf_opts.rf_dsc': (int, float),
        'mb.HH2015_rf_opts.rf_meltcrit': (int, float),
        'mb.HH2015_rf_opts.pp': (int, float),
        'mb.HH2015_rf_opts.rf_dens_top': (int, float),
        'mb.HH2015_rf_opts.rf_dens_bot': (int, float),
        'mb.HH2015_rf_opts.option_rf_limit_meltsnow': int,
        'rgi': dict,
        'rgi.rgi_relpath': str,
        'rgi.rgi_lat_colname': str,
        'rgi.rgi_lon_colname': str,
        'rgi.elev_colname': str,
        'rgi.indexname': str,
        'rgi.rgi_O1Id_colname': str,
        'rgi.rgi_glacno_float_colname': str,
        'rgi.rgi_cols_drop': list,
        'time': dict,
        'time.option_leapyear': int,
        'time.startmonthday': str,
        'time.endmonthday': str,
        'time.wateryear_month_start': int,
        'time.winter_month_start': int,
        'time.summer_month_start': int,
        'time.timestep': str,
        'constants': dict,
        'constants.density_ice': (int, float),
        'constants.density_water': (int, float),
        'constants.k_ice': (int, float),
        'constants.k_air': (int, float),
        'constants.ch_ice': (int, float),
        'constants.ch_air': (int, float),
        'constants.Lh_rf': (int, float),
        'constants.tolerance': float,
        'debug': dict,
        'debug.refreeze': bool,
        'debug.mb': bool,
    }

    # expected types of elements in lists
    LIST_ELEMENT_TYPES = {
        'setup.rgi_region01': int,
        'setup.glac_no_skip': float,
        'setup.glac_no': float,
        'calib.HH2015mod_params.params2opt': str,
        'calib.emulator_params.params2opt': str,
        'sim.out.sim_stats': str,
        'rgi.rgi_cols_drop': str,
    }
