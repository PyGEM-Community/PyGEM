"""
Python Glacier Evolution Model (PyGEM)

copyright © 2018 David Rounce <drounce@cmu.edu>

Distrubted under the MIT lisence
"""
import os
import shutil
import ruamel.yaml

__all__ = ["ConfigManager"]

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
        self.source_config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")
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
        print(f"Copied default configuration to {self.config_path}")
        
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
                raise KeyError(f"Unrecognized configuration key: {key}")
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
            keys = key.split(".")
            sub_data = config
            for sub_key in keys:
                if isinstance(sub_data, dict) and sub_key in sub_data:
                    sub_data = sub_data[sub_key]
                else:
                    raise KeyError(f"Missing required key in configuration: {key}")

            if not isinstance(sub_data, expected_type):
                raise TypeError(f"Invalid type for '{key}': expected {expected_type}, not {type(sub_data)}")

            # Check elements inside lists (if defined)
            if key in self.LIST_ELEMENT_TYPES and isinstance(sub_data, list):
                elem_type = self.LIST_ELEMENT_TYPES[key]
                if not all(isinstance(item, elem_type) for item in sub_data):
                    raise TypeError(f"Invalid type for elements in '{key}': expected all elements to be {elem_type}, but got {sub_data}")
    

    # expected config types
    EXPECTED_TYPES = {
        "root": str,
        "user": dict,
        "user.name":  (str, type(None)),
        "user.institution":  (str, type(None)),
        "user.email":  (str, type(None)),
        "setup": dict,
        "setup.rgi_region01": list,
        "setup.rgi_region02": str,
        "setup.glac_no_skip": (list, type(None)),
        "setup.glac_no": (list, type(None)),
        "setup.min_glac_area_km2": int,
        "setup.include_landterm": bool,
        "setup.include_laketerm": bool,
        "setup.include_tidewater": bool,
        "setup.include_frontalablation": bool,
        "oggm": dict,
        "oggm.base_url": str,
        "oggm.logging_level": str,
        "oggm.border": int,
        "oggm.oggm_gdir_relpath": str,
        "oggm.overwrite_gdirs": bool,
        "oggm.has_internet": bool,
        "climate": dict,
        "climate.ref_gcm_name": str,
        "climate.ref_startyear": int,
        "climate.ref_endyear": int,
        "climate.ref_wateryear": str,
        "climate.ref_spinupyears": int,
        "climate.gcm_name": str,
        "climate.scenario": (str, type(None)),
        "climate.gcm_startyear": int,
        "climate.gcm_endyear": int,
        "climate.gcm_wateryear": str,
        "climate.constantarea_years": int,
        "climate.gcm_spinupyears": int,
        "climate.hindcast": bool,
        "climate.paths": dict,
        "climate.paths.era5_relpath": str,
        "climate.paths.era5_temp_fn": str,
        "climate.paths.era5_tempstd_fn": str,
        "climate.paths.era5_prec_fn": str,
        "climate.paths.era5_elev_fn": str,
        "climate.paths.era5_pressureleveltemp_fn": str,
        "climate.paths.era5_lr_fn": str,
        "climate.paths.cmip5_relpath": str,
        "climate.paths.cmip5_fp_var_ending": str,
        "climate.paths.cmip5_fp_fx_ending": str,
        "climate.paths.cmip6_relpath": str,
        "climate.paths.cesm2_relpath": str,
        "climate.paths.cesm2_fp_var_ending": str,
        "climate.paths.cesm2_fp_fx_ending": str,
        "climate.paths.gfdl_relpath": str,
        "climate.paths.gfdl_fp_var_ending": str,
        "climate.paths.gfdl_fp_fx_ending": str,
        "calib": dict,
        "calib.option_calibration": str,
        "calib.priors_reg_fn": str,
        "calib.HH2015_params": dict,
        "calib.HH2015_params.tbias_init": int,
        "calib.HH2015_params.tbias_step": int,
        "calib.HH2015_params.kp_init": float,
        "calib.HH2015_params.kp_bndlow": float,
        "calib.HH2015_params.kp_bndhigh": int,
        "calib.HH2015_params.ddfsnow_init": float,
        "calib.HH2015_params.ddfsnow_bndlow": float,
        "calib.HH2015_params.ddfsnow_bndhigh": float,
        "calib.HH2015mod_params": dict,
        "calib.HH2015mod_params.tbias_init": int,
        "calib.HH2015mod_params.tbias_step": float,
        "calib.HH2015mod_params.kp_init": int,
        "calib.HH2015mod_params.kp_bndlow": float,
        "calib.HH2015mod_params.kp_bndhigh": int,
        "calib.HH2015mod_params.ddfsnow_init": float,
        "calib.HH2015mod_params.method_opt": str,
        "calib.HH2015mod_params.params2opt": list,
        "calib.HH2015mod_params.ftol_opt": float,
        "calib.HH2015mod_params.eps_opt": float,
        "calib.emulator_params": dict,
        "calib.emulator_params.emulator_sims": int,
        "calib.emulator_params.overwrite_em_sims": bool,
        "calib.emulator_params.opt_hh2015_mod": bool,
        "calib.emulator_params.tbias_step": float,
        "calib.emulator_params.tbias_init": int,
        "calib.emulator_params.kp_init": int,
        "calib.emulator_params.kp_bndlow": float,
        "calib.emulator_params.kp_bndhigh": int,
        "calib.emulator_params.ddfsnow_init": float,
        "calib.emulator_params.option_areaconstant": bool,
        "calib.emulator_params.tbias_disttype": str,
        "calib.emulator_params.tbias_sigma": int,
        "calib.emulator_params.kp_gamma_alpha": int,
        "calib.emulator_params.kp_gamma_beta": int,
        "calib.emulator_params.ddfsnow_disttype": str,
        "calib.emulator_params.ddfsnow_mu": float,
        "calib.emulator_params.ddfsnow_sigma": float,
        "calib.emulator_params.ddfsnow_bndlow": int,
        "calib.emulator_params.ddfsnow_bndhigh": float,
        "calib.emulator_params.method_opt": str,
        "calib.emulator_params.params2opt": list,
        "calib.emulator_params.ftol_opt": float,
        "calib.emulator_params.eps_opt": float,
        "calib.MCMC_params": dict,
        "calib.MCMC_params.option_use_emulator": bool,
        "calib.MCMC_params.emulator_sims": int,
        "calib.MCMC_params.tbias_step": float,
        "calib.MCMC_params.tbias_stepsmall": float,
        "calib.MCMC_params.option_areaconstant": bool,
        "calib.MCMC_params.mcmc_step": float,
        "calib.MCMC_params.n_chains": int,
        "calib.MCMC_params.mcmc_sample_no": int,
        "calib.MCMC_params.mcmc_burn_pct": int,
        "calib.MCMC_params.thin_interval": int,
        "calib.MCMC_params.ddfsnow_disttype": str,
        "calib.MCMC_params.ddfsnow_mu": float,
        "calib.MCMC_params.ddfsnow_sigma": float,
        "calib.MCMC_params.ddfsnow_bndlow": int,
        "calib.MCMC_params.ddfsnow_bndhigh": float,
        "calib.MCMC_params.kp_disttype": str,
        "calib.MCMC_params.tbias_disttype": str,
        "calib.MCMC_params.tbias_mu": int,
        "calib.MCMC_params.tbias_sigma": int,
        "calib.MCMC_params.tbias_bndlow": int,
        "calib.MCMC_params.tbias_bndhigh": int,
        "calib.MCMC_params.kp_gamma_alpha": int,
        "calib.MCMC_params.kp_gamma_beta": int,
        "calib.MCMC_params.kp_lognorm_mu": int,
        "calib.MCMC_params.kp_lognorm_tau": int,
        "calib.MCMC_params.kp_mu": int,
        "calib.MCMC_params.kp_sigma": float,
        "calib.MCMC_params.kp_bndlow": float,
        "calib.MCMC_params.kp_bndhigh": float,
        "calib.data": dict,
        "calib.data.massbalance": dict,
        "calib.data.massbalance.hugonnet2021_relpath": str,
        "calib.data.massbalance.hugonnet2021_fn": str,
        "calib.data.massbalance.hugonnet2021_facorrected_fn": str,
        "calib.data.frontalablation": dict,
        "calib.data.frontalablation.frontalablation_relpath": str,
        "calib.data.frontalablation.frontalablation_cal_fn": str,
        "calib.data.icethickness": dict,
        "calib.data.icethickness.h_consensus_relpath": str,
        "calib.icethickness_cal_frac_byarea": float,
        "sim": dict,
        "sim.option_dynamics": (str, type(None)),
        "sim.option_bias_adjustment": int,
        "sim.nsims": int,
        "sim.out": dict,
        "sim.out.sim_stats": list,
        "sim.out.export_all_simiters": bool,
        "sim.out.export_extra_vars": bool,
        "sim.out.export_binned_data": bool,
        "sim.out.export_binned_components": bool,
        "sim.out.export_binned_area_threshold": int,
        "sim.oggm_dynamics": dict,
        "sim.oggm_dynamics.cfl_number": float,
        "sim.oggm_dynamics.cfl_number_calving": float,
        "sim.oggm_dynamics.glena_reg_relpath": str,
        "sim.oggm_dynamics.use_reg_glena": bool,
        "sim.oggm_dynamics.fs": int,
        "sim.oggm_dynamics.glen_a_multiplier": int,
        "sim.icethickness_advancethreshold": int,
        "sim.terminus_percentage": int,
        "sim.params": dict,
        "sim.params.use_constant_lapserate": bool,
        "sim.params.kp": int,
        "sim.params.tbias": int,
        "sim.params.ddfsnow": float,
        "sim.params.ddfsnow_iceratio": float,
        "sim.params.precgrad": float,
        "sim.params.lapserate": float,
        "sim.params.tsnow_threshold": int,
        "sim.params.calving_k": float,
        "mb": dict,
        "mb.option_surfacetype_initial": int,
        "mb.include_firn": bool,
        "mb.include_debris": bool,
        "mb.debris_relpath": str,
        "mb.option_elev_ref_downscale": str,
        "mb.option_temp2bins": int,
        "mb.option_adjusttemp_surfelev": int,
        "mb.option_prec2bins": int,
        "mb.option_preclimit": int,
        "mb.option_accumulation": int,
        "mb.option_ablation": int,
        "mb.option_ddf_firn": int,
        "mb.option_refreezing": str,
        "mb.Woodard_rf_opts": dict,
        "mb.Woodard_rf_opts.rf_month": int,
        "mb.HH2015_rf_opts": dict,
        "mb.HH2015_rf_opts.rf_layers": int,
        "mb.HH2015_rf_opts.rf_dz": int,
        "mb.HH2015_rf_opts.rf_dsc": int,
        "mb.HH2015_rf_opts.rf_meltcrit": float,
        "mb.HH2015_rf_opts.pp": float,
        "mb.HH2015_rf_opts.rf_dens_top": int,
        "mb.HH2015_rf_opts.rf_dens_bot": int,
        "mb.HH2015_rf_opts.option_rf_limit_meltsnow": int,
        "rgi": dict,
        "rgi.rgi_relpath": str,
        "rgi.rgi_lat_colname": str,
        "rgi.rgi_lon_colname": str,
        "rgi.elev_colname": str,
        "rgi.indexname": str,
        "rgi.rgi_O1Id_colname": str,
        "rgi.rgi_glacno_float_colname": str,
        "rgi.rgi_cols_drop": list,
        "time": dict,
        "time.option_leapyear": int,
        "time.startmonthday": str,
        "time.endmonthday": str,
        "time.wateryear_month_start": int,
        "time.winter_month_start": int,
        "time.summer_month_start": int,
        "time.option_dates": int,
        "time.timestep": str,
        "constants": dict,
        "constants.density_ice": int,
        "constants.density_water": int,
        "constants.area_ocean": float,
        "constants.k_ice": float,
        "constants.k_air": float,
        "constants.ch_ice": int,
        "constants.ch_air": int,
        "constants.Lh_rf": int,
        "constants.tolerance": float,
        "constants.gravity": float,
        "constants.pressure_std": int,
        "constants.temp_std": float,
        "constants.R_gas": float,
        "constants.molarmass_air": float,
        "debug": dict,
        "debug.refreeze": bool,
        "debug.mb": bool,
    }

    # expected types of elements in lists
    LIST_ELEMENT_TYPES = {
        "setup.rgi_region01": int,
        "setup.glac_no_skip": float,
        "setup.glac_no": float,
        "calib.HH2015mod_params.params2opt": str,
        "calib.emulator_params.params2opt": str,
        "sim.out.sim_stats": str,
        "rgi.rgi_cols_drop": str,
    }
