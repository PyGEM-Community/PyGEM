"""
Python Glacier Evolution Model (PyGEM)

copyright © 2018 David Rounce <drounce@cmu.edu

Distributed under the MIT license

PyGEM classes and subclasses for model output datasets

For glacier simulations:
The two main parent classes are single_glacier(object) and compiled_regional(object)
Both of these have several subclasses which will inherit the necessary parent information
"""

import collections
import json
import os
import warnings
from dataclasses import dataclass
from datetime import datetime

import cftime
import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import median_abs_deviation

import pygem
from pygem.setup.config import ConfigManager

# instantiate ConfigManager
config_manager = ConfigManager()
# read the config
pygem_prms = config_manager.read_config()

__all__ = [
    'single_glacier',
    'glacierwide_stats',
    'binned_stats',
    'calc_stats_array',
]


@dataclass
class single_glacier:
    """
    Single glacier output dataset class for the Python Glacier Evolution Model.
    This serves as the parent class to both `output.glacierwide_stats` and `output.binned_stats`.

    Attributes
    ----------
    glacier_rgi_table : pd.DataFrame
        DataFrame containing metadata and characteristics of the glacier from the Randolph Glacier Inventory.
    dates_table : pd.DataFrame
        DataFrame containing the time series of dates associated with the model output.
    timestep : str
        The time step resolution ('monthly' or 'daily')
    sim_climate_name : str
        Name of the General Circulation Model (GCM) used for climate forcing.
    sim_climate_scenario : str
        Emission or climate sim_climate_scenario under which the simulation is run.
    realization : str
        Specific realization or ensemble member of the GCM simulation.
    nsims : int
        Number of simulation runs performed.
    modelprms : dict
        Dictionary containing model parameters used in the simulation.
    ref_startyear : int
        Start year of the reference period for model calibration or comparison.
    ref_endyear : int
        End year of the reference period for model calibration or comparison.
    sim_startyear : int
        Start year of the GCM forcing data used in the simulation.
    sim_endyear : int
        End year of the GCM forcing data used in the simulation.
    option_calibration : str
        Model calibration method.
    option_bias_adjustment : int
        Bias adjustment method applied to the climate input data
    """

    glacier_rgi_table: pd.DataFrame
    dates_table: pd.DataFrame
    timestep: str
    sim_climate_name: str
    sim_climate_scenario: str
    realization: str
    nsims: int
    modelprms: dict
    ref_startyear: int
    ref_endyear: int
    sim_startyear: int
    sim_endyear: int
    option_calibration: str
    option_bias_adjustment: str
    option_dynamics: str
    extra_vars: bool = False

    def __post_init__(self):
        """
        Initializes additional attributes after the dataclass fields are set.

        This method:
        - Retrieves and stores the PyGEM version.
        - Extracts field names in RGI glacier table.
        - Formats the glacier RGI ID as a string with five decimal places.
        - Extracts and zero-pads the primary region code from the RGI table.
        - Defines the output directory path for storing simulation results.
        - Calls setup functions to initialize and store filenames, time values, model parameters, and dictionaries.
        """
        self.pygem_version = pygem.__version__
        self.glac_values = np.array([self.glacier_rgi_table.name])
        self.glacier_str = '{0:0.5f}'.format(self.glacier_rgi_table['RGIId_float'])
        self.reg_str = str(self.glacier_rgi_table.O1Region).zfill(2)
        self.outdir = pygem_prms['root'] + '/Output/simulations/'
        self.set_fn()
        self._set_time_vals()
        self._model_params_record()
        self._init_dicts()

    def set_fn(self, outfn=None):
        """Set the dataset output file name.
        Parameters
        ----------
        outfn : str
            Output filename string.
        """
        if outfn:
            self.outfn = outfn
        else:
            self.outfn = self.glacier_str + '_' + self.sim_climate_name + '_'
            if self.sim_climate_scenario:
                self.outfn += f'{self.sim_climate_scenario}_'
            if self.realization:
                self.outfn += f'{self.realization}_'
            if self.option_calibration:
                self.outfn += f'{self.option_calibration}_'
            else:
                self.outfn += (
                    f'kp{self.modelprms["kp"]}_ddfsnow{self.modelprms["ddfsnow"]}_tbias{self.modelprms["tbias"]}_'
                )
                if 'lrbias' in self.modelprms:
                    self.outfn += f'lrbias{self.modelprms["lrbias"]}_'
            if self.sim_climate_name not in ['ERA-Interim', 'ERA5', 'COAWST']:
                self.outfn += f'ba{self.option_bias_adjustment}_'
            else:
                self.outfn += 'ba0_'
            if self.option_calibration:
                self.outfn += 'SETS_'
            self.outfn += f'{self.sim_startyear}_'
            self.outfn += f'{self.sim_endyear}_'

    def get_fn(self):
        """Return the output dataset filename."""
        return self.outfn

    def set_modelprms(self, modelprms):
        """
        Set the model parameters and update the dataset record.

        Parameters
        ----------
        modelprms : dict
            Dictionary containing model parameters used in the simulation.

        This method updates the `modelprms` attribute with the provided dictionary and esnures that
        the model parameter record is updated accordingly by calling `self._update_modelparams_record()`.
        """
        self.modelprms = modelprms
        # update model_params_record
        self._update_modelparams_record()

    def _set_time_vals(self):
        """Set output dataset time and year values from dates_table."""
        if pygem_prms['climate']['sim_wateryear'] == 'hydro':
            self.year_type = 'water year'
            self.annual_columns = np.unique(self.dates_table['wateryear'].values)[
                0 : int(self.dates_table.shape[0] / 12)
            ]
        elif pygem_prms['climate']['sim_wateryear'] == 'calendar':
            self.year_type = 'calendar year'
            self.annual_columns = np.unique(self.dates_table['year'].values)[0 : int(self.dates_table.shape[0] / 12)]
        elif pygem_prms['climate']['sim_wateryear'] == 'custom':
            self.year_type = 'custom year'
        self.time_values = [cftime.DatetimeGregorian(x.year, x.month, x.day) for x in self.dates_table['date']]
        # append additional year to self.year_values to account for mass and area at end of period
        self.year_values = self.annual_columns
        self.year_values = np.concatenate((self.year_values, np.array([self.annual_columns[-1] + 1])))

    def _model_params_record(self):
        """Build model parameters attribute dictionary to be saved to output dataset."""
        # get all locally defined variables from the pygem_prms, excluding imports, functions, and classes
        self.mdl_params_dict = {}
        # overwrite variables that are possibly different from pygem_input
        self.mdl_params_dict['ref_startyear'] = self.ref_startyear
        self.mdl_params_dict['ref_endyear'] = self.ref_endyear
        self.mdl_params_dict['sim_startyear'] = self.sim_startyear
        self.mdl_params_dict['sim_endyear'] = self.sim_endyear
        self.mdl_params_dict['sim_climate_name'] = self.sim_climate_name
        self.mdl_params_dict['realization'] = self.realization
        self.mdl_params_dict['sim_climate_scenario'] = self.sim_climate_scenario
        self.mdl_params_dict['option_calibration'] = self.option_calibration
        self.mdl_params_dict['option_bias_adjustment'] = self.option_bias_adjustment
        self.mdl_params_dict['option_dynamics'] = self.option_dynamics
        self.mdl_params_dict['timestep'] = self.timestep
        # record manually defined modelprms if calibration option is None
        if not self.option_calibration:
            self._update_modelparams_record()

    def _update_modelparams_record(self):
        """Update the values in the output dataset's model parameters dictionary."""
        for key, value in self.modelprms.items():
            self.mdl_params_dict[key] = value

    def _init_dicts(self):
        """Initialize output coordinate and attribute dictionaries."""
        self.output_coords_dict = collections.OrderedDict()
        self.output_coords_dict['RGIId'] = collections.OrderedDict([('glac', self.glac_values)])
        self.output_coords_dict['CenLon'] = collections.OrderedDict([('glac', self.glac_values)])
        self.output_coords_dict['CenLat'] = collections.OrderedDict([('glac', self.glac_values)])
        self.output_coords_dict['O1Region'] = collections.OrderedDict([('glac', self.glac_values)])
        self.output_coords_dict['O2Region'] = collections.OrderedDict([('glac', self.glac_values)])
        self.output_coords_dict['Area'] = collections.OrderedDict([('glac', self.glac_values)])
        self.output_attrs_dict = {
            'time': {
                'long_name': 'time',
                'year_type': self.year_type,
                'comment': 'start of the month',
            },
            'glac': {
                'long_name': 'glacier index',
                'comment': 'glacier index referring to glaciers properties and model results',
            },
            'year': {
                'long_name': 'years',
                'year_type': self.year_type,
                'comment': 'years referring to the start of each year',
            },
            'RGIId': {
                'long_name': 'Randolph Glacier Inventory ID',
                'comment': 'RGIv6.0',
            },
            'CenLon': {
                'long_name': 'center longitude',
                'units': 'degrees E',
                'comment': 'value from RGIv6.0',
            },
            'CenLat': {
                'long_name': 'center latitude',
                'units': 'degrees N',
                'comment': 'value from RGIv6.0',
            },
            'O1Region': {
                'long_name': 'RGI order 1 region',
                'comment': 'value from RGIv6.0',
            },
            'O2Region': {
                'long_name': 'RGI order 2 region',
                'comment': 'value from RGIv6.0',
            },
            'Area': {
                'long_name': 'glacier area',
                'units': 'm2',
                'comment': 'value from RGIv6.0',
            },
        }

    def create_xr_ds(self):
        """Create an xarrray dataset with placeholders for data arrays."""
        # Add variables to empty dataset and merge together
        count_vn = 0
        self.encoding = {}
        for vn in self.output_coords_dict.keys():
            count_vn += 1
            empty_holder = np.zeros(
                [len(self.output_coords_dict[vn][i]) for i in list(self.output_coords_dict[vn].keys())]
            )
            output_xr_ds_ = xr.Dataset(
                {vn: (list(self.output_coords_dict[vn].keys()), empty_holder)},
                coords=self.output_coords_dict[vn],
            )
            # Merge datasets of stats into one output
            if count_vn == 1:
                self.output_xr_ds = output_xr_ds_
            else:
                self.output_xr_ds = xr.merge((self.output_xr_ds, output_xr_ds_))
        noencoding_vn = ['RGIId']
        # Add attributes
        for vn in self.output_xr_ds.variables:
            try:
                self.output_xr_ds[vn].attrs = self.output_attrs_dict[vn]
            except:
                pass
            # Encoding (specify _FillValue, offsets, etc.)

            if vn not in noencoding_vn:
                self.encoding[vn] = {'_FillValue': None, 'zlib': True, 'complevel': 9}
        self.output_xr_ds['RGIId'].values = np.array([self.glacier_rgi_table.loc['RGIId']])
        self.output_xr_ds['CenLon'].values = np.array([self.glacier_rgi_table.CenLon])
        self.output_xr_ds['CenLat'].values = np.array([self.glacier_rgi_table.CenLat])
        self.output_xr_ds['O1Region'].values = np.array([self.glacier_rgi_table.O1Region])
        self.output_xr_ds['O2Region'].values = np.array([self.glacier_rgi_table.O2Region])
        self.output_xr_ds['Area'].values = np.array([self.glacier_rgi_table.Area * 1e6])

        self.output_xr_ds.attrs = {
            'source': f'PyGEMv{self.pygem_version}',
            'institution': pygem_prms['user']['institution'],
            'history': f'Created by {pygem_prms["user"]["name"]} ({pygem_prms["user"]["email"]}) on '
            + datetime.today().strftime('%Y-%m-%d'),
            'references': 'doi:10.1126/science.abo1324',
            'model_parameters': json.dumps(self.mdl_params_dict),
        }

    def get_xr_ds(self):
        """Return the xarray dataset."""
        return self.output_xr_ds

    def save_xr_ds(self):
        """Save the xarray dataset."""
        # export netcdf
        self.output_xr_ds.to_netcdf(self.outdir + self.outfn, encoding=self.encoding)
        # close datasets
        self.output_xr_ds.close()


@dataclass
class glacierwide_stats(single_glacier):
    """
    Single glacier-wide statistics dataset.

    This class extends `single_glacier` to store and manage glacier-wide statistical outputs.
    """

    def __post_init__(self):
        """
        Initializes additional attributes after the dataclass fields are set.

        This method:
        - Calls the parent class `__post_init__` to initialize glacier values,
          time stamps, and instantiate output dataset dictionarie.
        - Sets the output directory specific to glacier-wide statistics.
        - Updates the output dictionaries with required fields.
        """
        super().__post_init__()
        self._set_outdir()
        self._update_dicts()

    def _set_outdir(self):
        """Set the output directory path. Create if it does not already exist."""
        self.outdir += self.reg_str + '/' + self.sim_climate_name + '/'
        if self.sim_climate_name not in ['ERA-Interim', 'ERA5', 'COAWST']:
            self.outdir += self.sim_climate_scenario + '/'
        self.outdir += 'stats/'
        # Create filepath if it does not exist
        os.makedirs(self.outdir, exist_ok=True)

    def _update_dicts(self):
        """Update coordinate and attribute dictionaries specific to glacierwide_stats outputs"""
        self.output_coords_dict['glac_runoff'] = collections.OrderedDict(
            [('glac', self.glac_values), ('time', self.time_values)]
        )
        self.output_attrs_dict['glac_runoff'] = {
            'long_name': 'glacier-wide runoff',
            'units': 'm3',
            'temporal_resolution': self.timestep,
            'comment': 'runoff from the glacier terminus, which moves over time',
        }
        self.output_coords_dict['glac_area_annual'] = collections.OrderedDict(
            [('glac', self.glac_values), ('year', self.year_values)]
        )
        self.output_attrs_dict['glac_area_annual'] = {
            'long_name': 'glacier area',
            'units': 'm2',
            'temporal_resolution': 'annual',
            'comment': 'area at start of the year',
        }
        self.output_coords_dict['glac_mass_annual'] = collections.OrderedDict(
            [('glac', self.glac_values), ('year', self.year_values)]
        )
        self.output_attrs_dict['glac_mass_annual'] = {
            'long_name': 'glacier mass',
            'units': 'kg',
            'temporal_resolution': 'annual',
            'comment': 'mass of ice based on area and ice thickness at start of the year',
        }
        self.output_coords_dict['glac_mass_bsl_annual'] = collections.OrderedDict(
            [('glac', self.glac_values), ('year', self.year_values)]
        )
        self.output_attrs_dict['glac_mass_bsl_annual'] = {
            'long_name': 'glacier mass below sea level',
            'units': 'kg',
            'temporal_resolution': 'annual',
            'comment': 'mass of ice below sea level based on area and ice thickness at start of the year',
        }
        self.output_coords_dict['glac_ELA_annual'] = collections.OrderedDict(
            [('glac', self.glac_values), ('year', self.year_values)]
        )
        self.output_attrs_dict['glac_ELA_annual'] = {
            'long_name': 'annual equilibrium line altitude above mean sea level',
            'units': 'm',
            'temporal_resolution': 'annual',
            'comment': 'equilibrium line altitude is the elevation where the climatic mass balance is zero',
        }
        self.output_coords_dict['offglac_runoff'] = collections.OrderedDict(
            [('glac', self.glac_values), ('time', self.time_values)]
        )
        self.output_attrs_dict['offglac_runoff'] = {
            'long_name': 'off-glacier-wide runoff',
            'units': 'm3',
            'temporal_resolution': self.timestep,
            'comment': 'off-glacier runoff from area where glacier no longer exists',
        }

        # if nsims > 1, store median-absolute deviation metrics
        if self.nsims > 1:
            self.output_coords_dict['glac_runoff_mad'] = collections.OrderedDict(
                [('glac', self.glac_values), ('time', self.time_values)]
            )
            self.output_attrs_dict['glac_runoff_mad'] = {
                'long_name': 'glacier-wide runoff median absolute deviation',
                'units': 'm3',
                'temporal_resolution': self.timestep,
                'comment': 'runoff from the glacier terminus, which moves over time',
            }
            self.output_coords_dict['glac_area_annual_mad'] = collections.OrderedDict(
                [('glac', self.glac_values), ('year', self.year_values)]
            )
            self.output_attrs_dict['glac_area_annual_mad'] = {
                'long_name': 'glacier area median absolute deviation',
                'units': 'm2',
                'temporal_resolution': 'annual',
                'comment': 'area at start of the year',
            }
            self.output_coords_dict['glac_mass_annual_mad'] = collections.OrderedDict(
                [('glac', self.glac_values), ('year', self.year_values)]
            )
            self.output_attrs_dict['glac_mass_annual_mad'] = {
                'long_name': 'glacier mass median absolute deviation',
                'units': 'kg',
                'temporal_resolution': 'annual',
                'comment': 'mass of ice based on area and ice thickness at start of the year',
            }
            self.output_coords_dict['glac_mass_bsl_annual_mad'] = collections.OrderedDict(
                [('glac', self.glac_values), ('year', self.year_values)]
            )
            self.output_attrs_dict['glac_mass_bsl_annual_mad'] = {
                'long_name': 'glacier mass below sea level median absolute deviation',
                'units': 'kg',
                'temporal_resolution': 'annual',
                'comment': 'mass of ice below sea level based on area and ice thickness at start of the year',
            }
            self.output_coords_dict['glac_ELA_annual_mad'] = collections.OrderedDict(
                [('glac', self.glac_values), ('year', self.year_values)]
            )
            self.output_attrs_dict['glac_ELA_annual_mad'] = {
                'long_name': 'annual equilibrium line altitude above mean sea level median absolute deviation',
                'units': 'm',
                'temporal_resolution': 'annual',
                'comment': 'equilibrium line altitude is the elevation where the climatic mass balance is zero',
            }
            self.output_coords_dict['offglac_runoff_mad'] = collections.OrderedDict(
                [('glac', self.glac_values), ('time', self.time_values)]
            )
            self.output_attrs_dict['offglac_runoff_mad'] = {
                'long_name': 'off-glacier-wide runoff median absolute deviation',
                'units': 'm3',
                'temporal_resolution': self.timestep,
                'comment': 'off-glacier runoff from area where glacier no longer exists',
            }

        # optionally store extra variables
        if self.extra_vars:
            self.output_coords_dict['glac_prec'] = collections.OrderedDict(
                [('glac', self.glac_values), ('time', self.time_values)]
            )
            self.output_attrs_dict['glac_prec'] = {
                'long_name': 'glacier-wide precipitation (liquid)',
                'units': 'm3',
                'temporal_resolution': self.timestep,
                'comment': 'only the liquid precipitation, solid precipitation excluded',
            }
            self.output_coords_dict['glac_temp'] = collections.OrderedDict(
                [('glac', self.glac_values), ('time', self.time_values)]
            )
            self.output_attrs_dict['glac_temp'] = {
                'standard_name': 'air_temperature',
                'long_name': 'glacier-wide mean air temperature',
                'units': 'K',
                'temporal_resolution': self.timestep,
                'comment': (
                    'each elevation bin is weighted equally to compute the mean temperature, and '
                    'bins where the glacier no longer exists due to retreat have been removed'
                ),
            }
            self.output_coords_dict['glac_acc'] = collections.OrderedDict(
                [('glac', self.glac_values), ('time', self.time_values)]
            )
            self.output_attrs_dict['glac_acc'] = {
                'long_name': 'glacier-wide accumulation, in water equivalent',
                'units': 'm3',
                'temporal_resolution': self.timestep,
                'comment': 'only the solid precipitation',
            }
            self.output_coords_dict['glac_refreeze'] = collections.OrderedDict(
                [('glac', self.glac_values), ('time', self.time_values)]
            )
            self.output_attrs_dict['glac_refreeze'] = {
                'long_name': 'glacier-wide refreeze, in water equivalent',
                'units': 'm3',
                'temporal_resolution': self.timestep,
            }
            self.output_coords_dict['glac_melt'] = collections.OrderedDict(
                [('glac', self.glac_values), ('time', self.time_values)]
            )
            self.output_attrs_dict['glac_melt'] = {
                'long_name': 'glacier-wide melt, in water equivalent',
                'units': 'm3',
                'temporal_resolution': self.timestep,
            }
            self.output_coords_dict['glac_frontalablation'] = collections.OrderedDict(
                [('glac', self.glac_values), ('time', self.time_values)]
            )
            self.output_attrs_dict['glac_frontalablation'] = {
                'long_name': 'glacier-wide frontal ablation, in water equivalent',
                'units': 'm3',
                'temporal_resolution': self.timestep,
                'comment': (
                    'mass losses from calving, subaerial frontal melting, sublimation above the '
                    'waterline and subaqueous frontal melting below the waterline; positive values indicate mass lost like melt'
                ),
            }
            self.output_coords_dict['glac_massbaltotal'] = collections.OrderedDict(
                [('glac', self.glac_values), ('time', self.time_values)]
            )
            self.output_attrs_dict['glac_massbaltotal'] = {
                'long_name': 'glacier-wide total mass balance, in water equivalent',
                'units': 'm3',
                'temporal_resolution': self.timestep,
                'comment': 'total mass balance is the sum of the climatic mass balance and frontal ablation',
            }
            self.output_coords_dict['glac_snowline'] = collections.OrderedDict(
                [('glac', self.glac_values), ('time', self.time_values)]
            )
            self.output_attrs_dict['glac_snowline'] = {
                'long_name': 'transient snowline altitude above mean sea level',
                'units': 'm',
                'temporal_resolution': self.timestep,
                'comment': 'transient snowline is altitude separating snow from ice/firn',
            }
            self.output_coords_dict['glac_scaf'] = collections.OrderedDict(
                [('glac', self.glac_values), ('time', self.time_values)]
            )
            self.output_attrs_dict['glac_scaf'] = {
                'long_name': 'transient snow cover area fraction (SCAF)',
                'units': '-',
                'temporal_resolution': self.timestep,
                'comment': 'transient snow cover area fraction (SCAF) is fractional area of snow compared to total area',
            }
            self.output_coords_dict['glac_meltextent'] = collections.OrderedDict(
                [('glac', self.glac_values), ('time', self.time_values)]
            )
            self.output_attrs_dict['glac_meltextent'] = {
                'long_name': 'transient melt extent altitude above mean sea level',
                'units': 'm',
                'temporal_resolution': self.timestep,
                'comment': 'transient melt extent is altitude separating dry/wet (melting) snow',
            }
            self.output_coords_dict['glac_mass_change_ignored_annual'] = collections.OrderedDict(
                [('glac', self.glac_values), ('year', self.year_values)]
            )
            self.output_attrs_dict['glac_mass_change_ignored_annual'] = {
                'long_name': 'glacier mass change ignored',
                'units': 'kg',
                'temporal_resolution': 'annual',
                'comment': 'glacier mass change ignored due to flux divergence',
            }
            self.output_coords_dict['offglac_prec'] = collections.OrderedDict(
                [('glac', self.glac_values), ('time', self.time_values)]
            )
            self.output_attrs_dict['offglac_prec'] = {
                'long_name': 'off-glacier-wide precipitation (liquid)',
                'units': 'm3',
                'temporal_resolution': self.timestep,
                'comment': 'only the liquid precipitation, solid precipitation excluded',
            }
            self.output_coords_dict['offglac_refreeze'] = collections.OrderedDict(
                [('glac', self.glac_values), ('time', self.time_values)]
            )
            self.output_attrs_dict['offglac_refreeze'] = {
                'long_name': 'off-glacier-wide refreeze, in water equivalent',
                'units': 'm3',
                'temporal_resolution': self.timestep,
            }
            self.output_coords_dict['offglac_melt'] = collections.OrderedDict(
                [('glac', self.glac_values), ('time', self.time_values)]
            )
            self.output_attrs_dict['offglac_melt'] = {
                'long_name': 'off-glacier-wide melt, in water equivalent',
                'units': 'm3',
                'temporal_resolution': self.timestep,
                'comment': 'only melt of snow and refreeze since off-glacier',
            }
            self.output_coords_dict['offglac_snowpack'] = collections.OrderedDict(
                [('glac', self.glac_values), ('time', self.time_values)]
            )
            self.output_attrs_dict['offglac_snowpack'] = {
                'long_name': 'off-glacier-wide snowpack, in water equivalent',
                'units': 'm3',
                'temporal_resolution': self.timestep,
                'comment': 'snow remaining accounting for new accumulation, melt, and refreeze',
            }

            # if nsims > 1, store median-absolute deviation metrics
            if self.nsims > 1:
                self.output_coords_dict['glac_prec_mad'] = collections.OrderedDict(
                    [('glac', self.glac_values), ('time', self.time_values)]
                )
                self.output_attrs_dict['glac_prec_mad'] = {
                    'long_name': 'glacier-wide precipitation (liquid) median absolute deviation',
                    'units': 'm3',
                    'temporal_resolution': self.timestep,
                    'comment': 'only the liquid precipitation, solid precipitation excluded',
                }
                self.output_coords_dict['glac_temp_mad'] = collections.OrderedDict(
                    [('glac', self.glac_values), ('time', self.time_values)]
                )
                self.output_attrs_dict['glac_temp_mad'] = {
                    'standard_name': 'air_temperature',
                    'long_name': 'glacier-wide mean air temperature median absolute deviation',
                    'units': 'K',
                    'temporal_resolution': self.timestep,
                    'comment': (
                        'each elevation bin is weighted equally to compute the mean temperature, and '
                        'bins where the glacier no longer exists due to retreat have been removed'
                    ),
                }
                self.output_coords_dict['glac_acc_mad'] = collections.OrderedDict(
                    [('glac', self.glac_values), ('time', self.time_values)]
                )
                self.output_attrs_dict['glac_acc_mad'] = {
                    'long_name': 'glacier-wide accumulation, in water equivalent, median absolute deviation',
                    'units': 'm3',
                    'temporal_resolution': self.timestep,
                    'comment': 'only the solid precipitation',
                }
                self.output_coords_dict['glac_refreeze_mad'] = collections.OrderedDict(
                    [('glac', self.glac_values), ('time', self.time_values)]
                )
                self.output_attrs_dict['glac_refreeze_mad'] = {
                    'long_name': 'glacier-wide refreeze, in water equivalent, median absolute deviation',
                    'units': 'm3',
                    'temporal_resolution': self.timestep,
                }
                self.output_coords_dict['glac_melt_mad'] = collections.OrderedDict(
                    [('glac', self.glac_values), ('time', self.time_values)]
                )
                self.output_attrs_dict['glac_melt_mad'] = {
                    'long_name': 'glacier-wide melt, in water equivalent, median absolute deviation',
                    'units': 'm3',
                    'temporal_resolution': self.timestep,
                }
                self.output_coords_dict['glac_frontalablation_mad'] = collections.OrderedDict(
                    [('glac', self.glac_values), ('time', self.time_values)]
                )
                self.output_attrs_dict['glac_frontalablation_mad'] = {
                    'long_name': 'glacier-wide frontal ablation, in water equivalent, median absolute deviation',
                    'units': 'm3',
                    'temporal_resolution': self.timestep,
                    'comment': (
                        'mass losses from calving, subaerial frontal melting, sublimation above the '
                        'waterline and subaqueous frontal melting below the waterline'
                    ),
                }
                self.output_coords_dict['glac_massbaltotal_mad'] = collections.OrderedDict(
                    [('glac', self.glac_values), ('time', self.time_values)]
                )
                self.output_attrs_dict['glac_massbaltotal_mad'] = {
                    'long_name': 'glacier-wide total mass balance, in water equivalent, median absolute deviation',
                    'units': 'm3',
                    'temporal_resolution': self.timestep,
                    'comment': 'total mass balance is the sum of the climatic mass balance and frontal ablation',
                }
                self.output_coords_dict['glac_snowline_mad'] = collections.OrderedDict(
                    [('glac', self.glac_values), ('time', self.time_values)]
                )
                self.output_attrs_dict['glac_snowline_mad'] = {
                    'long_name': 'transient snowline above mean sea level median absolute deviation',
                    'units': 'm',
                    'temporal_resolution': self.timestep,
                    'comment': 'transient snowline is altitude separating snow from ice/firn',
                }
                self.output_coords_dict['glac_scaf_mad'] = collections.OrderedDict(
                    [('glac', self.glac_values), ('time', self.time_values)]
                )
                self.output_attrs_dict['glac_scaf_mad'] = {
                    'long_name': 'transient snow cover area fraction (SCAF) median absolute deviation',
                    'units': '-',
                    'temporal_resolution': self.timestep,
                    'comment': 'transient snow cover area fraction is fractional area of snow compared to total area',
                }
                self.output_coords_dict['glac_meltextent_mad'] = collections.OrderedDict(
                    [('glac', self.glac_values), ('time', self.time_values)]
                )
                self.output_attrs_dict['glac_meltextent_mad'] = {
                    'long_name': 'transient melt extent above mean sea level median absolute deviation',
                    'units': 'm',
                    'temporal_resolution': self.timestep,
                    'comment': 'transient melt extent is altitude separating dry/wet (melting) snow',
                }
                self.output_coords_dict['glac_mass_change_ignored_annual_mad'] = collections.OrderedDict(
                    [('glac', self.glac_values), ('year', self.year_values)]
                )
                self.output_attrs_dict['glac_mass_change_ignored_annual_mad'] = {
                    'long_name': 'glacier mass change ignored median absolute deviation',
                    'units': 'kg',
                    'temporal_resolution': 'annual',
                    'comment': 'glacier mass change ignored due to flux divergence',
                }
                self.output_coords_dict['offglac_prec_mad'] = collections.OrderedDict(
                    [('glac', self.glac_values), ('time', self.time_values)]
                )
                self.output_attrs_dict['offglac_prec_mad'] = {
                    'long_name': 'off-glacier-wide precipitation (liquid) median absolute deviation',
                    'units': 'm3',
                    'temporal_resolution': self.timestep,
                    'comment': 'only the liquid precipitation, solid precipitation excluded',
                }
                self.output_coords_dict['offglac_refreeze_mad'] = collections.OrderedDict(
                    [('glac', self.glac_values), ('time', self.time_values)]
                )
                self.output_attrs_dict['offglac_refreeze_mad'] = {
                    'long_name': 'off-glacier-wide refreeze, in water equivalent, median absolute deviation',
                    'units': 'm3',
                    'temporal_resolution': self.timestep,
                }
                self.output_coords_dict['offglac_melt_mad'] = collections.OrderedDict(
                    [('glac', self.glac_values), ('time', self.time_values)]
                )
                self.output_attrs_dict['offglac_melt_mad'] = {
                    'long_name': 'off-glacier-wide melt, in water equivalent, median absolute deviation',
                    'units': 'm3',
                    'temporal_resolution': self.timestep,
                    'comment': 'only melt of snow and refreeze since off-glacier',
                }
                self.output_coords_dict['offglac_snowpack_mad'] = collections.OrderedDict(
                    [('glac', self.glac_values), ('time', self.time_values)]
                )
                self.output_attrs_dict['offglac_snowpack_mad'] = {
                    'long_name': 'off-glacier-wide snowpack, in water equivalent, median absolute deviation',
                    'units': 'm3',
                    'temporal_resolution': self.timestep,
                    'comment': 'snow remaining accounting for new accumulation, melt, and refreeze',
                }


@dataclass
class binned_stats(single_glacier):
    """
    Single glacier binned dataset.

    This class extends `single_glacier` to store and manage binned glacier output data.

    Attributes
    ----------
    nbins : int
        Number of bins used to segment the glacier dataset.
    binned_components : bool
        Flag indicating whether additional binned components are included in the dataset.
    """

    nbins: int = 0
    binned_components: bool = False

    def __post_init__(self):
        """
        Initializes additional attributes after the dataclass fields are set.

        This method:
        - Calls the parent class `__post_init__` to initialize glacier values,
          time stamps, and instantiate output dataset dictionaries.
        - Creates an array of bin indices based on the number of bins.
        - Sets the output directory specific to binned statistics.
        - Updates the output dictionaries with required fields.
        """
        super().__post_init__()
        self.bin_values = np.arange(self.nbins)
        self._set_outdir()
        self._update_dicts()

    def _set_outdir(self):
        """Set the output directory path. Create if it does not already exist."""
        self.outdir += self.reg_str + '/' + self.sim_climate_name + '/'
        if self.sim_climate_name not in ['ERA-Interim', 'ERA5', 'COAWST']:
            self.outdir += self.sim_climate_scenario + '/'
        self.outdir += 'binned/'
        # Create filepath if it does not exist
        os.makedirs(self.outdir, exist_ok=True)

    def _update_dicts(self):
        """Update coordinate and attribute dictionaries specific to glacierwide_stats outputs"""
        self.output_coords_dict['bin_distance'] = collections.OrderedDict(
            [('glac', self.glac_values), ('bin', self.bin_values)]
        )
        self.output_attrs_dict['bin_distance'] = {
            'long_name': 'distance downglacier',
            'units': 'm',
            'comment': 'horizontal distance calculated from top of glacier moving downglacier',
        }
        self.output_coords_dict['bin_surface_h_initial'] = collections.OrderedDict(
            [('glac', self.glac_values), ('bin', self.bin_values)]
        )
        self.output_attrs_dict['bin_surface_h_initial'] = {
            'long_name': 'initial binned surface elevation',
            'units': 'm above sea level',
        }
        self.output_coords_dict['bin_area_annual'] = collections.OrderedDict(
            [
                ('glac', self.glac_values),
                ('bin', self.bin_values),
                ('year', self.year_values),
            ]
        )
        self.output_attrs_dict['bin_area_annual'] = {
            'long_name': 'binned glacier area',
            'units': 'm2',
            'temporal_resolution': 'annual',
            'comment': 'binned area at start of the year',
        }
        self.output_coords_dict['bin_mass_annual'] = collections.OrderedDict(
            [
                ('glac', self.glac_values),
                ('bin', self.bin_values),
                ('year', self.year_values),
            ]
        )
        self.output_attrs_dict['bin_mass_annual'] = {
            'long_name': 'binned ice mass',
            'units': 'kg',
            'temporal_resolution': 'annual',
            'comment': 'binned ice mass at start of the year',
        }
        self.output_coords_dict['bin_thick_annual'] = collections.OrderedDict(
            [
                ('glac', self.glac_values),
                ('bin', self.bin_values),
                ('year', self.year_values),
            ]
        )
        self.output_attrs_dict['bin_thick_annual'] = {
            'long_name': 'binned ice thickness',
            'units': 'm',
            'temporal_resolution': 'annual',
            'comment': 'binned ice thickness at start of the year',
        }
        self.output_coords_dict['bin_massbalclim_annual'] = collections.OrderedDict(
            [
                ('glac', self.glac_values),
                ('bin', self.bin_values),
                ('year', self.year_values),
            ]
        )
        self.output_attrs_dict['bin_massbalclim_annual'] = (
            {
                'long_name': 'binned climatic mass balance, in water equivalent',
                'units': 'm',
                'temporal_resolution': 'annual',
                'comment': 'climatic mass balance is computed before dynamics so can theoretically exceed ice thickness',
            },
        )
        self.output_coords_dict['bin_massbalclim'] = collections.OrderedDict(
            [
                ('glac', self.glac_values),
                ('bin', self.bin_values),
                ('time', self.time_values),
            ]
        )
        self.output_attrs_dict['bin_massbalclim'] = {
            'long_name': 'binned climatic mass balance, in water equivalent',
            'units': 'm',
            'temporal_resolution': self.timestep,
            'comment': 'climatic mass balance from the PyGEM mass balance module',
        }

        # optionally store binned mass balance components
        if self.binned_components:
            self.output_coords_dict['bin_accumulation'] = collections.OrderedDict(
                [
                    ('glac', self.glac_values),
                    ('bin', self.bin_values),
                    ('time', self.time_values),
                ]
            )
            self.output_attrs_dict['bin_accumulation'] = {
                'long_name': 'binned accumulation, in water equivalent',
                'units': 'm',
                'temporal_resolution': self.timestep,
                'comment': 'accumulation from the PyGEM mass balance module',
            }
            self.output_coords_dict['bin_melt'] = collections.OrderedDict(
                [
                    ('glac', self.glac_values),
                    ('bin', self.bin_values),
                    ('time', self.time_values),
                ]
            )
            self.output_attrs_dict['bin_melt'] = {
                'long_name': 'binned melt, in water equivalent',
                'units': 'm',
                'temporal_resolution': self.timestep,
                'comment': 'melt from the PyGEM mass balance module',
            }
            self.output_coords_dict['bin_refreeze'] = collections.OrderedDict(
                [
                    ('glac', self.glac_values),
                    ('bin', self.bin_values),
                    ('time', self.time_values),
                ]
            )
            self.output_attrs_dict['bin_refreeze'] = {
                'long_name': 'binned refreeze, in water equivalent',
                'units': 'm',
                'temporal_resolution': self.timestep,
                'comment': 'refreeze from the PyGEM mass balance module',
            }

        # if nsims > 1, store median-absolute deviation metrics
        if self.nsims > 1:
            self.output_coords_dict['bin_mass_annual_mad'] = collections.OrderedDict(
                [
                    ('glac', self.glac_values),
                    ('bin', self.bin_values),
                    ('year', self.year_values),
                ]
            )
            self.output_attrs_dict['bin_mass_annual_mad'] = {
                'long_name': 'binned ice mass median absolute deviation',
                'units': 'kg',
                'temporal_resolution': 'annual',
                'comment': 'mass of ice based on area and ice thickness at start of the year',
            }
            self.output_coords_dict['bin_thick_annual_mad'] = collections.OrderedDict(
                [
                    ('glac', self.glac_values),
                    ('bin', self.bin_values),
                    ('year', self.year_values),
                ]
            )
            self.output_attrs_dict['bin_thick_annual_mad'] = {
                'long_name': 'binned ice thickness median absolute deviation',
                'units': 'm',
                'temporal_resolution': 'annual',
                'comment': 'thickness of ice at start of the year',
            }
            self.output_coords_dict['bin_massbalclim_annual_mad'] = collections.OrderedDict(
                [
                    ('glac', self.glac_values),
                    ('bin', self.bin_values),
                    ('year', self.year_values),
                ]
            )
            self.output_attrs_dict['bin_massbalclim_annual_mad'] = {
                'long_name': 'binned climatic mass balance, in water equivalent, median absolute deviation',
                'units': 'm',
                'temporal_resolution': 'annual',
                'comment': 'climatic mass balance is computed before dynamics so can theoretically exceed ice thickness',
            }


def calc_stats_array(data, stats_cns=pygem_prms['sim']['out']['sim_stats']):
    """
    Calculate stats for a given variable.

    Parameters
    ----------
    data : np.array
        2D array with ensemble simulations (shape: [n_samples, n_ensembles])
    stats_cns : list, optional
        List of statistics to compute (e.g., ['mean', 'std', 'median'])

    Returns
    -------
    stats : np.array, or None
        Statistics related to a given variable.
    """

    # dictionary of functions to call for each stat in `stats_cns`
    stat_funcs = {
        'mean': lambda x: np.nanmean(x, axis=1),
        'std': lambda x: np.nanstd(x, axis=1),
        '2.5%': lambda x: np.nanpercentile(x, 2.5, axis=1),
        '25%': lambda x: np.nanpercentile(x, 25, axis=1),
        'median': lambda x: np.nanmedian(x, axis=1),
        '75%': lambda x: np.nanpercentile(x, 75, axis=1),
        '97.5%': lambda x: np.nanpercentile(x, 97.5, axis=1),
        'mad': lambda x: median_abs_deviation(x, axis=1, nan_policy='omit'),
    }

    # calculate statustics for each stat in `stats_cns`
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)  # Suppress All-NaN Slice Warnings
        stats_list = [stat_funcs[stat](data) for stat in stats_cns if stat in stat_funcs]

    # stack stats_list to numpy array
    return np.column_stack(stats_list) if stats_list else None
