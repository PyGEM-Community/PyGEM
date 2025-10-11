"""
Python Glacier Evolution Model (PyGEM)

copyright © 2018 David Rounce <drounce@cmu.edu

Distributed under the MIT license

class of climate data and functions associated with manipulating the dataset to be in the proper format
"""

import os
import warnings

import numpy as np

# External libraries
import pandas as pd
import xarray as xr

from pygem.setup.config import ConfigManager

# instantiate ConfigManager
config_manager = ConfigManager()
# read the config
pygem_prms = config_manager.read_config()


class GCM:
    """
    Global climate model data properties and functions used to automatically retrieve data.

    Attributes
    ----------
    name : str
        name of climate dataset.
    sim_climate_scenario : str
        rcp or ssp sim_climate_scenario (example: 'rcp26' or 'ssp585')
    realization : str
        realization from large ensemble (example: '1011.001' or '1301.020')
    """

    def __init__(self, name=str(), sim_climate_scenario=str(), realization=None):
        """
        Add variable name and specific properties associated with each gcm.
        """

        if pygem_prms['rgi']['rgi_lon_colname'] not in ['CenLon_360']:
            assert 1 == 0, (
                'Longitude does not use 360 degrees. Check how negative values are handled!'
            )

        # Source of climate data
        self.name = name

        # If multiple realizations from each model+sim_climate_scenario are being used,
        #   then self.realization = realization.
        # Otherwise, the realization attribute is not considered for single
        #   realization model+sim_climate_scenario simulations.
        if realization is not None:
            self.realization = realization

            # Set parameters for CESM2 Large Ensemble
            if self.name == 'smbb.f09_g17.LE2':
                # Standardized CESM2 Large Ensemble format (GCM/SSP)
                # Variable names
                self.temp_vn = 'tas'
                self.prec_vn = 'pr'
                self.elev_vn = 'orog'
                self.lat_vn = 'lat'
                self.lon_vn = 'lon'
                self.time_vn = 'time'
                # Variable filenames
                self.temp_fn = (
                    self.temp_vn
                    + '_mon_'
                    + sim_climate_scenario
                    + '_'
                    + name
                    + '-'
                    + realization
                    + '.cam.h0.1980-2100.nc'
                )
                self.prec_fn = (
                    self.prec_vn
                    + '_mon_'
                    + sim_climate_scenario
                    + '_'
                    + name
                    + '-'
                    + realization
                    + '.cam.h0.1980-2100.nc'
                )
                self.elev_fn = (
                    self.elev_vn
                    + '_fx_'
                    + sim_climate_scenario
                    + '_'
                    + name
                    + '.cam.h0.nc'
                )
                # Variable filepaths
                self.var_fp = (
                    pygem_prms['root']
                    + pygem_prms['climate']['paths']['cesm2_relpath']
                    + sim_climate_scenario
                    + pygem_prms['climate']['paths']['cesm2_fp_var_ending']
                )
                self.fx_fp = (
                    pygem_prms['root']
                    + pygem_prms['climate']['paths']['cesm2_relpath']
                    + sim_climate_scenario
                    + pygem_prms['climate']['paths']['cesm2_fp_fx_ending']
                )
                # Extra information
                # self.timestep = pygem_prms['time']['timestep']
                self.timestep = 'monthly'  # future scenario is always monthly timestep
                self.rgi_lat_colname = pygem_prms['rgi']['rgi_lat_colname']
                self.rgi_lon_colname = pygem_prms['rgi']['rgi_lon_colname']
                self.sim_climate_scenario = sim_climate_scenario

            # Set parameters for GFDL SPEAR Large Ensemble
            elif self.name == 'GFDL-SPEAR-MED':
                # Standardized GFDL SPEAR Large Ensemble format (GCM/SSP)
                # Variable names
                self.temp_vn = 'tas'
                self.prec_vn = 'pr'
                self.elev_vn = 'zsurf'
                self.lat_vn = 'lat'
                self.lon_vn = 'lon'
                self.time_vn = 'time'
                # Variable filenames
                self.temp_fn = (
                    self.temp_vn
                    + '_mon_'
                    + sim_climate_scenario
                    + '_'
                    + name
                    + '-'
                    + realization
                    + 'i1p1f1_gr3_1980-2100.nc'
                )
                self.prec_fn = (
                    self.prec_vn
                    + '_mon_'
                    + sim_climate_scenario
                    + '_'
                    + name
                    + '-'
                    + realization
                    + 'i1p1f1_gr3_1980-2100.nc'
                )
                self.elev_fn = (
                    self.elev_vn + '_fx_' + sim_climate_scenario + '_' + name + '.nc'
                )
                # Variable filepaths
                self.var_fp = (
                    pygem_prms['root']
                    + pygem_prms['climate']['paths']['gfdl_relpath']
                    + sim_climate_scenario
                    + pygem_prms['climate']['paths']['gfdl_fp_var_ending']
                )
                self.fx_fp = (
                    pygem_prms['root']
                    + pygem_prms['climate']['paths']['gfdl_relpath']
                    + sim_climate_scenario
                    + pygem_prms['climate']['paths']['gfdl_fp_fx_ending']
                )
                # Extra information
                # self.timestep = pygem_prms['time']['timestep']
                self.timestep = 'monthly'  # future scenario is always monthly timestep
                self.rgi_lat_colname = pygem_prms['rgi']['rgi_lat_colname']
                self.rgi_lon_colname = pygem_prms['rgi']['rgi_lon_colname']
                self.sim_climate_scenario = sim_climate_scenario

        else:
            self.realization = []

            # Set parameters for ERA5, ERA-Interim, and CMIP5 netcdf files
            if self.name == 'ERA5':
                # Variable names
                self.temp_vn = 't2m'
                self.tempstd_vn = 't2m_std'
                self.prec_vn = 'tp'
                self.elev_vn = 'z'
                self.lat_vn = 'latitude'
                self.lon_vn = 'longitude'
                self.time_vn = 'time'
                self.lr_vn = 'lapserate'
                # Variable filenames
                self.temp_fn = pygem_prms['climate']['paths']['era5_temp_fn']
                self.tempstd_fn = pygem_prms['climate']['paths']['era5_tempstd_fn']
                self.prec_fn = pygem_prms['climate']['paths']['era5_prec_fn']
                self.elev_fn = pygem_prms['climate']['paths']['era5_elev_fn']
                self.lr_fn = pygem_prms['climate']['paths']['era5_lr_fn']
                # Variable filepaths
                if pygem_prms['climate']['paths']['era5_relpath']:
                    self.var_fp = (
                        pygem_prms['root']
                        + pygem_prms['climate']['paths']['era5_relpath']
                    )
                    self.fx_fp = (
                        pygem_prms['root']
                        + pygem_prms['climate']['paths']['era5_relpath']
                    )
                else:
                    self.var_fp = ''
                    self.fx_fp = ''

                # Extra information
                self.timestep = pygem_prms['time']['timestep']
                self.rgi_lat_colname = pygem_prms['rgi']['rgi_lat_colname']
                self.rgi_lon_colname = pygem_prms['rgi']['rgi_lon_colname']

            # Standardized CMIP5 format (GCM/RCP)
            elif 'rcp' in sim_climate_scenario:
                # Variable names
                self.temp_vn = 'tas'
                self.prec_vn = 'pr'
                self.elev_vn = 'orog'
                self.lat_vn = 'lat'
                self.lon_vn = 'lon'
                self.time_vn = 'time'
                # Variable filenames
                self.temp_fn = (
                    self.temp_vn
                    + '_mon_'
                    + name
                    + '_'
                    + sim_climate_scenario
                    + '_r1i1p1_native.nc'
                )
                self.prec_fn = (
                    self.prec_vn
                    + '_mon_'
                    + name
                    + '_'
                    + sim_climate_scenario
                    + '_r1i1p1_native.nc'
                )
                self.elev_fn = (
                    self.elev_vn
                    + '_fx_'
                    + name
                    + '_'
                    + sim_climate_scenario
                    + '_r0i0p0.nc'
                )
                # Variable filepaths
                self.var_fp = (
                    pygem_prms['root']
                    + pygem_prms['climate']['paths']['cmip5_relpath']
                    + sim_climate_scenario
                    + pygem_prms['climate']['paths']['cmip5_fp_var_ending']
                )
                self.fx_fp = (
                    pygem_prms['root']
                    + pygem_prms['climate']['paths']['cmip5_relpath']
                    + sim_climate_scenario
                    + pygem_prms['climate']['paths']['cmip5_fp_fx_ending']
                )
                if not os.path.exists(self.var_fp) and os.path.exists(
                    pygem_prms['climate']['paths']['cmip5_relpath'] + name + '/'
                ):
                    self.var_fp = (
                        pygem_prms['root']
                        + pygem_prms['climate']['paths']['cmip5_relpath']
                        + name
                        + '/'
                    )
                if not os.path.exists(self.fx_fp) and os.path.exists(
                    pygem_prms['climate']['paths']['cmip5_relpath'] + name + '/'
                ):
                    self.fx_fp = (
                        pygem_prms['root']
                        + pygem_prms['climate']['paths']['cmip5_relpath']
                        + name
                        + '/'
                    )
                # Extra information
                # self.timestep = pygem_prms['time']['timestep']
                self.timestep = 'monthly'  # future scenario is always monthly timestep
                self.rgi_lat_colname = pygem_prms['rgi']['rgi_lat_colname']
                self.rgi_lon_colname = pygem_prms['rgi']['rgi_lon_colname']
                self.sim_climate_scenario = sim_climate_scenario

            # Standardized CMIP6 format (GCM/SSP)
            elif 'ssp' in sim_climate_scenario:
                # Variable names
                self.temp_vn = 'tas'
                self.prec_vn = 'pr'
                self.elev_vn = 'orog'
                self.lat_vn = 'lat'
                self.lon_vn = 'lon'
                self.time_vn = 'time'
                # Variable filenames
                self.temp_fn = (
                    name
                    + '_'
                    + sim_climate_scenario
                    + '_r1i1p1f1_'
                    + self.temp_vn
                    + '.nc'
                )
                self.prec_fn = (
                    name
                    + '_'
                    + sim_climate_scenario
                    + '_r1i1p1f1_'
                    + self.prec_vn
                    + '.nc'
                )
                self.elev_fn = name + '_' + self.elev_vn + '.nc'
                # Variable filepaths
                self.var_fp = (
                    pygem_prms['root']
                    + pygem_prms['climate']['paths']['cmip6_relpath']
                    + name
                    + '/'
                )
                self.fx_fp = (
                    pygem_prms['root']
                    + pygem_prms['climate']['paths']['cmip6_relpath']
                    + name
                    + '/'
                )
                # Extra information
                # self.timestep = pygem_prms['time']['timestep']
                self.timestep = 'monthly'  # future scenario is always monthly timestep
                self.rgi_lat_colname = pygem_prms['rgi']['rgi_lat_colname']
                self.rgi_lon_colname = pygem_prms['rgi']['rgi_lon_colname']
                self.sim_climate_scenario = sim_climate_scenario

    def importGCMfxnearestneighbor_xarray(self, filename, vn, main_glac_rgi):
        """
        Import time invariant (constant) variables and extract nearest neighbor.

        Note: cmip5 data used surface height, while ERA5 data is geopotential

        Parameters
        ----------
        filename : str
            filename of variable
        variablename : str
            variable name
        main_glac_rgi : pandas dataframe
            dataframe containing relevant rgi glacier information

        Returns
        -------
        glac_variable : numpy array
            array of nearest neighbor values for all the glaciers in model run (rows=glaciers, column=variable)
        """
        # Import netcdf file
        data = xr.open_dataset(self.fx_fp + filename)
        glac_variable = np.zeros(main_glac_rgi.shape[0])
        # If time dimension included, then set the time index (required for ERA Interim, but not for CMIP5 or COAWST)
        if 'time' in data[vn].coords:
            time_idx = 0
            #  ERA Interim has only 1 value of time, so index is 0
        # Find Nearest Neighbor
        if self.name == 'COAWST':
            for glac in range(main_glac_rgi.shape[0]):
                latlon_dist = (
                    (
                        data[self.lat_vn].values
                        - main_glac_rgi[self.rgi_lat_colname].values[glac]
                    )
                    ** 2
                    + (
                        data[self.lon_vn].values
                        - main_glac_rgi[self.rgi_lon_colname].values[glac]
                    )
                    ** 2
                ) ** 0.5
                latlon_nearidx = [
                    x[0] for x in np.where(latlon_dist == latlon_dist.min())
                ]
                lat_nearidx = latlon_nearidx[0]
                lon_nearidx = latlon_nearidx[1]
                glac_variable[glac] = data[vn][
                    latlon_nearidx[0], latlon_nearidx[1]
                ].values
        else:
            #  argmin() finds the minimum distance between the glacier lat/lon and the GCM pixel
            lat_nearidx = np.abs(
                main_glac_rgi[self.rgi_lat_colname].values[:, np.newaxis]
                - data.variables[self.lat_vn][:].values
            ).argmin(axis=1)
            lon_nearidx = np.abs(
                main_glac_rgi[self.rgi_lon_colname].values[:, np.newaxis]
                - data.variables[self.lon_vn][:].values
            ).argmin(axis=1)

            latlon_nearidx = list(zip(lat_nearidx, lon_nearidx))
            latlon_nearidx_unique = list(set(latlon_nearidx))

            glac_variable_dict = {}
            for latlon in latlon_nearidx_unique:
                try:
                    glac_variable_dict[latlon] = data[vn][
                        time_idx, latlon[0], latlon[1]
                    ].values
                except:
                    glac_variable_dict[latlon] = data[vn][latlon[0], latlon[1]].values

            glac_variable = np.array([glac_variable_dict[x] for x in latlon_nearidx])

        # Correct units if necessary (CMIP5 already in m a.s.l., ERA Interim is geopotential [m2 s-2])
        if vn == self.elev_vn:
            # If the variable has units associated with geopotential, then convert to m.a.s.l (ERA Interim)
            if 'units' in data[vn].attrs and (data[vn].attrs['units'] == 'm**2 s**-2'):
                # Convert m2 s-2 to m by dividing by gravity (ERA Interim states to use 9.80665)
                glac_variable = glac_variable / 9.80665
            # Elseif units already in m.a.s.l., then continue
            elif 'units' in data[vn].attrs and data[vn].attrs['units'] == 'm':
                pass
            # Otherwise, provide warning
            else:
                print('Check units of elevation from GCM is m.')

        return glac_variable

    def importGCMvarnearestneighbor_xarray(
        self,
        filename,
        vn,
        main_glac_rgi,
        dates_table,
        realizations=['r1i1p1f1', 'r4i1p1f1'],
        upscale_var_timestep=False,
        verbose=False,
    ):
        """
        Import time series of variables and extract nearest neighbor.

        Note: "NG" refers to a homogenized "new generation" of products from ETH-Zurich.
              The function is setup to select netcdf data using the dimensions: time, latitude, longitude (in that
              order). Prior to running the script, the user must check that this is the correct order of the dimensions
              and the user should open the netcdf file to determine the names of each dimension as they may vary.

        Parameters
        ----------
        filename : str
            filename of variable
        vn : str
            variable name
        main_glac_rgi : pandas dataframe
            dataframe containing relevant rgi glacier information
        dates_table: pandas dataframe
            dataframe containing dates of model run

        Returns
        -------
        glac_variable_series : numpy array
            array of nearest neighbor values for all the glaciers in model run (rows=glaciers, columns=time series)
        time_series : numpy array
            array of dates associated with the meteorological data (may differ slightly from those in table for monthly
            timestep, i.e., be from the beginning/middle/end of month)
        """
        # Import netcdf file
        if self.timestep == 'monthly':
            if not os.path.exists(self.var_fp + filename):
                if os.path.exists(
                    self.var_fp + filename.replace('r1i1p1f1', 'r4i1p1f1')
                ):
                    filename = filename.replace('r1i1p1f1', 'r4i1p1f1')
                if os.path.exists(self.var_fp + filename.replace('_native', '')):
                    filename = filename.replace('_native', '')

            data = xr.open_dataset(self.var_fp + filename)
        elif self.timestep == 'daily':
            year_start = pd.Timestamp(dates_table['date'].values[0]).year
            year_end = pd.Timestamp(dates_table['date'].values[-1]).year

            lats = main_glac_rgi[self.rgi_lat_colname].values
            lons = main_glac_rgi[self.rgi_lon_colname].values
            # define lat/lon window around all glaciers in run (bounds expanded to nearest 0.25 degrees)
            min_lat, max_lat = np.floor(lats.min() * 4) / 4, np.ceil(lats.max() * 4) / 4
            min_lon, max_lon = np.floor(lons.min() * 4) / 4, np.ceil(lons.max() * 4) / 4
            if 'YYYY' in filename:
                datasets = []
                for yr in range(year_start, year_end + 1):
                    data_yr = xr.open_dataset(
                        self.var_fp + filename.replace('YYYY', str(yr))
                    )
                    if 'valid_time' in data_yr.coords or 'valid_time' in data_yr.dims:
                        data_yr = data_yr.rename({'valid_time': self.time_vn})

                    # convert longitude from -180—180 to 0—360
                    if data_yr.longitude.min() < 0:
                        data_yr = data_yr.assign_coords(
                            longitude=(data_yr.longitude % 360)
                        )

                    # subset for desired lats and lons
                    data_yr = data_yr.sel(
                        latitude=slice(max_lat, min_lat),
                        longitude=slice(min_lon, max_lon),
                    )

                    # append data to list
                    datasets.append(data_yr)

                # combine along the time dimension
                data = xr.concat(datasets, dim=self.time_vn)

            else:
                data = xr.open_dataset(self.var_fp + filename)
                data = data.sel(
                    latitude=slice(max_lat, min_lat), longitude=slice(min_lon, max_lon)
                )

            # mask out leap days
            if pygem_prms['time']['option_leapyear'] == 0 and not upscale_var_timestep:
                time_index = pd.to_datetime(data[self.time_vn].values)
                mask = ~((time_index.month == 2) & (time_index.day == 29))
                data = data.isel({self.time_vn: mask})

            # Upscale timestep to match data (e.g., monthly lapserate for daily data)
            if upscale_var_timestep:
                # convert time to datetime
                time_monthly = pd.to_datetime(data[self.time_vn].values)
                var_monthly = data[vn]  # xarray DataArray with dims (time, lat, lon)

                # create empty DataArray for daily data
                daily_times = dates_table['date'].values
                daily_data = xr.DataArray(
                    np.zeros(
                        (len(daily_times), len(data.latitude), len(data.longitude))
                    ),
                    dims=(self.time_vn, 'latitude', 'longitude'),
                    coords={
                        self.time_vn: daily_times,
                        'latitude': data.latitude,
                        'longitude': data.longitude,
                    },
                    name=vn,
                )

                # loop through months and fill daily slots
                for i, t in enumerate(time_monthly):
                    # find all days in this month
                    idx = np.where(
                        (dates_table['year'] == t.year)
                        & (dates_table['month'] == t.month)
                    )[0]

                    # assign monthly values to these daily indices
                    daily_data[idx, :, :] = var_monthly.isel(time=i).values

                # convert to Dataset with data variable vn
                data = daily_data.to_dataset(name=vn)

        glac_variable_series = np.zeros((main_glac_rgi.shape[0], dates_table.shape[0]))

        # Check GCM provides required years of data
        years_check = pd.Series(data['time']).apply(lambda x: int(x.strftime('%Y')))
        assert years_check.max() >= dates_table.year.max(), (
            self.name + ' does not provide data out to ' + str(dates_table.year.max())
        )
        assert years_check.min() <= dates_table.year.min(), (
            self.name + ' does not provide data back to ' + str(dates_table.year.min())
        )

        # Determine the correct time indices
        if self.timestep == 'monthly':
            start_idx = (
                np.where(
                    pd.Series(data[self.time_vn]).apply(lambda x: x.strftime('%Y-%m'))
                    == dates_table['date'].apply(lambda x: x.strftime('%Y-%m'))[0]
                )
            )[0][0]
            end_idx = (
                np.where(
                    pd.Series(data[self.time_vn]).apply(lambda x: x.strftime('%Y-%m'))
                    == dates_table['date'].apply(lambda x: x.strftime('%Y-%m'))[
                        dates_table.shape[0] - 1
                    ]
                )
            )[0][0]
            #  np.where finds the index position where to values are equal
            #  pd.Series(data.variables[gcm_time_varname]) creates a pandas series of the time variable associated with
            #  the netcdf
            #  .apply(lambda x: x.strftime('%Y-%m')) converts the timestamp to a string with YYYY-MM to enable the
            #  comparison
            #    > different climate dta can have different date formats, so this standardization for comparison is
            #      important
            #      ex. monthly data may provide date on 1st of month or middle of month, so YYYY-MM-DD would not work
            #  The same processing is done for the dates_table['date'] to facilitate the comparison
            #  [0] is used to access the first date
            #  dates_table.shape[0] - 1 is used to access the last date
            #  The final indexing [0][0] is used to access the value, which is inside of an array containing extraneous
            #  information
        elif self.timestep == 'daily':
            start_idx = (
                np.where(
                    pd.Series(data[self.time_vn]).apply(
                        lambda x: x.strftime('%Y-%m-%d')
                    )
                    == dates_table['date']
                    .apply(lambda x: x.strftime('%Y-%m-%d'))
                    .iloc[0]
                )
            )[0][0]
            end_idx = (
                np.where(
                    pd.Series(data[self.time_vn]).apply(
                        lambda x: x.strftime('%Y-%m-%d')
                    )
                    == dates_table['date']
                    .apply(lambda x: x.strftime('%Y-%m-%d'))
                    .iloc[-1]
                )
            )[0][0]

        # Extract the time series
        time_series = pd.Series(data[self.time_vn][start_idx : end_idx + 1])
        # Find Nearest Neighbor
        if self.name == 'COAWST':
            for glac in range(main_glac_rgi.shape[0]):
                latlon_dist = (
                    (
                        data[self.lat_vn].values
                        - main_glac_rgi[self.rgi_lat_colname].values[glac]
                    )
                    ** 2
                    + (
                        data[self.lon_vn].values
                        - main_glac_rgi[self.rgi_lon_colname].values[glac]
                    )
                    ** 2
                ) ** 0.5
                latlon_nearidx = [
                    x[0] for x in np.where(latlon_dist == latlon_dist.min())
                ]
                lat_nearidx = latlon_nearidx[0]
                lon_nearidx = latlon_nearidx[1]
                glac_variable_series[glac, :] = data[vn][
                    start_idx : end_idx + 1, latlon_nearidx[0], latlon_nearidx[1]
                ].values
        else:
            #  argmin() finds the minimum distance between the glacier lat/lon and the GCM pixel; .values is used to
            #  extract the position's value as opposed to having an array
            lat_nearidx = np.abs(
                main_glac_rgi[self.rgi_lat_colname].values[:, np.newaxis]
                - data.variables[self.lat_vn][:].values
            ).argmin(axis=1)
            lon_nearidx = np.abs(
                main_glac_rgi[self.rgi_lon_colname].values[:, np.newaxis]
                - data.variables[self.lon_vn][:].values
            ).argmin(axis=1)
            # Find unique latitude/longitudes
            latlon_nearidx = list(zip(lat_nearidx, lon_nearidx))
            latlon_nearidx_unique = list(set(latlon_nearidx))

            # Create dictionary of time series for each unique latitude/longitude
            glac_variable_dict = {}
            for latlon in latlon_nearidx_unique:
                if 'expver' in data.keys():
                    expver_idx = 0
                    glac_variable_dict[latlon] = data[vn][
                        start_idx : end_idx + 1, expver_idx, latlon[0], latlon[1]
                    ].values
                else:
                    glac_variable_dict[latlon] = data[vn][
                        start_idx : end_idx + 1, latlon[0], latlon[1]
                    ].values

            # Check all glacier use appropriate climate data
            for i, latlon in enumerate(latlon_nearidx):
                rgi_id = main_glac_rgi[
                    pygem_prms['rgi']['rgi_glacno_float_colname']
                ].values[i]
                if (len(data[vn][self.lat_vn].values) == 1) or (
                    len(data[vn][self.lon_vn].values) == 1
                ):
                    if verbose:
                        warnings.warn(
                            f'{vn} data has only one latitude or longitude value; check that the correct data is being used',
                            Warning,
                            stacklevel=2,
                        )
                else:
                    lat_res = abs(np.diff(data[vn][self.lat_vn].values)[0])
                    lon_res = abs(np.diff(data[vn][self.lon_vn].values)[0])
                    lat_dd = abs(
                        main_glac_rgi[self.rgi_lat_colname].values[i]
                        - data[vn][self.lat_vn].values[latlon[0]]
                    )
                    lon_dd = abs(
                        main_glac_rgi[self.rgi_lon_colname].values[i]
                        - data[vn][self.lon_vn].values[latlon[1]]
                    )

                    assert lat_dd <= lat_res and lon_dd <= lon_res, (
                        f'Climate data pixel for {vn} too from glacier {rgi_id}: Δlat={lat_dd:.3f}, '
                        + f'Δlon={lon_dd:.3f}, res=({lat_res:.3f}, {lon_res:.3f})'
                    )

            # Convert to series
            glac_variable_series = np.array(
                [glac_variable_dict[x] for x in latlon_nearidx]
            )

        # Perform corrections to the data if necessary
        # Surface air temperature corrections
        if vn in ['tas', 't2m', 'T2']:
            if 'units' in data[vn].attrs and data[vn].attrs['units'] == 'K':
                # Convert from K to deg C
                glac_variable_series = glac_variable_series - 273.15
            else:
                print('Check units of air temperature from GCM is degrees C.')
        elif vn in ['t2m_std']:
            if 'units' in data[vn].attrs and data[vn].attrs['units'] not in ['C', 'K']:
                print(
                    'Check units of air temperature standard deviation from GCM is degrees C or K'
                )
        # Precipitation corrections
        # If the variable is precipitation
        elif vn in ['pr', 'tp', 'TOTPRECIP']:
            # If the variable has units and those units are meters (ERA Interim)
            if 'units' in data[vn].attrs and data[vn].attrs['units'] == 'm':
                pass
            # Elseif the variable has units and those units are kg m-2 s-1 (CMIP5/CMIP6)
            elif 'units' in data[vn].attrs and data[vn].attrs['units'] == 'kg m-2 s-1':
                # Convert from kg m-2 s-1 to m day-1
                glac_variable_series = glac_variable_series / 1000 * 3600 * 24
                #   (1 kg m-2 s-1) * (1 m3/1000 kg) * (3600 s / hr) * (24 hr / day) = (m day-1)
            # Elseif the variable has units and those units are mm (COAWST)
            elif 'units' in data[vn].attrs and data[vn].attrs['units'] == 'mm':
                glac_variable_series = glac_variable_series / 1000
            # Else check the variables units
            else:
                print('Check units of precipitation from GCM is meters per day.')
            if self.timestep == 'monthly' and self.name != 'COAWST':
                # Convert from meters per day to meters per month (COAWST data already 'monthly accumulated precipitation')
                if 'daysinmonth' in dates_table.columns:
                    glac_variable_series = (
                        glac_variable_series
                        * dates_table['daysinmonth'].values[np.newaxis, :]
                    )
        elif vn != self.lr_vn:
            print('Check units of air temperature or precipitation')

        return glac_variable_series, time_series
