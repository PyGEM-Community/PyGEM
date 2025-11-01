"""
Python Glacier Evolution Model (PyGEM)

copyright © 2025 Brandon Tober <btober@cmu.edu>, David Rounce <drounce@cmu.edu>

Distributed under the MIT license
"""

# Built-in libaries
import datetime
import logging
import os

import numpy as np
import pandas as pd

# External libraries
# Local libraries
from oggm import cfg
from oggm.utils import entity_task

# pygem imports
from pygem.setup.config import ConfigManager

# instantiate ConfigManager
config_manager = ConfigManager()
# read the config
pygem_prms = config_manager.read_config()

# Module logger
log = logging.getLogger(__name__)

# Add "meltextent_1d", "snowline_1d", and "scaf_1d" to the list of things that the GlacierDirectory understands
if 'meltextent_1d' not in cfg.BASENAMES:
    cfg.BASENAMES['meltextent_1d'] = (
        'meltextent_1d.json',
        '1D melt extent data',
    )
if 'snowline_1d' not in cfg.BASENAMES:
    cfg.BASENAMES['snowline_1d'] = (
        'snowline_1d.json',
        '1D snowline data',
    )
if 'scaf_1d' not in cfg.BASENAMES:
    cfg.BASENAMES['scaf_1d'] = (
        'scaf_1d.json',
        '1D snow cover area fraction (SCAF) data',
    )


@entity_task(log, writes=['meltextent_1d'])
def meltextent_1d_to_gdir(
    gdir,
):
    """
    Add 1d melt extent observations to the given glacier directory

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        where to write the data

    expected csv structure:
        Columns: 'date', 'z', 'z_min', 'z_max', 'direction'
            'date':         Observation date, stored as a string in 'YYYY-MM-DD' format
            'z':            Melt extent elevation (meters)
            'z_min':        Melt extent elevation minimum (meters)
            'z_max':        Melt extent elevation maximum (meters)
            'direction':    SAR path direction, stored as a string (e.g., 'ascending' or 'descending')
            'ref_dem':      Reference DEM used for elevation values
            'ref_dem_year': Reference DEM year for elevation value of observations (m a.s.l.) (e.g., 2013 if using COP30)
    """
    # get dataset file path
    meltextent_1d_fp = (
        f'{pygem_prms["root"]}/'
        f'{pygem_prms["calib"]["data"]["meltextent_1d"]["meltextent_1d_relpath"]}/'
        f'{gdir.rgi_id.split("-")[1]}_melt_extent_elev.csv'
    )

    # check for file
    if os.path.exists(meltextent_1d_fp):
        meltextent_1d_df = pd.read_csv(meltextent_1d_fp)
    else:
        log.debug('No melt extent data to load, skipping task.')
        raise Warning('No melt extent data to load')  # file not found, skip

    validate_meltextent_1d_structure(meltextent_1d_df)
    meltextent_1d_dict = meltextent_csv_to_dict(meltextent_1d_df)
    gdir.write_json(meltextent_1d_dict, 'meltextent_1d')


def validate_meltextent_1d_structure(data):
    """Validate that meltextent_1d CSV structure matches expected format."""

    required_cols = [
        'date',
        'z',
        'z_min',
        'z_max',
        'direction',
        'ref_dem',
        'ref_dem_year',
    ]
    for col in required_cols:
        if col not in data.columns:
            raise ValueError(f"Missing required column '{col}' in melt extent CSV.")

    # Validate dates
    dates = data['date']
    if not isinstance(dates, pd.Series) or len(dates) == 0:
        raise ValueError("'dates' must be a non-empty series.")
    for i, date_str in enumerate(dates):
        try:
            datetime.datetime.strptime(date_str, '%Y-%m-%d')
        except ValueError:
            raise ValueError(f"Invalid date format in 'dates[{i}]': {date_str}") from None

    # Validate z
    z = data['z']
    if not (isinstance(z, pd.Series) and len(z) == len(dates)):
        raise ValueError(f"'z' must be a series of length {len(dates)}.")
    if not all(isinstance(x, (int, float)) for x in z):
        raise ValueError("All 'z' values must be numeric.")

    # Validate z_min
    z_min = data['z_min']
    if not (isinstance(z_min, pd.Series) and len(z_min) == len(dates)):
        raise ValueError(f"'z_min' must be a series of length {len(dates)}.")
    if not all(isinstance(x, (int, float)) for x in z_min):
        raise ValueError("All 'z_min' values must be numeric.")

    # Validate z_max
    z_max = data['z_max']
    if not (isinstance(z_max, pd.Series) and len(z_max) == len(dates)):
        raise ValueError(f"'z_max' must be a series of length {len(dates)}.")
    if not all(isinstance(x, (int, float)) for x in z_max):
        raise ValueError("All 'z_max' values must be numeric.")

    # Validate direction
    direction = data['direction']
    if not (isinstance(direction, pd.Series) and len(direction) == len(dates)):
        raise ValueError(f"'direction' must be a series of length {len(dates)}.")
    if not all(isinstance(x, str) for x in direction):
        raise ValueError("All 'direction' values must be strings.")

    # Validate reference DEM
    ref_dem = data['ref_dem'].dropna().unique()
    for ref_dem_i in ref_dem:
        if not isinstance(ref_dem_i, (str)):
            raise TypeError(f"All 'ref_dem' values must be an string, but got {ref_dem_i} ({type(ref_dem_i).__name__}).")

    # Validate reference DEM year
    dem_year = data['ref_dem_year'].dropna().unique()
    for dem_year_i in dem_year:
        if not isinstance(dem_year_i, (int, np.integer)):
            raise TypeError(f"All 'ref_dem_year' must be integers, but got {dem_year_i} ({type(dem_year_i).__name__}).")

    return True


def meltextent_csv_to_dict(data):
    """Convert meltextent_1d CSV to JSON for OGGM ingestion."""
    dates = data['date'].astype(str).tolist()
    z = data['z'].astype(float).tolist()
    z_min = data['z_min'].astype(float).tolist()
    z_max = data['z_max'].astype(float).tolist()
    direction = data['direction'].astype(str).tolist()
    ref_dem = data['ref_dem'].astype(str).tolist()[0]
    ref_dem_year = data['ref_dem_year'].astype(int).tolist()[0]

    data_dict = {
        'date': dates,
        'z': z,
        'z_min': z_min,
        'z_max': z_max,
        'direction': direction,
        'ref_dem': ref_dem,
        'ref_dem_year': ref_dem_year,
    }
    return data_dict


@entity_task(log, writes=['snowline_1d'])
def snowline_1d_to_gdir(
    gdir,
):
    """
    Add 1d snowline observations to the given glacier directory

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        where to write the data

    expected csv structure:
        Columns: 'date', 'z', 'z_min', 'z_max', 'direction'
            'date':         Observation date, stored as a string in 'YYYY-MM-DD' format
            'z':            Snowline elevation (m a.s.l.)
            'z_min':        Snowline elevation minimum (m a.s.l.)
            'z_max':        Snowline elevation maximum (m a.s.l.)
            'direction':    SAR path direction, stored as a string (e.g., 'ascending' or 'descending')
            'ref_dem':      Reference DEM used for elevation values
            'ref_dem_year': Reference DEM year for elevation value of observations (m a.s.l.) (e.g., 2013 if using COP30)
    """
    # get dataset file path
    snowline_1d_fp = (
        f'{pygem_prms["root"]}/'
        f'{pygem_prms["calib"]["data"]["snowline_1d"]["snowline_1d_relpath"]}/'
        f'{gdir.rgi_id.split("-")[1]}_snowline_elev.csv'
    )

    # check for file
    if os.path.exists(snowline_1d_fp):
        snowline_1d_df = pd.read_csv(snowline_1d_fp)
    else:
        log.debug('No snowline data to load, skipping task.')
        raise Warning('No snowline data to load')  # file not found, skip

    validate_snowline_1d_structure(snowline_1d_df)
    snowline_1d_dict = snowline_csv_to_dict(snowline_1d_df)
    gdir.write_json(snowline_1d_dict, 'snowline_1d')


def validate_snowline_1d_structure(data):
    """Validate that snowline_1d CSV structure matches expected format."""

    required_cols = [
        'date',
        'z',
        'z_min',
        'z_max',
        'direction',
        'ref_dem',
        'ref_dem_year',
    ]
    for col in required_cols:
        if col not in data.columns:
            raise ValueError(f"Missing required column '{col}' in snowline CSV.")

    # Validate dates
    dates = data['date']
    if not isinstance(dates, pd.Series) or len(dates) == 0:
        raise ValueError("'dates' must be a non-empty series.")
    for i, date_str in enumerate(dates):
        try:
            datetime.datetime.strptime(date_str, '%Y-%m-%d')
        except ValueError:
            raise ValueError(f"Invalid date format in 'dates[{i}]': {date_str}") from None

    # Validate z
    z = data['z']
    if not (isinstance(z, pd.Series) and len(z) == len(dates)):
        raise ValueError(f"'z' must be a series of length {len(dates)}.")
    if not all(isinstance(x, (int, float)) for x in z):
        raise ValueError("All 'z' values must be numeric.")

    # Validate z_min
    z_min = data['z_min']
    if not (isinstance(z_min, pd.Series) and len(z_min) == len(dates)):
        raise ValueError(f"'z_min' must be a series of length {len(dates)}.")
    if not all(isinstance(x, (int, float)) for x in z_min):
        raise ValueError("All 'z_min' values must be numeric.")

    # Validate z_max
    z_max = data['z_max']
    if not (isinstance(z_max, pd.Series) and len(z_max) == len(dates)):
        raise ValueError(f"'z_max' must be a series of length {len(dates)}.")
    if not all(isinstance(x, (int, float)) for x in z_max):
        raise ValueError("All 'z_max' values must be numeric.")

    # Validate direction
    direction = data['direction']
    if not (isinstance(direction, pd.Series) and len(direction) == len(dates)):
        raise ValueError(f"'direction' must be a series of length {len(dates)}.")
    if not all(isinstance(x, str) for x in direction):
        raise ValueError("All 'direction' values must be strings.")

    # Validate reference DEM
    ref_dem = data['ref_dem'].dropna().unique()
    for ref_dem_i in ref_dem:
        if not isinstance(ref_dem_i, (str)):
            raise TypeError(f"All 'ref_dem' values must be an string, but got {ref_dem_i} ({type(ref_dem_i).__name__}).")

    # Validate reference DEM year
    dem_year = data['ref_dem_year'].dropna().unique()
    for dem_year_i in dem_year:
        if not isinstance(dem_year_i, (int, np.integer)):
            raise TypeError(f"All 'ref_dem_year' must be integers, but got {dem_year_i} ({type(dem_year_i).__name__}).")

    return True


def snowline_csv_to_dict(data):
    """Convert snowline_1d CSV to JSON for OGGM ingestion."""
    dates = data['date'].astype(str).tolist()
    z = data['z'].astype(float).tolist()
    z_min = data['z_min'].astype(float).tolist()
    z_max = data['z_max'].astype(float).tolist()
    direction = data['direction'].astype(str).tolist()
    ref_dem = data['ref_dem'].astype(str).tolist()[0]
    ref_dem_year = data['ref_dem_year'].astype(int).tolist()[0]

    data_dict = {
        'date': dates,
        'z': z,
        'z_min': z_min,
        'z_max': z_max,
        'direction': direction,
        'ref_dem': ref_dem,
        'ref_dem_year': ref_dem_year,
    }
    return data_dict

@entity_task(log, writes=['scaf_1d'])
def scaf_1d_to_gdir(
    gdir,
):
    """
    Add 1d snow cover area fraction (SCAF) observations to the given glacier directory

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        where to write the data

    expected csv structure:
        Columns: 'date', 'scaf', 'scaf_min', 'scaf_max', 'direction'
            'date':         Observation date, stored as a string in 'YYYY-MM-DD' format
            'scaf':         Snow cover area fraction (-) [note: 1 is glacier covered fully in snow]
            'scaf_min':     Snow cover area fraction minimum (-)
            'scaf_max':     Snow cover area fraction maximum (-)
            'direction':    SAR path direction, stored as a string (e.g., 'ascending' or 'descending')
            'ref_area':      Reference area used for SCAF values
            'ref_area_year': Reference area year for SCAF value of observations (-) (e.g., 2000 if using RGI)
    """
    # get dataset file path
    scaf_1d_fp = (
        f'{pygem_prms["root"]}/'
        f'{pygem_prms["calib"]["data"]["scaf_1d"]["scaf_1d_relpath"]}/'
        f'{gdir.rgi_id.split("-")[1]}_snowline_scaf.csv'
    )

    # check for file
    if os.path.exists(scaf_1d_fp):
        scaf_1d_df = pd.read_csv(scaf_1d_fp)
    else:
        log.debug('No snow cover area fraction (SCAF) data to load, skipping task.')
        raise Warning('No snow cover area fraction (SCAF) data to load')  # file not found, skip

    validate_scaf_1d_structure(scaf_1d_df)
    scaf_1d_dict = scaf_csv_to_dict(scaf_1d_df)
    gdir.write_json(scaf_1d_dict, 'scaf_1d')


def validate_scaf_1d_structure(data):
    """Validate that scaf_1d CSV structure matches expected format."""

    required_cols = [
        'date',
        'scaf',
        'scaf_min',
        'scaf_max',
        'direction',
        'ref_area',
        'ref_area_year',
    ]
    for col in required_cols:
        if col not in data.columns:
            raise ValueError(f"Missing required column '{col}' in snow cover area fraction (SCAF) CSV.")

    # Validate dates
    dates = data['date']
    if not isinstance(dates, pd.Series) or len(dates) == 0:
        raise ValueError("'dates' must be a non-empty series.")
    for i, date_str in enumerate(dates):
        try:
            datetime.datetime.strptime(date_str, '%Y-%m-%d')
        except ValueError:
            raise ValueError(f"Invalid date format in 'dates[{i}]': {date_str}") from None

    # Validate scaf
    scaf = data['scaf']
    if not (isinstance(scaf, pd.Series) and len(scaf) == len(dates)):
        raise ValueError(f"'scaf' must be a series of length {len(dates)}.")
    if not all(isinstance(x, (int, float)) for x in scaf):
        raise ValueError("All 'scaf' values must be numeric.")

    # Validate scaf_min
    scaf_min = data['scaf_min']
    if not (isinstance(scaf_min, pd.Series) and len(scaf_min) == len(dates)):
        raise ValueError(f"'scaf_min' must be a series of length {len(dates)}.")
    if not all(isinstance(x, (int, float)) for x in scaf_min):
        raise ValueError("All 'scaf_min' values must be numeric.")

    # Validate scaf_max
    scaf_max = data['scaf_max']
    if not (isinstance(scaf_max, pd.Series) and len(scaf_max) == len(dates)):
        raise ValueError(f"'scaf_max' must be a series of length {len(dates)}.")
    if not all(isinstance(x, (int, float)) for x in scaf_max):
        raise ValueError("All 'scaf_max' values must be numeric.")

    # Validate direction
    direction = data['direction']
    if not (isinstance(direction, pd.Series) and len(direction) == len(dates)):
        raise ValueError(f"'direction' must be a series of length {len(dates)}.")
    if not all(isinstance(x, str) for x in direction):
        raise ValueError("All 'direction' values must be strings.")

    # Validate reference DEM
    ref_area = data['ref_area'].dropna().unique()
    for ref_area_i in ref_area:
        if not isinstance(ref_area_i, (str)):
            raise TypeError(f"All 'ref_area' values must be an string, but got {ref_area_i} ({type(ref_area_i).__name__}).")

    # Validate reference DEM year
    area_year = data['ref_area_year'].dropna().unique()
    for area_year_i in area_year:
        if not isinstance(area_year_i, (int, np.integer)):
            raise TypeError(f"All 'ref_area_year' must be integers, but got {area_year_i} ({type(area_year_i).__name__}).")

    return True


def scaf_csv_to_dict(data):
    """Convert scaf_1d CSV to JSON for OGGM ingestion."""
    dates = data['date'].astype(str).tolist()
    scaf = data['scaf'].astype(float).tolist()
    scaf_min = data['scaf_min'].astype(float).tolist()
    scaf_max = data['scaf_max'].astype(float).tolist()
    direction = data['direction'].astype(str).tolist()
    ref_area = data['ref_area'].astype(str).tolist()[0]
    ref_area_year = data['ref_area_year'].astype(int).tolist()[0]

    data_dict = {
        'date': dates,
        'scaf': scaf,
        'scaf_min': scaf_min,
        'scaf_max': scaf_max,
        'direction': direction,
        'ref_area': ref_area,
        'ref_area_year': ref_area_year,
    }
    return data_dict
