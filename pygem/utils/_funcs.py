"""
Python Glacier Evolution Model (PyGEM)

copyright © 2018 David Rounce <drounce@cmu.edu>

Distributed under the MIT license

Functions that didn't fit into other modules
"""

import argparse
import json

import numpy as np

from pygem.setup.config import ConfigManager

# instantiate ConfigManager
config_manager = ConfigManager()
# read the config
pygem_prms = config_manager.read_config()


def str2bool(v):
    """
    Convert a string to a boolean.

    Accepts: "yes", "true", "t", "1" → True;
             "no", "false", "f", "0" → False.

    Raises an error if input is unrecognized.
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', '1', 'y'):
        return True
    elif v.lower() in ('no', 'false', 'f', '0', 'n'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def annualweightedmean_array(var, dates_table):
    """
    Calculate annual mean of variable according to the timestep.

    Monthly timestep will group every 12 months, so starting month is important.

    Parameters
    ----------
    var : np.ndarray
        Variable with monthly or daily timestep
    dates_table : pd.DataFrame
        Table of dates, year, month, days_in_step, wateryear, and season for each timestep
    Returns
    -------
    var_annual : np.ndarray
        Annual weighted mean of variable
    """
    if pygem_prms['time']['timestep'] == 'monthly':
        dayspermonth = dates_table['days_in_step'].values.reshape(-1, 12)
        #  creates matrix (rows-years, columns-months) of the number of days per month
        daysperyear = dayspermonth.sum(axis=1)
        #  creates an array of the days per year (includes leap years)
        weights = (dayspermonth / daysperyear[:, np.newaxis]).reshape(-1)
        #  computes weights for each element, then reshapes it from matrix (rows-years, columns-months) to an array,
        #  where each column (each monthly timestep) is the weight given to that specific month
        var_annual = (
            (var * weights[np.newaxis, :])
            .reshape(-1, 12)
            .sum(axis=1)
            .reshape(-1, daysperyear.shape[0])
        )
        #  computes matrix (rows - bins, columns - year) of weighted average for each year
        #  explanation: var*weights[np.newaxis,:] multiplies each element by its corresponding weight; .reshape(-1,12)
        #    reshapes the matrix to only have 12 columns (1 year), so the size is (rows*cols/12, 12); .sum(axis=1)
        #    takes the sum of each year; .reshape(-1,daysperyear.shape[0]) reshapes the matrix back to the proper
        #    structure (rows - bins, columns - year)
        # If averaging a single year, then reshape so it returns a 1d array
        if var_annual.shape[1] == 1:
            var_annual = var_annual.reshape(var_annual.shape[0])
    elif pygem_prms['time']['timestep'] == 'daily':
        var_annual = var.mean(1)
    else:
        # var_annual = var.mean(1)
        assert 1==0, 'add this functionality for weighting that is not monthly or daily'
    return var_annual


def haversine_dist(grid_lons, grid_lats, target_lons, target_lats):
    """
    Compute haversine distances between each (lon_target, lat_target)
    and all (grid_lons, grid_lats) positions.

    Parameters:
    - grid_lons: (ncol,) array of longitudes from data
    - grid_lats: (ncol,) array of latitudes from data
    - target_lons: (n_targets,) array of target longitudes
    - target_lats: (n_targets,) array of target latitudes

    Returns:
    - distances: (n_targets, ncol) array of distances in km to each grid location for each target
    """
    R = 6371.0  # Earth radius in kilometers

    # Convert degrees to radians
    grid_lons = np.radians(grid_lons)[np.newaxis, :]  # (1, ncol)
    grid_lats = np.radians(grid_lats)[np.newaxis, :]
    target_lons = np.radians(target_lons)[:, np.newaxis]  # (n_targets, 1)
    target_lats = np.radians(target_lats)[:, np.newaxis]

    dlon = grid_lons - target_lons
    dlat = grid_lats - target_lats

    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(target_lats) * np.cos(grid_lats) * np.sin(dlon / 2.0) ** 2
    )
    c = 2 * np.arcsin(np.sqrt(a))

    return R * c  # (n_targets, ncol)


def append_json(file_path, new_key, new_value):
    """
    Opens a JSON file, reads its content, adds a new key-value pair,
    and writes the updated data back to the file.

    :param file_path: Path to the JSON file
    :param new_key: The key to add
    :param new_value: The value to add
    """
    try:
        # Read the existing data
        with open(file_path, 'r') as file:
            data = json.load(file)

        # Ensure the JSON data is a dictionary
        if not isinstance(data, dict):
            raise ValueError('JSON file must contain a dictionary at the top level.')

        # Add the new key-value pair
        data[new_key] = new_value

        # Write the updated data back to the file
        with open(file_path, 'w') as file:
            json.dump(data, file)

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
    except json.JSONDecodeError:
        print('Error: The file does not contain valid JSON.')
    except Exception as e:
        print(f'An unexpected error occurred: {e}')
