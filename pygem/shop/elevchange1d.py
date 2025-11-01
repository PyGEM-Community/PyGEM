"""
Python Glacier Evolution Model (PyGEM)

copyright © 2025 Brandon Tober <btober@cmu.edu>, David Rounce <drounce@cmu.edu>

Distributed under the MIT license
"""

# Built-in libaries
import datetime
import json
import logging
import os

import numpy as np
import pandas as pd

# External libraries
# Local libraries
from oggm import cfg
from oggm.utils import entity_task
from scipy.stats import binned_statistic

# pygem imports
from pygem.setup.config import ConfigManager

# instantiate ConfigManager
config_manager = ConfigManager()
# read the config
pygem_prms = config_manager.read_config()


# Module logger
log = logging.getLogger(__name__)

# Add the new name "elev_change_1d" to the list of things that the GlacierDirectory understands
if 'elev_change_1d' not in cfg.BASENAMES:
    cfg.BASENAMES['elev_change_1d'] = (
        'elev_change_1d.json',
        '1D elevation change data',
    )


@entity_task(log, writes=['elev_change_1d'])
def dh_1d_to_gdir(
    gdir,
    dh_datadir=f'{pygem_prms["root"]}/{pygem_prms["calib"]["data"]["elev_change"]["dh_1d_relpath"]}/',
    filesuffix='',
    bin_spacing=None,
    bin_lowcut=None,
    bin_highcut=None,
    bin_cut_percentile=True,
):
    """
    Add 1D elevation change observations to the given glacier directory.

    Binned 1D elevation change data should be stored as a JSON or CSV file with the following equivalent formats.

    JSON file structure:
        {
            'ref_dem': str,
            'ref_dem_year': int,
            'dates': [
                (date_start_1, date_end_1),
                (date_start_2, date_end_2),
                ...
                (date_start_M, date_end_M)
            ],
            'bin_edges': [edge0, edge1, ..., edgeN],
            'bin_area': [area0, area1, ..., areaN-1],
            'dh': [
                [dh_bin1_period1, dh_bin2_period1, ..., dh_binN-1_period1],
                [dh_bin1_period2, dh_bin2_period2, ..., dh_binN-1_period2],
                ...
                [dh_bin1_periodM, dh_bin2_periodM, ..., dh_binN-1_periodM]
            ],
            'dh_sigma': [
                [dh_sigma_bin1_period1, dh_sigma_bin2_period1, ..., dh_sigma_binN-1_period1],
                [dh_sigma_bin1_period2, dh_sigma_bin2_period2, ..., dh_sigma_binN-1_period2],
                ...
                [dh_sigma_bin1_periodM, dh_sigma_bin2_periodM, ..., dh_sigma_binN-1_periodM]
            ],
        }

    Notes:
        - 'ref_dem' is the reference DEM used for elevation-binning.
        - 'ref_dem_year' is the acquisition year of the reference DEM.
        - Each element in 'dates' defines one elevation change period with a start and end date,
        stored as strings in 'YYYY-MM-DD' format.
        - 'bin_edges' should be a list length N containing the elevation values of each bin edge.
        - 'bin_area' should be a list of length N-1 containing the bin areas given by the 'ref_dem' (optional).
        - 'dh' should contain M lists of length N-1, where M is the number of periods and N is the number of bin edges. Units are in meters.
        - 'dh_sigma' should either be M lists of length N-1 (matching 'dh') or a single scalar value. Units are in meters.
        - Each list in 'dh' (and optionally 'dh_sigma') corresponds exactly to one period in 'dates'.

    CSV file structure:
        bin_start, bin_stop, bin_area, date_start, date_end, dh, dh_sigma, ref_dem, ref_dem_year
        edge0, edge1, area0, date_start_1, date_end_1, dh_bin1_period1, dh_sigma_bin1_period1, ref_dem, ref_dem_year
        edge1, edge2, area1, date_start_1, date_end_1, dh_bin2_period1, dh_sigma_bin2_period1, ref_dem, ref_dem_year
        ...
        edgeN-1, edgeN, areaN-1, date_start_1, date_end_1, dh_binN-1_period1, dh_sigma_binN-1_period1, ref_dem, ref_dem_year
        edge0, edge1, area0, date_start_2, date_end_2, dh_bin1_period2, dh_sigma_bin1_period2, ref_dem, ref_dem_year
        ...
        edgeN-1, edgeN, areaN-1, date_start_M, date_end_M, dh_binN-1_periodM, dh_sigma_binN-1_periodM, ref_dem, ref_dem_year

    Notes:
        - Each set of 'date_start' and 'date_end' defines one elevation change period.
        - Dates must be stored as strings in 'YYYY-MM-DD' format.
        - Rows with the same ('date_start', 'date_end') values correspond to a single period,
        with one row per elevation bin.
        - 'bin_area' should contain M × (N-1) entries, where M is the number of periods and N is the number of bin edges. Units are in square meters. (optional).
        - 'dh' should contain M × (N-1) entries, where M is the number of periods and N is the number of bin edges. Units are in meters.
        - 'dh_sigma' should contain M × (N-1) entries, where M is the number of periods and N is the number of bin edges. Units are in meters.
        - 'ref_dem' is constant for all rows and indicates the acquisition year of the reference DEM used for elevation-binning.
        - 'ref_dem_year' is constant for all rows and indicates the year of the reference DEM.

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        where to write the data

    """
    # get dataset file path
    elev_change_1d_fp = f'{dh_datadir}/{gdir.rgi_id.split("-")[1]}_elev_change_1d{filesuffix}'

    # Check for both .json and .csv extensions
    if os.path.exists(elev_change_1d_fp + '.json'):
        elev_change_1d_fp += '.json'
        with open(elev_change_1d_fp, 'r') as f:
            data = json.load(f)

    elif os.path.exists(elev_change_1d_fp + '.csv'):
        elev_change_1d_fp += '.csv'
        data = csv_to_elev_change_1d_dict(elev_change_1d_fp)

    else:
        log.debug('No binned elevation change data to load, skipping task.')
        raise Warning('No binned elevation data to load')  # file not found, skip

    validate_elev_change_1d_structure(data)

    # optionally cut bins
    if bin_lowcut is not None or bin_highcut is not None:
        data = filter_elev_change_1d_data(data, bin_lowcut, bin_highcut, bin_cut_percentile)

    # optionally rebin
    if bin_spacing:
        data = rebin_elev_change_1d_data(data, float(bin_spacing))

    # can't hurt to validate again after rebinning
    validate_elev_change_1d_structure(data)

    gdir.write_json(data, 'elev_change_1d')


def validate_elev_change_1d_structure(data):
    """Validate that elev_change_1d JSON structure matches expected format."""

    required_keys = ['ref_dem', 'ref_dem_year', 'bin_edges', 'dates', 'dh', 'dh_sigma']
    for key in required_keys:
        if key not in data:
            raise ValueError(f"Missing required key '{key}' in elevation change JSON.")

    # Validate bin_edges
    bin_edges = data['bin_edges']
    if not isinstance(bin_edges, list) or len(bin_edges) < 2:
        raise ValueError("'bin_edges' must be a list of at least two numeric values.")
    if not all(isinstance(x, (int, float)) for x in bin_edges):
        raise ValueError("All 'bin_edges' values must be numeric.")

    # Calculate bin_centers if missing
    if 'bin_centers' not in data:
        data['bin_centers'] = [0.5 * (bin_edges[i] + bin_edges[i + 1]) for i in range(len(bin_edges) - 1)]

    # Validate bin_centers
    bin_centers = data['bin_centers']
    if not isinstance(bin_centers, list) or len(bin_centers) != len(bin_edges) - 1:
        raise ValueError("'bin_centers' must be a list of length len(bin_edges)-1.")
    if not all(isinstance(x, (int, float)) for x in bin_centers):
        raise ValueError("All 'bin_centers' values must be numeric.")

    # Validate dates
    dates = data['dates']
    if not isinstance(dates, list) or len(dates) == 0:
        raise ValueError("'dates' must be a non-empty list of (start, end) tuples.")
    for i, d in enumerate(dates):
        if not (isinstance(d, (list, tuple)) and len(d) == 2):
            raise ValueError(f"'dates[{i}]' must be a 2-element list or tuple.")
        for j, date_str in enumerate(d):
            try:
                datetime.datetime.strptime(date_str, '%Y-%m-%d')
            except ValueError:
                raise ValueError(f"Invalid date format in 'dates[{i}][{j}]': {date_str}") from None

    # Validate dh
    dh = data['dh']
    M = len(dates)
    N = len(bin_edges) - 1
    if not (isinstance(dh, list) and len(dh) == M):
        raise ValueError(f"'dh' must have {M} entries, one per period in 'dates'.")
    for i, arr in enumerate(dh):
        if not (isinstance(arr, list) and len(arr) == N):
            raise ValueError(f"'dh[{i}]' must be a list of length {N}.")
        if not all(isinstance(x, (int, float)) for x in arr):
            raise ValueError(f"All 'dh[{i}]' values must be numeric.")

    # Validate sigma
    sigma = data['dh_sigma']
    if isinstance(sigma, (int, float)):
        # scalar is OK
        pass
    elif isinstance(sigma, list):
        if len(sigma) != M:
            raise ValueError(f"'sigma' must have {M} entries to match 'dates'.")
        for i, arr in enumerate(sigma):
            if isinstance(arr, list):
                if len(arr) != N:
                    raise ValueError(f"'sigma[{i}]' must be length {N}.")
                if not all(isinstance(x, (int, float)) for x in arr):
                    raise ValueError(f"All 'sigma[{i}]' values must be numeric.")
            elif not isinstance(arr, (int, float)):
                raise ValueError(f"'sigma[{i}]' must be numeric or a list of numeric values.")
    else:
        raise ValueError("'sigma' must be a list or scalar numeric value.")

    # Validate ref_dem
    ref_dem = data['ref_dem']
    if not isinstance(ref_dem, str):
        raise ValueError("'ref_dem' must be a string.")

    # Validate ref_dem_year
    ref_dem_year = data['ref_dem_year']
    if not isinstance(ref_dem_year, int):
        raise ValueError("'ref_dem_year' must be an integer year.")

    # Validate bin_area if present
    if 'bin_area' in data:
        bin_area = data['bin_area']
        if len(bin_area) != N:
            raise ValueError(f"'bin_area' must be a list of length {N}.")
        if not all(isinstance(x, (int, float)) for x in bin_area):
            raise ValueError("All 'bin_area' values must be numeric.")

    return True


def csv_to_elev_change_1d_dict(csv_path):
    """
    Convert a CSV with columns:
    bin_start, bin_stop, date_start, date_end, dh, dh_sigma, ref_dem, ref_dem_year
    into a dictionary structure matching elev_change_data format.
    """
    df = pd.read_csv(csv_path)

    required_cols = {
        'bin_start',
        'bin_stop',
        'date_start',
        'date_end',
        'dh',
        'dh_sigma',
        'ref_dem',
        'ref_dem_year',
    }
    if not required_cols.issubset(df.columns):
        raise ValueError(f'CSV must contain columns: {required_cols}')

    # Ensure sorted bins
    df = df.sort_values(['bin_start', 'date_start', 'date_end']).reset_index(drop=True)

    # Drop duplicate (bin_start, bin_stop) pairs
    df_unique_bins = df.drop_duplicates(subset=['bin_start', 'bin_stop']).sort_values('bin_start')

    # Define bin edges from unique bins
    bin_edges = np.concatenate(([df_unique_bins['bin_start'].iloc[0]], df_unique_bins['bin_stop'].values)).tolist()

    # Handle bin_area if present
    if 'bin_area' in df.columns:
        bin_area = df_unique_bins['bin_area'].tolist()
    else:
        bin_area = False

    # Validate reference DEM - should only be one unique string
    dems = df['ref_dem'].dropna().unique()
    if len(dems) != 1:
        raise ValueError(f"'ref_dem' must have exactly one unique value, but found {len(dems)}: {dems.tolist()}")
    dem = dems[0]
    if not isinstance(dem, str):
        raise TypeError(f"'ref_dem' must be a string, but got {type(dem).__name__}: {dem}")

    # Validate reference DEM year - should only be one constant integer value
    years = df['ref_dem_year'].dropna().unique()
    if len(years) != 1:
        raise ValueError(f"'ref_dem_year' must have exactly one unique value, but found {len(years)}: {years.tolist()}")
    dem_year = years[0]
    if not isinstance(dem_year, (int, np.integer)):
        raise TypeError(f"'ref_dem_year' must be an integer, but got {type(dem_year).__name__}: {dem_year}")
    # convert to plain int
    dem_year = int(dem_year)

    # Get all unique date pairs (preserving order)
    date_pairs = df[['date_start', 'date_end']].drop_duplicates().apply(tuple, axis=1).tolist()

    # Group by date pairs and collect dh, sigma
    dh_all, sigma_all = [], []
    for ds, de in date_pairs:
        subset = df[(df['date_start'] == ds) & (df['date_end'] == de)]
        subset = subset.sort_values('bin_start')
        dh_all.append(subset['dh'].tolist())
        sigma_all.append(subset['dh_sigma'].tolist())

    data = {
        'ref_dem': dem,
        'ref_dem_year': dem_year,
        'dates': [(str(ds), str(de)) for ds, de in date_pairs],
        'bin_edges': bin_edges,
        'bin_area': bin_area,
        'dh': dh_all,
        'dh_sigma': sigma_all,
    }

    if not bin_area:  # remove bin_area if flag is False
        del data['bin_area']

    return data


def rebin_elev_change_1d_data(data, bin_spacing):
    """
    Rebin elevation change data to new bin spacing.
    """
    bin_centers = data['bin_centers']
    # estimate bin width (assuming uniform)
    dz = np.abs(np.diff(bin_centers).mean())

    # optionally rebin
    if bin_spacing != dz:
        # define new bin edges
        new_edges = np.arange(min(data['bin_edges']), max(data['bin_edges']) + bin_spacing, bin_spacing)

        # simple mean of dh per new bin
        dh_rebinned, _, _ = binned_statistic(bin_centers, data['dh'], statistic='mean', bins=new_edges)

        # simple mean of dh_sigma per new bin
        dh_sigma_rebinned, _, _ = binned_statistic(bin_centers, data['dh_sigma'], statistic='mean', bins=new_edges)

        # sum of bin_area per new bin
        bin_area_rebinned, _, _ = binned_statistic(bin_centers, data['bin_area'], statistic='sum', bins=new_edges)

        # replace data values
        data['bin_edges'] = new_edges.tolist()
        data['bin_centers'] = [
            0.5 * (data['bin_edges'][i] + data['bin_edges'][i + 1]) for i in range(len(data['bin_edges']) - 1)
        ]
        data['bin_area'] = bin_area_rebinned.tolist()
        data['dh'] = dh_rebinned.tolist()
        data['dh_sigma'] = dh_sigma_rebinned.tolist()
        return data


def filter_elev_change_1d_data(data, bin_lowcut, bin_highcut, bin_cut_percentile):
    """
    Filter elevation change data to only include bins within specified elevation range.

    Parameters
    ----------
    data : dict
        Elevation change data dictionary.
    bin_lowcut : float or None
        Lower elevation cut-off. If None, no lower cut is applied.
    bin_highcut : float or None
        Upper elevation cut-off. If None, no upper cut is applied.
    bin_cut_percentile : bool
        If True, interpret low/high cut as percentiles of the elevation distribution.

    Returns
    -------
    data_filtered : dict
        Filtered elevation change data dictionary.
    """
    bin_centers = np.array(data['bin_centers'])
    bin_edges = np.array(data['bin_edges'])

    if bin_cut_percentile:
        if bin_lowcut is not None:
            bin_lowcut = np.percentile(bin_centers, bin_lowcut)
        if bin_highcut is not None:
            bin_highcut = np.percentile(bin_centers, bin_highcut)

    mask = np.ones_like(bin_centers, dtype=bool)
    if bin_lowcut is not None:
        mask &= bin_centers >= bin_lowcut
    if bin_highcut is not None:
        mask &= bin_centers <= bin_highcut

    # Ensure at least one bin remains
    if not np.any(mask):
        raise ValueError('No bins remain after filtering.')

    # Select corresponding edges (N+1)
    idx = np.where(mask)[0]
    bin_edges_filtered = bin_edges[idx[0] : idx[-1] + 2]

    data_filtered = data.copy()
    data_filtered['bin_centers'] = bin_centers[mask].tolist()
    data_filtered['bin_edges'] = bin_edges_filtered.tolist()

    if 'bin_area' in data:
        data_filtered['bin_area'] = np.array(data['bin_area'])[mask].tolist()

    data_filtered['dh'] = [np.array(dh_arr)[mask].tolist() for dh_arr in data['dh']]

    if isinstance(data['dh_sigma'], list):
        data_filtered['dh_sigma'] = [np.array(sigma_arr)[mask].tolist() for sigma_arr in data['dh_sigma']]
    else:
        data_filtered['dh_sigma'] = data['dh_sigma']

    return data_filtered
