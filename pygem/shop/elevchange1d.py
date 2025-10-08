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

# Add the new name "elev_change_1d" to the list of things that the GlacierDirectory understands
if 'elev_change_1d' not in cfg.BASENAMES:
    cfg.BASENAMES['elev_change_1d'] = (
        'elev_change_1d.json',
        '1D elevation change data',
    )


@entity_task(log, writes=['elev_change_1d'])
def elev_change_1d_to_gdir(
    gdir,
):
    """
    Add 1D elevation change observations to the given glacier directory.

    Binned 1D elevation change data should be stored as a JSON or CSV file with the following equivalent formats.

    JSON file structure:
        {
            'bin_edges': [edge0, edge1, ..., edgeN],
            'dates': [
                (date_start_1, date_end_1),
                (date_start_2, date_end_2),
                ...
                (date_start_M, date_end_M)
            ],
            'dh': [
                [dh_bin1_period1, dh_bin2_period1, ..., dh_binN_period1],
                [dh_bin1_period2, dh_bin2_period2, ..., dh_binN_period2],
                ...
                [dh_bin1_periodM, dh_bin2_periodM, ..., dh_binN_periodM]
            ],
            'dh_sigma': [
                [dh_sigma_bin1_period1, dh_sigma_bin2_period1, ..., dh_sigma_binN_period1],
                [dh_sigma_bin1_period2, dh_sigma_bin2_period2, ..., dh_sigma_binN_period2],
                ...
                [dh_sigma_bin1_periodM, dh_sigma_bin2_periodM, ..., dh_sigma_binN_periodM]
            ]
        }

    Notes:
        - Each element in 'dates' defines one elevation change period with a start and end date,
        stored as strings in 'YYYY-MM-DD' format.
        - Each list in 'dh' (and optionally 'dh_sigma') corresponds exactly to one period in 'dates'.
        - 'dh' should contain M lists of length N-1, where N is the number of bin edges. Units are in meters.
        - 'dh_sigma' should either be M lists of length N-1 (matching 'dh') or a single scalar value. Units are in meters.

    CSV file structure:
        bin_start, bin_stop, date_start, date_end, dh, dh_sigma
        edge0, edge1, date_start_1, date_end_1, dh_bin1_period1, dh_sigma_bin1_period1
        edge1, edge2, date_start_1, date_end_1, dh_bin2_period1, dh_sigma_bin2_period1
        ...
        edgeN-1, edgeN, date_start_1, date_end_1, dh_binN_period1, dh_sigma_binN_period1
        edge0, edge1, date_start_2, date_end_2, dh_bin1_period2, dh_sigma_bin1_period2
        ...
        edgeN-1, edgeN, date_start_M, date_end_M, dh_binN_periodM, dh_sigma_binN_periodM

    Notes:
        - Each set of 'date_start' and 'date_end' defines one elevation change period.
        - Dates must be stored as strings in 'YYYY-MM-DD' format.
        - Rows with the same ('date_start', 'date_end') values correspond to a single period,
        with one row per elevation bin.
        - 'dh' should contain M × (N-1) entries, where N is the number of bin edges and M is the number of periods. Units are in meters.
        - 'dh_sigma' should contain M × (N-1) entries, where N is the number of bin edges and M is the number of periods. Units are in meters.
    
    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        where to write the data

    """
    # get dataset file path
    elev_change_1d_fp = (
        f'{pygem_prms["root"]}/'
        f'{pygem_prms["calib"]["data"]["elev_change_1d"]["elev_change_1d_relpath"]}/'
        f'{gdir.rgi_id.split("-")[1]}_elev_change_1d'
    )

    # Check for both .json and .csv extensions
    if os.path.exists(elev_change_1d_fp + '.json'):
        elev_change_1d_fp += '.json'
        with open(elev_change_1d_fp, 'r') as f:
            data = json.load(f)
    
    elif os.path.exists(elev_change_1d_fp + '.csv'):
        elev_change_1d_fp += '.csv'
        data = csv_to_elev_change_1d_dict(elev_change_1d_fp)
    
    else:
        log.debug(f"No binned elevation change data to load, skipping task.")
        raise Warning('No binned elevation data to load')  # file not found, skip

    validate_elev_change_1d_structure(data)
    
    gdir.write_json(data, 'elev_change_1d')


def validate_elev_change_1d_structure(data):
    """Validate that elev_change_1d JSON structure matches expected format."""

    required_keys = ['bin_edges', 'dates', 'dh', 'sigma']
    for key in required_keys:
        if key not in data:
            raise ValueError(f"Missing required key '{key}' in elevation change JSON.")

    # Validate bin_edges
    bin_edges = data['bin_edges']
    if not isinstance(bin_edges, list) or len(bin_edges) < 2:
        raise ValueError("'bin_edges' must be a list of at least two numeric values.")
    if not all(isinstance(x, (int, float)) for x in bin_edges):
        raise ValueError("All 'bin_edges' values must be numeric.")

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
                raise ValueError(
                    f"Invalid date format in 'dates[{i}][{j}]': {date_str}"
                ) from None

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
    sigma = data['sigma']
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
                raise ValueError(
                    f"'sigma[{i}]' must be numeric or a list of numeric values."
                )
    else:
        raise ValueError("'sigma' must be a list or scalar numeric value.")

    return True


def csv_to_elev_change_1d_dict(csv_path):
    """
    Convert a CSV with columns:
    bin_start, bin_stop, date_start, date_end, dh, dh_sigma
    into a dictionary structure matching elev_change_data format.
    """
    df = pd.read_csv(csv_path)

    required_cols = {"bin_start", "bin_stop", "date_start", "date_end", "dh", "dh_sigma"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"CSV must contain columns: {required_cols}")

    # Ensure sorted bins
    df = df.sort_values(["bin_start", "date_start", "date_end"]).reset_index(drop=True)

    # Get all unique bin edges
    bin_edges = sorted(set(df["bin_start"]).union(df["bin_stop"]))

    # Get all unique date pairs (preserving order)
    date_pairs = (
        df[["date_start", "date_end"]]
        .drop_duplicates()
        .apply(tuple, axis=1)
        .tolist()
    )

    # Group by date pairs and collect dh, sigma
    dh_all, sigma_all = [], []
    for ds, de in date_pairs:
        subset = df[(df["date_start"] == ds) & (df["date_end"] == de)]
        subset = subset.sort_values("bin_start")
        dh_all.append(subset["dh"].tolist())
        sigma_all.append(subset["dh_sigma"].tolist())

    data = {
        "bin_edges": bin_edges,
        "dates": [(str(ds), str(de)) for ds, de in date_pairs],
        "dh": dh_all,
        "sigma": sigma_all,
    }

    return data