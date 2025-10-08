"""
Python Glacier Evolution Model (PyGEM)

copyright © 2025 Brandon Tober <btober@cmu.edu>, David Rounce <drounce@cmu.edu>

Distributed under the MIT license
"""

# Built-in libaries
import datetime
import json
import logging

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
    Add 1d elevation change observations to the given glacier directory

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        where to write the data

    expected json structure:
        {   'bin_edges':    [edge0, edge1, ..., edgeN],
            'dates':        [(period1_start, period1_end), (period2_start, period2_end), ... (periodM_start, periodM_end)],
            'dh':           [[dh_bin1_period1, dh_bin2_period1, ..., dh_binN_period1],
                            [dh_bin1_period2, dh_bin2_period2, ..., dh_binN_period2],
                            ...
                            [dh_bin1_periodM, dh_bin2_periodM, ..., dh_binN_periodM]],
            'sigma':        [[sigma_bin1_period1, sigma_bin2_period1, ..., sigma_binN_period1],
                            [sigma_bin1_period2, sigma_bin2_period2, ..., sigma_binN_period2],
                            ...
                            [sigma_bin1_periodM, sigma_bin2_periodM, ..., sigma_binN_periodM]],
        }
        note: 'dates' are tuples (or length-2 sublists) of the start and stop date of an individual elevation change record
        and are stored as strings in 'YYYY-MM-DD' format. 'dh' should M lists of length N-1,
        where N is the number of bin edges. 'sigma'  should eaither be M lists of shape N-1 a scalar value.
    """
    # get dataset file path
    elev_change_1d_fp = (
        f'{pygem_prms["root"]}/'
        f'{pygem_prms["calib"]["data"]["elev_change_1d"]["elev_change_1d_relpath"]}/'
        f'{gdir.rgi_id.split("-")[1]}_elev_change_1d.json'
    )

    try:
        with open(elev_change_1d_fp, 'r') as f:
            # Load JSON
            data = json.load(f)

        validate_elev_change_1d_structure(data)

    except Exception as err:
        log.error(f'Validation failed for {elev_change_1d_fp}: {err}')
        # optionally include traceback for debugging
        log.exception('Full traceback:')
        raise  # Re-raise so OGGM / your pipeline knows this task failed

    else:
        log.info('Binned elevation cahnge data added to glacier directory')
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
