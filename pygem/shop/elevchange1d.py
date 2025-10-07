"""
Python Glacier Evolution Model (PyGEM)

copyright © 2018 David Rounce <drounce@cmu.edu>

Distributed under the MIT license
"""

# Built-in libaries
import json
import logging
import os
import shutil

# External libraries
from datetime import timedelta

import numpy as np
import pandas as pd

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
    """
    # get dataset file path
    elev_change_1d_fp = f'{pygem_prms["root"]}/{pygem_prms["calib"]["data"]["elev_change_1d"]["elev_change_1d_relpath"]}'

    assert os.path.isdir(elev_change_1d_fp), (
        'Error, elevation change dataset does not exist: {elev_change_1d_fp}'
    )

    # check for glacier elevation change data
    if gdir.rgi_id not in os.listdir(elev_change_1d_fp):


    # read reference mass balance dataset and pull data of interest
    mb_df = pd.read_csv(mbdata_fp)
    mb_df_rgiids = list(mb_df[rgiid_cn])

    if gdir.rgi_id in mb_df_rgiids:
        # RGIId index
        rgiid_idx = np.where(gdir.rgi_id == mb_df[rgiid_cn])[0][0]

        # Glacier-wide mass balance
        mb_mwea = mb_df.loc[rgiid_idx, mb_cn]
        mb_mwea_err = mb_df.loc[rgiid_idx, mberr_cn]

        if mb_clim_cn in mb_df.columns:
            mb_clim_mwea = mb_df.loc[rgiid_idx, mb_clim_cn]
            mb_clim_mwea_err = mb_df.loc[rgiid_idx, mberr_clim_cn]
        else:
            mb_clim_mwea = None
            mb_clim_mwea_err = None

        t1_str, t2_str = mb_df.loc[rgiid_idx, 'period'].split('_')
        t1_datetime = pd.to_datetime(t1_str)
        t2_datetime = pd.to_datetime(t2_str)

        # remove one day from t2 datetime for proper indexing (ex. 2001-01-01 want to run through 2000-12-31)
        t2_datetime = t2_datetime - timedelta(days=1)
        # Number of years
        nyears = (t2_datetime + timedelta(days=1) - t1_datetime).days / 365.25

        # Record data
        mbdata = {
            key: value
            for key, value in {
                'mb_mwea': float(mb_mwea),
                'mb_mwea_err': float(mb_mwea_err),
                'mb_clim_mwea': float(mb_clim_mwea)
                if mb_clim_mwea is not None
                else None,
                'mb_clim_mwea_err': float(mb_clim_mwea_err)
                if mb_clim_mwea_err is not None
                else None,
                't1_str': t1_str,
                't2_str': t2_str,
                'nyears': nyears,
            }.items()
            if value is not None
        }
        mb_fn = gdir.get_filepath('mb_calib_pygem')
        with open(mb_fn, 'w') as f:
            json.dump(mbdata, f)