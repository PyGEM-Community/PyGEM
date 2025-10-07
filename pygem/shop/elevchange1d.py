"""
Python Glacier Evolution Model (PyGEM)

copyright © 2018 David Rounce <drounce@cmu.edu>

Distributed under the MIT license
"""

# Built-in libaries
import logging
import os
import shutil

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
    """
    # get dataset file path
    elev_change_1d_fp = f'{pygem_prms["root"]}/{pygem_prms["calib"]["data"]["elev_change_1d"]["elev_change_1d_relpath"]}/{gdir.rgi_id.split("-")[1]}_elev_change_1d.json'
    # check for glacier elevation change data
    if os.path.isfile(elev_change_1d_fp):
        # copy file to glacier directory
        shutil.copyfile(
            elev_change_1d_fp,
            gdir.get_filepath('elev_change_1d'),
        )
