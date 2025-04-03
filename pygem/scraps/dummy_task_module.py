import logging

import xarray as xr
from oggm import cfg
from oggm.utils import entity_task

# Module logger
log = logging.getLogger(__name__)

# Add the new name "my_netcdf_file" to the list of things that the GlacierDirectory understands
cfg.BASENAMES['my_netcdf_file'] = (
    'somefilename.nc',
    'This is just a documentation string',
)


@entity_task(log, writes=[])
def dummy_task(gdir, some_param=None):
    """Very dummy"""

    fpath = gdir.get_filepath('my_netcdf_file')
    da = xr.DataArray([1, 2, 3])
    da.to_netcdf(fpath)
