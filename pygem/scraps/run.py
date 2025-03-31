# Libs
import geopandas as gpd
from oggm import cfg, utils, workflow

# Initialize OGGM and set up the default run parameters
cfg.initialize()

# How many grid points around the glacier?
cfg.PARAMS["border"] = 10

# Make it robust
cfg.PARAMS["use_intersects"] = False
cfg.PARAMS["continue_on_error"] = True

# Local working directory (where OGGM will write its output)
cfg.PATHS["working_dir"] = utils.get_temp_dir("some_wd")

# RGI file
path = utils.get_rgi_region_file("11")
rgidf = gpd.read_file(path)

# Select only 2 glaciers
rgidf = rgidf.iloc[:2]

# Sort for more efficient parallel computing
rgidf = rgidf.sort_values("Area", ascending=False)

# Go - create the pre-processed glacier directories
gdirs = workflow.init_glacier_directories(rgidf)

# Our task now
from dummy_task_module import dummy_task

workflow.execute_entity_task(dummy_task, gdirs)

# See that we can read the new dummy data:
import xarray as xr

fpath = gdirs[0].get_filepath("my_netcdf_file")
print(xr.open_dataset(fpath))
