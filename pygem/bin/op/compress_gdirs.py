"""
Python Glacier Evolution Model (PyGEM)

copyright © 2024 Brandon Tober <btober@cmu.edu> David Rounce <drounce@cmu.edu>

Distributed under the MIT license

compress OGGM glacier directories
"""

import argparse

import geopandas as gpd
import oggm.cfg as cfg
from oggm import utils, workflow

# pygem imports
from pygem.setup.config import ConfigManager

# instantiate ConfigManager
config_manager = ConfigManager()
# read the config
pygem_prms = config_manager.read_config()

# Initialize OGGM subprocess
cfg.initialize(logging_level='WARNING')
cfg.PATHS['working_dir'] = f'{pygem_prms["root"]}/{pygem_prms["oggm"]["oggm_gdir_relpath"]}'
cfg.PARAMS['border'] = pygem_prms['oggm']['border']
cfg.PARAMS['use_multiprocessing'] = True


def compress_region(region):
    print(f'\n=== Compressing glacier directories for RGI Region: {region} ===')
    # Get glacier IDs from the RGI shapefile
    rgi_ids = gpd.read_file(utils.get_rgi_region_file(str(region).zfill(2), version='62'))['RGIId'].tolist()

    # Initialize glacier directories
    gdirs = workflow.init_glacier_directories(rgi_ids)

    # Tar the individual glacier directories
    workflow.execute_entity_task(utils.gdir_to_tar, gdirs, delete=True)

    # Tar the bundles
    utils.base_dir_to_tar(
        f'{cfg.PATHS["working_dir"]}/per_glacier/RGI60-{str(region).zfill(2)}',
        delete=True,
    )


def main():
    parser = argparse.ArgumentParser(description='Script to compress and store OGGM glacier directories')
    # add arguments
    parser.add_argument(
        '-rgi_region01',
        type=int,
        default=pygem_prms['setup']['rgi_region01'],
        help='Randoph Glacier Inventory region (can take multiple, e.g. `-run_region01 1 2 3`)',
        nargs='+',
    )
    parser.add_argument(
        '-ncores',
        action='store',
        type=int,
        default=1,
        help='number of simultaneous processes (cores) to use',
    )
    args = parser.parse_args()

    cfg.PARAMS['mp_processes'] = args.ncores

    for reg in args.rgi_region01:
        compress_region(reg)


if __name__ == '__main__':
    main()
