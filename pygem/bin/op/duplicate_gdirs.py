"""
Python Glacier Evolution Model (PyGEM)

copyright © 2024 Brandon Tober <btober@cmu.edu> David Rounce <drounce@cmu.edu>

Distrubted under the MIT lisence

duplicate OGGM glacier directories
"""
import argparse
import os
import shutil
# pygem imports
from pygem.setup.config import ConfigManager
# instantiate ConfigManager
config_manager = ConfigManager()
# read the config
pygem_prms = config_manager.read_config()

def main():
    parser = argparse.ArgumentParser(description="Script to make duplicate oggm glacier directories - primarily to avoid corruption if parellelizing runs on a single glacier")
    # add arguments
    parser.add_argument('-rgi_glac_number', type=str, default=None,
                        help='Randoph Glacier Inventory region')
    parser.add_argument('-num_copies', type=int, default=1,
                        help='Number of copies to create of the glacier directory data')
    args = parser.parse_args()
    num_copies = args.num_copies
    glac_num = args.rgi_glac_number

    if (glac_num is not None) and (num_copies)>1:
        reg,id = glac_num.split('.')
        reg = reg.zfill(2)
        thous = id[:2]
        
        root = pygem_prms['root'] + '/' + pygem_prms['oggm']['oggm_gdir_relpath']
        sfix = '/per_glacier/' + f'RGI60-{reg}/' + f'RGI60-{reg}.{thous}/'

        for n in range(num_copies):
            nroot = os.path.abspath(root.replace('gdirs',f'gdirs_{n+1}'))
            # duplicate structure
            os.makedirs(nroot + sfix + f'RGI60-{reg}.{id}', exist_ok=True)
            # copy directory data
            shutil.copytree(root + sfix + f'RGI60-{reg}.{id}', nroot + sfix + f'RGI60-{reg}.{id}', dirs_exist_ok=True)

    return

if __name__ == '__main__':
    main()