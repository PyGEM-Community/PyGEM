"""
Python Glacier Evolution Model (PyGEM)

copyright © 2018 David Rounce <drounce@cmu.edu>

Distrubted under the MIT lisence
"""

import os
import shutil

import yaml

# pygem_params file name
config_fn = "config.yaml"

# Define the base directory and the path to the configuration file
basedir = os.path.join(os.path.expanduser("~"), "PyGEM")
config_file = os.path.join(
    basedir, config_fn
)  # Path where you want the config file

# Get the source configuration file path from your package
package_dir = os.path.dirname(
    __file__
)  # Get the directory of the current script
source_config_file = os.path.join(package_dir, config_fn)  # Path to params.py


def ensure_config(overwrite=False):
    isfile = os.path.isfile(config_file)
    if isfile and overwrite:
        overwrite = None
        while overwrite is None:
            # Ask the user for a y/n response
            response = (
                input(
                    f"PyGEM configuration.yaml file already exist ({config_file}), do you wish to overwrite (y/n):"
                )
                .strip()
                .lower()
            )
            # Check if the response is valid
            if response == "y" or response == "yes":
                overwrite = True
            elif response == "n" or response == "no":
                overwrite = False
            else:
                print("Invalid input. Please enter 'y' or 'n'.")

    if (not isfile) or (overwrite):
        os.makedirs(basedir, exist_ok=True)  # Ensure the base directory exists
        shutil.copy(source_config_file, config_file)  # Copy the file
        print(f"Copied default configuration to {config_file}")


def read_config():
    """Read the configuration file and return its contents as a dictionary."""
    with open(config_file, "r") as f:
        config = yaml.safe_load(
            f
        )  # Use safe_load to avoid arbitrary code execution
    return config
