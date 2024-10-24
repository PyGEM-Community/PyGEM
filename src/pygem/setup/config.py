import os
import shutil
import sys
import yaml

# pygem_params file name
config_fn = 'config.yaml'

# Define the base directory and the path to the configuration file
basedir = os.path.join(os.path.expanduser('~'), 'PyGEM')
config_file = os.path.join(basedir, config_fn)  # Path where you want the config file

# Get the source configuration file path from your package
package_dir = os.path.dirname(__file__)  # Get the directory of the current script
source_config_file = os.path.join(package_dir, config_fn)  # Path to params.py

def ensure_config():
    """Check if the config file exists, and copy it if not."""
    if not os.path.isfile(config_file):
        os.makedirs(basedir, exist_ok=True)  # Ensure the base directory exists
        shutil.copy(source_config_file, config_file)  # Copy the file
        print(f"Copied default configuration to {config_file}")

def read_config():
    """Read the configuration file and return its contents as a dictionary."""
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)  # Use safe_load to avoid arbitrary code execution
    return config