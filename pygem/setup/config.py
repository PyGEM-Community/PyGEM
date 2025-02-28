"""
Python Glacier Evolution Model (PyGEM)

copyright © 2018 David Rounce <drounce@cmu.edu>

Distrubted under the MIT lisence
"""
import os
import shutil
import yaml
import argparse
from ruamel.yaml import YAML

class ConfigManager:
    def __init__(self, config_filename='config.yaml', base_dir=None):
        """initialize the ConfigManager class"""
        self.config_filename = config_filename
        self.base_dir = base_dir or os.path.join(os.path.expanduser('~'), 'PyGEM')
        self.config_path = os.path.join(self.base_dir, self.config_filename)
        self.source_config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")
    
    def ensure_config(self, overwrite=False):
        """Ensure the configuration file exists, creating or overwriting it if necessary"""
        if not os.path.isfile(self.config_path) or overwrite:
            self.create_config()

    def create_config(self):
        """Copy the default configuration file to the expected location"""
        if not os.path.exists(self.source_config_path):
            raise FileNotFoundError(f"Default config file not found at {self.source_config_path}, there may have been an installation issue")
        
        os.makedirs(self.base_dir, exist_ok=True)
        shutil.copy(self.source_config_path, self.config_path)
        print(f"Copied default configuration to {self.config_path}")
    
    def read_config(self):
        """Read the configuration file and return its contents as a dictionary."""
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def update_config(self, updates):
        """Update multiple keys in the YAML configuration file while preserving quotes and original types."""
        yaml = YAML()
        yaml.preserve_quotes = True  # Preserve quotes around string values

        with open(self.config_path, 'r') as file:
            config = yaml.load(file)

        for key, value in updates.items():
            keys = key.split('.')
            d = config
            # Traverse the keys up to the second-to-last
            for i, k in enumerate(keys[:-1]):
                if k not in d:
                    raise KeyError(f"No matching `{'.'.join(keys[:i+1])}` key found in the configuration file at path: {self.config_path}")
                d = d[k]

            final_key = keys[-1]

            # Ensure the final key exists before updating its value
            if final_key not in d:
                raise KeyError(f"No matching `{key}` key found in the configuration file at path: {self.config_path}")

            # Prevent replacing a dictionary with a non-dictionary value
            if isinstance(d[final_key], dict):
                raise ValueError(f"Cannot directly overwrite key `{key}` because it contains a dictionary.")
            
            d[final_key] = yaml.load(value)

        # Save the updated config back to the file
        with open(self.config_path, 'w') as file:
            yaml.dump(config, file)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.yaml', help='Path to the config file')
    parser.add_argument('updates', nargs='*', help='Key-value pairs to update in the config file')
    
    args = parser.parse_args()
    
    # Parse the updates into a dictionary
    updates = {}
    for update in args.updates:
        key, value = update.split('=')
        updates[key] = value
    
    config_manager = ConfigManager(config_filename=args.config)
    config_manager.update_config(updates)

if __name__ == '__main__':
    main()