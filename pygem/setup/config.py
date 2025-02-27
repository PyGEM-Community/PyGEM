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
        self.package_dir = os.path.dirname(__file__)
        self.source_config_path = os.path.join(self.package_dir, self.config_filename)
    
    def ensure_config(self, overwrite=False):
        """Ensure the configuration file exists"""
        if os.path.isfile(self.config_path) and not overwrite:
            return
        
        if os.path.isfile(self.config_path) and overwrite:
            overwrite = self._prompt_overwrite()
        
        if not os.path.isfile(self.config_path) or overwrite:
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

        try:
            with open(self.config_path, 'r') as file:
                config = yaml.load(file)

            for key, value in updates.items():
                keys = key.split('.')
                d = config
                for k in keys[:-1]:
                    if k not in d:
                        print(f"Key '{key}' not found in the config. Skipping update.")
                        break
                    d = d[k]
                else:
                    # Reparse value with YAML to infer correct type
                    if keys[-1] in d:
                        d[keys[-1]] = yaml.load(value)
                    else:
                        print(f"Key '{key}' not found in the config. Skipping update.")

            with open(self.config_path, 'w') as file:
                yaml.dump(config, file)

        except Exception as e:
            print(f"Error updating config: {e}")

    
    def _prompt_overwrite(self):
        """Prompt the user for confirmation before overwriting the config file."""
        while True:
            response = input(f"Configuration file already exists ({self.config_path}). Overwrite? (y/n): ").strip().lower()
            if response in ['y', 'yes']:
                return True
            elif response in ['n', 'no']:
                return False
            print("Invalid input. Please enter 'y' or 'n'.")

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