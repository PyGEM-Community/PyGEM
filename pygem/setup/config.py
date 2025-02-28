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
    def __init__(self, config_filename='config.yaml', base_dir=None, overwrite=False):
        """initialize the ConfigManager class"""
        self.config_filename = config_filename
        self.base_dir = base_dir or os.path.join(os.path.expanduser('~'), 'PyGEM')
        self.config_path = os.path.join(self.base_dir, self.config_filename)
        self.source_config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")
        self.overwrite = overwrite
        self.ensure_config()
    
    def ensure_config(self):
        """Ensure the configuration file exists, creating or overwriting it if necessary"""
        if not os.path.isfile(self.config_path) or self.overwrite:
            self.create_config()

    def create_config(self):
        """Copy the default configuration file to the expected location"""
        if not os.path.exists(self.source_config_path):
            raise FileNotFoundError(f"Default config file not found at {self.source_config_path}, there may have been an installation issue")
        
        os.makedirs(self.base_dir, exist_ok=True)
        shutil.copy(self.source_config_path, self.config_path)
        print(f"Copied default configuration to {self.config_path}")
        
    def read_config(self, validate=True):
        """Read the configuration file and return its contents as a dictionary.
        
        Args:
            validate (bool): Whether to compare with the default config
        """
        with open(self.config_path, 'r') as f:
            user_config = yaml.safe_load(f)

        if validate:
            self.compare_with_source()

        return user_config
    
    def update_config(self, updates):
        """Update multiple keys in the YAML configuration file while preserving quotes and original types.

        Args:
            updates (dict): Dictionary with key-value pairs to be updated
        """
        ryaml = YAML()
        ryaml.preserve_quotes = True  # Preserve quotes around string values

        with open(self.config_path, 'r') as file:
            config = ryaml.load(file)

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
                raise TypeError(f"Cannot directly overwrite key `{key}` because it contains a dictionary.")
            # Check if the original value is a string, and raise an error if a non-string type is passed
            if isinstance(d[final_key], str) and not isinstance(value, str):
                raise TypeError(f"Cannot update `{key}` with a non-string value: expected a string.")
            # Check if the original value is a string, and raise an error if a non-string type is passed
            if isinstance(d[final_key], bool) and not isinstance(value, bool):
                raise TypeError(f"Cannot update `{key}` with a non-bool value: expected a bool.")

            d[final_key] = ryaml.load(value)

        # Save the updated config back to the file
        with open(self.config_path, 'w') as file:
            ryaml.dump(config, file)

    def compare_with_source(self):
        """Compare the user's config with the default and raise errors for missing keys or type mismatches."""
        with open(self.source_config_path, 'r') as f:
            default_config = yaml.safe_load(f)
        with open(self.config_path, 'r') as f:
            user_config = yaml.safe_load(f)

        def _check(ref, test, path=""):
            if not isinstance(ref, dict) or not isinstance(test, dict):
                return
            
            for key in ref:
                current_path = f"{path}.{key}" if path else key
                if key not in test:
                    raise ValueError(f"Missing key in user config: {current_path}")

                ref_val, test_val = ref[key], test[key]

                # Allow any type if ref[key] is None
                if ref_val is not None:
                    # Ignore type mismatches if the source was a list but now a single value
                    if isinstance(ref_val, list) and not isinstance(test_val, list):
                        pass  # Ignore type mismatch
                    elif type(ref_val) != type(test_val):
                        raise TypeError(f"Type mismatch at {current_path}: "
                                        f"expected {type(ref_val)}, got {type(test_val)}")

                if isinstance(ref_val, dict):
                    _check(ref_val, test_val, current_path)

        _check(default_config, user_config)