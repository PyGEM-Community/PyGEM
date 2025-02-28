"""
Python Glacier Evolution Model (PyGEM)

copyright © 2018 David Rounce <drounce@cmu.edu>

Distrubted under the MIT lisence
"""
import os, yaml
from pygem.setup.config import ConfigManager

# Test case to check if the configuration is created or overwritten correctly
def test_ensure_config():
    config_manager = ConfigManager(overwrite=True)
    config_manager.ensure_config()

    # Check if the config file is created
    assert os.path.isfile(config_manager.config_path)
    
    # Check if the default config file was copied correctly
    with open(config_manager.config_path, 'r') as f:
        config = f.read()
    assert "sim" in config  # Check if a known key exists in the config

    # Test without overwriting
    config_manager_no_overwrite = ConfigManager(overwrite=False)
    config_manager_no_overwrite.ensure_config()

    # The config should not be overwritten if it already exists
    with open(config_manager_no_overwrite.config_path, 'r') as f:
        config_no_overwrite = f.read()
    assert config_no_overwrite == config  # No change should happen

# Test case to verify updating the config
def test_update_config():
    updates = {
        "sim.nsims": "5",
        "user.email": "updated@example.com",
        "constants.density_ice": "850"
    }
    
    config_manager = ConfigManager(overwrite=True)
    config_manager.update_config(updates)
    config = config_manager.read_config()
    
    assert config["sim"]["nsims"] == 5
    assert config["user"]["email"] == "updated@example.com"
    assert config["constants"]["density_ice"] == 850

# Test case to check if the `read_config` function works
def test_read_config():
    config_manager = ConfigManager(overwrite=True)
    config_manager.ensure_config()
    config = config_manager.read_config()

    # Check that the config reads a known key correctly
    assert "sim" in config
    assert isinstance(config["sim"], dict)

# Test case to check if the `compare_with_source` function works
def test_compare_with_source():
    config_manager = ConfigManager(overwrite=True)
    config_manager.ensure_config()
    
    # Modify the config file to simulate an error (e.g., remove a required key)
    config = config_manager.read_config()
    config["sim"].pop("nsims", None)  # Remove a required key
    
    # Write the modified config back to the file
    with open(config_manager.config_path, 'w') as f:
        yaml.dump(config, f)
    
    # Test if it raises a ValueError when comparing with the source config
    try:
        config_manager.compare_with_source()
        assert False, "Expected ValueError due to missing key"
    except ValueError:
        pass  # Expected error, test passes

# Test case for wrong type key-value overwrite
def test_update_config_wrong_type():
    # Test for overwriting a dict with a non-dict value
    updates = {
        "sim": "invalid_value"  # Trying to overwrite a dict with a non-dict value
    }
    
    config_manager = ConfigManager(overwrite=True)
    
    try:
        config_manager.update_config(updates)
        assert False, "Expected ValueError due to non-dict overwrite"
    except ValueError:
        pass  # Expected error, test passes

    # Test for attempting to overwrite the 'root' key (top-level structure) with a non-dict value
    updates_root = {
        "root": "invalid_root_value"  # Trying to overwrite the root of the config with a non-dict value
    }

    try:
        config_manager.update_config(updates_root)
        assert False, "Expected ValueError due to overwriting the root with a non-dict value"
    except ValueError:
        pass  # Expected error, test passes

# Test the `create_config` function when the source config doesn't exist
def test_create_config_missing_source():
    # Simulate the absence of the source config by removing it if it exists
    source_config_path = ConfigManager().source_config_path
    if os.path.exists(source_config_path):
        os.remove(source_config_path)
    
    config_manager = ConfigManager(overwrite=True)
    
    try:
        config_manager.create_config()
        assert False, "Expected FileNotFoundError due to missing source config"
    except FileNotFoundError:
        pass  # Expected error, test passes