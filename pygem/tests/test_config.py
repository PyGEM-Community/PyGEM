import os
import shutil
from pygem.setup.config import ConfigManager

def test_create_config():
    """Test that create_config creates the config file."""
    test_dir = os.path.join(os.getcwd(), "test_config")
    os.makedirs(test_dir, exist_ok=True)
    config_manager = ConfigManager(config_filename='config.yaml', base_dir=test_dir, overwrite=True)
    
    config_manager.create_config()
    assert os.path.isfile(config_manager.config_path), "Config file was not created"
    
    # Clean up
    shutil.rmtree(test_dir)

def test_update_config():
    """Test updating keys in the config file."""
    test_dir = os.path.join(os.getcwd(), "test_config")
    os.makedirs(test_dir, exist_ok=True)
    config_manager = ConfigManager(config_filename='config.yaml', base_dir=test_dir, overwrite=True)

    updates = {
        "sim.nsims": "5",
        "user.email": "updated@example.com",
        "constants.density_ice": "850"
    }
    
    config_manager.update_config(updates)
    config = config_manager.read_config()

    # Assert the updates were made correctly
    assert config["sim"]["nsims"] == 5
    assert config["user"]["email"] == "updated@example.com"
    assert config["constants"]["density_ice"] == 850
    
    # Clean up
    shutil.rmtree(test_dir)

def test_update_config_key_error():
    """Test that update_config raises an error if a key doesn't exist."""
    test_dir = os.path.join(os.getcwd(), "test_config")
    os.makedirs(test_dir, exist_ok=True)
    config_manager = ConfigManager(config_filename='config.yaml', base_dir=test_dir, overwrite=True)
    
    try:
        config_manager.update_config({"nonexistent.key": "value"})
    except KeyError:
        pass  # Expected error
    else:
        assert False, "KeyError not raised"
    
    # Clean up
    shutil.rmtree(test_dir)

def test_update_config_type_error():
    """Test that update_config raises an error if there is a type mismatch."""
    test_dir = os.path.join(os.getcwd(), "test_config")
    os.makedirs(test_dir, exist_ok=True)
    config_manager = ConfigManager(config_filename='config.yaml', base_dir=test_dir, overwrite=True)
    
    # try setting a dictionary key-value with an value that is not a dictionary
    try:
        config_manager.update_config({"sim": "not a dict"})
    except TypeError:
        pass  # Expected error
    else:
        assert False, "TypeError not raised"
    
    # try setting a bool key-value type with a non-bool value type
    try:
        config_manager.update_config({"setup.include_tidewater": -999})
    except TypeError:
        pass  # Expected error
    else:
        assert False, "TypeError not raised"
    
    # try setting a string key-value type with a non-string value type
    try:
        config_manager.update_config({"root": -999})
    except TypeError:
        pass  # Expected error
    else:
        assert False, "TypeError not raised"
    
    # Clean up
    shutil.rmtree(test_dir)