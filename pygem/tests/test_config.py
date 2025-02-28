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

def test_ensure_config_no_overwrite():
    """Test that create_config does not overwrite the existing config file when overwrite=False."""
    test_dir = os.path.join(os.getcwd(), "test_config")
    os.makedirs(test_dir, exist_ok=True)
    
    # First, create the config file with overwrite=True
    config_manager = ConfigManager(config_filename='config.yaml', base_dir=test_dir, overwrite=True)
    config_manager.create_config()  # This will create the file
    
    # Now, try initializing ConfigManager without overwrite=True
    config_manager_no_overwrite = ConfigManager(config_filename='config.yaml', base_dir=test_dir, overwrite=False)
    
    # Try to ensure the config - it should not overwrite the existing file
    try:
        config_manager_no_overwrite.ensure_config()
        # If it doesn't raise any exceptions, check that the file exists and is not modified
        assert os.path.isfile(config_manager_no_overwrite.config_path), "Config file should exist"
        original_mtime = os.path.getmtime(config_manager.config_path)
        
        # Trigger ensure_config again to confirm it does not modify the existing file
        config_manager_no_overwrite.ensure_config()
        assert os.path.getmtime(config_manager_no_overwrite.config_path) == original_mtime, "Config file should not be modified"
    except Exception as e:
        assert False, f"Unexpected error: {e}"
    
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