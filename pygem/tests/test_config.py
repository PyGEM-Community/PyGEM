"""
Python Glacier Evolution Model (PyGEM)

copyright © 2018 David Rounce <drounce@cmu.edu>

Distrubted under the MIT lisence
"""
from pygem.setup.config import ConfigManager

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