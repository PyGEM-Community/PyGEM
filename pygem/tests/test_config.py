import pathlib
import pytest
import yaml
from pygem.setup.config import ConfigManager


class TestConfigManager:
    """Tests for the ConfigManager class."""
    
    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        """Setup a ConfigManager instance for each test."""
        self.config_manager = ConfigManager(
            config_filename='config.yaml',
            base_dir=tmp_path,
            overwrite=True
        )
    
    def test_config_created(self, tmp_path):
        config_path = pathlib.Path(tmp_path) / 'config.yaml'
        assert config_path.is_file()

    def test_read_config(self):
        config = self.config_manager.read_config()
        assert isinstance(config, dict)
        assert "sim" in config
        assert "nsims" in config["sim"]

    def test_update_config_unrecognized_key_error(self):
        """Test that a KeyError is raised when updating a value with an unrecognized key."""
        with pytest.raises(KeyError, match="Unrecognized configuration key: invalid_key"):
            self.config_manager.update_config({"invalid_key": None})

    @pytest.mark.parametrize("key, invalid_value, expected_type, invalid_type", [
        ("sim.nsims", [1, 2, 3], "int", "list"),
        ("calib.HH2015_params.kp_init", "0.5", "float", "str"),
        ("setup.include_landterm", -999, "bool", "int"),
        ("rgi.rgi_cols_drop", "not-a-list", "list", "str"),
    ])
    def test_update_config_type_error(self, key, invalid_value, expected_type, invalid_type):
        """
        Test that a TypeError is raised when updating a value with a new value of a
        wrong type.
        """
        with pytest.raises(
            TypeError, 
            match=f"Invalid type for '{key.replace('.', '\\.')}':"
                  f" expected.*{expected_type}.*, not.*{invalid_type}.*"
        ):
            self.config_manager.update_config({key: invalid_value})

    def test_update_config_list_element_type_error(self):
        """
        Test that a TypeError is raised when updating a value with a new list value
        containing elements of a different type than expected.
        """
        key = "rgi.rgi_cols_drop"
        invalid_value = ["a", "b", 100]
        expected_type = "str"

        with pytest.raises(
            TypeError, 
            match=f"Invalid type for elements in '{key.replace('.', '\\.')}':"
                  f" expected all elements to be .*{expected_type}.*, but got.*{invalid_value}.*"
        ):
            self.config_manager.update_config({key: invalid_value})

    def test_compare_with_source(self):
        """Test that compare_with_source detects missing keys."""
        # Remove a key from the config file
        with open(self.config_manager.config_path, 'r') as f:
            config = yaml.safe_load(f)
        del config['sim']['nsims']
        with open(self.config_manager.config_path, 'w') as f:
            yaml.dump(config, f)

        with pytest.raises(KeyError, match=r"Missing required key in configuration: sim\.nsims"):
            self.config_manager.read_config(validate=True)

    def test_update_config(self):
        """Test that update_config updates the config file for all data types."""
        updates = {
            "sim.nsims": 5,                             # int
            "calib.HH2015_params.kp_init": 0.5,         # float
            "user.email": "updated@example.com",        # str
            "setup.include_landterm": False,            # bool
            "rgi.rgi_cols_drop": ['Item1', 'Item2'],    # list
        }

        # Values before updating
        config = self.config_manager.read_config()
        assert config["sim"]["nsims"] == 1
        assert config["calib"]["HH2015_params"]["kp_init"] == 1.5
        assert config["user"]["email"] == "drounce@cmu.edu"
        assert config["setup"]["include_landterm"] == True
        assert config["rgi"]["rgi_cols_drop"] == ["GLIMSId", "BgnDate", "EndDate", "Status", "Linkages", "Name"]
        
        self.config_manager.update_config(updates)
        config = self.config_manager.read_config()

        # Values after updating
        assert config["sim"]["nsims"] == 5
        assert config["calib"]["HH2015_params"]["kp_init"] == 0.5
        assert config["setup"]["include_landterm"] == False
        assert config["user"]["email"] == "updated@example.com"
        assert config["rgi"]["rgi_cols_drop"] == ["Item1", "Item2"]

    def test_update_config_dict(self):
        """Test that update_config updates the config file for nested dictionaries."""
        # Values before updating
        config = self.config_manager.read_config()
        assert config["user"]["name"] == "David Rounce"
        assert config["user"]["email"] == "drounce@cmu.edu"
        assert config["user"]["institution"] == "Carnegie Mellon University, Pittsburgh PA"

        updates = {
            "user": {
                "name": "New Name",
                "email": "New email",
                "institution": "New Institution",
            }
        }
        self.config_manager.update_config(updates)

        # Values after updating
        config = self.config_manager.read_config()
        assert config["user"]["name"] == "New Name"
        assert config["user"]["email"] == "New email"
        assert config["user"]["institution"] == "New Institution"
