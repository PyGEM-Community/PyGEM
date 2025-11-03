import os
import subprocess

import pytest

from pygem.setup.config import ConfigManager

# instantiate ConfigManager
config_manager = ConfigManager()
# update export_extra_vars to True before running tests
config_manager.update_config({'sim.out.export_extra_vars': True})


# Get all notebooks in the PyGEM-notebooks repository
nb_dir = os.environ.get('PYGEM_NOTEBOOKS_DIRPATH') or os.path.join(os.path.expanduser('~'), 'PyGEM-notebooks')
# TODO #54: Test all notebooks
# notebooks = [f for f in os.listdir(nb_dir) if f.endswith('.ipynb')]

# list of notebooks to test, in the desired order
notebooks = [
    'simple_test.ipynb',
    'advanced_test.ipynb',
    'dhdt_processing.ipynb',
    'advanced_test_spinup_elev_change_calib.ipynb',
    'advanced_test_tw.ipynb',
]


@pytest.mark.parametrize('notebook', notebooks)
def test_notebook(notebook):
    """
    Run pytest with nbmake on the specified notebook.

    This test is parameterized to run each notebook individually,
    preserving the order defined in the `notebooks` list.
    """
    subprocess.check_call(['pytest', '--nbmake', os.path.join(nb_dir, notebook)])
