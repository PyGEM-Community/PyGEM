import os
import subprocess

import pytest

# Get all notebooks in the PyGEM-notebooks repository
nb_dir = os.environ.get('PYGEM_NOTEBOOKS_DIRPATH') or os.path.join(
    os.path.expanduser('~'), 'PyGEM-notebooks'
)
# TODO #54: Test all notebooks
# notebooks = [f for f in os.listdir(nb_dir) if f.endswith('.ipynb')]

# list of notebooks to test, in the desired order
notebooks = [
    'simple_test.ipynb',
    'advanced_test.ipynb',
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
