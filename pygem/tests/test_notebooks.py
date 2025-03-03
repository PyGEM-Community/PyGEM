import os
import subprocess

import pytest

# Get all notebooks in the PyGEM-notebooks repository
nb_dir = os.environ.get("PYGEM_NOTEBOOKS_DIRPATH") or os.path.join(
    os.path.expanduser("~"), "PyGEM-notebooks"
)
notebooks = [f for f in os.listdir(nb_dir) if f.endswith(".ipynb")]


@pytest.mark.parametrize("notebook", notebooks)
def test_notebook(notebook):
    # TODO #54: Test all notebooks

    if notebook not in ("simple_test.ipynb", "advanced_test.ipynb", "advanced_test_tw.ipynb"):
        pytest.skip()
    subprocess.check_call(
        ["pytest", "--nbmake", os.path.join(nb_dir, notebook)]
    )