(install_pygem_target)=
# Installation
The Python Glacier Evolution Model has been packaged using Poetry and is hosted on the Python Package Index ([PyPI](https://pypi.org/project/pygem/)), to ensure that all dependencies install seamlessly. It is recommended that users create a [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/index.html) environment from which to install the model dependencies and core code. If you do not yet have conda installed, see [conda's documentation](https://docs.conda.io/projects/conda/en/latest/user-guide/install) for instructions.

Next, choose your preferred PyGEM installation option:<br>
- [**stable**](stable_install_target): this is the latest version that has been officially released to PyPI, with a fixed version number (e.g. v1.0.1). It is intended for general use.
- [**development**](dev_install_target): this is the development version of PyGEM hosted on [GitHub](https://github.com/PyGEM-Community/PyGEM/tree/dev). It might contain new features and bug fixes, but is also likely to continue to change until a new release is made. This is the recommended option if you want to work with the latest changes to the code. Note, this installation options require [git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git) software to be installed on your computer.

**Copyright note**: PyGEM's installation instructions are modified from that of [OGGM](https://docs.oggm.org/en/stable/installing-oggm.html)

(stable_install_target)=
## Stable install
The simplest **stable** installation method is to use an environment file. Right-click and save PyGEM's recommended environment file from [this link](https://raw.githubusercontent.com/PyGEM-Community/PyGEM/refs/heads/master/docs/pygem_environment.yml).

From the folder where you saved the file, run `conda env create -f pygem_environment.yml`.
```{note}
By default the environment will be named `pygem`. A different name can be specified in the environment file.
```

(dev_install_target)=
## Development install
Install the [development version](https://github.com/PyGEM-Community/PyGEM/tree/dev) of PyGEM in your conda environment using pip:
```
pip uninstall pygem
pip install git+https://github.com/PyGEM-Community/pygem/@dev
```

If you intend to access and make your own edits to the model's source code, see the [contribution guide](contributing_pygem_target).

(setup_target)=
# Setup
Following installation, an initialization script should be executed.

The initialization script accomplishes two things:
1. Initializes the PyGEM configuration file *~/PyGEM/config.yaml*. If this file already exists, an overwrite prompt will appear.
2. Downloads and unzips a series of sample data files to *~/PyGEM/*, which can also be manually downloaded [here](https://drive.google.com/file/d/1Wu4ZqpOKxnc4EYhcRHQbwGq95FoOxMfZ/view?usp=drive_link).

Run the initialization script by entering the following in the terminal:
```
initialize
```

# Demonstration Notebooks
A series of accompanying Jupyter notebooks has been produced for demonstrating the functionality of PyGEM. These are hosted in the [PyGEM-notebooks repository](https://github.com/PyGEM-Community/PyGEM-notebooks).
