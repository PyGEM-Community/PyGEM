(install_pygem_target)=
# Installation
The Python Glacier Evolution Model has been packaged using Poetry and is hosted on the Python Package Index ([PyPI](https://pypi.org/project/pygem/)), such that all dependencies should install seamlessly. It is recommended that users create a [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/index.html) environment from which to install the model dependencies and core code. If you do not yet have conda installed, see [conda's documentation](https://docs.conda.io/projects/conda/en/latest/user-guide/install) for instructions.

Next, choose your preferred PyGEM installation option:<br>
- [**stable**](stable_install_target): this is the latest version officially released and has a fixed version number (e.g. v1.0.1).
- [**dev**](dev_install_target): this is the development version. It might contain new features and bug fixes, but is also likely to continue to change until a new release is made. This is the recommended way if you want to work with the latest changes to the code.
- [**dev+**](dev+_install_target): this is the recommended way if you plan to contribute to the PyGEM and make your own changes to the code.

```{note}
The **dev** and **dev+** installation options require [git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git) software to be installed on your computer.
```
**Copyright note**: PyGEM's installation instructions are modified from that of [OGGM](https://docs.oggm.org/en/stable/installing-oggm.html)

(stable_install_target)=
## Stable install
The simplest **stable** installation method is to use an environment file. Right-click and save PyGEM's recommended environment file from [this link](https://github.com/PyGEM-Community/PyGEM/tree/master/docs/pygem_env.yml).

From the folder where you saved the file, run `conda env create -f pygem_environment.yml`.
```{note}
By default the environment will be named `pygem`. A different name can be specified in the environment file.
```

(dev_install_target)=
## Dev install
Install the [development version](https://github.com/PyGEM-Community/PyGEM/tree/dev) of PyGEM in your conda environment using pip:
```
pip uninstall pygem
pip install git+https://github.com/PyGEM-Community/pygem/@dev
```

(dev+_install_target)=
## Dev+ install
If you intend to access and make your own edits to the model, is is recommended that you clone the source code locally to your computer.

Either clone [PyGEM's GitHub repository](https://github.com/PyGEM-Community/PyGEM) directly, or initiate your own fork to clone. See [here](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/fork-a-repo) for instructions on how to fork a repo.

If PyGEM was already installed in your conda environment it is recommended that you first uninstall:
```
pip uninstall pygem
```

Next, clone PyGEM. This will place the code at your current directory, so you may wish to navigate to a desired location in your terminal before cloning:
```
git clone https://github.com/PyGEM-Community/PyGEM.git
```
If you opted to create your own fork, clone using appropriate repo URL: `git clone https://github.com/YOUR-USERNAME/PyGEM.git`

Navigate to root project directory:
```
cd pygem
```

Install PyGEM in 'editable' mode:
```
pip install -e .
```

Installing a package in editable mode (also called development mode) creates a symbolic link to your source code directory (*/path/to/your/PyGEM/clone*), rather than copying the package files into the site-packages directory. This allows you to modify the package code without reinstalling it. Changes to the source code take effect immediately without needing to reinstall the package, thus efficiently facilitating development.<br>

To contribute to PyGEM's development, see the [contribution guide](contributing_pygem_target).

(setup_target)=
# Setup
Following installation, an initialization script should to be executed.

The initialization script accomplishes two things:
1. Initializes the PyGEM configuration file *~/PyGEM/config.yaml*. If this file already exists, an overwrite prompt will appear.
2. Downloads and unzips a series of sample data files to *~/PyGEM/*, which can also be manually downloaded [here](https://drive.google.com/file/d/1Wu4ZqpOKxnc4EYhcRHQbwGq95FoOxMfZ/view?usp=drive_link).

Run the initialization script by entering the following in the terminal:
```
initialize
```

# Demonstration Notebooks
A series of accompanying Jupyter notebooks have been produced for demonstrating the functionality of PyGEM. These can be acquired from [GitHub](https://github.com/PyGEM-Community/PyGEM-notebooks).