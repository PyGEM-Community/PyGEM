## Python Glacier Evolution Model (PyGEM) TESTING

Overview: Python Glacier Evolution Model (PyGEM) is an open-source glacier evolution model coded in Python that models the transient evolution of glaciers. Each glacier is modeled independently using a monthly timestep. PyGEM has a modular framework that allows different schemes to be used for model calibration or model physics (e.g., climatic mass balance, glacier dynamics).

Manual: Details concerning the model physics, installation, and running the model may be found [here](https://pygem.readthedocs.io/en/latest/).

Usage: PyGEM is meant for large-scale glacier evolution modeling.  PyGEM<1.0.0 are no longer being actively being supported.

***

### Installation
PyGEM can be downloaded from the Python Package Index ([PyPI](https://pypi.org/project/pygem/)).  We recommend creating a dedicated [Anaconda](https://anaconda.org/) environment to house PyGEM.
```
conda create --name <environment_name> python=3.12
conda activate <environment_name>
pip install pygem
```
This will install all PyGEM dependencies within your conda environment, and set up PyGEM command line tools to run core model scripts.

***

### Setup
Following installation, an initialization script should to be executed.

The initialization script accomplishes two things:
1. Initializes the PyGEM configuration file *~/PyGEM/config.yaml*.  If this file already exists, an overwrite prompt will appear.
2. Downloads and unzips a set of sample data files to *~/PyGEM/*, which can also be manually downloaded [here](https://drive.google.com/file/d/1Wu4ZqpOKxnc4EYhcRHQbwGq95FoOxMfZ/view?usp=drive_link).

Run the initialization script by entering the following in the terminal:
```
initialize
```

***

### Development
Please report any bugs [here](https://github.com/PyGEM-Community/PyGEM/issues).

If you are interested in contributing to further development of PyGEM, we recommend forking [PyGEM](https://github.com/PyGEM-Community/PyGEM) and then [cloning](https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository) onto your local machine.

Note, if PyGEM was already installed via PyPI, first uninstall:
```
pip uninstall pygem
````

You can then use pip to install your locally cloned fork of PyGEM in 'editable' mode to easily facilitate development like so:
```
pip install -e /path/to/your/cloned/pygem/fork/
```


***

### Model Testing

To support model testing and demonstration, a suite of Jupyter notebooks can be found within a separate [PyGEM-notebooks](https://github.com/PyGEM-Community/PyGEM-notebooks) repository. 

***

<table style="width: 100%;">
  <tr>
    <td style="width: 50%;"><b>Version</b></td>
    <td style="width: 50%;">
      <a href="https://pypi.python.org/pypi/pygem"><img src="https://img.shields.io/pypi/v/pygem.svg"></a>
      &nbsp;
      <a href="https://pypi.python.org/pypi/pygem"><img src="https://img.shields.io/pypi/pyversions/pygem.svg"></a>
    </td>
  </tr>
  <tr>
    <td style="width: 50%;"><b>Citation</b></td>
    <td style="width: 50%;">
      <a href="https://www.science.org/doi/10.1126/science.abo1324"><img src="https://img.shields.io/badge/citation-Rounce%20et%20al.%20(2023;%20Science)-orange.svg"></a>
    </td>
  </tr>
  <tr>
    <td style="width: 50%;"><b>License</b></td>
    <td style="width: 50%;">
      <a href="https://github.com/PyGEM-Community/PyGEM/blob/master/LICENSE.txt"><img src="https://img.shields.io/pypi/l/pygem.svg"></a>
    </td>
  </tr>
  <tr>
    <td style="width: 50%;"><b>Systems</b></td>
    <td style="width: 50%;">
      - Ubuntu 20.04, 22.04 <br>
      - Red Hat Enterprise Linux (RHEL) 8.8 <br>
      - macOS (Intel & Apple Silicon) <br>
      - <em>Note</em>, we suggest that Windows users install<br>PyGEM using either the Windows Subsystem<br>for Linux or Oracle VirtualBox
    </td>
  </tr>
</table>
