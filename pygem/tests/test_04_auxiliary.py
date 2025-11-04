import glob
import os
import subprocess

import pytest

from pygem.setup.config import ConfigManager

"""
Test suite to any necessary aux"""


@pytest.fixture(scope='module')
def rootdir():
    config_manager = ConfigManager()
    pygem_prms = config_manager.read_config()
    return pygem_prms['root']


def test_simulation_massredistribution_dynamics(rootdir):
    """
    Test the run_simulation CLI script with the "MassRedistributionCurves" dynamical option.
    """

    # Run run_simulation CLI script
    subprocess.run(
        [
            'run_simulation',
            '-rgi_glac_number',
            '1.03622',
            '-option_calibration',
            'MCMC',
            '-sim_climate_name',
            'ERA5',
            '-sim_startyear',
            '2000',
            '-sim_endyear',
            '2019',
            '-nsims',
            '1',
            '-option_dynamics',
            'MassRedistributionCurves',
            '-outputfn_sfix',
            'mrcdynamics_',
        ],
        check=True,
    )

    # Check if output files were created
    outdir = os.path.join(rootdir, 'Output', 'simulations', '01', 'ERA5')
    output_files = glob.glob(os.path.join(outdir, '**', '*_mrcdynamics_all.nc'), recursive=True)
    assert output_files, f'Simulation output file not found in {outdir}'


def test_simulation_no_dynamics(rootdir):
    """
    Test the run_simulation CLI script with no dynamics option.
    """

    # Run run_simulation CLI script
    subprocess.run(
        [
            'run_simulation',
            '-rgi_glac_number',
            '1.03622',
            '-option_calibration',
            'MCMC',
            '-sim_climate_name',
            'ERA5',
            '-sim_startyear',
            '2000',
            '-sim_endyear',
            '2019',
            '-nsims',
            '1',
            '-option_dynamics',
            'None',
            '-outputfn_sfix',
            'nodynamics_',
        ],
        check=True,
    )

    # Check if output files were created
    outdir = os.path.join(rootdir, 'Output', 'simulations', '01', 'ERA5')
    output_files = glob.glob(os.path.join(outdir, '**', '*_nodynamics_all.nc'), recursive=True)
    assert output_files, f'Simulation output file not found in {outdir}'
