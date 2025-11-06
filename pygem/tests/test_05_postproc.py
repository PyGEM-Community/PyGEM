import glob
import os
import subprocess

import numpy as np
import pytest
import xarray as xr

from pygem.setup.config import ConfigManager


@pytest.fixture(scope='module')
def rootdir():
    config_manager = ConfigManager()
    pygem_prms = config_manager.read_config()
    return pygem_prms['root']


def test_postproc_subannual_mass(rootdir):
    """
    Test the postproc_subannual_mass CLI script.
    """
    simdir = os.path.join(rootdir, 'Output', 'simulations', '01', 'CESM2', 'ssp245', 'stats')

    # Run postproc_monthyl_mass CLI script
    subprocess.run(['postproc_subannual_mass', '-simdir', simdir], check=True)


def test_postproc_binned_subannual_thick(rootdir):
    """
    Test the postproc_binned_subannual_thick CLI script.
    """
    simdir = os.path.join(rootdir, 'Output', 'simulations', '01', 'CESM2', 'ssp245', 'binned')

    # Run postproc_monthyl_mass CLI script
    subprocess.run(['postproc_binned_subannual_thick', '-simdir', simdir], check=True)


def test_postproc_compile_simulations(rootdir):
    """
    Test the postproc_compile_simulations CLI script.
    """

    # Run postproc_compile_simulations CLI script
    subprocess.run(
        [
            'postproc_compile_simulations',
            '-rgi_region01',
            '01',
            '-option_calibration',
            'MCMC',
            '-sim_climate_name',
            'CESM2',
            '-sim_climate_scenario',
            'ssp245',
            '-sim_startyear',
            '2000',
            '-sim_endyear',
            '2100',
        ],
        check=True,
    )

    # Check if output files were created
    compdir = os.path.join(rootdir, 'Output', 'simulations', 'compile', 'glacier_stats')
    output_files = glob.glob(os.path.join(compdir, '**', '*.nc'), recursive=True)
    assert output_files, f'No output files found in {compdir}'


def test_check_compiled_product(rootdir):
    """
    Verify the contents of the files created by postproc_compile_simulations.
    """
    # skip variables that are not in the compiled products
    vars_to_skip = [
        'glac_temp',
        'glac_mass_change_ignored_annual',
        'offglac_prec',
        'offglac_refreeze',
        'offglac_melt',
        'offglac_snowpack',
    ]

    simpath = os.path.join(
        rootdir,
        'Output',
        'simulations',
        '01',
        'CESM2',
        'ssp245',
        'stats',
        '1.03622_CESM2_ssp245_MCMC_ba1_10sets_2000_2100_all.nc',
    )
    compdir = os.path.join(rootdir, 'Output', 'simulations', 'compile', 'glacier_stats')

    with xr.open_dataset(simpath) as simds:
        # loop through vars
        vars_to_check = [name for name, var in simds.variables.items() if len(var.dims) > 1]
        vars_to_check = [item for item in vars_to_check if item not in vars_to_skip]

        for var in vars_to_check:
            # skip mad
            if 'mad' in var:
                continue
            simvar = simds[var]
            comppath = os.path.join(compdir, var, '01')
            comppath = glob.glob(f'{comppath}/R01_{var}*.nc')[0]
            assert os.path.isfile(comppath), f'Compiled product not found for {var} at {comppath}'
            with xr.open_dataset(comppath) as compds:
                compvar = compds[var]

                # verify coords (compiled product has one more dimension for the `model`)
                assert compvar.ndim == simvar.ndim + 1

                # pull data values
                simvals = simvar.values
                compvals = compvar.values[0, :, :]  # 0th index is the glacier index

                # check that compiled product has same shape as original data
                assert (
                    simvals.shape == compvals.shape
                ), f'Compiled product shape {compvals.shape} does not match original data shape {simvals.shape}'
                # check that compiled product matches original data
                assert np.allclose(
                    simvals, compvals, rtol=1e-8, atol=1e-12, equal_nan=True
                ), f'Compiled product for {var} does not match original data'
