"""
Python Glacier Evolution Model (PyGEM)

copyright © 2025 Brandon Tober <btober@cmu.edu>, David Rounce <drounce@cmu.edu>

Distributed under the MIT license
"""

# Built-in libaries
import os
import numpy as np
import pandas as pd

# pygem imports
from pygem.setup.config import ConfigManager

# instantiate ConfigManager
config_manager = ConfigManager()
# read the config
pygem_prms = config_manager.read_config()


def meltextent_1d_calib_tbias(gdir, z_step=20, mo_cutoff=3):
    """
    Add 1d melt extent observations to the given glacier directory

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        where to write the data
    z_step : elevation bin step
    mo_cutoff : month cutoff for melt onset (e.g., 3 indicated no melt onset before March)
    """
    # read meltextent json
    meltextent_1d_dict = gdir.read_json('meltextent_1d')

    # extract melt onset from melt extent
    meltonset_df = meltextent_to_meltonset(meltextent_1d_dict, z_step=z_step, mo_cutoff=mo_cutoff)

    # calibrate t_bias with melt onset data




    # # all of these are lists
    # data_dict = {
    #     'date': dates,
    #     'z': z,
    #     'z_min': z_min,
    #     'z_max': z_max,
    #     'direction': direction,
    #     'ref_dem': ref_dem,
    #     'ref_dem_year': ref_dem_year,
    # }



def meltextent_to_meltonset(meltextent_1d_dict, z_step=20, mo_cutoff=3):
    """
    Convert melt extent to melt onset

    Parameters
    ----------
    meltextent_1d_dict : loaded melt extent dictionary
    z_step : elevation bin step
    mo_cutoff : month cutoff for melt onset (e.g., 3 indicated no melt onset before March)
    """
    # Convert dictionary to DataFrame for easier handling
    df = pd.DataFrame(meltextent_1d_dict)
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df = df.set_index('date')

    # Prepare result DataFrame
    years = sorted(df['year'].unique())

    # Get elevation bins of melt
    elevation_bins = np.arange(min(df['z_min']), max(df['z_max'])+z_step, z_step)
    if len(elevation_bins) <= 1:
        raise ValueError(f"Insufficient melt extent elevation bins (melt extent must span more than one elevation bin)")

    # Prepare output: rows = elevation bins, columns = years
    meltonset_df = pd.DataFrame(index=elevation_bins, columns=years)

    # Find onset date for each elevation and year
    for yr in years:
        df_year = df[df['year'] == yr]
        for elev in elevation_bins:
            subset = df_year[df_year['z'] >= elev]

            if not subset.empty:
                # Earliest date that reached this elevation or higher
                meltonset_df.loc[elev, yr] = subset.index.min()

                # Restrict to remove dates before mo_cutoff of that year
                subset_after_date = subset[subset.index >= pd.Timestamp(f'{yr}-{str(mo_cutoff).zfill(2)}-01')]
                if not subset_after_date.empty:
                    meltonset_df.loc[elev, yr] = subset_after_date.index.min()

        # Ignore onset of lowest elevation bin
        meltonset_df.loc[elevation_bins[0], yr] = meltonset_df.loc[elevation_bins[1], yr]

    meltonset_df = meltonset_df.map(lambda x: float(x.strftime('%Y%m%d')) if pd.notnull(x) else 0)
    meltonset_df = meltonset_df.reset_index().rename(columns={'index': 'Elev'})
    return meltonset_df
