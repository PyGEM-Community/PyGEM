"""
Python Glacier Evolution Model (PyGEM)

copyright © 2018 David Rounce <drounce@cmu.edu>

Distrubted under the MIT lisence
"""
import os
import logging

import numpy as np
import pandas as pd
import pickle
import rasterio
import xarray as xr

from oggm import cfg
from oggm.utils import entity_task
from oggm.core.gis import rasterio_to_gdir
from oggm.utils import ncDataset
# pygem imports
from pygem.setup.config import ConfigManager
# instantiate ConfigManager
config_manager = ConfigManager()
# read the config
pygem_prms = config_manager.read_config()

if not 'consensus_mass' in cfg.BASENAMES:
    cfg.BASENAMES['consensus_mass'] = ('consensus_mass.pkl', 'Glacier mass from consensus ice thickness data')
if not 'consensus_h' in cfg.BASENAMES:
    cfg.BASENAMES['consensus_h'] = ('consensus_h.tif', 'Raster of consensus ice thickness data')

# Module logger
log = logging.getLogger(__name__)

@entity_task(log, writes=['consensus_mass'])
def consensus_gridded(gdir, h_consensus_fp=f"{pygem_prms['root']}/{pygem_prms['calib']['data']['icethickness']['h_consensus_relpath']}", add_mass=True, add_to_gridded=True):
    """Bin consensus ice thickness and add total glacier mass to the given glacier directory
    
    Updates the 'inversion_flowlines' save file and creates new consensus_mass.pkl
    
    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        where to write the data
    """
    # If binned mb data exists, then write to glacier directory
    h_fn = h_consensus_fp + 'RGI60-' + gdir.rgi_region + '/' + gdir.rgi_id + '_thickness.tif'
    assert os.path.exists(h_fn), 'Error: h_consensus_fullfn for ' + gdir.rgi_id + ' does not exist.'
        
    # open consensus ice thickness estimate
    h_dr = rasterio.open(h_fn, 'r', driver='GTiff')
    h = h_dr.read(1).astype(rasterio.float32)
    
    # Glacier mass [kg]
    glacier_mass_raw = (h * h_dr.res[0] * h_dr.res[1]).sum() * pygem_prms['constants']['density_ice']
#    print(glacier_mass_raw)

    if add_mass:
        # Pickle data
        consensus_fn = gdir.get_filepath('consensus_mass')
        with open(consensus_fn, 'wb') as f:
            pickle.dump(glacier_mass_raw, f)
        
    
    if add_to_gridded:
        rasterio_to_gdir(gdir, h_fn, 'consensus_h', resampling='bilinear')
        output_fn = gdir.get_filepath('consensus_h')
        # append the debris data to the gridded dataset
        with rasterio.open(output_fn) as src:
            grids_file = gdir.get_filepath('gridded_data')
            with ncDataset(grids_file, 'a') as nc:
                # Mask values
                glacier_mask = nc['glacier_mask'][:] 
                data = src.read(1) * glacier_mask
                # Pixel area
                pixel_m2 = abs(gdir.grid.dx * gdir.grid.dy)
                # Glacier mass [kg] reprojoected (may lose or gain mass depending on resampling algorithm)
                glacier_mass_reprojected = (data * pixel_m2).sum() * pygem_prms['constants']['density_ice']
                # Scale data to ensure conservation of mass during reprojection
                data_scaled = data * glacier_mass_raw / glacier_mass_reprojected
#                glacier_mass = (data_scaled * pixel_m2).sum() * pygem_prms['constants']['density_ice']
#                print(glacier_mass)
                
                # Write data
                vn = 'consensus_h'
                if vn in nc.variables:
                    v = nc.variables[vn]
                else:
                    v = nc.createVariable(vn, 'f8', ('y', 'x', ), zlib=True)
                v.units = 'm'
                v.long_name = 'Consensus ice thicknness'
                v[:] = data_scaled
    

@entity_task(log, writes=['inversion_flowlines'])
def consensus_binned(gdir):
    """Bin consensus ice thickness ice estimates.
    
    Updates the 'inversion_flowlines' save file.
    
    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        where to write the data
    """
    flowlines = gdir.read_pickle('inversion_flowlines')
    fl = flowlines[0]
    
    assert len(flowlines) == 1, 'Error: binning debris data set up only for single flowlines at present'
    
    # Add binned debris thickness and enhancement factors to flowlines
    ds = xr.open_dataset(gdir.get_filepath('gridded_data'))
    glacier_mask = ds['glacier_mask'].values
    topo = ds['topo_smoothed'].values
    h = ds['consensus_h'].values

    # Only bin on-glacier values
    idx_glac = np.where(glacier_mask == 1)
    topo_onglac = topo[idx_glac]
    h_onglac = h[idx_glac]

    # Bin edges        
    nbins = len(fl.dis_on_line)
    z_center = (fl.surface_h[0:-1] + fl.surface_h[1:]) / 2
    z_bin_edges = np.concatenate((np.array([topo[idx_glac].max() + 1]), 
                                  z_center, 
                                  np.array([topo[idx_glac].min() - 1])))
    # Loop over bins and calculate the mean debris thickness and enhancement factor for each bin
    h_binned = np.zeros(nbins) 
    for nbin in np.arange(0,len(z_bin_edges)-1):
        bin_max = z_bin_edges[nbin]
        bin_min = z_bin_edges[nbin+1]
        bin_idx = np.where((topo_onglac < bin_max) & (topo_onglac >= bin_min))
        try:
            h_binned[nbin] = h_onglac[bin_idx].mean()
        except:
            h_binned[nbin] = 0
            
    fl.consensus_h = h_binned
    
    # Overwrite pickle
    gdir.write_pickle(flowlines, 'inversion_flowlines')
        