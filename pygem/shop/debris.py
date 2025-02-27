"""
Python Glacier Evolution Model (PyGEM)

copyright © 2018 David Rounce <drounce@cmu.edu>

Distrubted under the MIT lisence
"""
import os
import logging

import numpy as np
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

"""
To-do list:
  - Add binned debris-covered area to flowlines
  - Fabien may have better way of processing debris rasters to gridded data without exporting .tif
"""

# Module logger
log = logging.getLogger(__name__)

# Add the new name "hd" to the list of things that the GlacierDirectory understands
if not 'debris_hd' in cfg.BASENAMES:
    cfg.BASENAMES['debris_hd'] = ('debris_hd.tif', 'Raster of debris thickness data')
if not 'debris_ed' in cfg.BASENAMES:
    cfg.BASENAMES['debris_ed'] = ('debris_ed.tif', 'Raster of debris enhancement factor data')

@entity_task(log, writes=['debris_hd', 'debris_ed'])
def debris_to_gdir(gdir, debris_dir=f"{pygem_prms['root']}/{pygem_prms['mb']['debris_relpath']}", add_to_gridded=True, hd_max=5, hd_min=0, ed_max=10, ed_min=0):
    """Reproject the debris thickness and enhancement factor files to the given glacier directory
    
    Variables are exported as new files in the glacier directory.
    Reprojecting debris data from one map proj to another is done. 
    We use bilinear interpolation to reproject the velocities to the local glacier map.
    
    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        where to write the data
    """
    
    assert os.path.exists(debris_dir), "Error: debris directory does not exist."

    hd_dir = debris_dir + 'hd_tifs/' + gdir.rgi_region + '/'
    ed_dir = debris_dir + 'ed_tifs/' + gdir.rgi_region + '/'
    
    glac_str_nolead = str(int(gdir.rgi_region)) + '.' + gdir.rgi_id.split('-')[1].split('.')[1]
    
    # If debris thickness data exists, then write to glacier directory
    if os.path.exists(hd_dir + glac_str_nolead + '_hdts_m.tif'):
        hd_fn = hd_dir + glac_str_nolead + '_hdts_m.tif'
    elif os.path.exists(hd_dir + glac_str_nolead + '_hdts_m_extrap.tif'):
        hd_fn = hd_dir + glac_str_nolead + '_hdts_m_extrap.tif'
    else: 
        hd_fn = None
        
    if hd_fn is not None:
        rasterio_to_gdir(gdir, hd_fn, 'debris_hd', resampling='bilinear')
    if add_to_gridded and hd_fn is not None:
        output_fn = gdir.get_filepath('debris_hd')
        
        # append the debris data to the gridded dataset
        with rasterio.open(output_fn) as src:
            grids_file = gdir.get_filepath('gridded_data')
            with ncDataset(grids_file, 'a') as nc:
                # Mask values
                glacier_mask = nc['glacier_mask'][:] 
                data = src.read(1) * glacier_mask
                data[data>hd_max] = 0
                data[data<hd_min] = 0
                
                # Write data
                vn = 'debris_hd'
                if vn in nc.variables:
                    v = nc.variables[vn]
                else:
                    v = nc.createVariable(vn, 'f8', ('y', 'x', ), zlib=True)
                v.units = 'm'
                v.long_name = 'Debris thicknness'
                v[:] = data
        
    # If debris enhancement factor data exists, then write to glacier directory
    if os.path.exists(ed_dir + glac_str_nolead + '_meltfactor.tif'):
        ed_fn = ed_dir + glac_str_nolead + '_meltfactor.tif'
    elif os.path.exists(ed_dir + glac_str_nolead + '_meltfactor_extrap.tif'):
        ed_fn = ed_dir + glac_str_nolead + '_meltfactor_extrap.tif'
    else: 
        ed_fn = None
        
    if ed_fn is not None:
        rasterio_to_gdir(gdir, ed_fn, 'debris_ed', resampling='bilinear')
    if add_to_gridded and ed_fn is not None:
        output_fn = gdir.get_filepath('debris_ed')
        # append the debris data to the gridded dataset
        with rasterio.open(output_fn) as src:
            grids_file = gdir.get_filepath('gridded_data')
            with ncDataset(grids_file, 'a') as nc:
                # Mask values
                glacier_mask = nc['glacier_mask'][:] 
                data = src.read(1) * glacier_mask
                data[data>ed_max] = 1
                data[data<ed_min] = 1
                # Write data
                vn = 'debris_ed'
                if vn in nc.variables:
                    v = nc.variables[vn]
                else:
                    v = nc.createVariable(vn, 'f8', ('y', 'x', ), zlib=True)
                v.units = '-'
                v.long_name = 'Debris enhancement factor'
                v[:] = data



@entity_task(log, writes=['inversion_flowlines'])
def debris_binned(gdir, ignore_debris=False, fl_str='inversion_flowlines'):
    """Bin debris thickness and enhancement factors.
    
    Updates the 'inversion_flowlines' save file.
    
    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        where to write the data
    """
    # Nominal glaciers will throw error, so make sure inversion_flowlines exist
    try:
        flowlines = gdir.read_pickle(fl_str)
        fl = flowlines[0]
        
        assert len(flowlines) == 1, 'Error: binning debris only works for single flowlines at present'
        
    except:
        flowlines = None        
    
    if flowlines is not None:
        # Add binned debris thickness and enhancement factors to flowlines
        if os.path.exists(gdir.get_filepath('debris_hd')) and ignore_debris==False:
            ds = xr.open_dataset(gdir.get_filepath('gridded_data'))
            glacier_mask = ds['glacier_mask'].values
            topo = ds['topo_smoothed'].values
            hd = ds['debris_hd'].values
            ed = ds['debris_ed'].values
        
            # Only bin on-glacier values
            idx_glac = np.where(glacier_mask == 1)
            topo_onglac = topo[idx_glac]
            hd_onglac = hd[idx_glac]
            ed_onglac = ed[idx_glac]
    
            # Bin edges        
            nbins = len(fl.dis_on_line)
            z_center = (fl.surface_h[0:-1] + fl.surface_h[1:]) / 2
            z_bin_edges = np.concatenate((np.array([topo[idx_glac].max() + 1]), 
                                          z_center, 
                                          np.array([topo[idx_glac].min() - 1])))
            # Loop over bins and calculate the mean debris thickness and enhancement factor for each bin
            hd_binned = np.zeros(nbins)
            ed_binned = np.ones(nbins)    
            for nbin in np.arange(0,len(z_bin_edges)-1):
                bin_max = z_bin_edges[nbin]
                bin_min = z_bin_edges[nbin+1]
                bin_idx = np.where((topo_onglac < bin_max) & (topo_onglac >= bin_min))[0]
                # Debris thickness and enhancement factors for on-glacier bins
                if len(bin_idx) > 0:
                    hd_binned[nbin] = np.nanmean(hd_onglac[bin_idx])
                    ed_binned[nbin] = np.nanmean(ed_onglac[bin_idx])
                    hd_terminus = hd_binned[nbin]
                    ed_terminus = ed_binned[nbin]
                # Debris thickness and enhancement factors for bins below the present-day glacier
                #  assume an advancing glacier will have debris thickness equal to the terminus
                elif np.mean([bin_min, bin_max]) < topo[idx_glac].min():
                    hd_binned[nbin] = hd_terminus
                    ed_binned[nbin] = ed_terminus
                else:
                    hd_binned[nbin] = 0
                    ed_binned[nbin] = 1        
                    
            fl.debris_hd = hd_binned
            fl.debris_ed = ed_binned
            
        else:
            nbins = len(fl.dis_on_line)
            fl.debris_hd = np.zeros(nbins)
            fl.debris_ed = np.ones(nbins)
        
        # Overwrite pickle
        gdir.write_pickle(flowlines, fl_str)
        