"""
Python Glacier Evolution Model (PyGEM)

copyright © 2025 Brandon Tober <btober@cmu.edu>, David Rounce <drounce@cmu.edu>

Distributed under the MIT license

Raster reprojection and processing to glacier directory framework adapted from the Open Global Glacier Model (OGGM) shop/hugonnet_maps.py module.
"""

import glob
import logging
import os
import re
import warnings

import numpy as np
import pandas as pd
import xarray as xr
from packaging.version import Version

try:
    import rasterio
    from rasterio import MemoryFile
    from rasterio.warp import Resampling, calculate_default_transform, reproject

    try:
        # rasterio V > 1.0
        from rasterio.merge import merge as merge_tool
    except ImportError:
        from rasterio.tools.merge import merge as merge_tool
except ImportError:
    pass
import geopandas as gpd
from oggm import cfg, tasks, utils
from shapely.geometry import box

from pygem.setup.config import ConfigManager
from pygem.utils._funcs import parse_period

config_manager = ConfigManager()
# read the config
pygem_prms = config_manager.read_config()


# Module logger
log = logging.getLogger(__name__)

data_basedir = f'{pygem_prms["root"]}/{pygem_prms["calib"]["data"]["elev_change"]["dhdt_2d_relpath"]}/'


def raster_overlaps_glacier(raster_path, glacier_geom):
    """Return True if raster overlaps glacier extent (reprojects if needed)."""
    with rasterio.open(raster_path) as src:
        geom = box(*src.bounds)
        if src.crs.to_string() != 'EPSG:4326':
            geom = gpd.GeoSeries([geom], crs=src.crs).to_crs('EPSG:4326').iloc[0]
        return geom.intersects(glacier_geom)


@utils.entity_task(log, writes=['gridded_data'])
def dhdt_to_gdir(
    gdir,
    raster_path=None,
    period='',
    period_format=pygem_prms['calib']['data']['massbalance']['massbalance_period_date_format'],
    period_delimiter=pygem_prms['calib']['data']['massbalance']['massbalance_period_delimiter'],
    verbose=False,
):
    """Add 2d dhdt data to this glacier directory.

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        the glacier directory to process
    raster_path : str, optional
        A path to a single raster file or a directory containing raster files.
        If None, defaults to the standard data_basedir.
    period: str, optional
        A string indicating the time period of the dhdt data (e.g. '2000-01-01_2020-01-01').
    verbose : bool, optional
    """
    # step 0. determine base directory
    if raster_path is None:
        basedir = os.path.normpath(data_basedir)
    elif os.path.isfile(raster_path):
        # single file
        tif_files = [raster_path]
        basedir = os.path.dirname(raster_path)
    elif os.path.isdir(raster_path):
        basedir = raster_path
        tif_files = glob.glob(os.path.join(basedir, '*.tif'))
    else:
        raise ValueError(f"Provided raster_path '{raster_path}' is not valid.")

    # step 1. get glacier bounds
    lon_ex, lat_ex = gdir.extent_ll
    lon_ex = [np.floor(lon_ex[0]) - 1e-9, np.ceil(lon_ex[1]) + 1e-9]
    lat_ex = [np.floor(lat_ex[0]) - 1e-9, np.ceil(lat_ex[1]) + 1e-9]
    # define glacier bounding box
    glacier_bounds = box(lon_ex[0], lat_ex[0], lon_ex[1], lat_ex[1])

    # step 2. if raster_path was a directory or None, look for tif files
    if raster_path is None or os.path.isdir(raster_path):
        RO1_basedir = os.path.join(basedir, f'{gdir.rgi_region.zfill(2)}')
        if os.path.isdir(RO1_basedir):
            tif_files = glob.glob(os.path.join(RO1_basedir, '*.tif'))
        else:
            tif_files = glob.glob(os.path.join(basedir, '*.tif'))

    if not tif_files:
        log.info(f'No rasters found for glacier {gdir.rgi_id}')
        return

    # match files by rgi_id if possible
    glac_no = '{0:0.5f}'.format(float(gdir.rgi_id.split('-')[1]))
    flist = [
        f
        for f in tif_files
        if (gdir.rgi_id in os.path.basename(f) or glac_no in os.path.basename(f))
        and raster_overlaps_glacier(f, glacier_bounds)
    ]

    # if no RGI ID match, look for any raster overlapping glacier extent
    if not flist:
        flist = [f for f in tif_files if raster_overlaps_glacier(f, glacier_bounds)]

    if verbose:
        print(f'Found {len(flist)} files for {gdir.rgi_id}:', [f.split('/')[-1] for f in flist])

    if not flist:
        log.info(f'No rasters overlap glacier {gdir.rgi_id}')
        return

    # search for date strings in files
    pattern = r'\d{4}-\d{2}-\d{2}_\d{4}-\d{2}-\d{2}'
    matches = list({re.search(pattern, f).group(0) for f in flist if re.search(pattern, f)})

    if matches:
        if len(set(matches)) != 1:
            raise ValueError(
                f'It seems the dhdt files for glacier {gdir.rgi_id} may cover different date ranges: {set(matches)}'
            )
        period = matches[0]
    if period:
        t1, t2 = parse_period(period, date_format=period_format, delimiter=period_delimiter)

    # A glacier area can cover more than one tile:
    if len(flist) == 1:
        dem_dss = [rasterio.open(flist[0])]  # if one tile, just open it
        file_crs = dem_dss[0].crs
        dem_data = rasterio.band(dem_dss[0], 1)
        if Version(rasterio.__version__) >= Version('1.0'):
            src_transform = dem_dss[0].transform
        else:
            src_transform = dem_dss[0].affine
        nodata = dem_dss[0].meta.get('nodata', None)
    else:
        dem_dss = [rasterio.open(s) for s in flist]  # list of rasters

        # make sure all files have the same crs and reproject if needed;
        # defining the target crs to the one most commonly used, minimizing
        # the number of files for reprojection
        crs_list = np.array([dem_ds.crs.to_string() for dem_ds in dem_dss])
        unique_crs, crs_counts = np.unique(crs_list, return_counts=True)
        file_crs = rasterio.crs.CRS.from_string(unique_crs[np.argmax(crs_counts)])

        if len(unique_crs) != 1:
            # more than one crs, we need to do reprojection
            memory_files = []
            for i, src in enumerate(dem_dss):
                if file_crs != src.crs:
                    transform, width, height = calculate_default_transform(
                        src.crs, file_crs, src.width, src.height, *src.bounds
                    )
                    kwargs = src.meta.copy()
                    kwargs.update({'crs': file_crs, 'transform': transform, 'width': width, 'height': height})

                    reprojected_array = np.empty(shape=(src.count, height, width), dtype=src.dtypes[0])
                    # just for completeness; even the data only has one band
                    for band in range(1, src.count + 1):
                        reproject(
                            source=rasterio.band(src, band),
                            destination=reprojected_array[band - 1],
                            src_transform=src.transform,
                            src_crs=src.crs,
                            dst_transform=transform,
                            dst_crs=file_crs,
                            resampling=Resampling.nearest,
                        )

                    memfile = MemoryFile()
                    with memfile.open(**kwargs) as mem_dst:
                        mem_dst.write(reprojected_array)
                    memory_files.append(memfile)
                else:
                    memfile = MemoryFile()
                    with memfile.open(**src.meta) as mem_src:
                        mem_src.write(src.read())
                    memory_files.append(memfile)

            with rasterio.Env():
                datasets_to_merge = [memfile.open() for memfile in memory_files]
                nodata = datasets_to_merge[0].meta.get('nodata', None)
                dem_data, src_transform = merge_tool(datasets_to_merge, nodata=nodata)
                # close datasets
            for mf in memory_files:
                mf.close()
            for ds_merge in datasets_to_merge:
                ds_merge.close()
        else:
            # only one single crs occurring, no reprojection needed
            nodata = dem_dss[0].meta.get('nodata', None)
            dem_data, src_transform = merge_tool(dem_dss, nodata=nodata)

    # Set up profile for writing output
    with rasterio.open(gdir.get_filepath('dem')) as dem_ds:
        dst_array = dem_ds.read().astype(np.float32)
        dst_array[:] = np.nan
        profile = dem_ds.profile
        transform = dem_ds.transform
        dst_crs = dem_ds.crs

    # Set up profile for writing output
    profile.update(
        {
            'nodata': np.nan,
        }
    )

    resampling = Resampling.bilinear

    with MemoryFile() as dest:
        reproject(
            # Source parameters
            source=dem_data,
            src_crs=file_crs,
            src_transform=src_transform,
            src_nodata=nodata,
            # Destination parameters
            destination=dst_array,
            dst_transform=transform,
            dst_crs=dst_crs,
            dst_nodata=np.nan,
            # Configuration
            resampling=resampling,
        )
        dest.write(dst_array)

    for dem_ds in dem_dss:
        dem_ds.close()

    # Write
    with utils.ncDataset(gdir.get_filepath('gridded_data'), 'a') as nc:
        vn = 'dhdt'
        if vn in nc.variables:
            v = nc.variables[vn]
        else:
            v = nc.createVariable(
                vn,
                'f4',
                (
                    'y',
                    'x',
                ),
                zlib=True,
                fill_value=np.nan,
            )
        v.units = 'm'
        ln = 'dhdt'
        v.long_name = ln
        data_str = ' '.join(flist) if len(flist) > 1 else flist[0]
        v.data_source = data_str
        v.period = period
        v.t1 = t1.strftime('%Y-%m-%d') if period else ''
        v.t2 = t2.strftime('%Y-%m-%d') if period else ''
        v[:] = np.squeeze(dst_array).astype(np.float32)


@utils.entity_task(log)
def dhdt_statistics(gdir, compute_massbalance=True):
    """Gather statistics about the dhdt data."""

    d = dict()

    # Easy stats - this should always be possible
    d['rgi_id'] = gdir.rgi_id
    d['rgi_region'] = gdir.rgi_region
    d['rgi_subregion'] = gdir.rgi_subregion
    d['rgi_area_km2'] = gdir.rgi_area_km2
    d['area_km2'] = 0
    d['perc_cov'] = 0
    d['avg_dhdt'] = np.nan

    try:
        with xr.open_dataset(gdir.get_filepath('gridded_data')) as ds:
            dhdt = ds['dhdt'].where(ds['glacier_mask'], np.nan).load()
            gridded_area = ds['glacier_mask'].sum() * gdir.grid.dx**2 * 1e-6
            d['area_km2'] = float((~dhdt.isnull()).sum() * gdir.grid.dx**2 * 1e-6)
            d['perc_cov'] = float(d['area_km2'] / gridded_area)
            with warnings.catch_warnings():
                # This can trigger an empty mean warning
                warnings.filterwarnings('ignore', category=RuntimeWarning)
                d['avg_dhdt'] = np.nanmean(dhdt.data)
        if compute_massbalance:
            # convert to m w.e. yr^-1
            d['dmdtda'] = (
                d['avg_dhdt'] * pygem_prms['constants']['density_ice'] / pygem_prms['constants']['density_water']
            )
    except (FileNotFoundError, AttributeError, KeyError):
        pass

    return d


@utils.global_task(log)
def compile_hugonnet_statistics(gdirs, filesuffix='', path=True):
    """Gather as much statistics as possible about a list of glaciers.

    It can be used to do result diagnostics and other stuffs.

    Parameters
    ----------
    gdirs : list of :py:class:`oggm.GlacierDirectory` objects
        the glacier directories to process
    filesuffix : str
        add suffix to output file
    path : str, bool
        Set to "True" in order  to store the info in the working directory
        Set to a path to store the file to your chosen location
    """
    from oggm.workflow import execute_entity_task

    out_df = execute_entity_task(dhdt_statistics, gdirs)

    out = pd.DataFrame(out_df).set_index('rgi_id')

    if path:
        if path is True:
            out.to_csv(os.path.join(cfg.PATHS['working_dir'], ('dhdt_statistics' + filesuffix + '.csv')))
        else:
            out.to_csv(path)

    return out


def dh_1d(
    gdir,
    dhdt_error=None,
    ref_dem_year=None,
    outdir=(f'{pygem_prms["root"]}/{pygem_prms["calib"]["data"]["elev_change"]["dh_1d_relpath"]}/'),
    verbose=False,
):
    """Convert the 2d dhdt data to a 1d elevation change profile and save to gdir.

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        the glacier directory to process
    """
    tasks.elevation_band_flowline(
        gdir,
        bin_variables=['dhdt'],
    )
    df = pd.read_csv(gdir.get_filepath('elevation_band_flowline'), index_col=0)

    # get dhdt time period

    with xr.open_dataset(gdir.get_filepath('gridded_data')) as ds:
        ds = ds.load()
    t2 = pd.to_datetime(ds.dhdt.attrs['t2'])
    t1 = pd.to_datetime(ds.dhdt.attrs['t1'])

    # compute bin edges
    bin_centers = df['bin_elevation'].values

    # estimate bin width (assuming uniform)
    dz = np.diff(bin_centers).mean()

    # compute edges
    bin_edges = np.concatenate(([bin_centers[0] - dz / 2], bin_centers[:-1] + dz / 2, [bin_centers[-1] + dz / 2]))

    # get bin start
    bin_start = bin_edges[:-1]
    # get bin end
    bin_end = bin_edges[1:]
    # get bin area
    bin_area = df['area']
    # compute dh - dhdt * nyears
    dh = df['dhdt'] * (t2 - t1).days / 365.25
    # get gdir ref dem
    ref_dem = gdir.dem_info.split('\n')[0]
    # need a reference dem year that the data was binned to for proper dynamic model calibration.
    # default oggm is COP90, which has variable reference year depending on region
    if not ref_dem_year:
        match = list(set(re.findall(r'\b\d{4}-\d{4}\b', gdir.dem_info)))[0]
        if match:
            ref_dem_year = round(sum(int(y) for y in match.split('-')) / 2)
    if not ref_dem_year:
        raise ValueError(f'Could not determine reference DEM year from gdir.dem_info: {gdir.dem_info}')

    if not dhdt_error:
        raise ValueError('dhdt_error must be provided to compute elevation change uncertainty.')
    dh_error = dhdt_error * (t2 - t1).days / 365.25
    # save as csv
    out_df = pd.DataFrame(
        {
            'bin_start': bin_start,
            'bin_stop': bin_end,
            'bin_area': bin_area,
            'date_start': t1.strftime('%Y-%m-%d'),
            'date_end': t2.strftime('%Y-%m-%d'),
            'dh': dh,
            'dh_sigma': dh_error,
            'ref_dem': ref_dem,
            'ref_dem_year': ref_dem_year,
        }
    )

    outfpath = os.path.join(os.path.normpath(outdir), f'{gdir.rgi_id.split("-")[1]}_elev_change_1d.csv')
    out_df.to_csv(outfpath, index=False)
    if verbose:
        print(f'Saved dh_1d to {outfpath}')
