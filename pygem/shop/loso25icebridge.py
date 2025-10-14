"""
Python Glacier Evolution Model (PyGEM)

copyright © 2024 Brandon Tober <btober@cmu.edu>, David Rounce <drounce@cmu.edu>

Distributed under the MIT license

NASA Operation IceBridge data and processing class
"""

import datetime
import glob
import json
import os
import re
import warnings
from datetime import timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# pygem imports
from pygem.setup.config import ConfigManager

# instantiate ConfigManager
config_manager = ConfigManager()
# read the config
pygem_prms = config_manager.read_config()

__all__ = ['oib']


class oib:
    def __init__(
        self,
        oib_datpath,
        rgi6id='',
        rgi7id='',
        rgi7_rgi6_linksfp='RGI2000-v7.0-G-01_alaska-rgi6_links.csv',
    ):
        self.oib_datpath = oib_datpath
        if os.path.isfile(rgi7_rgi6_linksfp):
            self.rgi7_6_df = pd.read_csv(rgi7_rgi6_linksfp)
            self.rgi7_6_df['rgi7_id'] = self.rgi7_6_df['rgi7_id'].str.split('RGI2000-v7.0-G-').str[1]
            self.rgi7_6_df['rgi6_id'] = self.rgi7_6_df['rgi6_id'].str.split('RGI60-').str[1]
        self.rgi6id = rgi6id
        self.rgi7id = rgi7id
        self.name = None
        # instatntiate dictionary to hold all data - store the data by survey date, with each key containing a list with the binned differences and uncertainties (diffs, sigma)
        self.oib_diffs = {}
        self.dbl_diffs = {}
        self.bin_edges = None
        self.bin_centers = None
        self.bin_area = None
        # automatically map rgi6 and rgi7 ids if one is provided
        if self.rgi6id and not self.rgi7id:
            self._rgi6torgi7id()
        elif self.rgi7id and not self.rgi6id:
            self._rgi7torgi6id()

    def get_diffs(self):
        return self.oib_diffs

    def set_diffs(self, diffs_dict):
        self.oib_diffs = dict(sorted(diffs_dict.items()))

    def set_dbldiffs(self, diffs_dict):
        self.dbl_diffs = diffs_dict

    def get_dbldiffs(self):
        return self.dbl_diffs

    def set_centers(self, centers):
        self.bin_centers = centers

    def get_centers(self):
        return self.bin_centers

    def set_edges(self, edges):
        self.bin_edges = edges

    def get_edges(self):
        return self.bin_edges

    def set_area(self, area):
        self.bin_area = area

    def get_area(self):
        return self.bin_area

    def get_name(self):
        return self.name

    def get_diff_inds_map(self):
        return self.diff_inds_map

    def _rgi6torgi7id(self, debug=False):
        """
        return RGI version 7 glacier id for a given RGI version 6 id

        """
        self.rgi6id = self.rgi6id.split('.')[0].zfill(2) + '.' + self.rgi6id.split('.')[1]
        df_sub = self.rgi7_6_df.loc[self.rgi7_6_df['rgi6_id'] == self.rgi6id, :]

        if len(df_sub) == 1:
            self.rgi7id = df_sub.iloc[0]['rgi7_id']
            if debug:
                print(f'RGI6:{self.rgi6id} -> RGI7:{self.rgi7id}')
        elif len(df_sub) == 0:
            raise IndexError(f'No matching RGI7Id for {self.rgi6id}')
        elif len(df_sub) > 1:
            self.rgi7id = df_sub.sort_values(by='rgi6_area_fraction', ascending=False).iloc[0]['rgi7_id']

    def _rgi7torgi6id(self, debug=False):
        """
        return RGI version 6 glacier id for a given RGI version 7 id

        """
        self.rgi7id = self.rgi7id.split('-')[0].zfill(2) + '-' + self.rgi7id.split('-')[1]
        df_sub = self.rgi7_6_df.loc[self.rgi7_6_df['rgi7_id'] == self.rgi7id, :]
        if len(df_sub) == 1:
            self.rgi6id = df_sub.iloc[0]['rgi6_id']
            if debug:
                print(f'RGI7:{self.rgi7id} -> RGI6:{self.rgi6id}')
        elif len(df_sub) == 0:
            raise IndexError(f'No matching RGI6Id for {self.rgi7id}')
        elif len(df_sub) > 1:
            self.rgi6id = df_sub.sort_values(by='rgi7_area_fraction', ascending=False).iloc[0]['rgi6_id']

    def load(self):
        """
        load Operation IceBridge data
        """
        oib_fpath = glob.glob(f'{self.oib_datpath}/diffstats5_*{self.rgi7id}*.json')
        if len(oib_fpath) == 0:
            return
        else:
            oib_fpath = oib_fpath[0]
        # load diffstats file
        with open(oib_fpath, 'rb') as f:
            self.oib_dict = json.load(f)
            self.name = _split_by_uppercase(self.oib_dict['glacier_shortname'])

    def set_diff_inds_map(self, dates_table):
        """
        Store mapping of date pairs in deltah['dates'] to their indices in dates_table.
        """
        # create a dictionary mapping datetime values to their indices
        index_map = {value: idx for idx, value in enumerate(dates_table.date.tolist())}
        # map each date pair in deltah['dates'] to their indices, if not found store (None,None)
        self.diff_inds_map = [
            (index_map.get(val1), index_map.get(val2)) if val1 in index_map and val2 in index_map else (None, None)
            for val1, val2 in self.dbl_diffs['dates']
        ]

    def parsediffs(self, debug=False):
        """
        parse COP30-relative elevation differences
        """
        # get seasons stored in oib diffs
        seasons = list(set(self.oib_dict.keys()).intersection(['march', 'may', 'august']))
        diffs_dict = {}
        for ssn in seasons:
            for yr in list(self.oib_dict[ssn].keys()):
                # get survey date
                doy_int = int(np.ceil(self.oib_dict[ssn][yr]['mean_doy']))
                dt_obj = datetime.datetime.strptime(f'{int(yr)}-{doy_int}', '%Y-%j')
                # get survey data and filter by pixel count
                diffs = np.asarray(self.oib_dict[ssn][yr]['bin_vals']['bin_median_diffs_vec'])
                counts = np.asarray(self.oib_dict[ssn][yr]['bin_vals']['bin_count_vec'])
                # uncertainty represented by IQR
                sigmas = np.asarray(self.oib_dict[ssn][yr]['bin_vals']['bin_interquartile_range_diffs_vec'])
                # add [diffs, sigma, counts] to master dictionary
                diffs_dict[_round_to_nearest_month(dt_obj)] = [diffs, sigmas, counts]
        # Sort the dictionary by date keys
        self.set_diffs(diffs_dict)

        if debug:
            print(
                f'OIB survey dates:\n{", ".join([str(dt.year) + "-" + str(dt.month) + "-" + str(dt.day) for dt in list(self.oib_diffs.keys())])}'
            )
        # get bin centers
        self.set_centers(
            (
                np.asarray(self.oib_dict[ssn][list(self.oib_dict[ssn].keys())[0]]['bin_vals']['bin_start_vec'])
                + np.asarray(self.oib_dict[ssn][list(self.oib_dict[ssn].keys())[0]]['bin_vals']['bin_stop_vec'])
            )
            / 2
        )
        self.set_area(np.asarray(self.oib_dict['aad_dict']['hist_bin_areas_m2']))
        edges = list(self.oib_dict[ssn][list(self.oib_dict[ssn].keys())[0]]['bin_vals']['bin_start_vec'])
        edges.append(self.oib_dict[ssn][list(self.oib_dict[ssn].keys())[0]]['bin_vals']['bin_stop_vec'][-1])
        self.set_edges(np.asarray(edges))

    def terminus_mask(self, debug=False, inplace=False):
        """
        create mask of missing terminus ice using the last OIB survey.

        parameters:
        - debug: bool, whether to plot debug information.
        - inplace: bool, whether to modify in place.
        """
        diffs = self.get_diffs()
        x = self.get_centers()
        oib_diffs_masked = {}
        survey_dates = list(diffs.keys())
        inds = list(range(len(survey_dates)))[::-1]  # reverse index list
        diffs_arr = [v[0] for v in diffs.values()]  # extract first array from lists
        # find the lowest bin where area is nonzero
        lowest_bin = np.nonzero(self.get_area())[0][0]
        x = x[lowest_bin : lowest_bin + 50]
        idx = None
        mask = []

        try:
            for i in inds:
                tmp = diffs_arr[i][lowest_bin : lowest_bin + 50]

                if np.isnan(tmp).all():
                    continue  # skip if all NaN
                else:
                    # interpolate over ant nans and then find peak/trouch
                    goodmask = ~np.isnan(tmp)
                    tmp = np.interp(x, x[goodmask], tmp[goodmask])
                    # identify peak/trough based on survey year
                    if survey_dates[i].year > 2013:
                        idx = np.nanargmin(tmp) + lowest_bin  # look for a trough
                    else:
                        idx = np.nanargmax(-tmp) + lowest_bin  # look for a peak
                    mask = np.arange(0, idx + 1, 1)  # create mask range
                    break  # stop once the first valid index is found
            if debug:
                plt.figure()
                cmap = plt.cm.rainbow(np.linspace(0, 1, len(inds)))
                for i in inds[::-1]:
                    plt.plot(
                        diffs_arr[i],
                        label=f'{survey_dates[i].year}:{survey_dates[i].month}:{survey_dates[i].day}',
                        c=cmap[i],
                    )
                if idx is not None:
                    plt.axvline(idx, c='k', ls=':')
                plt.legend(loc='upper right')
                plt.show()

        except Exception as err:
            if debug:
                print(f'_terminus_mask error: {err}')
            mask = []

        # apply mask while preserving list structure
        for k, v in diffs.items():
            v_list = [arr.copy().astype(float) if isinstance(arr, np.ndarray) else arr for arr in v]
            for i in range(len(v_list)):
                if isinstance(v_list[i], np.ndarray):
                    v_list[i][mask] = np.nan  # apply mask

            oib_diffs_masked[k] = v_list  # store modified list

        if inplace:
            self.set_diffs(oib_diffs_masked)
        else:
            return dict(sorted(oib_diffs_masked.items()))

    def get_dhdt(self):
        """
        compute thinning rate per year
        """
        # get nyears between surveys for averaging - round to nearest year
        self.dbl_diffs['nyears'] = np.array(
            [round((t[1] - t[0]).total_seconds() / (365.25 * 24 * 60 * 60)) for t in self.dbl_diffs['dates']]
        ).astype(float)
        # mask any diffs where nyears < 1 (shouldn't ever be the case anyways, but just in case)
        self.dbl_diffs['nyears'][self.dbl_diffs['nyears'] < 1] = np.nan
        # get annual averages
        self.dbl_diffs['dhdt'] = self.dbl_diffs['dh'] / self.dbl_diffs['nyears']

    def surge_mask(self, ela=0, threshold=10, inplace=False):
        """
        mask surges based on some maximum thinning rate threshold below the ELA.
        this simply masks an entire survey if the maximum thinning rate below the ELA is above the threshold value.

        parameters:
        - ela: float, equilibrium line altitude
        - threshold: float, maximum thinning rate (m/yr) below the ELA
        - inplace: bool, whether to modify in place
        """
        # instantiate masked dbl diffs dictionary
        oib_dbl_diffs_masked = {}
        # check if dhdt computed
        if 'dhdt' not in self.get_dbldiffs().keys():
            self.get_dhdt()
        # get dbl diffs
        dbl_diffs = self.get_dbldiffs()
        # get elevation values
        centers = self.get_centers()
        # ablation area mask
        abl_mask = centers < ela  # boolean mask
        # identify columns (survey pairs) to mask
        cols2mask = (dbl_diffs['dhdt'][abl_mask] > threshold).any(axis=0)
        # retain only non-masked survey pairs
        oib_dbl_diffs_masked['dates'] = [dt for i, dt in enumerate(dbl_diffs['dates']) if not cols2mask[i]]
        oib_dbl_diffs_masked['nyears'] = [y for i, y in enumerate(dbl_diffs['nyears']) if not cols2mask[i]]
        oib_dbl_diffs_masked['dh'] = dbl_diffs['dh'][:, ~cols2mask]
        oib_dbl_diffs_masked['dhdt'] = dbl_diffs['dhdt'][:, ~cols2mask]
        oib_dbl_diffs_masked['sigma'] = dbl_diffs['sigma'][:, ~cols2mask]

        # apply changes in-place or return results
        if inplace:
            self.set_dbldiffs(oib_dbl_diffs_masked)

        else:
            return oib_dbl_diffs_masked

    def rebin(self, agg=100, inplace=False):
        """
        rebin to specified bin sizes.

        parameters:
        - agg: int, bin size
        - inplace: bool, whether to modify in place
        """
        oib_diffs_rebin = {}

        centers = self.get_centers()
        edges = self.get_edges()

        # define new coarser edges (e.g., 0, 100, 200, …)
        start = np.floor(edges[0] / agg) * agg
        end = np.ceil(edges[-1] / agg) * agg
        new_edges = np.arange(start, end + agg, agg)

        # suppress warnings for NaN-related operations
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)

            for i, (k, v) in enumerate(self.get_diffs().items()):
                # Eensure v is a list of three NumPy arrays
                if not (isinstance(v, list) and len(v) == 3 and all(isinstance(arr, np.ndarray) for arr in v)):
                    raise ValueError(f"Expected list of 3 NumPy arrays for key '{k}', but got {v} of type {type(v)}")

                # perform binning
                if i == 0:
                    y = stats.binned_statistic(x=centers, values=v[0], statistic=np.nanmedian, bins=new_edges)[0]
                else:
                    y = stats.binned_statistic(x=centers, values=v[0], statistic=np.nanmedian, bins=new_edges)[0]

                s = stats.binned_statistic(x=centers, values=v[1], statistic=np.nanmedian, bins=new_edges)[0]
                c = stats.binned_statistic(x=centers, values=v[2], statistic=np.nanmedian, bins=new_edges)[0]

                # store results
                oib_diffs_rebin[k] = [y, s, c]

            # compute binned area
            area = stats.binned_statistic(
                x=centers,
                values=self.get_area(),
                statistic=np.nanmedian,
                bins=new_edges,
            )[0]

        # compute new bin centers
        centers = (new_edges[:-1] + new_edges[1:]) / 2

        # apply changes in-place or return results
        if inplace:
            self.set_diffs(oib_diffs_rebin)
            self.set_edges(new_edges)
            self.set_centers(centers)
            self.set_area(area)
        else:
            return oib_diffs_rebin, new_edges, centers, area

    # double difference all oib diffs from the same season 1+ year apart
    def dbl_diff(self, tolerance_months=0):
        # prepopulate dbl_diffs dictionary object will structure with dates, dh, sigma
        # where dates is a tuple for each double differenced array in the format of (date1,date2),
        # where date1's cop30 differences were subtracted from date2's to get the dh values for that time span,
        # and the sigma was taken as the mean sigma from each date
        self.dbl_diffs['dates'] = []
        self.dbl_diffs['dh'] = []
        self.dbl_diffs['sigma'] = []

        # convert keys to a sorted list
        sorted_dates = list(self.oib_diffs.keys())
        # iterate through sorted dates
        for i, date1 in enumerate(sorted_dates[:-1]):
            for j in range(i + 1, len(sorted_dates)):
                date2 = sorted_dates[j]
                delta_mon = date2.month - date1.month
                delta_mon = ((date2.year - date1.year) * 12) + delta_mon
                # calculate the modulus to find how far the difference is from the closest multiple of 12
                rem = abs(delta_mon % 12)
                # check if the difference is approximately an integer multiple of years ± n month
                if rem <= tolerance_months or rem >= 12 - tolerance_months:
                    self.dbl_diffs['dates'].append((date1, date2))
                    # self.dbl_diffs['dh'].append((self.oib_diffs[date2][0] - self.oib_diffs[date1][0]) / round(delta_mon/12))
                    self.dbl_diffs['dh'].append(self.oib_diffs[date2][0] - self.oib_diffs[date1][0])
                    # self.dbl_diffs['sigma'].append(np.sqrt((self.oib_diffs[date2][1])**2 + (self.oib_diffs[date1][1])**2))
                    self.dbl_diffs['sigma'].append(self.oib_diffs[date2][1] + self.oib_diffs[date1][1])
                    break  # Stop looking for further matches for date1

        # column stack dh and sigmas into single 2d array
        if len(self.dbl_diffs['dh']) > 0:
            self.dbl_diffs['dh'] = np.column_stack(self.dbl_diffs['dh'])
            self.dbl_diffs['sigma'] = np.column_stack(self.dbl_diffs['sigma'])
            # get rid of any all-nan dbl-diffs (where cop30 offsets may not have overlapped)
            mask = ~np.all(np.isnan(self.dbl_diffs['dh']), axis=0)
            self.dbl_diffs['dates'] = [val for val, keep in zip(self.dbl_diffs['dates'], mask) if keep]
            self.dbl_diffs['dh'] = self.dbl_diffs['dh'][:, mask]
            self.dbl_diffs['sigma'] = self.dbl_diffs['sigma'][:, mask]

        # check if deltah is all nan
        if np.isnan(self.dbl_diffs['dh']).all():
            self.dbl_diffs['dh'] = None
            self.dbl_diffs['sigma'] = None

    def filter_on_pixel_count(self, pctl=15, inplace=False):
        """
        filter oib diffs by perntile pixel count

        parameters:
        - pctl: int, percentile
        - inplace: bool, whether to modify in place
        """
        oib_diffs_filt = {}
        for k, v in self.get_diffs().items():
            arr = v[2].astype(float)  # convert 2nd tuple element (count) to float
            arr[arr == 0] = np.nan  # replace any 0 counts to nan
            mask = arr < np.nanpercentile(arr, pctl)
            v_list = list(v)
            # Apply mask only to numpy arrays
            for i in range(len(v_list)):
                if isinstance(v_list[i], np.ndarray):
                    v_list[i] = v_list[i].copy().astype(float)  # ensure modification doesn't affect original
                    v_list[i][mask] = np.nan  # apply mask

            # set key with updated list of arrays
            oib_diffs_filt[k] = v_list

        if inplace:
            self.set_diffs(oib_diffs_filt)
        else:
            return oib_diffs_filt

    def remove_outliers_zscore(self, zscore=3, inplace=False):
        """
        z-score filter based on sigma-obs

        parameters:
        - zscore: int, z-score
        - inplace: bool, whether to modify in place
        """
        oib_diffs_filt = {}

        for k, v in self.get_diffs().items():
            arr = v[1].astype(float)  # convert sigma-obs to float
            if not np.isnan(arr).all():
                mean = np.nanmean(arr)
                std = np.nanstd(arr)
            else:
                mean = np.nan
                std = np.nan

            # avoid division by zero
            if std == 0 or np.isnan(std):
                mask = np.full(arr.shape, False)  # no outliers if std is zero
            else:
                mask = np.abs((arr - mean) / std) >= zscore  # boolean mask

            v_list = list(v)
            for i in range(len(v_list)):
                if isinstance(v_list[i], np.ndarray) and np.any(mask):
                    v_list[i] = v_list[i].copy().astype(float)  # copy only if we modify it
                    v_list[i][mask] = np.nan  # apply NaN mask

            # set key with updated list of arrays
            oib_diffs_filt[k] = v_list

        if inplace:
            self.set_diffs(oib_diffs_filt)
        else:
            return oib_diffs_filt

    def save_elevchange1d(
        self,
        outdir=f'{pygem_prms["root"]}/{pygem_prms["calib"]["data"]["elev_change_1d"]["elev_change_1d_relpath"]}',
        csv=False,
    ):
        """
        Save elevation change data in a format compatible with elevchange1d module.

        Parameters
        ----------
        outdir : str
            Directory to save the elevation change data.

        format will be a JSON file with the following structure:
        {
            'ref_dem': str,
            'ref_dem_year': int,
            'dates':        [(period1_start, period1_end), (period2_start, period2_end), ... (periodM_start, periodM_end)],
            'bin_edges':    [edge0, edge1, ..., edgeN],
            'bin_area':     [area0, area1, ..., areaN],
            'dh':           [[dh_bin1_period1, dh_bin2_period1, ..., dh_binN_period1],
                            [dh_bin1_period2, dh_bin2_period2, ..., dh_binN_period2],
                            ...
                            [dh_bin1_periodM, dh_bin2_periodM, ..., dh_binN_periodM]],
            'dh_sigma':        [[sigma_bin1_period1, sigma_bin2_period1, ..., sigma_binN_period1],
                            [sigma_bin1_period2, sigma_bin2_period2, ..., sigma_binN_period2],
                            ...
                            [sigma_bin1_periodM, sigma_bin2_periodM, ..., sigma_binN_periodM]],
        }
        note: 'ref_dem' is the reference DEM used for elevation-binning.
        'ref_dem_year' is the acquisition year of the reference DEM.
        'dates' are tuples (or length-2 sublists) of the start and stop date of an individual elevation change record
        and are stored as strings in 'YYYY-MM-DD' format.
        'bin_edges' should be a list length N containing the elevation values of each bin edge.
        'bin_area' should be a list of length N-1 containing the bin areas given by the 'ref_dem' (optional).
        'dh' should be M lists of length N-1, where M is the number of date pairs and N is the number of bin edges.
        'dh_sigma'  should eaither be M lists of shape N-1 a scalar value.

        """
        # Ensure output directory exists
        os.makedirs(outdir, exist_ok=True)
        # Prepare data for saving
        elev_change_data = {
            'ref_dem': 'COP30',
            'ref_dem_year': 2013,  # hardcoded for now since all OIB diffs are relative to COP30 (2013)
            'dates': [(dt[0].strftime('%Y-%m-%d'), dt[1].strftime('%Y-%m-%d')) for dt in self.dbl_diffs['dates']],
            'bin_edges': self.bin_edges.tolist(),
            'bin_area': self.bin_area.tolist(),
            'dh': self.dbl_diffs['dh'].T.tolist(),
            'dh_sigma': self.dbl_diffs['sigma'].T.tolist(),
        }

        # Save to JSON file
        outfp = os.path.join(outdir, f'{self.rgi6id}_elev_change_1d.json')
        with open(outfp, 'w') as f:
            json.dump(elev_change_data, f)

        if csv:
            edges = np.array(elev_change_data['bin_edges'])
            ref_dem = elev_change_data['ref_dem']
            ref_year = elev_change_data['ref_dem_year']
            bin_area = np.array(elev_change_data.get('bin_area', [np.nan] * (len(edges) - 1)))

            records = []
            for period_idx, (start, end) in enumerate(elev_change_data['dates']):
                dh_vals = np.array(elev_change_data['dh'][period_idx])
                dh_sig = np.array(elev_change_data['dh_sigma'][period_idx])
                for i in range(len(edges) - 1):
                    records.append(
                        {
                            'bin_start': edges[i],
                            'bin_stop': edges[i + 1],
                            'bin_area': bin_area[i],
                            'date_start': start,
                            'date_end': end,
                            'dh': dh_vals[i],
                            'dh_sigma': dh_sig[i],
                            'ref_dem': ref_dem,
                            'ref_dem_year': ref_year,
                        }
                    )

            df = pd.DataFrame(records)
            outfp = os.path.join(outdir, f'{self.rgi6id}_elev_change_1d.csv')
            df.to_csv(outfp, index=False)


def _split_by_uppercase(text):
    """Add space before each uppercase letter (except at the start of the string."""
    return re.sub(r'(?<!^)(?=[A-Z])', ' ', text)


def _round_to_nearest_month(dt):
    """Round a datetime object to the nearest month."""
    if dt.day >= 15:
        # Round up to the first day of next month
        next_month = dt.replace(day=1) + timedelta(days=32)
        return next_month.replace(day=1)
    else:
        # Round down to the first day of the current month
        return dt.replace(day=1)

    ### not fully working yet ###
    # def _savgol_smoother(self, window=5, poly=2, inplace=False):
    #     """
    #     smooths an array using Savitzky-Golay filter with NaN handling

    #     parameters:
    #     - window: int, window size
    #     - poly: int, polynomial degree
    #     - inplace: bool, whether to modify in place
    #     """
    #     oib_diffs_filt = {}
    #     x = self.get_centers()

    #     for k, v in self.get_diffs().items():
    #         filtered_data = []
    #         for i in range(2):  # apply filtering to both v[0] and v[1]
    #             arr = v[i].astype(float)  # convert to float
    #             mask = np.isnan(arr)
    #             data_filled = np.copy(arr)
    #             # interpolate NaNs
    #             data_filled[mask] = np.interp(x[mask], x[~mask], arr[~mask])
    #             # apply Savitzky-Golay filter
    #             smoothed = signal.savgol_filter(data_filled, window_length=window, polyorder=poly)
    #             # restore NaNs
    #             smoothed[mask] = np.nan
    #             filtered_data.append(smoothed)

    #         # populate filtered dictionary with smoothed data
    #         oib_diffs_filt[k] = [filtered_data[0], filtered_data[1], v[2]]

    #     if inplace:
    #         self.set_diffs(oib_diffs_filt)
    #     else:
    #         return oib_diffs_filt
