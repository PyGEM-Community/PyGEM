(model_output_overview_target)=
# Model Output 
The model outputs a variety of data including monthly mass balance and its components (accumulation, melt, refreezing, frontal ablation, glacier runoff), and annual mass, mass below sea level, and area. Results are written as a netcdf file (.nc) for each glacier. If multiple simulations are performed (e.g., for Monte Carlo simulations), then statistics related to the median and median absolute deviation are output for each parameter. In addition to this standard glacier-wide output, binned output is also available, which include the bins surface elevation, volume, thickness, and climatic mass balance annually.

## Post-processing Data
PyGEM simulations are output for each glacier individually. For most analyses, it is useful to aggregate or merge analyses to a regional or global scale. PyGEM's *postproc.compile_simulations.py* is designed to do just so.

This script is designed to aggregate by region, scenario, and variable. For example the following call will result in 8 output files, the annual glacier mass and the annual glacier area for each specified scenario, for each specified region

 ```
compile_simulations -rgi_region 01 02 -scenario ssp245 ssp585 -gcm_startyear2000 -gcm_endyear 2100 -vars glac_mass_annual glac_area_annual
 ```
 See below for more information.

## Analyzing Results
Various Jupyter Notebooks are provided for analyzing PyGEM results in a separate [GitHub repository](https://github.com/PyGEM-Community/PyGEM-notebooks).

- **analyze_regional_change.ipynb.ipynb** <br>This notebook demonstrates how to aggregate simulations by region and plot the glacier mass, area, and runoff changes.
```{figure} _static/analyze_glacier_change_region11.png
---
width: 100%
---
```
- **analyze_glacier_change_byWatershed.ipynb** <br>This notebook can be used to aggregate glacier mass, area, and runoff into watersheds; specifically, it will create new netcdf files per watershed such that after the initial aggregation, analyses can be performed much more rapidly. The notebook continues to show an example plot of glacier mass, area, and runoff changes for each watershed in an example region:
```{figure} _static/R06_change.png
---
width: 50%
---
```
```{note}
This notebook assumes that you have a "dictionary", i.e., a .csv file, that has each glacier of interest and the watershed (or other grouping) name associated with each glacier.
```
- **analyze_glacier_change_CrossSection.ipynb** <br>This notebook can be used to plot cross sections of an individual glacier's ice thickness over time for an ensemble of GCMs:
```{figure} _static/15.03733_profile_2100_ssps.png
---
width: 100%
---
```