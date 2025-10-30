(run_inversion_overview_target)=
# run_inversion.py
This script will perform ice thickness inversion while calibrating the ice viscosity ("Glen A") model parameter such that the modeled ice volume roughly matches the ice volume estimates from [Farinotti et al. (2019)](https://www.nature.com/articles/s41561-019-0300-3) for each RGI region. Run the script as follows:

```
run_inversion -rgi_region01 <region>
```

## Script Structure

Broadly speaking, the script follows:
* Load glaciers
* Load climate data
* Compute apparent mass balance and invert for initial ice thickness
* Use minimization to find agreement between our modeled and [Farinotti et al. (2019)](https://www.nature.com/articles/s41561-019-0300-3) modeled ice thickness estimates for each RGI region
* Export the calibrated parameters

## Special Considerations
The regional Glen A value is calibrated by inverting for the ice thickness of all glaciers in a given region without considering calving (all glaciers are considered land-terminating). After the "best" Glen A value is determined, a final round of ice thickness inversion is performed for tidewater glaciers with calving turned **on**. Running this script will by default export the regionally calibrated Glen A values to the path specfied by `sim.oggm_dynamics.glen_a_regional_relpath` in *~/PyGEM/config.yaml'*. The calibrated inversion parameters also get stored within a given glacier directories *diagnostics.json* file, e.g.:
```
{"dem_source": "COPDEM90", "flowline_type": "elevation_band", "apparent_mb_from_any_mb_residual": 2893.2237556771674, "inversion_glen_a": 3.784593106855888e-24, "inversion_fs": 0}
```