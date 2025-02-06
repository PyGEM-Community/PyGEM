(run_calibration_target)=
# run_calibration.py
This script will calibrate the mass balance model parameters (degree-day factor of snow, precipitation factor, and temperature bias). 

When running the script, the calibration option can be specified in the *~/PyGEM/config.yaml* configuration file or passed via the command line as follows:
```
run_calibration -option_calibration <calib_option>
```

If successful, the script will run without errors and the following will be generated:
* ../Output/calibration/\[glacno\]-modelprms_dict.json

This is a JSON file that contains the calibration data. If the file already exists, the calibrated model option will be added to the existing .json file.

## Script Structure
While most users may just want to run the model, those interested in developing new calibration schemes should be aware of the general structure of the script.  Broadly speaking, the script follows:
* Load glaciers
* Load reference climate data
* Load glacier data (area, etc.)
* Load mass balance data
* Calibrate the model parameters
  - "Modular" calibration options are included as an if/elif/else statement.
* Export model parameters

## View Output
The .json file stores a dictionary and each calibration option is a key to the dictionary. The model parameters are also stored in a dictionary (i.e., a dictionary within a dictionary) with each model parameter being a key to the dictionary that provides access to a list of values for that specific model parameter. The following shows an example of how to print a list of the precipitation factors ($k_{p}$) for the calibration option specified in the input file:

```
with open(modelprms_fullfn, 'r') as f:
    modelprms_dict = json.load(f)
print(modelprms_dict[pygem_prms.option_calibration][‘kp’])
```

## Special Considerations
Typically, the glacier area is assumed to be constant (<em>option_dynamics=None</em>), i.e., the glacier geometry is not updated, to reduce computational expense.

Current calibration options rely solely on glacier-wide geodetic mass balance estimates.