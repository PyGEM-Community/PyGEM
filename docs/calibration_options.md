(calibration_target)=
# Model Calibration
Several calibration options exist, which vary with respect to complexity and computational expense. These include the relatively straightforward approach of [Huss and Hock (2015)](https://www.frontiersin.org/articles/10.3389/feart.2015.00054/full), which is referred to as ‘HH2015’ to more complex Bayesian approaches from [Rounce et al. (2023)](https://www.science.org/doi/10.1126/science.abo1324) (‘MCMC’) ([Table 1](cal_options_table_target). At present, the options all use geodetic glacier-wide mass balance data for each glacier in units of meters of water equivalent (m w.e.) per year ([Hugonnet et al. (2021)](https://www.nature.com/articles/s41586-021-03436-z). The calibration is done assuming the glacier area remains constant to avoid mass balance-ice thickness circularity issues.

(cal_options_table_target)=

| Calibration option | Overview | Reference |
| :--- | :--- | :--- |
| ['HH2015'](HH2015_target) | Finds single set of parameters.<br>Varies in order: $f_{snow}$, $k_{p}$, $T_{bias}$ | [Huss and Hock, 2015](https://www.frontiersin.org/articles/10.3389/feart.2015.00054/full) |
| ['HH2015mod'](HH2015mod_target) | Finds single set of parameters.<br>Varies in order: $k_{p}$, $T_{bias}$ | [Rounce et al., 2020](https://www.cambridge.org/core/journals/journal-of-glaciology/article/quantifying-parameter-uncertainty-in-a-largescale-glacier-evolution-model-using-bayesian-inference-application-to-high-mountain-asia/61D8956E9A6C27CC1A5AEBFCDADC0432) |
| ['emulator'](emulator_target) | Creates emulator for ['MCMC'](MCMC_target).<br>Finds single set of parameters with emulator following ['HH2015mod'](HH2015mod_target) | [Rounce et al., 2023](https://www.science.org/doi/10.1126/science.abo1324) |
| ['MCMC'](MCMC_target) | Finds many sets of parameters using Bayesian inference. Setting `calib.MCMC_params.option_use_emulator=True` in ~/PyGEM/config.yaml will run Bayesian inference using the mass balance emulator. Setting `calib.MCMC_params.option_use_emulator=False` (or when performing dynamical calibration against elevation change data)$^*$ will run Bayesian inference calibration with full model simulations.<br>Varies $f_{snow}$, $k_{p}$, $T_{bias}$, (optionally $\rho_{ablation}$, $\rho_{accumulation}$)$^*$ |  [Rounce et al., 2020](https://www.cambridge.org/core/journals/journal-of-glaciology/article/quantifying-parameter-uncertainty-in-a-largescale-glacier-evolution-model-using-bayesian-inference-application-to-high-mountain-asia/61D8956E9A6C27CC1A5AEBFCDADC0432); [2023](https://www.science.org/doi/10.1126/science.abo1324) |
| [Future options](cal_custom_target) | Stay tuned for new options coming in 2023/2024! | | 

The output of each calibration is a .json file that holds a dictionary of the calibration options and the subsequent model parameters.  Thus, the .json file will store several calibration options.  Each calibration option is a key to the dictionary. The model parameters are also stored in a dictionary (i.e., a dictionary within a dictionary) with each model parameter being a key to the dictionary that provides access to a list of values for that specific model parameter. The following shows an example of how to print a list of the precipitation factors ($k_{p}$) for the calibration option specified in the input file:

```
with open(modelprms_fullfn, 'r') as f:
    modelprms_dict = json.load(f)
print(modelprms_dict[pygem_prms.option_calibration][‘kp’])
```

The calibration options are each discussed below.  We recommend using the MCMC calibration option (Rounce et al. [2020a](https://www.cambridge.org/core/journals/journal-of-glaciology/article/quantifying-parameter-uncertainty-in-a-largescale-glacier-evolution-model-using-bayesian-inference-application-to-high-mountain-asia/61D8956E9A6C27CC1A5AEBFCDADC0432), [2020b](https://www.frontiersin.org/articles/10.3389/feart.2019.00331/full), [2023](https://www.science.org/doi/10.1126/science.abo1324)) as this enables the user to quantify the uncertainty associated with the model parameters in the simulations; however, it is very computationally expensive. The methods from [Huss and Hock (2015)](https://www.frontiersin.org/articles/10.3389/feart.2015.00054/full) provide a computationally cheap alternative. 

```{note}
Running these options is performed using *run_calibration.py* (see [Model Workflow](workflow_cal_prms_target)). Additionally, there are two other calibration scripts to calibrate the [ice viscosity model parameter](workflow_cal_glena_target) using the *run_calibration_icethickness_consensus.py* and the [frontal ablation parameter](calibration_frontalablation_target) for marine-terminating glaciers using the *run_calibration_frontalablation.py*.
```

(HH2015_target)=
## HH2015
The calibration option **‘HH2015’** follows the calibration steps from [Huss and Hock (2015)](https://www.frontiersin.org/articles/10.3389/feart.2015.00054/full). Specifically, the precipitation factor is initially adjusted between 0.8-2.0. If agreement between the observed and modeled mass balance is not reached, then the degree-day factor of snow is adjusted between 1.75-4.5 mm d$^{-1}$ K$^{-1}$. Note that the ratio of the degree-day factor of ice to snow is set to 2, so both parameters are adjusted simultaneously. Lastly, if agreement is still not achieved, then the temperature bias is adjusted.

(HH2015mod_target)=
## HH2015mod
The calibration option **‘HH2015mod’** is a modification of the calibration steps from [Huss and Hock (2015)](https://www.frontiersin.org/articles/10.3389/feart.2015.00054/full) that are used to generate the prior distributions for the MCMC methods [(Rounce et al. 2020a)](https://www.cambridge.org/core/journals/journal-of-glaciology/article/quantifying-parameter-uncertainty-in-a-largescale-glacier-evolution-model-using-bayesian-inference-application-to-high-mountain-asia/61D8956E9A6C27CC1A5AEBFCDADC0432)
. Since the MCMC methods used degree-day factors of snow based on previous studies, only the precipitation factor and temperature bias are calibrated. The precipitation factor varies from 0.5-3 and if agreement is not reached between the observed and modeled mass balance, then the temperature bias is varied. Note the limits on the precipitation factor are estimated based on a rough estimate of the precipitation factors needed for the modeled winter mass balance of reference glacier to match the observations.

However, if you plan to use the MCMC methods, you are suggested to use the **‘emulator’** calibration option described below, which follows the same steps, but creates an emulator to run the mass balance simulations for each potential parameter set to reduce the computational expense.

(emulator_target)=
## Emulator applying HH2015mod
The calibration option **‘emulator’** creates an independent emulator for each glacier that is derived by performing 100 present-day simulations based on randomly sampled model parameter sets and then fitting a Gaussian Process to these parameter-response pairs. This model replaces the mass balance model within the MCMC sampler (see Bayesian inference using MCMC below), which tests showed reduces the computational expense by two orders of magnitude. In the event that a single set of model parameters is desired, the emulator is also used to derive a set of model parameters following the same steps as ‘HH2015mod’.

```{note}
The ‘emulator’ calibration option will generate both the .json file of the model parmaters as well as the model simulations and emulators for each glacier stored in a subdirectory named 'emulator'.
```

```{note}
The ‘emulator’ calibration option needs to be run before the ‘MCMC’ option.
```

(MCMC_target)=
## Bayesian inference using Markov Chain Monte Carlo methods
The calibration option **‘MCMC’** is the recommended option. Details of the methods are provided by Rounce et al. ([2020a](https://www.cambridge.org/core/journals/journal-of-glaciology/article/quantifying-parameter-uncertainty-in-a-largescale-glacier-evolution-model-using-bayesian-inference-application-to-high-mountain-asia/61D8956E9A6C27CC1A5AEBFCDADC0432), [2023](https://www.science.org/doi/10.1126/science.abo1324)). In short, Bayesian inference is performed using Markov Chain Monte Carlo (MCMC) methods, which requires a mass balance observation (including the uncertainty represented by a standard deviation) and prior distributions. In an ideal world, we would have enough data to use broad prior distributions (e.g., uniform distributions), but unfortunately the model is overparameterized meaning there are an infinite number of parameter sets that give us a perfect fit. We therefore must use an empirical Bayes approach by which we use a simple optimization scheme (the **‘HH2015mod’** calibration option) to generate our prior distributions at the regional scale, and then use these prior distributions for the Bayesian inference. The prior distribution for the degree-day factor is based on previous data ([Braithwaite 2008](https://www.cambridge.org/core/journals/journal-of-glaciology/article/temperature-and-precipitation-climate-at-the-equilibriumline-altitude-of-glaciers-expressed-by-the-degreeday-factor-for-melting-snow/6C2362F61B7DE7F153247A039736D54C)), while the temperature bias and precipitation factor are derived using a simple optimization scheme based on each RGI Order 2 subregion. The temperature bias assumes a normal distribution and the precipitation factor assumes a gamma distribution to ensure positivity. Glacier-wide winter mass balance data ([WGMS 2020](https://wgms.ch/data_databaseversions/)) are used to determine a reasonable upper-level constraint for the precipitation factor for the simple optimization scheme.

The MCMC methods thus require several steps. First, set `calib.option_calibration = emulator` in *~/PyGEM/config.yaml* (or pass as command-line arguemnt, as shown below). This creates an emulator that helps speed up the simulations within the MCMC methods and helps generate an initial calibration to generate the regional priors. Run this initial calibration:
```
run_calibration -option_calibration emulator
```
The regional priors are then determined by running the following:
```
run_mcmc_prior
```
This will output a .csv file that has the distributions for the temperature bias and precipitation factors for each Order 2 RGI subregion. This file is located in the calibration subdirectory within the Output directory.

Once the regional priors are set, the MCMC methods can be performed.  Set `['calib']['option_calibration'] = MCMC` in *~/PyGEM/config.yaml* (or pass as command-line arguemnt, as shown below), then run the following:
```
run_calibration -option_calibration MCMC
```
In order to reduce the file size, the parameter sets are thinned by a factor of 10. This is reasonable given the correlation between subsequent parameter sets during the Markov Chain, but can be adjusted if thinning is not desired (change value to 1 in the input file).

```{note}
**'MCMC_fullsim'** is another calibration option that runs full model simulations within the MCMC methods instead of using the emulator. It is computationally very expensive but allows one to assess the emulators impact on the MCMC methods.
```

(cal_custom_target)=
## Customized Calibration Routines
As new observations become available, we envision the calibration routines will need to change to leverage these observations. The only real limitation in developing a calibration routine is that the dictionary stored as a .json file needs to be consistent such that the calibration option is consistent with the run_simulation.py script.

(calibration_frontalablation_target)=
## Frontal Ablation Parameter for Marine-terminating Glaciers
Marine-terminating glaciers have an additional frontal ablation parameter that is calibrated at the glacier-scale to match frontal ablation data [(Osmanoglu et al. 2013;](https://www.cambridge.org/core/journals/annals-of-glaciology/article/surface-velocity-and-ice-discharge-of-the-ice-cap-on-king-george-island-antarctica/62E511405ADD31A43FF52CDBC727A9D0) [2014;](https://tc.copernicus.org/articles/8/1807/2014/) [Minowa et al. 2021;](https://www.sciencedirect.com/science/article/pii/S0012821X21000704) [Kochtitzky et al. 2022](https://www.nature.com/articles/s41467-022-33231-x)). Marine-terminating glaciers require a special procedure for calibration to avoid circularity issues. The initial ice thickness is estimated using the mass balance parameters assuming the glacier is land-terminating and a forward simulation from 2000-2020 estimates the frontal ablation. If a dynamic instability error occurs (8% of glaciers for [Rounce et al. 2023](https://www.science.org/doi/10.1126/science.abo1324)), the glacier dynamics model uses [mass redistribution curves](mass_redistribution_curves_target) instead. For quality control, we combined the frontal ablation and geodetic mass balance observations to estimate climatic mass balances. For some glaciers, the resulting climatic mass balances are unrealistic due to errors in the RGI outlines and/or poor glacier thickness and velocity data used in frontal ablation calculations. For these glaciers, we assume frontal ablation is overestimated and reduce the frontal ablation to ensure the climatic mass balance is within three standard deviations of the regional mean from the geodetic mass balance data. The Antarctic and Subantarctic have the sparsest frontal ablation data, so the region’s median frontal ablation parameter and corresponding standard deviation is used for glaciers without data.

The frontal ablation calibration runs through several steps on a regional basis. First, the frontal ablation data are merged together into a single dataset. Next, the frontal ablation parameter is calibrated for each marine-terminating glacier in a given region. All the frontal ablation parameters are then merged into a single file for each region. Lastly, the reference climatic-basal mass balance data is updated by removing the frontal ablation from the total mass change.

To perform frontal ablation calibration, simply call the run_calibration_frontalablation python script (by default all regions are calibrated):
```
run_calibration_frontalablation  (optionally pass -rgi_region01 <region>)
```

## Ice Viscosity Parameter
The ice viscosity parameter will affect the ice thickness inversion and dynamical evolution of the glacier. The ice viscosity parameter is currently calibrated such that the volume of ice at the regional scale is consistent with the regional ice volumes from [Farinotti et al. (2019)](https://www.nature.com/articles/s41561-019-0300-3).