## Ablation
There are currently two model options for ablation. Both model options use a degree-day model ($f$). 

### Option 1: monthly temperatures
The first calculates ablation ($a$) using the mean monthly temperature:
$$a=f_{snow/ice/firn/debris} \cdot T_{m}^{+} \cdot n$$

where $f$ is the degree-day factor of snow, ice, firn, or debris (m w.e. d-1 °C$^{-1}$), $T_{m}^{+}$ is the positive monthly mean temperature (°C), and $n$ is the number of days per month. 

### Option 2: monthly temperatures with daily variance
The second option incorporates the daily variance associated with the temperature for each month according to [Huss and Hock (2015)](https://www.frontiersin.org/articles/10.3389/feart.2015.00054/full):
$$a=f_{snow/ice/firn/debris} \cdot \sum_{i=1}^{ndays} T_{d,i}^{+} $$

where $T_{d}$ is the daily positive mean air temperature and is estimated by superimposing random variability from the standard deviation of the daily temperature for each month.

The degree-day factors for snow, ice, firn, and debris depend on the surface type option that is chosen by the user (see Section 5). The values of $f$ for these various surface types are assumed to be related to one another to reduce the number of model parameters. The default ratio of the $f_{snow}$ to the $f_{ice}$ is 0.7, and $f_{firn}$ is assumed to be the mean of the $f_{snow}$ and $f_{ice}$; however, the user may change these values in the input file if desired. The values for $f_{debris}$ are equal to $f_{ice}$ multiplied by the mean sub-debris melt enhancement factor [(Rounce et al. 2021)](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2020GL091311) for the given elevation bin.

### Temperature at elevation bins
Temperature for each elevation bin ($T_{bin}$) is determined by selecting the temperature from the gridded climate data ($T_{gcm}$) based on the nearest neighbor, which is then downscaled to the elevation bins on the glacier according to:
$$T_{bin} = T_{gcm} + lr_{gcm} \cdot (z_{ref} - z_{gcm}) + lr_{glac} \cdot (z_{bin} - z_{ref}) + T_{bias}$$

where $lr_{gcm}$ and $lr_{glac}$ are lapse rates (°C m-1) associated with downscaling the climate data to the glacier and then over the glacier elevation bins, respectively; $z_{ref}$, $z_{gcm}$, and $z_{bin}$ are the elevations from the glacier’s reference point (median or mean elevation), the climate data, and the elevation bin, respectively; and $T_{bias}$ is the temperature bias. The temperature bias is one of three model parameters that is calibrated and serves to account for any biases resulting from the use of coarse climate data that is unable to capture local topographic variations. By default, the $lr_{gcm}$ and $lr_{glac}$ are assumed to be equal.