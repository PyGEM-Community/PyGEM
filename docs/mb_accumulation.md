## Accumulation
Accumulation ($c$) is calculated for each elevation bin as a function of the precipitation ($P_{bin}$), air temperature ($T_{bin}$), and the snow temperature threshold ($T_{snow}$).  There are two options for estimating accumulation based on how to classify precipitation as liquid or solid. 

### Option 1: Threshold +/- 1$^{\circ}$C
The first (default) option is to estimate the ratio of liquid and solid precipitation based on the air temperature:
$$c = \delta \cdot P_{bin}$$

where $\delta=1$; if $T_{bin} \leq T_{snow}-1$

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; $\delta=0$; if $T_{bin} \geq T_{snow}+1$

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; $\delta=0.5-(T_{bin}-T_{snow})/2$; if $T_{snow}-1 < T_{bin} < T_{snow}+1$

<br>where $P_{bin}$ is the monthly precipitation and $\delta$ is the fraction of solid precipitation each month. $T_{snow}$ typically ranges from 0 – 2 $^{\circ}$C ([Radić and Hock, 2011](https://www.nature.com/articles/ngeo1052); [Huss and Hock, 2015](https://www.frontiersin.org/articles/10.3389/feart.2015.00054/full)) and is typically assumed to be 1$^{\circ}$C.  

### Option 2: Single threshold
The alternative option is to classify precipitation as snow or rain based on a single threshold.

### Precipitation at elevation bins
Precipitation at each elevation bin of the glacier ($P_{bin}$) is determined by selecting the precipitation from the gridded climate data ($P_{gcm}$) based on the nearest neighbor, which is then downscaled to the elevation bins on the glacier:
$$P_{bin} = P_{GCM} \cdot k_{p} \cdot (1 + d_{prec} \cdot (z_{bin} - z_{ref}))$$
<br>where $k_{p}$ is the precipitation factor and $d_{prec}$ is the precipitation gradient. The precipitation factor is a model parameter that is used to adjust from the climate data to the glacier, which could be caused by local topographic effects due to differences in elevation, rain shadow effects, etc. The precipitation gradient is another model parameter, which is used to redistribute the precipitation along the glacier and can be thought of as a precipitation lapse rate. Typical values for the precipitation gradient vary from 0.01 – 0.025% m$^{-1}$([Huss and Hock, 2015](https://www.frontiersin.org/articles/10.3389/feart.2015.00054/full) who cited [WGMS, 2012](https://wgms.ch/products_fog/)). The default assumes a precipitation gradient of 0.01% m$^{-1}$ to reduce the number of model parameters.

Additionally, for glaciers with high relief (> 1000 m), the precipitation in the uppermost 25% of the glacier’s elevation is reduced using an exponential function ([Huss and Hock, 2015](https://www.frontiersin.org/articles/10.3389/feart.2015.00054/full)):
$$P_{bin,exp} = P_{bin} \cdot exp(\frac{z_{bin} - z_{75\%}}{z_{max} - z_{75\%}}) $$
where $P_{bin,exp}$ is the adjusted precipitation, and $z_{max}$ and $z_{75\%}$ are the elevation of the glacier maximum and the glacier’s 75th percentile elevation, respectively. The adjusted precipitation cannot be lower than 87.5% of the maximum precipitation on the glacier. This adjustment accounts for the reduced air moisture and increased wind erosion at higher elevations ([Benn and Lehmkuhl, 2000](https://risweb.st-andrews.ac.uk/portal/en/researchoutput/mass-balance-and-equilibriumline-altitudes-of-glaciers-in-highmountain-environments(080f17fc-33dd-4805-bc97-a5aaa018a457)/export.html)).