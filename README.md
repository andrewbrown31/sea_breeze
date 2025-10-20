# Sea breeze detection

This repository has code that is intended to detect sea breezes from numerical weather model output. The code relies on packages including xarray/dask, scipy, metpy, scikit-image, pyproj and pandas (see [requirements](#requirements)). 

Contents
* [Code structure](#code-structure)
* [Examples](#examples)
* [Requirements](#requirements)
* [Contributing](#contributing)
* [Citing](#citing)
* [WxSysLib](#wxsyslib)

## Code structure
The code is organised into three sequential steps:

1) [Pre-processing model data](#pre-processing-model-data)
2) [Calculating sea breeze diagnostics](#sea-breeze-diagnostics)
3) [Object identification and filtering](#sea-breeze-filtering)

### Pre-processing model data

These are functions for loading and pre-processing model data, and can be found in [`load_model_data`](load_model_data.py). This includes loading temperature, moisture, wind, and static variables from four different model datasets hosted on the [NCI data catalog](https://geonetwork.nci.org.au/geonetwork/srv/eng/catalog.search#/home) (requires the code to be run locally on the NCI system). There is also a function that finds the angle of dominant coastline orientation from a land-sea mask ([`get_coastline_angle_kernel`](load_model_data.py#L639)). This is important for finding the onshore component of the wind that is used in most sea breeze diagnostics (with thanks to Ewan Short for developing this method and code with input from Jarrah Harrison-Lofthouse).

### Sea breeze diagnostics

These are functions for calculating sea breeze diagnostics, and can be found in [`sea_breeze_funcs`](sea_breeze_funcs.py). There are currently three key diagnostics:

* Kinematic moisture frontogenesis (`kinematic_frontogenesis`)
* Sea breeze index (`calc_sbi`)
* Fuzzy logic algorithm (`fuzzy_function_combine`)

Details on each diagnostic can be found in the relevant docstrings. For the fuzzy logic algoirithm, hourly changes in wind, temperature, and moisture need to be first calculated using the `hourly_change` function within [`sea_breeze_funcs`](sea_breeze_funcs.py). It is recommended to calculate hourly changes for the entire period of interest, as the algorithm needs to find percentiles of the distribution of each hourly change.


### Sea breeze filtering

These functions convert the sea breeze diagnostics to a binary mask of candidate sea breeze objects, which are then refined by applying several filters of known sea breeze characteristics. Functions can be found in [`sea_breeze_filters`](sea_breeze_filters.py). There are several filters that can be applied in `filter_3d` (see docstring). Output from `filter_3d` is a binary dataset of sea breeze objects. The [`filter.py`](filter.py) script can be used to drive the filtering processing

For calculating the threshold to mask with, the [`percentile`](sea_breeze_filters.py#L72) function can be used prior to the filtering. That way, the percentile threshold can be taken from a distribution over a longer period (say, several months) than the filtering period (say, a single day).

### Notes

* For km-scale data, large amounts of memory may be required for diagnostic calculation and model pre-processing.

* In practice, diagnostics should be saved to disk, and reloaded for filtering. This is because each is a computationally heavy task, especially for km-scale model data. The zarr format is used for saving output for efficiency.

* There are a variety of settings in the filtering script, and it is noted that the `land_sea_temperature_filter` currently causes significant slowdown due to radial searching around sea breeze objects.

## Examples

### Local example

An example notebook demonstrating these three steps using ERA5 data is available [here](example_notebooks/era5_sea_breeze_identification.ipynb), with sample ERA5 data provided in the `example_data` directory.

### On the gadi HPC system

An example notebook demonstrating these three steps using [AUS2200](https://dx.doi.org/10.25914/w95d-q328) data is available [here](example_notebooks/aus2200_sea_breeze_identification.ipynb). This example is intended to run on the Australian NCI Gadi supercomputer, with access to the [ACCESS-NRI conda environment on xp65](https://docs.access-hive.org.au/getting_started/environments/) and [AUS2200 on the bs94 project](https://dx.doi.org/10.25914/w95d-q328).


## Requirements
The code was devloped on the [`analysis3-25.06` conda environment](https://docs.access-hive.org.au/getting_started/environments/) through the `xp65` NCI project. This code builds on the following packages:
* cartopy                                 0.24.0
* dask                                    2025.5.1
* metpy                                   1.7.0
* numpy                                   1.26.4
* pandas                                  2.2.3
* pyproj                                  3.6.1
* scikit-image                            0.25.2 
* scipy                                   1.15.2
* xarray                                  2025.4.0
* netCDF4                                 1.7.2 

An minimum working environment can be build using conda:
```
conda create -n sea_breeze_env -c conda-forge cartopy=0.24.0 dask=2025.5.1 metpy=1.7.0 numpy=1.26.4 pandas=2.2.3 pyproj=3.6.1 scikit-image=0.25.2 scipy=1.15.2 xarray=2025.4.0 netCDF4=1.7.2
```

In addition, for loading ERA5 data hosted on the NCI within the `load_model_data.py` module, the following packages are required:
* [access-nri-intake](https://docs.access-hive.org.au/model_evaluation/data/model_catalogs/) 1.2.3

## Contributing

If would like to make changes to improve this code, please reach out or make an issue!

## Citing
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.16938245.svg)](https://doi.org/10.5281/zenodo.17220916)

## WxSysLib
This code was developed with funding from the ARC Centre of Excellence for 21st Century Weather. The code is also available in the Centre's [WxSysLib](https://github.com/21centuryweather/WxSysLib/tree/main/utils/diagnostics/sea_breeze) repository. However, the version provided here should be considered the most up to date.
