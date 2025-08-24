# Sea breeze detection

This repository has code that is intended to detect sea breezes from numerical weather model output. The code relies on xarray/dask, numpy, scipy, metpy, scikit-image, pyproj and pandas. The code is organised into three sequential steps:

1) [Pre-processing model data](#pre-processing-model-data)
2) [Calculating sea breeze diagnostics](#sea-breeze-diagnostics)
3) [Object identification and filtering](#sea-breeze-filtering)

## Pre-processing model data

These are functions for loading and pre-processing model data, and can be found in [`load_model_data`](load_model_data.py). This includes loading temperature, moisture, wind, and static variables from four different model datasets hosted on the [NCI data catalog](https://geonetwork.nci.org.au/geonetwork/srv/eng/catalog.search#/home) (requires the code to be run locally on the NCI system). There is also a function that finds the angle of dominant coastline orientation from a land-sea mask ([`get_coastline_angle_kernel`](load_model_data.py#L639)). This is important for finding the onshore component of the wind that is used in most sea breeze diagnostics (with thanks to Ewan Short for developing this method and code with input from Jarrah Harrison-Lofthouse).

## Sea breeze diagnostics

These are functions for calculating sea breeze diagnostics, and can be found in [`sea_breeze_funcs`](sea_breeze_funcs.py). There are currently three key diagnostics:

* Kinematic moisture frontogenesis (`kinematic_frontogenesis`)
* Sea breeze index (`calc_sbi`)
* Fuzzy logic algorithm (`fuzzy_function_combine`)

Details on each diagnostic can be found in the relevant docstrings. For the fuzzy logic algoirithm, hourly changes in wind, temperature, and moisture need to be first calculated using the `hourly_change` function within [`sea_breeze_funcs`](sea_breeze_funcs.py). It is recommended to calculate hourly changes for the entire period of interest, as the algorithm needs to find percentiles of the distribution of each hourly change.

<!-- For processing the diagnostic functions, python scripts exist in each of the model directories (e.g. [`aus2200/aus2200_sbi.py`](https://github.com/andrewbrown31/sea_breeze_analysis/blob/main/aus2200/aus2200_sbi.py) calculates the sea breeze index from AUS2200) as well bash scripts that submit those python scripts to the PBS queue (e.g. [`aus2200/aus2200_sbi.sh`](https://github.com/andrewbrown31/sea_breeze_analysis/blob/main/aus2200/diagnostic_jobs/aus2200_sbi_smooth4_2016.sh)). Note that for km-scale data (such as AUS2200 or BARRA-C), large amounts of memory may be required. -->


## Sea breeze filtering

These functions convert the sea breeze diagnostics to a binary mask of candidate sea breeze objects, which are then refined by applying several filters of known sea breeze characteristics. Functions can be found in [`sea_breeze_filters`](sea_breeze_filters.py). There are several filters that can be applied in `filter_3d` (see docstring). Output from `filter_3d` is a binary dataset of sea breeze objects. The [`filter.py`](filter.py) script can be used to drive the filtering processing

<!-- as well bash scripts that submit those python scripts to the PBS queue (e.g. [`aus2200/filter.sh`](https://github.com/andrewbrown31/sea_breeze_analysis/blob/main/aus2200/filter_jobs/filter_smooth_s4.sh)). Again, for km-scale model data large amounts of memory may be required. There are a variety of settings in the filtering script, and it is noted that the `land_sea_temperature_filter` currently causes significant slowdown due to radial searching around sea breeze objects. -->

For calculating the threshold to mask with, the [`percentile`](sea_breeze_filters.py#L72) function can be used prior to the filtering. That way, the percentile threshold can be taken from a distribution over a longer period (say, several months) than the filtering period (say, a single day).

## Example 

An example notebook demonstrating these three steps is available [here](example_notebooks/sea_breeze_detection_example.ipynb). It applies each of the diagnostics and the filter to a day of [AUS2200](https://dx.doi.org/10.25914/w95d-q328) data. This example is intended to run on the Australian NCI Gadi supercomputer, with access to the [ACCESS-NRI conda environment on xp65](https://docs.access-hive.org.au/getting_started/environments/) and [AUS2200 on the bs94 project](https://dx.doi.org/10.25914/w95d-q328).

## Notes

* For km-scale data, large amounts of memory may be required for diagnostic calculation and model pre-processing.

* In practice, diagnostics should be saved to disk, and reloaded for filtering. This is because each is a computationally heavy task, especially for km-scale model data. The zarr format is used for saving output for efficiency.

* There are a variety of settings in the filtering script, and it is noted that the `land_sea_temperature_filter` currently causes significant slowdown due to radial searching around sea breeze objects.

## Contributing

If would like to make changes to improve this code, please reach out or make an issue!

