import xarray as xr
from sea_breeze import sea_breeze_funcs
from dask.distributed import Client

if __name__ == "__main__":

    #Set up dask client
    client = Client()

    #Set up paths to sea_breeze_funcs data output and other inputs
    path = "/g/data/ng72/ab4502/"
    hourly_change_path = path+ "sea_breeze_detection/era5/F_hourly_*.zarr"

    #Load the hourly change dataset
    hourly_change_ds = xr.open_mfdataset(
        hourly_change_path, engine="zarr"
        )
    
    #Combine the fuzzy functions
    fuzzy = sea_breeze_funcs.fuzzy_function_combine(
        hourly_change_ds.wind_change,
        hourly_change_ds.q_change,
        hourly_change_ds.t_change,
        combine_method="mean")    
    
    #Save
    fuzzy.chunk({"time":744,"lat":-1,"lon":-1}).to_zarr(path + "sea_breeze_detection/era5/fuzzy_201301010000_201802282300.zarr",mode="w")