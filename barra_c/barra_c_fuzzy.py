import xarray as xr
from sea_breeze import sea_breeze_funcs
from dask.distributed import Client

if __name__ == "__main__":

    #Set up dask client
    client = Client()

    #Set up paths to sea_breeze_funcs data output and other inputs
    path = "/g/data/gb02/ab4502/"
    hourly_change_path = path+ "sea_breeze_detection/barra_c/F_hourly_201601010000_201601312300.nc"

    #Load the hourly change dataset
    hourly_change_ds = xr.open_dataset(
        hourly_change_path, chunks={"time":1,"lat":-1,"lon":-1}
        )
    
    #Combine the fuzzy functions
    fuzzy = sea_breeze_funcs.fuzzy_function_combine(
        hourly_change_ds.wind_change,
        hourly_change_ds.q_change,
        hourly_change_ds.t_change,
        combine_method="mean")    
    
    #Save
    fuzzy.to_netcdf(path + "sea_breeze_detection/barra_c/fuzzy_mean_201601010000_201601312300.nc")