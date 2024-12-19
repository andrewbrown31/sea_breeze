from sea_breeze import load_model_data
from dask.distributed import Client

if __name__ == "__main__":

    #Initiate distributed dask client on the Gadi HPC
    client = Client()

    #This slice is the outer bounds of the AUS2200 domain
    lon_slice = slice(108,159)
    lat_slice = slice(-45.7,-6.831799)       

    _, lsm, cl = load_model_data.load_era5_static(lon_slice,lat_slice,"2016-01-01 00:00","2016-01-01 00:00")
    load_model_data.get_coastline_angle_kernel(
        load_model_data.remove_era5_inland_lakes(lsm,cl),
        R=20,
        compute=True,
        latlon_chunk_size=50,
        save=True,
        path_to_save="/g/data/gb02/ab4502/coastline_data/era5.nc")
