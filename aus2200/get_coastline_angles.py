from sea_breeze import load_model_data
from dask.distributed import Client

if __name__ == "__main__":

    #Initiate distributed dask client on the Gadi HPC
    client = Client()

    #This slice is the outer bounds of the AUS2200 domain
    lon_slice = slice(108,159)
    lat_slice = slice(-45.7,-6.831799)       

    #AUS2200
    _, aus2200_lsm = load_model_data.load_aus2200_static("mjo-neutral",lon_slice,lat_slice)
    aus2200_angles = load_model_data.get_coastline_angle_kernel(
        aus2200_lsm,
        R=4,
        compute=True,
        latlon_chunk_size=8,
        save=True,
        path_to_save="/g/data/gb02/ab4502/coastline_data/aus2200.nc")