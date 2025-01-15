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
        R=50,
        compute=True,
        latlon_chunk_size=50,
        save=True,
        path_to_save="/g/data/gb02/ab4502/coastline_data/era5.nc")
    
    #ERA5 global
    _, era5_lsm, era5_cl = load_model_data.load_era5_static(slice(0,360),slice(-90,90),"2023-01-01 00:00","2023-01-01 00:00")
    era5_lsm["lon"] = ((era5_lsm.lon - 180) % 360) - 180
    era5_lsm = era5_lsm.sortby("lon")
    era5_cl["lon"] = ((era5_cl.lon - 180) % 360) - 180
    era5_cl = era5_cl.sortby("lon")
    era5_angles = load_model_data.get_coastline_angle_kernel(
        load_model_data.remove_era5_inland_lakes(
            era5_lsm,era5_cl),
        R=50,
        compute=True,
        latlon_chunk_size=10,
        save=True,
        path_to_save="/g/data/gb02/ab4502/coastline_data/era5_global_angles.nc")
