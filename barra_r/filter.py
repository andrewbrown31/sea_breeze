import xarray as xr
from sea_breeze import sea_breeze_filters, load_model_data, utils
from dask.distributed import Client
import os
import pandas as pd

if __name__ == "__main__":

    #Set up dask client
    client = Client()

    #Set up paths to sea_breeze_funcs data output
    path = "/g/data/gb02/ab4502/"
    field_path = path + "sea_breeze_detection/barra_r/Fc_201601010000_201601312300.nc"
    hourly_change_path = path+ "sea_breeze_detection/barra_r/F_hourly_201601010000_201601312300.nc"

    #Set up path to coastline angle data
    angle_ds_path = path + "coastline_data/barra_r.nc"
    
    #Set up output paths
    props_df_out_path = path+"barra_r/props_df_Fc_201601010000_201601312300.csv" 
    filter_out_path = path+"barra_r/filtered_mask_Fc_201601010000_201601312300.csv" 

    #Set up domain bounds and variable name from field_path dataset
    field_name = "Fc"
    t1 = "2016-01-21 12:00"
    t2 = "2016-01-22 12:00"
    lat_slice = slice(-45.7,-6.9)
    lon_slice = slice(108,158.5)

    #Set up filtering options
    kwargs = {
                "orientation_filter":True,
                "dist_to_coast_filter":True,
                "land_sea_temperature_filter":True,                    
                "output_land_sea_temperature_diff":True,
                "temperature_change_filter":True,
                "humidity_change_filter":True,
                "wind_change_filter":True,
                }      
    kwargs["props_df_output_path"] = props_df_out_path

    #Load optional extra data for filtering: tas, lsm and coastline angles
    ta = load_model_data.load_barra_variable(
        "tas",t1,t2,"AUS-11","1hr",lat_slice,lon_slice)
    _,lsm = load_model_data.load_barra_static(
        "AUS-11",lon_slice,lat_slice)
    angle_ds = load_model_data.get_coastline_angle_kernel(
        compute=False,path_to_load=angle_ds_path,lat_slice=lat_slice,lon_slice=lon_slice)
    
    #Run the filtering
    filtered_mask = filter_ds_driver(field_path,
                     field_name,
                     lat_slice,
                     lon_slice,
                     save_mask=True,
                     filter_out_path=filter_out_path,
                     t1=t1,
                     t2=t2,
                     hourly_change_path=hourly_change_path,
                     ta=ta,
                     lsm=lsm,
                     angle_ds=angle_ds,
                     kwargs=kwargs)