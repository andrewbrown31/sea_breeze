import xarray as xr
from sea_breeze import sea_breeze_filters, load_model_data, sea_breeze_funcs
from dask.distributed import Client
import os
import pandas as pd
from dask.distributed import Client, progress
import warnings

if __name__ == "__main__":

    #Set up dask client
    client = Client()

    #Ignore warnings for runtime errors (divide by zero etc)
    warnings.simplefilter("ignore")

    #Set up paths to sea_breeze_funcs data output and other inputs
    path = "/g/data/gb02/ab4502/"
    fc_field_path = path + "sea_breeze_detection/era5/Fc_201601010000_201601312300.nc"
    f_field_path = path + "sea_breeze_detection/era5/F_201601010000_201601312300.nc"
    hourly_change_path = path+ "sea_breeze_detection/era5/F_hourly_201601010000_201601312300.nc"
    sbi_field_path = path+ "sea_breeze_detection/era5/sbi_201601010000_201601312300.nc"
    angle_ds_path = path + "coastline_data/era5.nc"
    
    #Set up domain bounds and variable name from field_path dataset
    t1 = "2016-01-01 00:00"
    t2 = "2016-01-31 23:00"
    lat_slice = slice(-45.7,-6.9)
    lon_slice = slice(108,158.5)

    #Set up filtering options
    kwargs = {
        "orientation_filter":True,
        "aspect_filter":True,
        "area_filter":True,        
        "land_sea_temperature_filter":True,                    
        "dist_to_coast_filter":False,
        "output_land_sea_temperature_diff":False,
        "temperature_change_filter":False,
        "humidity_change_filter":False,
        "wind_change_filter":False,
        "time_filter":False,
        "orientation_tol":45,
        "area_thresh_pixels":10,
        "aspect_thresh":2,
        "land_sea_temperature_diff_thresh":0
        }

    #Load data for filtering: Fc, hourly_change, tas, lsm and coastline angles
    Fc = xr.open_dataset(
        fc_field_path,chunks="auto"
        ).Fc.sel(lat=lat_slice,lon=lon_slice,time=slice(t1,t2)) 
    sbi = xr.open_dataset(
        sbi_field_path,chunks="auto"
        ).sbi.sel(lat=lat_slice,lon=lon_slice,time=slice(t1,t2)) 
    F = xr.open_dataset(
        f_field_path,chunks="auto"
        ).F.sel(lat=lat_slice,lon=lon_slice,time=slice(t1,t2)) 
    hourly_change_ds = xr.open_dataset(
        hourly_change_path,chunks="auto"
        ).sel(lat=lat_slice,lon=lon_slice,time=slice(t1,t2))
    ta = load_model_data.load_era5_variable(
        ["2t"],t1,t2,lon_slice,lat_slice,chunks="auto"
        )["2t"]["t2m"]
    _,lsm,_ = load_model_data.load_era5_static(
        lon_slice,lat_slice,t1,t2
        )
    angle_ds = load_model_data.get_coastline_angle_kernel(
        compute=False,path_to_load=angle_ds_path,lat_slice=lat_slice,lon_slice=lon_slice
        )
    
    #Compute the fuzzy function from houry changes
    fuzzy = sea_breeze_funcs.fuzzy_function_combine(
        hourly_change_ds.wind_change,
        hourly_change_ds.q_change,
        hourly_change_ds.t_change,
        combine_method="mean")
    
    for field_name, field in zip(["fuzzy_mean","Fc","F","sbi"],[fuzzy,Fc,F,sbi]):
    
        print(field_name)

        #Set up output paths
        props_df_out_path = path+\
            "sea_breeze_detection/era5/props_df_"+\
                field_name+"_"+\
                    pd.to_datetime(t1).strftime("%Y%m%d%H%M")+"_"+\
                        pd.to_datetime(t2).strftime("%Y%m%d%H%M")+".csv" 
        filter_out_path = path+\
            "sea_breeze_detection/era5/filtered_mask_"+\
                field_name+"_"+\
                    pd.to_datetime(t1).strftime("%Y%m%d%H%M")+"_"+\
                        pd.to_datetime(t2).strftime("%Y%m%d%H%M")+".nc"

        #Run the filtering
        filtered_mask = sea_breeze_filters.filter_3d(
            field,
            hourly_change_ds=hourly_change_ds,
            ta=ta,
            lsm=lsm,
            angle_ds=angle_ds,
            p=99.5,
            save_mask=True,
            filter_out_path=filter_out_path,
            props_df_output_path=props_df_out_path,
            skipna=False,
            **kwargs)