import xarray as xr
from sea_breeze import sea_breeze_filters, load_model_data, sea_breeze_funcs
from dask.distributed import Client
import os
import pandas as pd
from dask.distributed import Client, progress
import warnings
import argparse

if __name__ == "__main__":

    #Set up argument parser
    parser = argparse.ArgumentParser(
        prog="ERA5 filtering",
        description="This program loads a sea breeze diagnostic field from ERA5 and applies a series of filters to it"
    )
    parser.add_argument("--model",default="era5",type=str,help="Model directory name for input/output. Could be era5 (default)")
    parser.add_argument("--filter_name",default="",type=str,help="Filter name to add to the output file names")
    args = parser.parse_args()

    #Set up dask client
    client = Client()

    #Ignore warnings for runtime errors (divide by zero etc)
    warnings.simplefilter("ignore")

    #Set up paths to sea_breeze_funcs data output and other inputs
    model = args.model
    filter_name = args.filter_name
    path = "/g/data/ng72/ab4502/"
    fc_field_path = path + "sea_breeze_detection/"+model+"/Fc_201601010000_201601312300.nc"
    f_field_path = path + "sea_breeze_detection/"+model+"/F_201601010000_201601312300.nc"
    sbi_field_path = path+ "sea_breeze_detection/"+model+"/sbi_201601010000_201601312300.nc"
    fuzzy_field_path = path+ "sea_breeze_detection/"+model+"/fuzzy_201601010000_201601312300.nc"

    #Set up paths to other datasets that can be used for additional filtering
    hourly_change_path = path+ "sea_breeze_detection/era5/F_hourly_201601010000_201601312300.nc"
    angle_ds_path = path + "coastline_data/era5.nc"
    
    #Set up domain bounds and variable name from field_path dataset
    t1 = "2016-01-06 06:00"
    #t2 = "2016-01-06 06:00"
    t2 = "2016-01-12 23:00"
    lat_slice = slice(-45.7,-6.9)
    lon_slice = slice(108,158.5)

    #Set up filtering options
    kwargs = {
        "orientation_filter":True,
        "aspect_filter":True,
        "area_filter":True,        
        "land_sea_temperature_filter":True,                    
        "temperature_change_filter":False,
        "humidity_change_filter":False,
        "wind_change_filter":False,
        "propagation_speed_filter":True,
        "dist_to_coast_filter":False,
        "output_land_sea_temperature_diff":False,        
        "time_filter":False,
        "orientation_tol":45,
        "area_thresh_pixels":12,
        "aspect_thresh":2,
        "land_sea_temperature_diff_thresh":0,
        "propagation_speed_thresh":0,
        }

    #Load sea breeze diagnostics
    Fc = xr.open_dataset(
        fc_field_path,chunks={}
        ).Fc.sel(lat=lat_slice,lon=lon_slice,time=slice(t1,t2)) 
    sbi = xr.open_dataset(
        sbi_field_path,chunks={}
        ).sbi.sel(lat=lat_slice,lon=lon_slice,time=slice(t1,t2)) 
    F = xr.open_dataset(
        f_field_path,chunks={}
        ).F.sel(lat=lat_slice,lon=lon_slice,time=slice(t1,t2)) 
    fuzzy = xr.open_dataset(
        fuzzy_field_path,chunks={}
        )["__xarray_dataarray_variable__"].sel(lat=lat_slice,lon=lon_slice,time=slice(t1,t2))
    
    #Load other datasets that can be used for additional filtering
    angle_ds = load_model_data.get_coastline_angle_kernel(
        compute=False,path_to_load=angle_ds_path,lat_slice=lat_slice,lon_slice=lon_slice
        )
    hourly_change_ds = xr.open_dataset(
        hourly_change_path,chunks={}
        ).sel(lat=lat_slice,lon=lon_slice,time=slice(t1,t2)).chunk({"time":1,"lat":-1,"lon":-1})
    ta = load_model_data.load_era5_variable(
        ["2t"],t1,t2,lon_slice,lat_slice,chunks={}
        )["2t"]["t2m"].chunk({"time":1,"lat":-1,"lon":-1})
    uas = load_model_data.load_era5_variable(
        ["10u"],t1,t2,lon_slice,lat_slice,chunks={}
        )["10u"]["u10"].chunk({"time":1,"lat":-1,"lon":-1})
    vas = load_model_data.load_era5_variable(
        ["10v"],t1,t2,lon_slice,lat_slice,chunks={}
        )["10v"]["v10"]   .chunk({"time":1,"lat":-1,"lon":-1})
    uprime,vprime = sea_breeze_funcs.rotate_wind(
        uas,
        vas,
        angle_ds["angle_interp"])
    _,lsm,_ = load_model_data.load_era5_static(
        lon_slice,lat_slice,t1,t2
        )
    
    for field_name, field in zip(["fuzzy_mean","Fc","F","sbi"],[fuzzy,Fc,F,sbi]):
    
        print(field_name)

        #Set up output paths
        props_df_out_path = path+\
            "sea_breeze_detection/"+model+"/props_df_"+filter_name+"_"+\
                field_name+"_"+\
                    pd.to_datetime(t1).strftime("%Y%m%d%H%M")+"_"+\
                        pd.to_datetime(t2).strftime("%Y%m%d%H%M")+".csv" 
        filter_out_path = path+\
            "sea_breeze_detection/"+model+"/filtered_mask_"+filter_name+"_"+\
                field_name+"_"+\
                    pd.to_datetime(t1).strftime("%Y%m%d%H%M")+"_"+\
                        pd.to_datetime(t2).strftime("%Y%m%d%H%M")+".zarr"

        #Run the filtering
        filtered_mask = sea_breeze_filters.filter_3d(
            field.chunk({"time":1,"lat":-1,"lon":-1}),
            hourly_change_ds=hourly_change_ds,
            ta=ta,
            lsm=lsm,
            angle_ds=angle_ds,
            vprime=vprime,
            p=99.5,
            save_mask=True,
            filter_out_path=filter_out_path,
            props_df_out_path=props_df_out_path,
            skipna=False,
            **kwargs)