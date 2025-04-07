import xarray as xr
from sea_breeze import sea_breeze_filters, load_model_data, sea_breeze_funcs, utils
from dask.distributed import Client
import pandas as pd
from dask.distributed import Client, progress
import warnings
import argparse

if __name__ == "__main__":

    #Set up argument parser
    parser = argparse.ArgumentParser(
        prog="BARRA-R filtering",
        description="This program loads a sea breeze diagnostic field from BARRA-R and applies a series of filters to it"
    )
    parser.add_argument("--model",default="barra_c",type=str,help="Model directory name for input/output. Could be barra_r (default)")
    parser.add_argument("--filter_name",default="",type=str,help="Filter name to add to the output file names")
    args = parser.parse_args()

    #Set up dask client
    client = Client()

    #Ignore warnings for runtime errors (divide by zero etc)
    warnings.simplefilter("ignore")

    #Set up paths to sea breeze diagnostics (output from sea_breeze_funcs.py)
    model = args.model
    filter_name = args.filter_name    
    path = "/g/data/ng72/ab4502/"
    fc_field_path = path + "sea_breeze_detection/"+model+"/Fc_201601010000_201601312300.nc"
    f_field_path = path + "sea_breeze_detection/"+model+"/F_201601010000_201601312300.nc"
    fuzzy_path = path+ "sea_breeze_detection/"+model+"/fuzzy_mean_201601010000_201601312300.nc"

    #Set up paths to other datasets that can be used for additional filtering.
    hourly_change_path = path+ "sea_breeze_detection/barra_r/F_hourly_201601010000_201601312300.nc"
    angle_ds_path = path + "coastline_data/barra_r.nc"
    
    #Set up domain bounds and variable name from field_path dataset
    t1 = "2016-01-06 00:00"
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
        fc_field_path,chunks={"time":1,"lat":-1,"lon":-1}
        ).Fc.sel(lat=lat_slice,lon=lon_slice,time=slice(t1,t2)) 
    fuzzy = xr.open_dataset(
        fuzzy_path,chunks={"time":1,"lat":-1,"lon":-1}
        )["__xarray_dataarray_variable__"].sel(lat=lat_slice,lon=lon_slice,time=slice(t1,t2)) 
    F = xr.open_dataset(
        f_field_path,chunks={"time":1,"lat":-1,"lon":-1}
        ).F.sel(lat=lat_slice,lon=lon_slice,time=slice(t1,t2)) 
    
    #Load other datasets that can be used for additional filtering
    angle_ds = load_model_data.get_coastline_angle_kernel(
        compute=False,path_to_load=angle_ds_path,lat_slice=lat_slice,lon_slice=lon_slice
        )
    hourly_change_ds = xr.open_dataset(
        hourly_change_path,chunks={"time":1,"lat":-1,"lon":-1}
        ).sel(lat=lat_slice,lon=lon_slice,time=slice(t1,t2))
    ta = load_model_data.load_barra_variable(
        "tas",t1,t2,"AUS-11","1hr",lat_slice,lon_slice,chunks={"time":1,"lat":-1,"lon":-1}
        )
    uas = load_model_data.load_barra_variable(
        "uas",t1,t2,"AUS-11","1hr",lat_slice,lon_slice,chunks={"time":1,"lat":-1,"lon":-1}
        )
    vas = load_model_data.load_barra_variable(
        "vas",t1,t2,"AUS-11","1hr",lat_slice,lon_slice,chunks={"time":1,"lat":-1,"lon":-1}
        )
    uprime, vprime = sea_breeze_funcs.rotate_wind(
        uas,
        vas,
        angle_ds["angle_interp"])
    _,lsm = load_model_data.load_barra_static(
        "AUS-11",lon_slice,lat_slice
        )

    for field_name, field in zip(["fuzzy_mean","Fc","F"],[fuzzy,Fc,F]):
    
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
            field,
            hourly_change_ds=hourly_change_ds,
            ta=ta,
            lsm=lsm,
            angle_ds=angle_ds,
            vprime=vprime.drop("height"),
            p=99.5,
            save_mask=True,
            filter_out_path=filter_out_path,
            props_df_out_path=props_df_out_path,
            skipna=False,
            **kwargs)