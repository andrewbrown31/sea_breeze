import xarray as xr
from sea_breeze import sea_breeze_filters, load_model_data, sea_breeze_funcs
from dask.distributed import Client
import pandas as pd
from dask.distributed import Client, progress
import warnings
import argparse

if __name__ == "__main__":

    #Set up argument parser
    parser = argparse.ArgumentParser(
        prog="AUS2200 filtering",
        description="This program loads a sea breeze diagnostic field from AUS2200 and applies a series of filters to it"
    )
    parser.add_argument("--model",default="aus2200",type=str,help="Model directory name for input/output. Could be aus2200 (default) or maybe aus2200_pert")
    args = parser.parse_args()

    #Set up dask client
    client = Client()

    #Ignore warnings for runtime errors (divide by zero etc)
    warnings.simplefilter("ignore")

    #Set up paths to sea breeze diagnostics (output from sea_breeze_funcs.py)
    model = args.model
    path = "/g/data/gb02/ab4502/"
    fc_field_path = path + "sea_breeze_detection/"+model+"/Fc_mjo-elnino_201601010000_201601312300.zarr"
    f_field_path = path + "sea_breeze_detection/"+model+"/F_mjo-elnino_201601010000_201601312300.zarr"
    sbi_path = path+ "sea_breeze_detection/"+model+"/sbi_mjo-elnino_201601010100_201601312300.zarr"
    fuzzy_path = path+ "sea_breeze_detection/"+model+"/fuzzy_mjo-elnino_201601010000_201601312300.zarr"

    #Set up paths to other datasets that can be used for additional filtering. These always use standard aus2200 resolution, so no need to specify model (aus2200 or aus2200_pert or aus2200_smooth_s2)
    hourly_change_path = path+ "sea_breeze_detection/aus2200/F_hourly_mjo-elnino_201601010000_201601312300.zarr"
    angle_ds_path = path + "coastline_data/aus2200.nc"
    
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
        "temperature_change_filter":True,
        "humidity_change_filter":True,
        "wind_change_filter":True,
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
    F = xr.open_dataset(
        f_field_path,chunks={"time":1,"lat":-1,"lon":-1}
        ).F.sel(lat=lat_slice,lon=lon_slice,time=slice(t1,t2))
    fuzzy = xr.open_dataset(
        fuzzy_path,chunks={"time":1,"lat":-1,"lon":-1}
        )["__xarray_dataarray_variable__"].sel(lat=lat_slice,lon=lon_slice,time=slice(t1,t2))
    sbi = xr.open_dataset(
        sbi_path,chunks={"time":1,"lat":-1,"lon":-1}
        ).sbi.sel(lat=lat_slice,lon=lon_slice,time=slice(t1,t2))
    
    #Load other datasets that can be used for additional filtering
    angle_ds = load_model_data.get_coastline_angle_kernel(
        compute=False,path_to_load=angle_ds_path,lat_slice=lat_slice,lon_slice=lon_slice
        )
    hourly_change_ds = xr.open_dataset(
        hourly_change_path,chunks={},drop_variables="dqu_dt"
        ).sel(lat=lat_slice,lon=lon_slice,time=slice(t1,t2)).chunk({"time":1,"lat":-1,"lon":-1})
    ta = load_model_data.round_times(
        load_model_data.load_aus2200_variable(
        "tas",
        t1,
        t2,
        "mjo-elnino",
        lon_slice,
        lat_slice,
        "10min",
        chunks={"time":1,"lat":-1,"lon":-1}),
        "10min")
    aus2200_vas = load_model_data.round_times(
        load_model_data.load_aus2200_variable(
            "vas",
            t1,
            t2,
            "mjo-elnino",
            lon_slice,
            lat_slice,
            "10min",
            chunks={"time":1,"lat":-1,"lon":-1},
            staggered="lat",
            smooth=False),
            "10min")
    aus2200_uas = load_model_data.round_times(
        load_model_data.load_aus2200_variable(
            "uas",
            t1,
            t2,
            "mjo-elnino",
            lon_slice,
            lat_slice,
            "10min",
            chunks={"time":1,"lat":-1,"lon":-1},
            staggered="lon",
            smooth=False),
            "10min")
    ta = ta.sel(time=ta.time.dt.minute==0)
    aus2200_vas = aus2200_vas.sel(time=aus2200_vas.time.dt.minute==0)
    aus2200_uas = aus2200_uas.sel(time=aus2200_uas.time.dt.minute==0)    
    uprime, vprime = sea_breeze_funcs.rotate_wind(
        aus2200_uas,
        aus2200_vas,
        angle_ds["angle_interp"])
    orog, lsm = load_model_data.load_aus2200_static(
        "mjo-elnino",
        lon_slice,
        lat_slice)
    
    for field_name, field in zip(["Fc","F","fuzzy","sbi"],[Fc,F,fuzzy,sbi]):
    
        print(field_name)

        #Set up output paths
        props_df_out_path = path+\
            "sea_breeze_detection/"+model+"/props_df_"+\
                field_name+"_"+\
                    pd.to_datetime(t1).strftime("%Y%m%d%H%M")+"_"+\
                        pd.to_datetime(t2).strftime("%Y%m%d%H%M")+".csv" 
        filter_out_path = path+\
            "sea_breeze_detection/"+model+"/filtered_mask_"+\
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
            p=99.75,
            save_mask=True,
            filter_out_path=filter_out_path,
            props_df_out_path=props_df_out_path,
            skipna=False,
            **kwargs)