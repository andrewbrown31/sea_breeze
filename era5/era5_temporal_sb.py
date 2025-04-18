from sea_breeze import load_model_data, sea_breeze_funcs, sea_breeze_filters
from dask.distributed import Client
from dask.distributed import progress
import pandas as pd
import os
import argparse
import metpy.calc as mpcalc

if __name__ == "__main__":

    #Set up argument parser
    parser = argparse.ArgumentParser(
        prog="ERA5 temporal sea breeze functions",
        description="This program applies temporal sea breeze functions to a chosen period of ERA5 data, such as the hourly difference in temperature, humidity and wind"
    )
    parser.add_argument("t1",type=str,help="Start time (Y-m-d H:M)")
    parser.add_argument("t2",type=str,help="End time (Y-m-d H:M)")
    parser.add_argument("--model",default="era5",type=str,help="Model name to save the output under.")
    parser.add_argument("--lat1",default=-45.7,type=float,help="Start latitude")
    parser.add_argument("--lat2",default=-6.9,type=float,help="End latitude")
    parser.add_argument("--lon1",default=108,type=float,help="Start longitude")
    parser.add_argument("--lon2",default=158.5,type=float,help="End longitude")
    args = parser.parse_args()

    #Initiate distributed dask client on the Gadi HPC
    client = Client()

    #Set the domain bounds
    lat_slice=slice(args.lat1,args.lat2)
    lon_slice=slice(args.lon1,args.lon2)    

    #Set time slice and model name    
    t1 = args.t1
    t2 = args.t2                

    #Load ERA5
    chunks = {"time":-1,"lat":{},"lon":{}}
    orog, lsm, _ = load_model_data.load_era5_static(
        lon_slice,
        lat_slice,
        t1,
        t2)
    era5_uas = load_model_data.load_era5_variable(
            ["10u"],
            t1,
            t2,
            lon_slice,
            lat_slice,
            chunks=chunks)["10u"]["u10"]
    era5_vas = load_model_data.load_era5_variable(
            ["10v"],
            t1,
            t2,
            lon_slice,
            lat_slice,
            chunks=chunks)["10v"]["v10"]
    era5_2d = load_model_data.load_era5_variable(
            ["2d"],
            t1,
            t2,
            lon_slice,
            lat_slice,
            chunks=chunks)["2d"]["d2m"]
    era5_ps = load_model_data.load_era5_variable(
            ["sp"],
            t1,
            t2,
            lon_slice,
            lat_slice,
            chunks=chunks)["sp"]["sp"]
    era5_huss = mpcalc.specific_humidity_from_dewpoint(era5_ps,era5_2d)            
    era5_tas = load_model_data.load_era5_variable(
            ["2t"],
            t1,
            t2,
            lon_slice,
            lat_slice,
            chunks=chunks)["2t"]["t2m"]
    angle_ds = load_model_data.get_coastline_angle_kernel(
        lsm,
        compute=False,
        lat_slice=lat_slice,
        lon_slice=lon_slice,
        path_to_load="/g/data/ng72/ab4502/coastline_data/era5.nc")

    #Calc hourly change conditions
    F_hourly = sea_breeze_funcs.hourly_change(
        era5_huss,
        era5_tas,
        era5_uas,
        era5_vas,
        angle_ds["angle_interp"],
        lat_chunk="auto",
        lon_chunk="auto"
    )    

    #Setup out paths
    out_path = "/g/data/ng72/ab4502/sea_breeze_detection/"+args.model+"/"   
    F_hourly_fname = "F_hourly_"+pd.to_datetime(t1).strftime("%Y%m%d%H%M")+"_"+\
                        (pd.to_datetime(t2).strftime("%Y%m%d%H%M"))+".zarr"               
    if os.path.isdir(out_path):
        pass
    else:
        os.mkdir(out_path)   

    #Save the output
    print("INFO: Computing hourly changes...")
    F_hourly_save = F_hourly.to_zarr(out_path+F_hourly_fname,compute=False,mode="w")
    progress(F_hourly_save.persist())    
