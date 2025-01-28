from sea_breeze import load_model_data, sea_breeze_funcs, sea_breeze_filters
from dask.distributed import Client
from dask.distributed import progress
import pandas as pd
import os
import metpy.calc as mpcalc
import argparse

if __name__ == "__main__":

    #Set up argument parser
    parser = argparse.ArgumentParser(
        prog="AUS2200 frontogenesis",
        description="This program applies frontogenesis functions to a chosen period of ERA5 data"
    )
    parser.add_argument("t1",type=str,help="Start time (Y-m-d H:M)")
    parser.add_argument("t2",type=str,help="End time (Y-m-d H:M)")
    parser.add_argument("--model",default="era5",type=str,help="Model name to save the output under")
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

    #Load AUS2200 model level winds, BLH and static info
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
            chunks=chunks)["10u"]["u10"].chunk({"lat":-1,"lon":-1})
    era5_vas = load_model_data.load_era5_variable(
            ["10v"],
            t1,
            t2,
            lon_slice,
            lat_slice,
            chunks=chunks)["10v"]["v10"].chunk({"lat":-1,"lon":-1})
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
    
    angle_ds = load_model_data.get_coastline_angle_kernel(
        lsm,
        compute=False,
        lat_slice=lat_slice,
        lon_slice=lon_slice,
        path_to_load="/g/data/gb02/ab4502/coastline_data/era5_global_angles.nc")

    #Calc 2d kinematic moisture frontogenesis
    F = sea_breeze_funcs.kinematic_frontogenesis(
        era5_huss,
        era5_uas,
        era5_vas
    )

    #Calc coast-relative 2d kinematic moisture frontogenesis
    Fc = sea_breeze_funcs.coast_relative_frontogenesis(
        era5_huss,
        era5_uas,
        era5_vas,
        angle_ds["angle_interp"]
    )

    #Setup out paths
    out_path = "/g/data/gb02/ab4502/sea_breeze_detection/"+args.model+"/"
    F_fname = "F_"+pd.to_datetime(t1).strftime("%Y%m%d%H%M")+"_"+\
                    (pd.to_datetime(t2).strftime("%Y%m%d%H%M"))+".nc"   
    Fc_fname = "Fc_"+pd.to_datetime(t1).strftime("%Y%m%d%H%M")+"_"+\
                        (pd.to_datetime(t2).strftime("%Y%m%d%H%M"))+".nc"       
    F_dqdt_fname = "F_dqdt_"+pd.to_datetime(t1).strftime("%Y%m%d%H%M")+"_"+\
                        (pd.to_datetime(t2).strftime("%Y%m%d%H%M"))+".nc"       
    F_hourly_fname = "F_hourly_"+pd.to_datetime(t1).strftime("%Y%m%d%H%M")+"_"+\
                        (pd.to_datetime(t2).strftime("%Y%m%d%H%M"))+".nc"               
    if os.path.isdir(out_path):
        pass
    else:
        os.mkdir(out_path)   

    #Save the output
    print("INFO: Computing frontogenesis...")
    F_save = F.to_netcdf(out_path+F_fname,compute=False,engine="netcdf4")
    progress(F_save.persist())
    print("INFO: Computing coast-relative frontogenesis...")
    Fc_save = Fc.to_netcdf(out_path+Fc_fname,compute=False,engine="netcdf4")
    progress(Fc_save.persist())
