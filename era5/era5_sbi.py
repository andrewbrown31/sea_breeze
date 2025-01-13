from sea_breeze import load_model_data, sea_breeze_funcs, sea_breeze_filters
from dask.distributed import Client
from dask.distributed import progress
import xarray as xr
import pandas as pd
import os
import argparse
import numpy as np

if __name__ == "__main__":

    #Set up argument parser
    parser = argparse.ArgumentParser(
        prog="ERA5 sea breeze index",
        description="This program applies the sea breeze index to a chosen period of ERA5 data"
    )
    parser.add_argument("t1",type=str,help="Start time (Y-m-d H:M)")
    parser.add_argument("t2",type=str,help="End time (Y-m-d H:M)")
    parser.add_argument("--model",default="era5",type=str,help="Model name to save the output under. Could be era5 (default) or maybe era5_pert")
    parser.add_argument("--lat1",default=-45.7,type=float,help="Start latitude")
    parser.add_argument("--lat2",default=-6.5,type=float,help="End latitude")
    parser.add_argument("--lon1",default=108,type=float,help="Start longitude")
    parser.add_argument("--lon2",default=-158.5,type=float,help="End longitude")
    parser.add_argument("--hgt1",default=0,type=float,help="Start height to load from disk")
    parser.add_argument("--hgt2",default=4500,type=float,help="End height to load from disk")    
    parser.add_argument('--subtract_mean',default=False,action=argparse.BooleanOptionalAction,help="Subtract mean to calculate SBI on perturbation winds")
    parser.add_argument("--height_method",default="blh",type=str,help="Either blh or static")
    args = parser.parse_args()

    if args.subtract_mean:
        print("Subtracting daily mean...")
    else:
        print("Not subtracting daily mean...")

    #Initiate distributed dask client on the Gadi HPC
    client = Client()

    #Set the domain bounds
    lat_slice=slice(args.lat1,args.lat2)
    lon_slice=slice(args.lon1,args.lon2)    

    #Set time slice and model name    
    t1 = args.t1
    t2 = args.t2
    hgt1 = args.hgt1
    hgt2 = args.hgt2    
    hgt_slice=np.arange(hgt1,hgt2+100,100)

    #Set SBI settings and chunk settings
    subtract_mean = args.subtract_mean
    height_method = args.height_method
    height_mean = False                     

    #Load ERA5 model level winds, BLH and static info
    chunks = {"time":{},"latitude":{},"longitude":{},"hybrid":-1}
    era5_wind, lsm = load_model_data.load_era5_ml_and_interp(
            t1,
            t2,
            lat_slice,
            lon_slice,
            heights=hgt_slice,
            chunks=chunks)
    era5_zmla = load_model_data.load_era5_variable(
            ["blh"],
            t1,
            t2,
            lon_slice,
            lat_slice,
            chunks=chunks)
    angle_ds = load_model_data.get_coastline_angle_kernel(
        lsm,
        compute=False,
        lat_slice=lat_slice,
        lon_slice=lon_slice,
        path_to_load="/g/data/gb02/ab4502/coastline_data/era5_global_angles.nc")

    #Calc SBI
    sbi = sea_breeze_funcs.calc_sbi(era5_wind,
                                angle_ds,
                                height_mean=height_mean,
                                subtract_mean=subtract_mean,
                                height_method=height_method,
                                blh_da=era5_zmla["blh"]["blh"],
                                vert_coord="height")
    #print(sbi)

    #Save output
    out_path = "/g/data/gb02/ab4502/sea_breeze_detection/"+args.model+"/"
    sbi_fname = "sbi_"+pd.to_datetime(t1).strftime("%Y%m%d%H%M")+"_"+\
                    (pd.to_datetime(t2).strftime("%Y%m%d%H%M"))+".nc"   
    if os.path.isdir(out_path):
        pass
    else:
        os.mkdir(out_path)   
    sbi_save = sbi.to_netcdf(out_path+sbi_fname,compute=False,engine="netcdf4")
    progress(sbi_save.persist())
