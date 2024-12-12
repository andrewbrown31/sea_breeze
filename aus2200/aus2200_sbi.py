from sea_breeze import load_model_data, sea_breeze_funcs, sea_breeze_filters
from dask.distributed import Client
from dask.distributed import progress
import xarray as xr
import pandas as pd
import os
import argparse

if __name__ == "__main__":

    #Set up argument parser
    parser = argparse.ArgumentParser(
        prog="AUS2200 sea breeze index",
        description="This program applies the sea breeze index to a chosen period of AUS2200 data"
    )
    parser.add_argument("t1",type=str,help="Start time (Y-m-d H:M)")
    parser.add_argument("t2",type=str,help="End time (Y-m-d H:M)")
    parser.add_argument("--model",default="aus2200",type=str,help="Model name to save the output under. Could be aus2200 (default) or maybe aus2200_pert")
    parser.add_argument("--lat1",default=-45.7,type=float,help="Start latitude")
    parser.add_argument("--lat2",default=-6.5,type=float,help="End latitude")
    parser.add_argument("--lon1",default=108,type=float,help="Start longitude")
    parser.add_argument("--lon2",default=-158.5,type=float,help="End longitude")
    parser.add_argument("--hgt1",default=0,type=float,help="Start height to load from disk")
    parser.add_argument("--hgt2",default=5000,type=float,help="End height to load from disk")    
    parser.add_argument("-e","--exp_id",default="mjo-elnino",type=str,help="Experiment id for AUS2200 mjo runs")
    parser.add_argument('--subtract_mean',default=False,action=argparse.BooleanOptionalAction,help="Subtract mean to calculate SBI on perturbation winds")
    parser.add_argument("--lev_chunk",default=0,type=int,help="Chunk size for vertical level dim. Default is on-disk chunks")
    parser.add_argument("--time_chunk",default=0,type=int,help="Chunk size for time dim. Default is on-disk chunks")
    parser.add_argument("--lon_chunk",default=0,type=int,help="Chunk size for lon dim. Default is on-disk chunks")
    parser.add_argument("--lat_chunk",default=0,type=int,help="Chunk size for lat dim. Default is on-disk chunks")    
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
    exp_id = args.exp_id
    hgt1 = args.hgt1
    hgt2 = args.hgt2    
    hgt_slice=slice(hgt1,hgt2)

    #Set SBI settings and chunk settings
    subtract_mean = args.subtract_mean
    height_method = "blh"
    height_mean = False
    if args.lev_chunk==0:
        lev_chunk = {}
    else:
        lev_chunk = args.lev_chunk
    if args.time_chunk==0:
        time_chunk = {}
    else:
        time_chunk = args.time_chunk
    if args.lon_chunk==0:
        lon_chunk = {}
    else:
        lon_chunk = args.lon_chunk
    if args.lat_chunk==0:
        lat_chunk = {}
    else:
        lat_chunk = args.lat_chunk                        

    #Load AUS2200 model level winds, BLH and static info
    chunks = {"lev":lev_chunk,"time":time_chunk,"lat":lat_chunk,"lon":lon_chunk}
    orog, lsm = load_model_data.load_aus2200_static(
        "mjo-elnino",
        lon_slice,
        lat_slice)
    aus2200_va = load_model_data.round_times(
        load_model_data.load_aus2200_variable(
            "va",
            t1,
            t2,
            "mjo-elnino",
            lon_slice,
            lat_slice,
            "1hr",
            chunks=chunks,
            staggered="lat",
            hgt_slice=hgt_slice),
              "1hr")
    aus2200_ua = load_model_data.round_times(
        load_model_data.load_aus2200_variable(
            "ua",
            t1,
            t2,
            "mjo-elnino",
            lon_slice,
            lat_slice,
            "1hr",
            chunks=chunks,
            staggered="lon",
            hgt_slice=hgt_slice),
              "1hr")
    aus2200_zmla = load_model_data.round_times(
        load_model_data.load_aus2200_variable(
            "zmla",
            t1,
            t2,
            "mjo-elnino",
            lon_slice,
            lat_slice,
            "1hr",
            chunks=chunks,
            staggered="time"),
              "1hr")
    angle_ds = load_model_data.get_coastline_angle_kernel(
        lsm,
        compute=False,
        lat_slice=lat_slice,
        lon_slice=lon_slice,
        path_to_load="/g/data/gb02/ab4502/coastline_data/aus2200_v3.nc")

    #Calc SBI
    aus2200_wind = xr.Dataset({"u":aus2200_ua,"v":aus2200_va})
    sbi = sea_breeze_funcs.calc_sbi(aus2200_wind,
                                angle_ds,
                                subtract_mean=subtract_mean,
                                height_method=height_method,
                                blh_da=aus2200_zmla,
                                vert_coord="lev")

    #Save output
    out_path = "/g/data/gb02/ab4502/sea_breeze_detection/"+args.model+"/"
    sbi_fname = "sbi_"+exp_id+"_"+pd.to_datetime(t1).strftime("%Y%m%d%H%M")+"_"+\
                    (pd.to_datetime(t2).strftime("%Y%m%d%H%M"))+".nc"   
    if os.path.isdir(out_path):
        pass
    else:
        os.mkdir(out_path)   
    sbi_save = sbi.to_netcdf(out_path+sbi_fname,compute=False)
    progress(sbi_save.persist())
