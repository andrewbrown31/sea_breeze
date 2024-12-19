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
        prog="BARRA-R frontogenesis",
        description="This program applies frontogenesis functions to a chosen period of BARRA-R data"
    )
    parser.add_argument("t1",type=str,help="Start time (Y-m-d H:M)")
    parser.add_argument("t2",type=str,help="End time (Y-m-d H:M)")
    parser.add_argument("--model",default="barra_r",type=str,help="Model name to save the output under")
    parser.add_argument("--lat1",default=-45.7,type=float,help="Start latitude")
    parser.add_argument("--lat2",default=-6.5,type=float,help="End latitude")
    parser.add_argument("--lon1",default=108,type=float,help="Start longitude")
    parser.add_argument("--lon2",default=-158.5,type=float,help="End longitude")
    parser.add_argument("--time_chunk",default=0,type=int,help="Chunk size for time dim. Default is on-disk chunks")
    parser.add_argument("--lon_chunk",default=0,type=int,help="Chunk size for lon dim. Default is on-disk chunks")
    parser.add_argument("--lat_chunk",default=0,type=int,help="Chunk size for lat dim. Default is on-disk chunks")    
    args = parser.parse_args()

    #Initiate distributed dask client on the Gadi HPC
    client = Client()

    #Set the domain bounds
    lat_slice=slice(args.lat1,args.lat2)
    lon_slice=slice(args.lon1,args.lon2)    

    #Set time slice and model name    
    t1 = args.t1
    t2 = args.t2
    exp_id = args.exp_id

    #Setup settings and chunk settings
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
    chunks = {"time":time_chunk,"lat":lat_chunk,"lon":lon_chunk}
    orog, lsm = load_model_data.load_barra_static(
        "AUST-04",
        lon_slice,
        lat_slice)
    vas = load_model_data.load_barra_variable(
            "vas",
            t1,
            t2,
            "AUST-04",
            "1hr",
            lat_slice,
            lon_slice,
            chunks=chunks)
    uas = load_model_data.load_barra_variable(
            "uas",
            t1,
            t2,
            "AUST-04",
            "1hr",
            lat_slice,
            lon_slice,
            chunks=chunks)
    huss = load_model_data.load_barra_variable(
            "huss",
            t1,
            t2,
            "AUST-04",
            "1hr",
            lat_slice,
            lon_slice,
            chunks=chunks)
    tas = load_model_data.load_barra_variable(
            "tas",
            t1,
            t2,
            "AUST-04",
            "1hr",
            lat_slice,
            lon_slice,
            chunks=chunks)     
    angle_ds = load_model_data.get_coastline_angle_kernel(
        lsm,
        compute=False,
        lat_slice=lat_slice,
        lon_slice=lon_slice,
        path_to_load="/g/data/gb02/ab4502/coastline_data/barra_c_angles_v2.nc")

    #Calc 2d kinematic moisture frontogenesis
    F = sea_breeze_funcs.kinematic_frontogenesis(
        huss,
        uas,
        vas
    )

    #Calc coast-relative 2d kinematic moisture frontogenesis
    Fc = sea_breeze_funcs.coast_relative_frontogenesis(
        huss,
        uas,
        vas,
        angle_ds
    )

    #Calc moisture flux gradient
    F_dqdt = sea_breeze_funcs.moisture_flux_gradient(
        huss,
        uas,
        vas,
        angle_ds,
        lat_chunk="auto",
        lon_chunk="auto"
    )

    #Calc hourly change conditions
    F_hourly = sea_breeze_funcs.hourly_change(
        huss,
        tas,
        uas,
        vas,
        angle_ds,
        lat_chunk="auto",
        lon_chunk="auto"
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
    F_save = F.to_netcdf(out_path+F_fname,compute=False)
    progress(F_save.persist())
    print("INFO: Computing coast-relative frontogenesis...")
    Fc_save = Fc.to_netcdf(out_path+Fc_fname,compute=False)
    progress(Fc_save.persist())
    print("INFO: Computing moisture flux change...")
    F_dqdt_save = F_dqdt.to_netcdf(out_path+F_dqdt_fname,compute=False)
    progress(F_dqdt_save.persist())
    print("INFO: Computing hourly changes...")
    F_hourly_save = F_hourly.to_netcdf(out_path+F_hourly_fname,compute=False)
    progress(F_hourly_save.persist())    
