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
        prog="BARRA-R frontogenesis",
        description="This program applies frontogenesis functions to a chosen period of BARRA-R data"
    )
    parser.add_argument("t1",type=str,help="Start time (Y-m-d H:M)")
    parser.add_argument("t2",type=str,help="End time (Y-m-d H:M)")
    parser.add_argument("--model",default="barra_r",type=str,help="Model name to save the output under.")
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
    orog, lsm = load_model_data.load_barra_static(
        "AUS-11",
        lon_slice,
        lat_slice)
    vas = load_model_data.load_barra_variable(
            "vas",
            t1,
            t2,
            "AUS-11",
            "1hr",
            lat_slice,
            lon_slice,
            chunks=chunks)
    uas = load_model_data.load_barra_variable(
            "uas",
            t1,
            t2,
            "AUS-11",
            "1hr",
            lat_slice,
            lon_slice,
            chunks=chunks)
    huss = load_model_data.load_barra_variable(
            "huss",
            t1,
            t2,
            "AUS-11",
            "1hr",
            lat_slice,
            lon_slice,
            chunks=chunks)
    tas = load_model_data.load_barra_variable(
            "tas",
            t1,
            t2,
            "AUS-11",
            "1hr",
            lat_slice,
            lon_slice,
            chunks=chunks)     
    angle_ds = load_model_data.get_coastline_angle_kernel(
        lsm,
        compute=False,
        lat_slice=lat_slice,
        lon_slice=lon_slice,
        path_to_load="/g/data/gb02/ab4502/coastline_data/barra_r.nc")

    #Calc moisture flux gradient
    F_dqdt = sea_breeze_funcs.moisture_flux_gradient(
        huss,
        uas,
        vas,
        angle_ds["angle_interp"],
        lat_chunk="auto",
        lon_chunk="auto"
    )

    #Calc hourly change conditions
    F_hourly = sea_breeze_funcs.hourly_change(
        huss,
        tas,
        uas,
        vas,
        angle_ds["angle_interp"],
        lat_chunk="auto",
        lon_chunk="auto"
    )    

    #Setup out paths
    out_path = "/g/data/gb02/ab4502/sea_breeze_detection/"+args.model+"/"
    F_dqdt_fname = "F_dqdt_"+pd.to_datetime(t1).strftime("%Y%m%d%H%M")+"_"+\
                        (pd.to_datetime(t2).strftime("%Y%m%d%H%M"))+".nc"       
    F_hourly_fname = "F_hourly_"+pd.to_datetime(t1).strftime("%Y%m%d%H%M")+"_"+\
                        (pd.to_datetime(t2).strftime("%Y%m%d%H%M"))+".nc"               
    if os.path.isdir(out_path):
        pass
    else:
        os.mkdir(out_path)   

    #Save the output
    print("INFO: Computing moisture flux change...")
    F_dqdt_save = F_dqdt.to_netcdf(out_path+F_dqdt_fname,compute=False,engine="netcdf4")
    progress(F_dqdt_save.persist())
    print("INFO: Computing hourly changes...")
    F_hourly_save = F_hourly.to_netcdf(out_path+F_hourly_fname,compute=False,engine="netcdf4")
    progress(F_hourly_save.persist())    
