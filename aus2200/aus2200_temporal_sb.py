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
        prog="AUS2200 frontogenesis",
        description="This program applies temporal sea breeze functions to a chosen period of AUS2200 data, such as the hourly difference in temperature, humidity and wind"
    )
    parser.add_argument("t1",type=str,help="Start time (Y-m-d H:M)")
    parser.add_argument("t2",type=str,help="End time (Y-m-d H:M)")
    parser.add_argument("--model",default="aus2200",type=str,help="Model name to save the output under. Could be aus2200 (default) or maybe aus2200_pert")
    parser.add_argument("--lat1",default=-45.7,type=float,help="Start latitude")
    parser.add_argument("--lat2",default=-6.9,type=float,help="End latitude")
    parser.add_argument("--lon1",default=108,type=float,help="Start longitude")
    parser.add_argument("--lon2",default=158.5,type=float,help="End longitude")
    parser.add_argument("-e","--exp_id",default="mjo-elnino",type=str,help="Experiment id for AUS2200 mjo runs") 
    parser.add_argument('--smooth',default=False,action=argparse.BooleanOptionalAction,help="Smooth the data before calculating diagnostics")
    parser.add_argument("--sigma",default=2,type=int,help="Sigma for smoothing")
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

    #Setup out paths
    out_path = "/g/data/gb02/ab4502/sea_breeze_detection/"+args.model+"/"
    F_hourly_fname = "F_hourly_"+exp_id+"_"+pd.to_datetime(t1).strftime("%Y%m%d%H%M")+"_"+\
                        (pd.to_datetime(t2).strftime("%Y%m%d%H%M"))+".zarr"               
    if os.path.isdir(out_path):
        pass
    else:
        os.mkdir(out_path)   

    #Set up smoothing axes
    if args.smooth:
        smooth_axes = ["lat","lon"]   
    else:
        smooth_axes = None                                    

    #Load AUS2200 model level winds, BLH and static info
    if args.smooth:
        chunks = {"time":1,"lat":-1,"lon":-1}
    else:
        chunks = {"time":-1,"lat":{},"lon":{}}
    orog, lsm = load_model_data.load_aus2200_static(
        "mjo-elnino",
        lon_slice,
        lat_slice)
    aus2200_vas = load_model_data.round_times(
        load_model_data.load_aus2200_variable(
            "vas",
            t1,
            t2,
            "mjo-elnino",
            lon_slice,
            lat_slice,
            "10min",
            chunks=chunks,
            staggered="lat",
            smooth=args.smooth,
            sigma=args.sigma,
            smooth_axes=smooth_axes),
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
            chunks=chunks,
            staggered="lon",
            smooth=args.smooth,
            sigma=args.sigma,
            smooth_axes=smooth_axes),
              "10min")
    aus2200_hus = load_model_data.round_times(
        load_model_data.load_aus2200_variable(
            "hus",
            t1,
            t2,
            "mjo-elnino",
            lon_slice,
            lat_slice,
            "10min",
            chunks=chunks,
            smooth=args.smooth,
            sigma=args.sigma,
            smooth_axes=smooth_axes),
              "10min")
    aus2200_tas = load_model_data.round_times(
            load_model_data.load_aus2200_variable(
                "tas",
                t1,
                t2,
                "mjo-elnino",
                lon_slice,
                lat_slice,
                "10min",
                chunks=chunks,
                smooth=args.smooth,
                sigma=args.sigma,
                smooth_axes=smooth_axes),
                "10min")    
    angle_ds = load_model_data.get_coastline_angle_kernel(
        lsm,
        compute=False,
        lat_slice=lat_slice,
        lon_slice=lon_slice,
        path_to_load="/g/data/gb02/ab4502/coastline_data/aus2200.nc",
        smooth=args.smooth,
        sigma=args.sigma)

    #Just do the hourly data
    aus2200_vas = aus2200_vas.sel(time=aus2200_vas.time.dt.minute==0)
    aus2200_uas = aus2200_uas.sel(time=aus2200_uas.time.dt.minute==0)
    aus2200_hus = aus2200_hus.sel(time=aus2200_hus.time.dt.minute==0)
    aus2200_tas = aus2200_tas.sel(time=aus2200_tas.time.dt.minute==0)

    #Rechunk
    # aus2200_hus = aus2200_hus.chunk({"time":-1,"lat":lat_chunk,"lon":lon_chunk})
    # aus2200_uas = aus2200_uas.chunk({"time":-1,"lat":lat_chunk,"lon":lon_chunk})
    # aus2200_vas = aus2200_vas.chunk({"time":-1,"lat":lat_chunk,"lon":lon_chunk})
    # aus2200_tas = aus2200_tas.chunk({"time":-1,"lat":lat_chunk,"lon":lon_chunk})

    #Calc hourly change conditions
    F_hourly = sea_breeze_funcs.hourly_change(
        aus2200_hus,
        aus2200_tas,
        aus2200_uas,
        aus2200_vas,
        angle_ds["angle_interp"],
        lat_chunk=200,
        lon_chunk=200
    )    
    print("INFO: Computing hourly changes...")
    progress(F_hourly)

    F_hourly = F_hourly.assign_attrs(
        smooth=args.smooth,
        sigma=args.sigma,
    )

    F_hourly_save = F_hourly.to_zarr(out_path+F_hourly_fname,compute=False,mode="w")
    progress(F_hourly_save.persist())    
    