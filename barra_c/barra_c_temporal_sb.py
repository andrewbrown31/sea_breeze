from dask.distributed import Client
from dask.distributed import progress
import pandas as pd
import os
import argparse
import metpy.calc as mpcalc

if __name__ == "__main__":

    #Set up argument parser
    parser = argparse.ArgumentParser(
        prog="BARRA-C temporal sea breeze functions",
        description="This program applies temporal sea breeze functions to a chosen period of BARRA-C data, such as the hourly difference in temperature, humidity and wind"
    )
    parser.add_argument("t1",type=str,help="Start time (Y-m-d H:M)")
    parser.add_argument("t2",type=str,help="End time (Y-m-d H:M)")
    parser.add_argument("--model",default="barra_c",type=str,help="Model name to save the output under.")
    parser.add_argument("--lat1",default=-45.7,type=float,help="Start latitude")
    parser.add_argument("--lat2",default=-6.9,type=float,help="End latitude")
    parser.add_argument("--lon1",default=108,type=float,help="Start longitude")
    parser.add_argument("--lon2",default=158.5,type=float,help="End longitude")
    parser.add_argument('--smooth',default=False,action=argparse.BooleanOptionalAction,help="Smooth the data before calculating diagnostics")
    parser.add_argument("--sigma",default=2,type=int,help="Sigma for smoothing")
    args = parser.parse_args()
    args = parser.parse_args()

    #Initiate distributed dask client on the Gadi HPC
    #https://opus.nci.org.au/spaces/DAE/pages/155746540/Set+up+a+Dask+Cluster
    #https://distributed.dask.org/en/latest/plugins.html#nanny-plugins
    client = Client(scheduler_file=os.environ["DASK_PBS_SCHEDULER"])
    from distributed.diagnostics.plugin import UploadDirectory
    client.register_plugin(UploadDirectory(
        "/home/548/ab4502/working/sea_breeze")
        )
    #client.upload_file("sea_breeze_funcs.py")
    #client.upload_file("load_model_data.py")
    from sea_breeze import load_model_data, sea_breeze_funcs
    #client = Client()

    #Set the domain bounds
    lat_slice=slice(args.lat1,args.lat2)
    lon_slice=slice(args.lon1,args.lon2)    

    #Set time slice and model name    
    t1 = args.t1
    t2 = args.t2    

    #Set up smoothing axes
    if args.smooth:
        smooth_axes = ["lat","lon"]   
    else:
        smooth_axes = None                        

    #Load BARRA-C model level winds, BLH and static info
    if args.smooth:
        chunks = {"time":1,"lat":-1,"lon":-1}
    else:
        chunks = {"time":-1,"lat":{},"lon":{}}
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
            chunks=chunks,
            smooth=args.smooth,
            sigma=args.sigma,
            smooth_axes=smooth_axes)
    uas = load_model_data.load_barra_variable(
            "uas",
            t1,
            t2,
            "AUST-04",
            "1hr",
            lat_slice,
            lon_slice,
            chunks=chunks,
            smooth=args.smooth,
            sigma=args.sigma,
            smooth_axes=smooth_axes)
    huss = load_model_data.load_barra_variable(
            "huss",
            t1,
            t2,
            "AUST-04",
            "1hr",
            lat_slice,
            lon_slice,
            chunks=chunks,
            smooth=args.smooth,
            sigma=args.sigma,
            smooth_axes=smooth_axes)
    tas = load_model_data.load_barra_variable(
            "tas",
            t1,
            t2,
            "AUST-04",
            "1hr",
            lat_slice,
            lon_slice,
            chunks=chunks,
            smooth=args.smooth,
            sigma=args.sigma,
            smooth_axes=smooth_axes)     
    angle_ds = load_model_data.get_coastline_angle_kernel(
        lsm,
        compute=False,
        lat_slice=lat_slice,
        lon_slice=lon_slice,
        path_to_load="/g/data/ng72/ab4502/coastline_data/barra_c.nc",
        smooth=args.smooth,
        sigma=args.sigma)

    #Calc hourly change conditions
    F_hourly = sea_breeze_funcs.hourly_change(
        huss,
        tas,
        uas,
        vas,
        angle_ds["angle_interp"],
        lat_chunk=100,
        lon_chunk=100
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
