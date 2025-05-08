from sea_breeze import load_model_data, sea_breeze_funcs, utils
from dask.distributed import Client
from dask.distributed import progress
import pandas as pd
import os


if __name__ == "__main__":

    #Initiate distributed dask client on the Gadi HPC
    client = Client()

    #Set the domain bounds
    lat_slice, lon_slice = utils.get_perth_large_bounds()

    #Set time slice and model name    
    t1 = "2016-01-06 00:00:00"
    t2 = "2016-01-10 00:00:00"
    exp_id = "mjo-elnino2016"
    model = "aus2200_test"

    #Set SBI settings and chunk settings
    smooth=True
    sigma=4
    time_chunk = 1
    lon_chunk = -1 
    lat_chunk = -1
    smooth_axes = ["lat","lon"]   

    #Load AUS2200 model level winds, BLH and static info
    chunks = {"time":time_chunk,"lat":lat_chunk,"lon":lon_chunk}
    orog, lsm = load_model_data.load_aus2200_static(
        exp_id,
        lon_slice,
        lat_slice)
    aus2200_vas = load_model_data.round_times(
        load_model_data.load_aus2200_variable(
            "vas",
            t1,
            t2,
            exp_id,
            lon_slice,
            lat_slice,
            "10min",
            chunks=chunks,
            staggered="lat",
            smooth=smooth,
            sigma=sigma,
            smooth_axes=smooth_axes),
              "10min")
    aus2200_uas = load_model_data.round_times(
        load_model_data.load_aus2200_variable(
            "uas",
            t1,
            t2,
            exp_id,
            lon_slice,
            lat_slice,
            "10min",
            chunks=chunks,
            staggered="lon",
            smooth=smooth,
            sigma=sigma,
            smooth_axes=smooth_axes),
              "10min")
    aus2200_hus = load_model_data.load_aus2200_variable(
        "hus",
        t1,
        t2,
        exp_id,
        lon_slice,
        lat_slice,
        "1hr",
        smooth=smooth,
        sigma=sigma,
        smooth_axes=smooth_axes,
        hgt_slice=slice(0,10),
        chunks=chunks).sel(lev=5)
    angle_ds = load_model_data.get_coastline_angle_kernel(
        lsm,
        compute=False,
        lat_slice=lat_slice,
        lon_slice=lon_slice,
        path_to_load="/g/data/ng72/ab4502/coastline_data/aus2200.nc",
        smooth=smooth,
        sigma=sigma)
    
    #Just do the hourly data
    aus2200_vas = aus2200_vas.sel(time=aus2200_vas.time.dt.minute==0)
    aus2200_uas = aus2200_uas.sel(time=aus2200_uas.time.dt.minute==0)    

    #Calc 2d kinematic moisture frontogenesis
    F = sea_breeze_funcs.kinematic_frontogenesis(
        aus2200_hus,
        aus2200_uas,
        aus2200_vas
    )

    #Setup out paths
    out_path = "/g/data/ng72/ab4502/sea_breeze_detection/"+model+"/"
    F_fname = "F_"+exp_id+"_"+pd.to_datetime(t1).strftime("%Y%m%d%H%M")+"_"+\
                    (pd.to_datetime(t2).strftime("%Y%m%d%H%M"))+".zarr"   

    if os.path.isdir(out_path):
        pass
    else:
        os.mkdir(out_path)   

    #Add smoothing attrs
    F = F.assign_attrs(
        smooth=smooth,
        sigma=sigma,
    )

    #Save the output
    print("INFO: Computing frontogenesis...")
    F_save = F.to_zarr(out_path+F_fname,compute=False,mode="w")
    progress(F_save.persist())

    #Close the dask client
    client.close()
    print("INFO: Finished")