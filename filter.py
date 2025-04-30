import xarray as xr
from sea_breeze import sea_breeze_filters, load_model_data, sea_breeze_funcs
from dask.distributed import Client
from sea_breeze.utils import load_diagnostics
import pandas as pd
from dask.distributed import Client
import warnings
import argparse

def load_era5_filtering_data(lon_slice,lat_slice,t1,t2,base_path):

    """
    Load the data needed for filtering from ERA5.
    Args:
        lon_slice (slice): The longitude slice to load.
        lat_slice (slice): The latitude slice to load.
        t1 (str): The start time of the data to load.
        t2 (str): The end time of the data to load.
        base_path (str): The base path for the data.
        model (str): The model name.
    Returns:
        angle_ds (xarray.Dataset): The dataset containing the coastline angle.
        hourly_change_ds (xarray.Dataset): The dataset containing the hourly change.
        ta (xarray.DataArray): The dataset containing the temperature.
        uas (xarray.DataArray): The dataset containing the u-component of the wind.
        vas (xarray.DataArray): The dataset containing the v-component of the wind.
        uprime (xarray.DataArray): The rotated u-component of the wind.
        vprime (xarray.DataArray): The rotated v-component of the wind.
        lsm (xarray.DataArray): The land-sea mask.
    """

    angle_ds_path = base_path +\
        "coastline_data/era5.nc"
    angle_ds = load_model_data.get_coastline_angle_kernel(
        compute=False,path_to_load=angle_ds_path,lat_slice=lat_slice,lon_slice=lon_slice
        )
    ta = load_model_data.load_era5_variable(
        ["2t"],t1,t2,lon_slice,lat_slice,chunks={}
        )["2t"]["t2m"].chunk({"time":1,"lat":-1,"lon":-1})
    uas = load_model_data.load_era5_variable(
        ["10u"],t1,t2,lon_slice,lat_slice,chunks={}
        )["10u"]["u10"].chunk({"time":1,"lat":-1,"lon":-1})
    vas = load_model_data.load_era5_variable(
        ["10v"],t1,t2,lon_slice,lat_slice,chunks={}
        )["10v"]["v10"]   .chunk({"time":1,"lat":-1,"lon":-1})
    uprime,vprime = sea_breeze_funcs.rotate_wind(
        uas,
        vas,
        angle_ds["angle_interp"])
    _,lsm,_ = load_model_data.load_era5_static(
        lon_slice,lat_slice,t1,t2
        )

    return angle_ds, ta, uas, vas, uprime, vprime, lsm

def load_barra_r_filtering_data(lon_slice,lat_slice,t1,t2,base_path):

    angle_ds_path = base_path +\
        "coastline_data/barra_r.nc"
    angle_ds = load_model_data.get_coastline_angle_kernel(
        compute=False,path_to_load=angle_ds_path,lat_slice=lat_slice,lon_slice=lon_slice
        )
    ta = load_model_data.load_barra_variable(
        "tas",t1,t2,"AUS-11","1hr",lat_slice,lon_slice,chunks={"time":1,"lat":-1,"lon":-1}
        )
    uas = load_model_data.load_barra_variable(
        "uas",t1,t2,"AUS-11","1hr",lat_slice,lon_slice,chunks={"time":1,"lat":-1,"lon":-1}
        )
    vas = load_model_data.load_barra_variable(
        "vas",t1,t2,"AUS-11","1hr",lat_slice,lon_slice,chunks={"time":1,"lat":-1,"lon":-1}
        )
    uprime, vprime = sea_breeze_funcs.rotate_wind(
        uas,
        vas,
        angle_ds["angle_interp"])
    _,lsm = load_model_data.load_barra_static(
        "AUS-11",lon_slice,lat_slice
        )

    return angle_ds, ta, uas, vas, uprime.drop("height"), vprime.drop("height"), lsm

def load_barra_c_filtering_data(lon_slice,lat_slice,t1,t2,base_path):

    angle_ds_path = base_path +\
        "coastline_data/barra_c.nc"
    angle_ds = load_model_data.get_coastline_angle_kernel(
        compute=False,path_to_load=angle_ds_path,lat_slice=lat_slice,lon_slice=lon_slice
        )
    ta = load_model_data.load_barra_variable(
        "tas",t1,t2,"AUST-04","1hr",lat_slice,lon_slice,chunks={"time":1,"lat":-1,"lon":-1}
        )
    uas = load_model_data.load_barra_variable(
        "uas",t1,t2,"AUST-04","1hr",lat_slice,lon_slice,chunks={"time":1,"lat":-1,"lon":-1}
        )
    vas = load_model_data.load_barra_variable(
        "vas",t1,t2,"AUST-04","1hr",lat_slice,lon_slice,chunks={"time":1,"lat":-1,"lon":-1}
        )
    uprime, vprime = sea_breeze_funcs.rotate_wind(
        uas,
        vas,
        angle_ds["angle_interp"])
    _,lsm = load_model_data.load_barra_static(
        "AUST-04",lon_slice,lat_slice
        )

    return angle_ds, ta, uas, vas, uprime.drop("height"), vprime.drop("height"), lsm

def load_aus2200_filtering_data(lon_slice,lat_slice,t1,t2,base_path,exp_id):

    #Load other datasets that can be used for additional filtering
    angle_ds_path = base_path +\
        "coastline_data/aus2200.nc"
    angle_ds = load_model_data.get_coastline_angle_kernel(
        compute=False,path_to_load=angle_ds_path,lat_slice=lat_slice,lon_slice=lon_slice
        )
    ta = load_model_data.load_aus2200_variable(
        "ta",
        t1,
        t2,
        exp_id,
        lon_slice,
        lat_slice,
        "1hr",
        smooth=False,
        hgt_slice=slice(0,10),
        chunks={"time":1,"lat":-1,"lon":-1}).sel(lev=5)
    vas = load_model_data.round_times(
        load_model_data.load_aus2200_variable(
            "vas",
            t1,
            t2,
            exp_id,
            lon_slice,
            lat_slice,
            "10min",
            chunks={"time":1,"lat":-1,"lon":-1},
            staggered="lat",
            smooth=False),
            "10min")
    uas = load_model_data.round_times(
        load_model_data.load_aus2200_variable(
            "uas",
            t1,
            t2,
            exp_id,
            lon_slice,
            lat_slice,
            "10min",
            chunks={"time":1,"lat":-1,"lon":-1},
            staggered="lon",
            smooth=False),
            "10min")
    vas = vas.sel(time=vas.time.dt.minute==0)
    uas = uas.sel(time=uas.time.dt.minute==0)    
    uprime, vprime = sea_breeze_funcs.rotate_wind(
        uas,
        vas,
        angle_ds["angle_interp"])
    orog, lsm = load_model_data.load_aus2200_static(
        "mjo-elnino2016",
        lon_slice,
        lat_slice)

    return angle_ds, ta, uas, vas, uprime, vprime, lsm

if __name__ == "__main__":

    #Set up argument parser
    parser = argparse.ArgumentParser(
        prog="Filtering",
        description="This program loads a sea breeze diagnostic field from the sea_breeze_funcs module, and applies a series of filters to it"
    )

    #Add arguments
    parser.add_argument("--field_name",required=True,type=str,help="Name of the field to filter")
    parser.add_argument("--t1",required=True,type=str,help="Start time of the field to filter")
    parser.add_argument("--t2",required=True,type=str,help="End time of the field to filter")
    parser.add_argument("--model",default="era5",type=str,help="Model directory name for input/output. Could be era5 (default)")
    parser.add_argument("--filter_name",default="",type=str,help="Filter name to add to the output file names")
    parser.add_argument("--threshold",default="fixed",type=str,help="Threshold to use for the filter. Could be fixed (default) or percentile")
    parser.add_argument("--p",default=99.5,type=float,help="Percentile to use for the filter. Only used if threshold is percentile. Default is 99.5")
    parser.add_argument("--threshold_value",default=0.5,type=float,help="Threshold value to use for the filter. Only used if threshold is fixed. Default is 0.5")
    parser.add_argument("--exp_id",default="None",type=str,help="For AUS2200, need the experiment ID to load the data. Default is None")
    args = parser.parse_args()

    #Set up dask client
    client = Client()

    #Ignore warnings for runtime errors (divide by zero etc)
    warnings.simplefilter("ignore")

    #Set up paths to sea_breeze_funcs data output and other inputs
    model = args.model
    filter_name = args.filter_name
    field_name = args.field_name
    t1 = args.t1
    t2 = args.t2
    threshold = args.threshold
    threshold_value = args.threshold_value
    p = args.p
    exp_id = args.exp_id

    #Set up lat/lon slices
    lat_slice = slice(-45.7,-6.9)
    lon_slice = slice(108,158.5)

    #Set base path
    base_path = "/g/data/ng72/ab4502/"
    
    #Set up filtering options
    kwargs = {
        "orientation_filter":True,
        "aspect_filter":True,
        "area_filter":True,        
        "land_sea_temperature_filter":True,                    
        "temperature_change_filter":False,
        "humidity_change_filter":False,
        "wind_change_filter":False,
        "propagation_speed_filter":False,
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
    # if field_name == "fuzzy":
    #     field = load_diagnostics(field_name,model)["__xarray_dataarray_variable__"]\
    #         .sel(lat=lat_slice,lon=lon_slice,time=slice(t1,t2))\
    #             .chunk({"time":1,"lat":-1,"lon":-1})
    # else:
    #     field = load_diagnostics(field_name,model)[field_name]\
    #         .sel(lat=lat_slice,lon=lon_slice,time=slice(t1,t2))\
    #             .chunk({"time":1,"lat":-1,"lon":-1})
    field = load_diagnostics(field_name,model)\
        .sel(lat=lat_slice,lon=lon_slice,time=slice(t1,t2))\
            .chunk({"time":1,"lat":-1,"lon":-1})

    #Load other datasets that can be used for additional filtering
    if "era5" in model:
        angle_ds, ta, uas, vas, uprime, vprime, lsm = load_era5_filtering_data(
            lon_slice,lat_slice,t1,t2,base_path
            )
    elif "barra_r" in model:
        angle_ds, ta, uas, vas, uprime, vprime, lsm = load_barra_r_filtering_data(
            lon_slice,lat_slice,t1,t2,base_path
            )
    elif "barra_c" in model:
        angle_ds, ta, uas, vas, uprime, vprime, lsm = load_barra_c_filtering_data(
            lon_slice,lat_slice,t1,t2,base_path
            )
    elif "aus2200" in model:
        angle_ds, ta, uas, vas, uprime, vprime, lsm = load_aus2200_filtering_data(
            lon_slice,lat_slice,t1,t2,base_path,exp_id
            )
    
    #Set up output paths
    props_df_out_path = base_path+\
        "sea_breeze_detection/"+model+"/props_df/props_df_"+filter_name+"_"+\
            field_name+"_"+\
                pd.to_datetime(t1).strftime("%Y%m%d%H%M")+"_"+\
                    pd.to_datetime(t2).strftime("%Y%m%d%H%M")+".csv" 
    filter_out_path = base_path+\
        "sea_breeze_detection/"+model+"/filters/filtered_mask_"+filter_name+"_"+\
            field_name+"_"+\
                pd.to_datetime(t1).strftime("%Y%m%d%H%M")+"_"+\
                    pd.to_datetime(t2).strftime("%Y%m%d%H%M")+".zarr"

    #Run the filtering
    filtered_mask = sea_breeze_filters.filter_3d(
        field,
        hourly_change_ds=None,
        ta=ta,
        lsm=lsm,
        angle_ds=angle_ds,
        vprime=vprime,
        threshold=threshold,
        threshold_value=threshold_value,
        p=99.5,
        save_mask=True,
        filter_out_path=filter_out_path,
        props_df_out_path=props_df_out_path,
        skipna=False,
        **kwargs)

    #Close client
    client.close()