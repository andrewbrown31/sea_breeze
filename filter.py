from dask.distributed import Client
import pandas as pd
import warnings
import argparse
import os

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

    parser.add_argument("--orientation_filter",default=True,action=argparse.BooleanOptionalAction,help="Whether to apply the orientation filter. Default is True")
    parser.add_argument("--aspect_filter",default=True,action=argparse.BooleanOptionalAction,help="Whether to apply the aspect filter. Default is True")
    parser.add_argument("--area_filter",default=True,action=argparse.BooleanOptionalAction,help="Whether to apply the area filter. Default is True")
    parser.add_argument("--land_sea_temperature_filter",default=True,action=argparse.BooleanOptionalAction,help="Whether to apply the land sea temperature filter. Default is True")
    parser.add_argument("--temperature_change_filter",default=False,action=argparse.BooleanOptionalAction,help="Whether to apply the temperature change filter. Default is False")
    parser.add_argument("--humidity_change_filter",default=False,action=argparse.BooleanOptionalAction,help="Whether to apply the humidity change filter. Default is False")
    parser.add_argument("--wind_change_filter",default=False,action=argparse.BooleanOptionalAction,help="Whether to apply the wind change filter. Default is False")
    parser.add_argument("--propagation_speed_filter",default=True,action=argparse.BooleanOptionalAction,help="Whether to apply the propagation speed filter. Default is False")
    parser.add_argument("--dist_to_coast_filter",default=False,action=argparse.BooleanOptionalAction,help="Whether to apply the distance to coast filter. Default is False")
    parser.add_argument("--output_land_sea_temperature_diff",default=False,action=argparse.BooleanOptionalAction,help="Whether to output the land sea temperature difference. Default is False")
    parser.add_argument("--time_filter",default=False,action=argparse.BooleanOptionalAction,help="Whether to apply the time filter. Default is False")
    parser.add_argument("--orientation_tol",default=45,type=float,help="Orientation tolerance for the orientation filter. Default is 45")
    parser.add_argument("--area_thresh_pixels",default=12,type=int,help="Area threshold in pixels for the area filter. Default is 12")
    parser.add_argument("--aspect_thresh",default=2,type=float,help="Aspect threshold for the aspect filter. Default is 2")
    parser.add_argument("--land_sea_temperature_diff_thresh",default=0,type=float,help="Land sea temperature difference threshold for the land sea temperature filter. Default is 0")
    parser.add_argument("--propagation_speed_thresh",default=0,type=float,help="Propagation speed threshold for the propagation speed filter. Default is 0")
    args = parser.parse_args()

    #Set up dask client
    #client = Client()
    #https://opus.nci.org.au/spaces/DAE/pages/155746540/Set+up+a+Dask+Cluster
    #https://distributed.dask.org/en/latest/plugins.html#nanny-plugins
    client = Client(scheduler_file=os.environ["DASK_PBS_SCHEDULER"])
    from distributed.diagnostics.plugin import UploadDirectory
    client.register_plugin(UploadDirectory(
        "/home/548/ab4502/working/sea_breeze")
        )
    from sea_breeze import sea_breeze_filters, utils

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
    orientation_filter = args.orientation_filter
    aspect_filter = args.aspect_filter
    area_filter = args.area_filter
    land_sea_temperature_filter = args.land_sea_temperature_filter
    temperature_change_filter = args.temperature_change_filter
    humidity_change_filter = args.humidity_change_filter
    wind_change_filter = args.wind_change_filter
    propagation_speed_filter = args.propagation_speed_filter
    dist_to_coast_filter = args.dist_to_coast_filter
    output_land_sea_temperature_diff = args.output_land_sea_temperature_diff
    time_filter = args.time_filter
    orientation_tol = args.orientation_tol
    area_thresh_pixels = args.area_thresh_pixels
    aspect_thresh = args.aspect_thresh
    land_sea_temperature_diff_thresh = args.land_sea_temperature_diff_thresh
    propagation_speed_thresh = args.propagation_speed_thresh

    #Set up lat/lon slices
    lat_slice = slice(-45.7,-6.9)
    lon_slice = slice(108,158.5)

    #Set base path
    base_path = "/g/data/ng72/ab4502/"
    
    #Set up filtering options
    kwargs = {
        "orientation_filter":orientation_filter,
        "aspect_filter":aspect_filter,
        "area_filter":area_filter,        
        "land_sea_temperature_filter":land_sea_temperature_filter,                    
        "temperature_change_filter":temperature_change_filter,
        "humidity_change_filter":humidity_change_filter,
        "wind_change_filter":wind_change_filter,
        "propagation_speed_filter":propagation_speed_filter,
        "dist_to_coast_filter":dist_to_coast_filter,
        "output_land_sea_temperature_diff":output_land_sea_temperature_diff,        
        "time_filter":time_filter,
        "orientation_tol":orientation_tol,
        "area_thresh_pixels":area_thresh_pixels,
        "aspect_thresh":aspect_thresh,
        "land_sea_temperature_diff_thresh":land_sea_temperature_diff_thresh,
        "propagation_speed_thresh":propagation_speed_thresh,
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
    field = utils.load_diagnostics(field_name,model)\
        .sel(lat=lat_slice,lon=lon_slice,time=slice(t1,t2))\
            .chunk({"time":1,"lat":-1,"lon":-1})

    #Load other datasets that can be used for additional filtering
    if "era5" in model:
        angle_ds, ta, uas, vas, uprime, vprime, lsm = utils.load_era5_filtering_data(
            lon_slice,lat_slice,t1,t2,base_path
            )
    elif "barra_r" in model:
        angle_ds, ta, uas, vas, uprime, vprime, lsm = utils.load_barra_r_filtering_data(
            lon_slice,lat_slice,t1,t2,base_path
            )
    elif "barra_c" in model:
        angle_ds, ta, uas, vas, uprime, vprime, lsm = utils.load_barra_c_filtering_data(
            lon_slice,lat_slice,t1,t2,base_path
            )
    elif "aus2200" in model:
        angle_ds, ta, uas, vas, uprime, vprime, lsm = utils.load_aus2200_filtering_data(
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