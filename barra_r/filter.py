import xarray as xr
from sea_breeze import sea_breeze_filters, load_model_data
from dask.distributed import Client
import os
import pandas as pd

def process_time_slice(time_slice,**kwargs):
    ds, df = sea_breeze_filters.filter_sea_breeze(time_slice.squeeze(), **kwargs)
    return ds

if __name__ == "__main__":

    #Set up dask client
    client = Client()

    #Load sea breeze function from sea_breeze_funcs, get time and lat/lon bounds
    field = xr.open_dataset("/g/data/gb02/ab4502/sea_breeze_detection/barra_r/Fc_201601010000_201601312300.nc",chunks="auto").Fc.isel(time=slice(0,5))
    lat_slice = slice(field.lat.min().values,field.lat.max().values)
    lon_slice = slice(field.lon.min().values,field.lon.max().values)
    t1 = field.time.min().values
    t2 = field.time.max().values

    #Load extra data for sea breeze filtering: air temperature, angle of coast, and land sea mask
    angle_ds = load_model_data.get_coastline_angle_kernel(compute=False,path_to_load="/g/data/gb02/ab4502/coastline_data/barra_r.nc",lat_slice=lat_slice,lon_slice=lon_slice)
    ta = load_model_data.load_barra_variable("tas",t1,t2,"AUS-11","1hr",lat_slice,lon_slice)
    _,lsm = load_model_data.load_barra_static("AUS-11",lon_slice,lat_slice)

    #Mask based on percentile values
    thresh = sea_breeze_filters.percentile(field,95)
    mask = sea_breeze_filters.binary_mask(field, thresh)

    #Set up kwargs for filtering
    props_df_out_path = "/scratch/gb02/ab4502/tmp/props_df.csv"
    kwargs = {
                "orientation_filter":True,
                "dist_to_coast_filter":True,
                "land_sea_temperature_filter":False,
                "angle_ds":angle_ds,
                "lsm":lsm.compute(),
                "props_df_output_path":props_df_out_path
            }    

    #We will apply the filtering using map_blocks. So first, need to re-chunk and create a "template" from the first time step
    mask = mask.chunk({"time":1,"lat":-1,"lon":-1})
    template,props_df_template = sea_breeze_filters.filter_sea_breeze(mask.isel(time=0), **kwargs)
    template = template.chunk({"time":1}).reindex({"time":mask.time},fill_value=0).chunk({"time":1})

    #Setup the output dafaframe for object properties    
    def initialise_props_df_output(props_df_out_path,props_df_template):
        os.remove(props_df_out_path)
        pd.DataFrame(columns=props_df_template.columns).to_csv(props_df_out_path,index=False)
    initialise_props_df_output(props_df_out_path,props_df_template)

    #Apply the filtering
    filtered_mask = mask.map_blocks(
        process_time_slice,
        kwargs=kwargs,
        template=template).compute()
