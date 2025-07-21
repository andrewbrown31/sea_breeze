import numpy as np
import metpy.calc as mpcalc
import xarray as xr
#import xesmf as xe
import skimage
import glob
from sea_breeze import load_model_data, sea_breeze_funcs
import os

def load_diagnostics(field,model):

    """
    Load the sea breeze diagnostics from the specified model and field.
    Args:
        field (str): The field to load. Options are "F", "Fc", "sbi", "fuzzy", "F_hourly"
        model (str): The model to load. Options are "era5", "barra_r", "barra_c_smooth_s2", "aus2200_smooth_s4".
    Returns:
        xr.DataArray: The loaded data.
    """

    path = "/g/data/ng72/ab4502/sea_breeze_detection"

    # Construct the file path and open the dataset. If the field is "fuzzy", use there should be only one file
    if field == "fuzzy":
        ds = xr.open_dataset(f"{path}/{model}/fuzzy_201301010000_201802282300.zarr",engine="zarr",chunks={})["__xarray_dataarray_variable__"]
    else:
        #If the field is not "fuzzy", we need to open multiple files. Get the file names using glob
        # and open them using xarray
        if "aus2200" in model:
            fn1 = glob.glob(f"{path}/{model}/{field}_mjo-neutral2013_20130101??00_201301312300.zarr")
            fn2 = glob.glob(f"{path}/{model}/{field}_mjo-neutral2013_20130201??00_201302282300.zarr")
            fn3 = glob.glob(f"{path}/{model}/{field}_mjo-elnino2016_20160101??00_201601312300.zarr")
            fn4 = glob.glob(f"{path}/{model}/{field}_mjo-elnino2016_20160201??00_201602292300.zarr")
            fn5 = glob.glob(f"{path}/{model}/{field}_mjo-lanina2018_20180101??00_201801312300.zarr")
            fn6 = glob.glob(f"{path}/{model}/{field}_mjo-lanina2018_20180201??00_201802282300.zarr")
            fn_list = fn1+fn2+fn3+fn4+fn5+fn6
        else:
            fn1 = f"{path}/{model}/{field}_201301010000_201301312300.zarr"
            fn2 = f"{path}/{model}/{field}_201302010000_201302282300.zarr"
            fn3 = f"{path}/{model}/{field}_201601010000_201601312300.zarr"
            fn4 = f"{path}/{model}/{field}_201602010000_201602292300.zarr"
            fn5 = f"{path}/{model}/{field}_201801010000_201801312300.zarr"
            fn6 = f"{path}/{model}/{field}_201802010000_201802282300.zarr"            
            fn_list = [fn1]+[fn2]+[fn3]+[fn4]+[fn5]+[fn6]
            exist = [os.path.exists(fn) for fn in fn_list]
            fn_list = np.array(fn_list)[exist]

        
        ds = xr.open_mfdataset(
            fn_list,
            engine="zarr")[field]

    return ds   

def metpy_grid_area(lon,lat):
    """
    From a grid of latitudes and longitudes, calculate the grid spacing in x and y, and the area of each grid cell in km^2
    """
    xx,yy=np.meshgrid(lon,lat)
    dx,dy=mpcalc.lat_lon_grid_deltas(xx, yy)
    dx=np.pad(dx,((0,0),(0,1)),mode="edge")
    dy=np.pad(dy,((0,1),(0,0)),mode="edge")
    return dx.to("km"),dy.to("km"),(dx*dy).to("km^2")

def get_aus_bounds():
    """
    For Australia
    """
    lat_slice = slice(-45.7,-6.9)
    lon_slice = slice(108,158.5)
    return lat_slice, lon_slice

def get_seaus_bounds():
    """
    For southeast Australia
    """
    lat_slice=slice(-45,-30)
    lon_slice=slice(140,155)
    return lat_slice, lon_slice

def get_perth_bounds():
    """
    From rid 70
    """
    lat_slice = slice(-33.7406830440922, -31.0427169559078)
    lon_slice = slice(114.269565254344, 117.464434745656)
    return lat_slice, lon_slice

def get_perth_large_bounds():
    lat_slice=slice(-38,-30)
    lon_slice=slice(112,120)
    return lat_slice, lon_slice

def get_darwin_bounds():
    """
    From rid 63
    """
    lat_slice = slice(-13.8059830440922, -11.1080169559078)
    lon_slice = slice(129.543506224276, 132.306493775724)
    return lat_slice, lon_slice   

def get_darwin_large_bounds():
    """
    From rid 63
    """
    lat_slice = slice(-17, -9)
    lon_slice = slice(127, 135)
    return lat_slice, lon_slice   

def get_weipa_bounds():
    """
    From rid 63
    """
    lat_slice = slice(-13.8059830440922, -11.1080169559078)
    lon_slice = slice(129.543506224276, 132.306493775724)
    return lat_slice, lon_slice   

def get_gippsland_bounds():
    lat_slice = slice(-39.5, -36.5)
    lon_slice = slice(146, 149)
    return lat_slice, lon_slice 

# def regrid(da,new_lon,new_lat):
#     """
#     Regrid a dataarray to a new grid
#     """
    
#     ds_out = xr.Dataset({"lat":new_lat,"lon":new_lon})
#     regridder = xe.Regridder(da,ds_out,"bilinear")
#     dr_out = regridder(da,keep_attrs=True)

#     return dr_out

def binary_closing_time_slice(time_slice,disk_radius=1):
    out_ds = xr.DataArray(skimage.morphology.binary_closing(time_slice.squeeze(), skimage.morphology.disk(disk_radius)),
                          dims=time_slice.squeeze().dims, coords=time_slice.squeeze().coords)
    out_ds = out_ds.expand_dims("time")
    out_ds["time"] = time_slice.time
    return out_ds

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


def local_time(ds):
    lst_da_ls = []
    for h in np.arange(0,24):
        lst = [dt.datetime(2000,1,1,h) + dt.timedelta(hours=l / 180 * 12) for l in ds.lon.values]
        lst = np.array(pd.to_datetime(lst).round("h").hour)
        #lst = np.array(pd.to_datetime(lst))
        lst_arr = np.repeat(lst[np.newaxis,:],ds.lat.shape,axis=0)
        lst_da = xr.DataArray(data=lst_arr,dims=["lat","lon"],coords={"lat":ds.lat,"lon":ds.lon})
        lst_da_ls.append(lst_da)
    
    lst_da = xr.concat(lst_da_ls,dim="hour")
    lst_da = lst_da.assign_coords({"hour":np.arange(0,24)})

    return lst_da

def local_time_grid(ds):

    timedelta_lon = np.array([np.timedelta64(int(l / 180 * 12 * 60 * 60), "s") for l in ds.lon])
    timedelta_lon_time = np.repeat(timedelta_lon[np.newaxis,:],ds.time.shape,axis=0)

    time = ds.time.values
    time_lon = np.repeat(time[:,np.newaxis],ds.lon.shape,axis=1)

    lst = time_lon + timedelta_lon_time

    #round
    lst = (lst + np.timedelta64(30,"m")).astype("datetime64[h]")

    lst_da = xr.DataArray(
        np.repeat(lst[:,np.newaxis,:].astype("datetime64[ns]"),ds.lat.shape[0],axis=1),
        dims=["time","lat","lon"])
    
    hour_da = lst_da.dt.hour

    groups = ds.groupby(hour_da).mean("time")