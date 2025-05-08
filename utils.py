import numpy as np
import metpy.calc as mpcalc
import xarray as xr
import xesmf as xe
import skimage
import glob

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
            fn1 = glob.glob(f"{path}/{model}/{field}_mjo-neutral2013_20130101??00_201301312300.zarr")[0]
            fn2 = glob.glob(f"{path}/{model}/{field}_mjo-neutral2013_20130201??00_201302282300.zarr")[0]
            fn3 = glob.glob(f"{path}/{model}/{field}_mjo-elnino2016_20160101??00_201601312300.zarr")[0]
            fn4 = glob.glob(f"{path}/{model}/{field}_mjo-elnino2016_20160201??00_201602292300.zarr")[0]
            fn5 = glob.glob(f"{path}/{model}/{field}_mjo-lanina2018_20180101??00_201801312300.zarr")[0]
            fn6 = glob.glob(f"{path}/{model}/{field}_mjo-lanina2018_20180201??00_201802282300.zarr")[0]
        else:
            fn1 = f"{path}/{model}/{field}_201301010000_201301312300.zarr"
            fn2 = f"{path}/{model}/{field}_201302010000_201302282300.zarr"
            fn3 = f"{path}/{model}/{field}_201601010000_201601312300.zarr"
            fn4 = f"{path}/{model}/{field}_201602010000_201602292300.zarr"
            fn5 = f"{path}/{model}/{field}_201801010000_201801312300.zarr"
            fn6 = f"{path}/{model}/{field}_201802010000_201802282300.zarr"
        
        ds = xr.open_mfdataset(
            [fn1,fn2,fn3,fn4,fn5,fn6],
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

def regrid(da,new_lon,new_lat):
    """
    Regrid a dataarray to a new grid
    """
    
    ds_out = xr.Dataset({"lat":new_lat,"lon":new_lon})
    regridder = xe.Regridder(da,ds_out,"bilinear")
    dr_out = regridder(da,keep_attrs=True)

    return dr_out

def binary_closing_time_slice(time_slice,disk_radius=1):
    out_ds = xr.DataArray(skimage.morphology.binary_closing(time_slice.squeeze(), skimage.morphology.disk(disk_radius)),
                          dims=time_slice.squeeze().dims, coords=time_slice.squeeze().coords)
    out_ds = out_ds.expand_dims("time")
    out_ds["time"] = time_slice.time
    return out_ds