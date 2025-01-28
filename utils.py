import numpy as np
import metpy.calc as mpcalc
import xarray as xr
import xesmf as xe

def metpy_grid_area(lon,lat):
    """
    From a grid of latitudes and longitudes, calculate the grid spacing in x and y, and the area of each grid cell in km^2
    """
    xx,yy=np.meshgrid(lon,lat)
    dx,dy=mpcalc.lat_lon_grid_deltas(xx, yy)
    dx=np.pad(dx,((0,0),(0,1)),mode="edge")
    dy=np.pad(dy,((0,1),(0,0)),mode="edge")
    return dx.to("km"),dy.to("km"),(dx*dy).to("km^2")

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

def regrid(da,new_lon,new_lat):
    """
    Regrid a dataarray to a new grid
    """
    
    ds_out = xr.Dataset({"lat":new_lat,"lon":new_lon})
    regridder = xe.Regridder(da,ds_out,"bilinear")
    dr_out = regridder(da,keep_attrs=True)

    return dr_out