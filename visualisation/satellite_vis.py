import xarray as xr
import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dask.distributed import Client
import xmovie
from sea_breeze.load_obs import load_hima, get_lat_lons
from sea_breeze import utils

def plot_hima(ds,fig,tt,framedim,proj):

    ax = plt.axes(projection=proj)
    c = ds.isel({framedim:tt}).plot(cmap="Greys_r",vmin=-0.2,vmax=1.5,ax=ax,add_colorbar=False)
    ax.set_title(ds.isel(time=tt).time.values)
    fig.colorbar(c,extend="both")
    ax.coastlines()
    ax.gridlines(draw_labels=["bottom","left"],ls=":")
    plt.close(fig)
    
    return None, None 

if __name__ == "__main__":

    client = Client()

    #Settings for amination
    domain_name = "perth"
    t1 = "2016-1-06 21:30"
    t2 = "2016-1-12 11:00"
    step = "30min"
    lat_slice,lon_slice = utils.get_perth_large_bounds()

    #Load a single time step of himawari data for approximately the whole Australia region
    ds,proj,ccrs_proj = load_hima(dt.datetime(2016,1,10,4))

    #From the himawari dataset, return lat lons using the accompanying projection. Note that the lat lons aren't evenly spaced
    lons, lats = get_lat_lons(ds, proj)

    #Slice the x and y coordinates from lat lon bounds
    min_lat = lat_slice.start
    max_lat = lat_slice.stop
    min_lon = lon_slice.start
    max_lon = lon_slice.stop
    y_ind, x_ind = np.where((lons >= min_lon) & (lons <= max_lon) & (lats <= max_lat) & (lats >= min_lat))
    ymin, ymax = (ds["y"][y_ind].min().values, ds["y"][y_ind].max().values)
    xmin, xmax = (ds["x"][x_ind].min().values, ds["x"][x_ind].max().values)

    #Now load a series of times, for the smaller region of interest defined above
    ds_list = []
    for t in pd.date_range(t1,t2,freq=step):
        print(t)
        temp_ds,proj,ccrs_proj = load_hima(t,x_slice = slice(xmin, xmax),y_slice = slice(ymax,ymin))
        ds_list.append(temp_ds)
    ds_multiple_times = xr.concat(ds_list,"time")

    #Create movie
    mov = xmovie.Movie(
    ds_multiple_times.channel_0003_corrected_reflectance,
    pixelwidth=1200,
    pixelheight=1200,
    plotfunc=plot_hima,
    proj=ccrs_proj,
    input_check=False,
)
    
    #Save movie
    mov.save(
        "/scratch/ng72/ab4502/temp_figs/sat_"+domain_name+"_"+t1+"_"+t2+".mp4",
        overwrite_existing=True,
        parallel=True,
        progress=True,
        framerate=5)  