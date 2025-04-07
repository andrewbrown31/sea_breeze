import xarray as xr
from sys import stdout
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
import cartopy.crs as ccrs
import pandas as pd
import matplotlib.animation as animation
from sea_breeze import sea_breeze_funcs, utils, load_obs
from dask.distributed import Client
import xmovie
import os
import glob

def plot_pcolormesh(da,vmin=None,vmax=None,save=True,fname="temp",cmap="viridis"):
    plt.figure(figsize=(10,5))
    ax = plt.axes(projection=ccrs.PlateCarree())
    da.plot.pcolormesh(ax=ax,transform=ccrs.PlateCarree(),vmin=vmin,vmax=vmax)
    ax.coastlines()
    if save:
        plt.savefig("/scratch/ng72/ab4502/temp_figs/"+fname+".png",bbox_inches="tight",dpi=300)

def animate_pcolormesh(datasets, figsize=(20,5), rows=None, cols=None, vmins=None, vmaxs=None, fname="temp", cmaps=None, titles=None):
    N = len(datasets)
    if (rows == None) | (cols == None):
        cols = N
        rows = 1
    fig, axes = plt.subplots(rows, cols, figsize=(20, 5), subplot_kw={'projection': ccrs.PlateCarree()})
    
    if vmins is None:
        vmins = [None] * N
    if vmaxs is None:
        vmaxs = [None] * N
    if cmaps is None:
        cmaps = ["viridis"] * N
    if titles is None:
        titles = [""] * N
    
    meshes = []
    for i, (da, ax, vmin, vmax, cmap, title) in enumerate(zip(datasets, axes, vmins, vmaxs, cmaps,titles)):
        mesh = da.isel(time=0).plot.pcolormesh(ax=ax, transform=ccrs.PlateCarree(), add_colorbar=False, vmin=vmin, vmax=vmax, cmap=cmap)
        plt.colorbar(mesh, ax=ax, orientation='vertical', pad=0.02, aspect=50)
        ax.coastlines()
        ax.set_title(title + "\n" + str(da.isel(time=0).time.values))
        meshes.append(mesh)

    def update(frame):
        stdout.write("\rPlotting frame number %d/%d" % (frame, (datasets[0].time.shape[0] - 1)))
        stdout.flush()
        for i, (da, ax, vmin, vmax, cmap, title) in enumerate(zip(datasets, axes, vmins, vmaxs, cmaps, titles)):
            ax.clear()
            da.isel(time=frame).plot.pcolormesh(ax=ax, transform=ccrs.PlateCarree(), add_colorbar=False, vmin=vmin, vmax=vmax, cmap=cmap)
            ax.coastlines()
            ax.set_title(title + "\n" + str(da.isel(time=frame).time.values))

    ani = animation.FuncAnimation(fig, update, frames=len(datasets[0].time), repeat=False)
    ani.save('/g/data/ng72/ab4502/figs/animations/' + fname + '.mp4', writer='ffmpeg', dpi=300)
    #plt.close()

def plotmasks_1fields(ds,fig,tt,framedim,vmins,vmaxs,field_titles,cmap,**kwargs):

    ax1, ax2, ax3 = fig.subplots(1,3,subplot_kw={"projection":ccrs.PlateCarree()}).flatten()
    ds["field1"].isel({framedim:tt}).plot(ax=ax1,vmin=vmins[0],vmax=vmaxs[0],transform=ccrs.PlateCarree(),extend="both",cmap=cmap)
    ds["mask1"].isel({framedim:tt}).plot(ax=ax2,vmin=0,vmax=1,transform=ccrs.PlateCarree(),cmap="Blues")
    ds["filtered_mask1"].isel({framedim:tt}).plot(ax=ax3,vmin=0,vmax=1,transform=ccrs.PlateCarree(),cmap="Blues")

    for ax in [ax1, ax2, ax3]:
        ax.coastlines()
        xr.plot.contourf(
            ds["variance_interp"],ax=ax,levels=[0.,0.5],hatches=["","/////"],colors="none",add_colorbar=False)
    
    for ax,i in zip([ax1],[0,1,2]):
        ax.set_title(field_titles[i])
    for ax in [ax2]:
        ax.set_title("SB mask")
    for ax in [ax3]:        
        ax.set_title("SB filter")

    fig.suptitle(ds.isel(time=tt).time.values)
    fig.subplots_adjust(wspace=0.3)

    plt.close(fig)
    
    return None, None

def plotmasks_2fields(ds,fig,tt,framedim,vmins,vmaxs,field_titles,**kwargs):

    ax1, ax2, ax3, ax4, ax5, ax6 = fig.subplots(2,3,subplot_kw={"projection":ccrs.PlateCarree()}).flatten()

    ds["field1"].isel({framedim:tt}).plot(ax=ax1,vmin=vmins[0],vmax=vmaxs[0],transform=ccrs.PlateCarree(),extend="both")
    ds["mask1"].isel({framedim:tt}).plot(ax=ax2,vmin=0,vmax=1,transform=ccrs.PlateCarree(),cmap="Blues")
    ds["filtered_mask1"].isel({framedim:tt}).plot(ax=ax3,vmin=0,vmax=1,transform=ccrs.PlateCarree(),cmap="Blues")

    ds["field2"].isel({framedim:tt}).plot(ax=ax4,vmin=vmins[1],vmax=vmaxs[1],transform=ccrs.PlateCarree(),extend="both")
    ds["mask2"].isel({framedim:tt}).plot(ax=ax5,vmin=0,vmax=1,transform=ccrs.PlateCarree(),cmap="Blues")
    ds["filtered_mask2"].isel({framedim:tt}).plot(ax=ax6,vmin=0,vmax=1,transform=ccrs.PlateCarree(),cmap="Blues")

    for ax in [ax1, ax2, ax3, ax4, ax5, ax6]:
        ax.coastlines()
        xr.plot.contourf(
            ds["variance_interp"],ax=ax,levels=[0.,0.5],hatches=["","/////"],colors="none",add_colorbar=False)
    
    for ax,i in zip([ax1, ax4],[0,1,2]):
        ax.set_title(field_titles[i])
    for ax in [ax2, ax5]:
        ax.set_title("SB mask")
    for ax in [ax3, ax6]:        
        ax.set_title("SB filter")

    fig.suptitle(ds.isel(time=tt).time.values)
    fig.subplots_adjust(wspace=0.3)

    plt.close(fig)
    
    return None, None

def plotmasks_3fields(ds,fig,tt,framedim,vmins,vmaxs,field_titles,**kwargs):

    ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9 = fig.subplots(3,3,subplot_kw={"projection":ccrs.PlateCarree()}).flatten()
    ds["field1"].isel({framedim:tt}).plot(ax=ax1,vmin=vmins[0],vmax=vmaxs[0],transform=ccrs.PlateCarree(),extend="both")
    ds["mask1"].isel({framedim:tt}).plot(ax=ax2,vmin=0,vmax=1,transform=ccrs.PlateCarree(),cmap="Blues")
    ds["filtered_mask1"].isel({framedim:tt}).plot(ax=ax3,vmin=0,vmax=1,transform=ccrs.PlateCarree(),cmap="Blues")

    ds["field2"].isel({framedim:tt}).plot(ax=ax4,vmin=vmins[1],vmax=vmaxs[1],transform=ccrs.PlateCarree(),extend="both")
    ds["mask2"].isel({framedim:tt}).plot(ax=ax5,vmin=0,vmax=1,transform=ccrs.PlateCarree(),cmap="Blues")
    ds["filtered_mask2"].isel({framedim:tt}).plot(ax=ax6,vmin=0,vmax=1,transform=ccrs.PlateCarree(),cmap="Blues")

    ds["field3"].isel({framedim:tt}).plot(ax=ax7,vmin=vmins[2],vmax=vmaxs[2],transform=ccrs.PlateCarree(),extend="both")
    ds["mask3"].isel({framedim:tt}).plot(ax=ax8,vmin=0,vmax=1,transform=ccrs.PlateCarree(),cmap="Blues")
    ds["filtered_mask3"].isel({framedim:tt}).plot(ax=ax9,vmin=0,vmax=1,transform=ccrs.PlateCarree(),cmap="Blues")

    for ax in [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9]:
        ax.coastlines()
        xr.plot.contourf(
            ds["variance_interp"],ax=ax,levels=[0.,0.5],hatches=["","/////"],colors="none",add_colorbar=False)
    
    for ax,i in zip([ax1, ax4, ax7],[0,1,2]):
        ax.set_title(field_titles[i])
    for ax in [ax2, ax5, ax8]:
        ax.set_title("SB mask")
    for ax in [ax3, ax6, ax9]:        
        ax.set_title("SB filter")

    fig.suptitle(ds.isel(time=tt).time.values)
    fig.subplots_adjust(wspace=0.3)

    plt.close(fig)
    
    return None, None
    
def plotmasks_4models(ds,fig,tt,framedim,vmins,vmaxs,field_titles,cmaps,**kwargs):

    ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10, ax11, ax12, ax13, ax14, ax15, ax16 = fig.subplots(4,4,subplot_kw={"projection":ccrs.PlateCarree()}).flatten()
    ds["era5/era5_field1"].isel({framedim:tt}).plot(ax=ax1,vmin=vmins[0][0],vmax=vmaxs[0][0],transform=ccrs.PlateCarree(),cmap=cmaps[0],extend="both")
    ds["barra_r/barra_r_field1"].isel({framedim:tt}).plot(ax=ax2,vmin=vmins[0][1],vmax=vmaxs[0][1],transform=ccrs.PlateCarree(),cmap=cmaps[0],extend="both")
    ds["barra_c/barra_c_field1"].isel({framedim:tt}).plot(ax=ax3,vmin=vmins[0][2],vmax=vmaxs[0][2],transform=ccrs.PlateCarree(),cmap=cmaps[0],extend="both")
    ds["aus2200/aus2200_field1"].isel({framedim:tt}).plot(ax=ax4,vmin=vmins[0][3],vmax=vmaxs[0][3],transform=ccrs.PlateCarree(),cmap=cmaps[0],extend="both")

    ds["era5/era5_field2"].isel({framedim:tt}).plot(ax=ax5,vmin=vmins[1][0],vmax=vmaxs[1][0],transform=ccrs.PlateCarree(),cmap=cmaps[1],extend="both")
    ds["barra_r/barra_r_field2"].isel({framedim:tt}).plot(ax=ax6,vmin=vmins[1][1],vmax=vmaxs[1][1],transform=ccrs.PlateCarree(),cmap=cmaps[1],extend="both")
    ds["barra_c/barra_c_field2"].isel({framedim:tt}).plot(ax=ax7,vmin=vmins[1][2],vmax=vmaxs[1][2],transform=ccrs.PlateCarree(),cmap=cmaps[1],extend="both")
    ds["aus2200/aus2200_field2"].isel({framedim:tt}).plot(ax=ax8,vmin=vmins[1][3],vmax=vmaxs[1][3],transform=ccrs.PlateCarree(),cmap=cmaps[1],extend="both")

    ds["era5/era5_field3"].isel({framedim:tt}).plot(ax=ax9,vmin=vmins[2][0],vmax=vmaxs[2][0],transform=ccrs.PlateCarree(),cmap=cmaps[2],extend="both")
    ds["barra_r/barra_r_field3"].isel({framedim:tt}).plot(ax=ax10,vmin=vmins[2][1],vmax=vmaxs[2][1],transform=ccrs.PlateCarree(),cmap=cmaps[2],extend="both")
    ds["barra_c/barra_c_field3"].isel({framedim:tt}).plot(ax=ax11,vmin=vmins[2][2],vmax=vmaxs[2][2],transform=ccrs.PlateCarree(),cmap=cmaps[2],extend="both")
    ds["aus2200/aus2200_field3"].isel({framedim:tt}).plot(ax=ax12,vmin=vmins[2][3],vmax=vmaxs[2][3],transform=ccrs.PlateCarree(),cmap=cmaps[2],extend="both")

    ds["era5/era5_field4"].isel({framedim:tt}).plot(ax=ax13,vmin=vmins[3][0],vmax=vmaxs[3][0],transform=ccrs.PlateCarree(),cmap=cmaps[3],extend="both")
    ds["barra_r/barra_r_field4"].isel({framedim:tt}).plot(ax=ax14,vmin=vmins[3][1],vmax=vmaxs[3][1],transform=ccrs.PlateCarree(),cmap=cmaps[3],extend="both")
    ds["barra_c/barra_c_field4"].isel({framedim:tt}).plot(ax=ax15,vmin=vmins[3][2],vmax=vmaxs[3][2],transform=ccrs.PlateCarree(),cmap=cmaps[3],extend="both")
    ds["aus2200/aus2200_field4"].isel({framedim:tt}).plot(ax=ax16,vmin=vmins[3][3],vmax=vmaxs[3][3],transform=ccrs.PlateCarree(),cmap=cmaps[3],extend="both")            

    for ax in [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10, ax11, ax12, ax13, ax14, ax15, ax16]:
        ax.coastlines()
        # xr.plot.contourf(
        #     ds["variance_interp"],ax=ax,levels=[0.,0.5],hatches=["","/////"],colors="none",add_colorbar=False)
    
    for ax,i in zip([ax1, ax2, ax3, ax4],["ERA5","BARRA-R","BARRA-C","AUS2200"]):
        ax.set_title(i + " " + field_titles[0])
    for ax,i in zip([ax5, ax6, ax7, ax8],["ERA5","BARRA-R","BARRA-C","AUS2200"]):
        ax.set_title(i + " " + field_titles[1])
    for ax,i in zip([ax9, ax10, ax11, ax12],["ERA5","BARRA-R","BARRA-C","AUS2200"]):
        ax.set_title(i + " " + field_titles[2])
    for ax,i in zip([ax13, ax14, ax15, ax16],["ERA5","BARRA-R","BARRA-C","AUS2200"]):
        ax.set_title(i + " " + field_titles[3])

    fig.suptitle(ds.isel(time=tt).time.values)
    fig.subplots_adjust(wspace=0.3)

    plt.close(fig)
    
    return None, None

def plotmasks_4fields(ds,fig,tt,framedim,vmins,vmaxs,field_titles,**kwargs):

    ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10, ax11, ax12 = fig.subplots(4,3,subplot_kw={"projection":ccrs.PlateCarree()}).flatten()
    ds["field1"].isel({framedim:tt}).plot(ax=ax1,vmin=vmins[0],vmax=vmaxs[0],transform=ccrs.PlateCarree(),extend="both")
    ds["mask1"].isel({framedim:tt}).plot(ax=ax2,vmin=0,vmax=1,transform=ccrs.PlateCarree(),cmap="Blues")
    ds["filtered_mask1"].isel({framedim:tt}).plot(ax=ax3,vmin=0,vmax=1,transform=ccrs.PlateCarree(),cmap="Blues")

    ds["field2"].isel({framedim:tt}).plot(ax=ax4,vmin=vmins[1],vmax=vmaxs[1],transform=ccrs.PlateCarree(),extend="both")
    ds["mask2"].isel({framedim:tt}).plot(ax=ax5,vmin=0,vmax=1,transform=ccrs.PlateCarree(),cmap="Blues")
    ds["filtered_mask2"].isel({framedim:tt}).plot(ax=ax6,vmin=0,vmax=1,transform=ccrs.PlateCarree(),cmap="Blues")

    ds["field3"].isel({framedim:tt}).plot(ax=ax7,vmin=vmins[2],vmax=vmaxs[2],transform=ccrs.PlateCarree(),extend="both")
    ds["mask3"].isel({framedim:tt}).plot(ax=ax8,vmin=0,vmax=1,transform=ccrs.PlateCarree(),cmap="Blues")
    ds["filtered_mask3"].isel({framedim:tt}).plot(ax=ax9,vmin=0,vmax=1,transform=ccrs.PlateCarree(),cmap="Blues")

    ds["field4"].isel({framedim:tt}).plot(ax=ax10,vmin=vmins[3],vmax=vmaxs[3],transform=ccrs.PlateCarree(),extend="both")
    ds["mask4"].isel({framedim:tt}).plot(ax=ax11,vmin=0,vmax=1,transform=ccrs.PlateCarree(),cmap="Blues")
    ds["filtered_mask4"].isel({framedim:tt}).plot(ax=ax12,vmin=0,vmax=1,transform=ccrs.PlateCarree(),cmap="Blues")

    for ax in [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10, ax11, ax12]:
        ax.coastlines()
        xr.plot.contourf(
            ds["variance_interp"],ax=ax,levels=[0.,0.5],hatches=["","/////"],colors="none",add_colorbar=False)
    
    for ax,i in zip([ax1, ax4, ax7, ax10],[0,1,2,3]):
        ax.set_title(field_titles[i])
    for ax in [ax2, ax5, ax8, ax11]:
        ax.set_title("SB mask")
    for ax in [ax3, ax6, ax9, ax12]:        
        ax.set_title("SB filter")

    fig.suptitle(ds.isel(time=tt).time.values)
    fig.subplots_adjust(wspace=0.3)

    plt.close(fig)
    
    return None, None

def plotmasks_4fields_only(ds,fig,tt,framedim,vmins,vmaxs,field_titles,**kwargs):

    ax1, ax2, ax3, ax4 = fig.subplots(2,2,subplot_kw={"projection":ccrs.PlateCarree()}).flatten()
    ds["field1"].isel({framedim:tt}).plot(ax=ax1,vmin=vmins[0],vmax=vmaxs[0],transform=ccrs.PlateCarree(),extend="both")
    ds["field2"].isel({framedim:tt}).plot(ax=ax2,vmin=vmins[1],vmax=vmaxs[1],transform=ccrs.PlateCarree(),extend="both")
    ds["field3"].isel({framedim:tt}).plot(ax=ax3,vmin=vmins[2],vmax=vmaxs[2],transform=ccrs.PlateCarree(),extend="both")
    ds["field4"].isel({framedim:tt}).plot(ax=ax4,vmin=vmins[3],vmax=vmaxs[3],transform=ccrs.PlateCarree(),extend="both")

    for ax in [ax1, ax2, ax3, ax4]:
        ax.coastlines()
        xr.plot.contourf(
            ds["variance_interp"],ax=ax,levels=[0.,0.5],hatches=["","/////"],colors="none",add_colorbar=False)
    
    for ax,i in zip([ax1, ax2, ax3, ax4],[0,1,2,3]):
        ax.set_title(field_titles[i])

    fig.suptitle(ds.isel(time=tt).time.values)
    fig.subplots_adjust(wspace=0.3)

    plt.close(fig)
    
    return None, None   

def xmovie_animation_plotmasks_1fields(ds,vmins,vmaxs,field_titles=["Field1","Field2"],cmap="viridis"):

    mov = xmovie.Movie(
        ds,
        pixelwidth=3000,
        pixelheight=1000,
        plotfunc=plotmasks_1fields,
        vmins=vmins,
        vmaxs=vmaxs,
        input_check=False,
        field_titles=field_titles,
        cmap=cmap
    )
    mov.save(
        "/scratch/ng72/ab4502/temp_figs/animation.mp4",
        overwrite_existing=True,
        parallel=True,
        progress=True,
        framerate=3)

def xmovie_animation_plotmasks_2fields(ds,vmins,vmaxs,field_titles=["Field1","Field2"]):

    mov = xmovie.Movie(
        ds,
        pixelwidth=2560,
        pixelheight=1280,
        plotfunc=plotmasks_2fields,
        vmins=vmins,
        vmaxs=vmaxs,
        input_check=False,
        field_titles=field_titles
    )
    mov.save(
        "/scratch/ng72/ab4502/temp_figs/animation.mp4",
        overwrite_existing=True,
        parallel=True,
        progress=True,
        framerate=5)

def xmovie_animation_plotmasks_3fields(ds,vmins,vmaxs,field_titles=["Field1","Field2","Field3"]):

    mov = xmovie.Movie(
        ds,
        pixelwidth=1920,
        pixelheight=1920,
        plotfunc=plotmasks_3fields,
        vmins=vmins,
        vmaxs=vmaxs,
        input_check=False,
        field_titles=field_titles
    )
    mov.save(
        "/scratch/ng72/ab4502/temp_figs/animation.mp4",
        overwrite_existing=True,
        parallel=True,
        progress=True,
        framerate=5)

def xmovie_animation_plotmasks_4fields(ds,vmins,vmaxs,field_titles=["Field1","Field2","Field3","Field4"]):

    mov = xmovie.Movie(
        ds,
        pixelwidth=2500,
        pixelheight=1920,
        plotfunc=plotmasks_4fields,
        vmins=vmins,
        vmaxs=vmaxs,
        input_check=False,
        field_titles=field_titles
    )
    mov.save(
        "/scratch/ng72/ab4502/temp_figs/animation.mp4",
        overwrite_existing=True,
        parallel=True,
        progress=True,
        framerate=5)    
    
def xmovie_animation_plotmasks_4fields_only(ds,vmins,vmaxs,field_titles=["Field1","Field2","Field3"]):

    mov = xmovie.Movie(
        ds,
        pixelwidth=1280,
        pixelheight=1280,
        plotfunc=plotmasks_4fields_only,
        vmins=vmins,
        vmaxs=vmaxs,
        input_check=False,
        field_titles=field_titles
    )
    mov.save(
        "/scratch/ng72/ab4502/temp_figs/animation.mp4",
        overwrite_existing=True,
        parallel=True,
        progress=True,
        framerate=5)     
    
def xmovie_animation_plotmasks_4models(ds,vmins,vmaxs,cmaps,field_titles=["Field1","Field2","Field3","Field4"],outname=None):

    mov = xmovie.Movie(
        ds,
        pixelwidth=2560,
        pixelheight=1920,
        plotfunc=plotmasks_4models,
        vmins=vmins,
        vmaxs=vmaxs,
        cmaps=cmaps,
        input_check=False,
        field_titles=field_titles
    )
    if outname is None:
        outname = "animation"
    mov.save(
        "/scratch/ng72/ab4502/temp_figs/"+outname+".mp4",
        overwrite_existing=True,
        parallel=False,
        progress=True,
        framerate=5)     

def plot_radar(ds,fig,tt,framedim,latitude,longitude,vmin,vmax,cmap,**kwargs):

    ax = plt.axes(projection=ccrs.PlateCarree())
    c = ax.pcolormesh(longitude,latitude,ds.isel({framedim:tt}),vmin=vmin,vmax=vmax,cmap=cmap)
    ax.set_title(ds.isel(time=tt).time.values)
    fig.colorbar(c,extend="both")
    ax.coastlines()
    plt.close(fig)
    
    return None, None        

def xmovie_animation_plot_radar(ds,latitude,longitude,vmin,vmax,cmap,outname=None):

    mov = xmovie.Movie(
        ds,
        pixelwidth=1200,
        pixelheight=1200,
        plotfunc=plot_radar,
        latitude=latitude,
        longitude=longitude,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        input_check=False,
    )
    if outname is None:
        outname = "animation"
    mov.save(
        "/scratch/ng72/ab4502/temp_figs/"+outname+".mp4",
        overwrite_existing=True,
        parallel=True,
        progress=True,
        framerate=5)  

def radar_animation(rid,times,field,z):

    #Load radar data
    radar_ds = load_obs.load_radar_level1b(rid,times)

    #Subset to every half an hour
    #radar_ds = radar_ds.sel(time=np.in1d(radar_ds.time.dt.minute,[0,30]))
    
    #Get radar data coordinates
    longitude = radar_ds.isel({"time":0}).longitude.values
    latitude = radar_ds.isel({"time":0}).latitude.values

    #Plot radar data
    xmovie_animation_plot_radar(
        radar_ds[field].sel(z=z).persist(),
        latitude,
        longitude,
        vmin=-10,
        vmax=10,
        cmap="RdBu",
        outname=rid+"_"+field+"_"+str(z)+"_"+times[0].strftime("%Y%m%d%H")+"_"+times[1].strftime("%Y%m%d%H")) 
    
    [os.remove(f) for f in glob.glob("/scratch/ng72/ab4502/radar/"+rid+"*_grid.nc")];

def barra_c_animation(lat_slice,lon_slice,time_slice):

    #Load data
    barra_c_F = xr.open_dataset(
            "/g/data/ng72/ab4502/sea_breeze_detection/barra_c/F_201601010000_201601312300.nc",
            chunks="auto").F
    barra_c_Fc = xr.open_dataset(
            "/g/data/ng72/ab4502/sea_breeze_detection/barra_c/Fc_201601010000_201601312300.nc",
            chunks="auto").Fc    
    barrac_fuzzy = xr.open_dataset(
            "/g/data/ng72/ab4502/sea_breeze_detection/barra_c/fuzzy_mean_201601010000_201601312300.nc",
            chunks="auto")["__xarray_dataarray_variable__"]   
    barra_c_fuzzy_mask = xr.open_dataset(
        "/g/data/ng72/ab4502/sea_breeze_detection/barra_c/filtered_mask_fuzzy_mean_201601010000_201601312300.nc",
        chunks="auto")
    barra_c_F_mask = xr.open_dataset(
            "/g/data/ng72/ab4502/sea_breeze_detection/barra_c/filtered_mask_F_201601010000_201601312300.nc",
            chunks="auto")    
    barra_c_Fc_mask = xr.open_dataset(
            "/g/data/ng72/ab4502/sea_breeze_detection/barra_c/filtered_mask_Fc_201601010000_201601312300.nc",
            chunks="auto")    
    angle_ds = xr.open_dataset('/g/data/ng72/ab4502/coastline_data/barra_c.nc')

    plot_ds = xr.Dataset(
            {"field1":barrac_fuzzy,
            "mask1":(barra_c_fuzzy_mask.all_labels >= 1),
            "filtered_mask1":barra_c_fuzzy_mask.mask,
            "field2":barra_c_Fc,
            "mask2":(barra_c_Fc_mask.all_labels >= 1),
            "filtered_mask2":barra_c_Fc_mask.mask,
            "field3":barra_c_F,
            "mask3":(barra_c_F_mask.all_labels >= 1),
            "filtered_mask3":barra_c_F_mask.mask,
            "variance_interp":angle_ds["variance_interp"]
             }
             ).sel(lat=lat_slice,lon=lon_slice,time=time_slice).chunk({"time":1,"lat":-1,"lon":-1}).persist()

    xmovie_animation_plotmasks_3fields(
        plot_ds,
        vmins=[0,-60,-60],
        vmaxs=[0.5,60,60],
        field_titles=["Fuzzy function","Coast F","F"])

def barra_r_animation(lat_slice,lon_slice,time_slice):

    #Load data
    barra_r_hourly = xr.open_dataset(
        "/g/data/ng72/ab4502/sea_breeze_detection/barra_r/F_hourly_201601010000_201601312300.nc",
        chunks="auto")
    barra_r_F = xr.open_dataset(
            "/g/data/ng72/ab4502/sea_breeze_detection/barra_r/F_201601010000_201601312300.nc",
            chunks="auto").F
    barra_r_Fc = xr.open_dataset(
            "/g/data/ng72/ab4502/sea_breeze_detection/barra_r/Fc_201601010000_201601312300.nc",
            chunks="auto").Fc    
    barrar_fuzzy = sea_breeze_funcs.fuzzy_function_combine(
        barra_r_hourly.wind_change,
        barra_r_hourly.q_change,
        barra_r_hourly.t_change,
        combine_method="mean")
    barra_r_fuzzy_mask = xr.open_dataset(
        "/g/data/ng72/ab4502/sea_breeze_detection/barra_r/filtered_mask_fuzzy_mean_201601010000_201601312300.nc",
        chunks="auto")
    barra_r_F_mask = xr.open_dataset(
            "/g/data/ng72/ab4502/sea_breeze_detection/barra_r/filtered_mask_F_201601010000_201601312300.nc",
            chunks="auto")    
    barra_r_Fc_mask = xr.open_dataset(
            "/g/data/ng72/ab4502/sea_breeze_detection/barra_r/filtered_mask_Fc_201601010000_201601312300.nc",
            chunks="auto")    
    angle_ds = xr.open_dataset('/g/data/ng72/ab4502/coastline_data/barra_r.nc')

    plot_ds = xr.Dataset(
            {"field1":barrar_fuzzy,
            "mask1":(barra_r_fuzzy_mask.all_labels >= 1),
            "filtered_mask1":barra_r_fuzzy_mask.mask,
            "field2":barra_r_Fc,
            "mask2":(barra_r_Fc_mask.all_labels >= 1),
            "filtered_mask2":barra_r_Fc_mask.mask,
            "field3":barra_r_F,
            "mask3":(barra_r_F_mask.all_labels >= 1),
            "filtered_mask3":barra_r_F_mask.mask,
            "variance_interp":angle_ds["variance_interp"]
             }
             ).sel(lat=lat_slice,lon=lon_slice,time=time_slice).chunk({"time":1,"lat":-1,"lon":-1}).persist()

    xmovie_animation_plotmasks_3fields(
        plot_ds,
        vmins=[0,-20,-20],
        vmaxs=[0.5,20,20],
        field_titles=["Fuzzy function","Coast F","F"])

def aus2200_animation(lat_slice,lon_slice,time_slice):

    #Load funcs
    aus2200_F = xr.open_dataset(
            "/g/data/ng72/ab4502/sea_breeze_detection/aus2200_smooth_s4/F_mjo-elnino_201601010000_201601312300.zarr",
            chunks={}).F
    aus2200_Fc = xr.open_dataset(
            "/g/data/ng72/ab4502/sea_breeze_detection/aus2200_smooth_s4/Fc_mjo-elnino_201601010000_201601312300.zarr",
            chunks={}).Fc    
    aus2200_sbi = xr.open_dataset(
            "/g/data/ng72/ab4502/sea_breeze_detection/aus2200_smooth_s4/sbi_mjo-elnino_201601010100_201601312300.zarr",
            chunks={}).sbi
    aus2200_fuzzy = xr.open_dataset(
            "/g/data/ng72/ab4502/sea_breeze_detection/aus2200_smooth_s4/fuzzy_mjo-elnino_201601010000_201601312300.zarr",
            chunks={})["__xarray_dataarray_variable__"]
    # aus2200_Fc = xr.open_dataset(
    #         "/g/data/ng72/ab4502/sea_breeze_detection/aus2200/Fc_mjo-elnino_201601010000_201601312300.zarr",
    #         chunks="auto").Fc   
    # aus2200_Fc_smooth = xr.open_dataset(
    #         "/g/data/ng72/ab4502/sea_breeze_detection/aus2200_smooth_s2/Fc_mjo-elnino_201601010000_201601312300.zarr",
    #         chunks="auto").Fc    

    #Load masks
    aus2200_fuzzy_mask = xr.open_dataset(
        "/g/data/ng72/ab4502/sea_breeze_detection/aus2200_smooth_s4/filtered_mask_no_hourly_change_fuzzy_201601060000_201601122300.zarr",
        chunks={})
    aus2200_F_mask = xr.open_dataset(
        "/g/data/ng72/ab4502/sea_breeze_detection/aus2200_smooth_s4/filtered_mask_no_hourly_change_F_201601060000_201601122300.zarr",
        chunks={})
    aus2200_Fc_mask = xr.open_dataset(
        "/g/data/ng72/ab4502/sea_breeze_detection/aus2200_smooth_s4/filtered_mask_no_hourly_change_Fc_201601060000_201601122300.zarr",
        chunks={})
    aus2200_sbi_mask = xr.open_dataset(
        "/g/data/ng72/ab4502/sea_breeze_detection/aus2200_smooth_s4/filtered_mask_no_hourly_change_sbi_201601060000_201601122300.zarr",
        chunks={})
    # aus2200_Fc_mask = xr.open_dataset(
    #     "/g/data/ng72/ab4502/sea_breeze_detection/aus2200/filtered_mask_Fc_201601010000_201601312300.zarr",
    #     chunks="auto")
    # aus2200_Fc_mask_smooth = xr.open_dataset(
    #     "/g/data/ng72/ab4502/sea_breeze_detection/aus2200_smooth_s2/filtered_mask_Fc_201601010000_201601312300.zarr",
    #     chunks="auto")


    angle_ds = xr.open_dataset('/g/data/ng72/ab4502/coastline_data/aus2200.nc')

    plot_ds = xr.Dataset(
            {"field1":aus2200_fuzzy,
            "mask1":(aus2200_fuzzy_mask.all_labels >= 1),
            "filtered_mask1":aus2200_fuzzy_mask.mask,
            "field2":aus2200_Fc,
            "mask2":(aus2200_Fc_mask.all_labels >= 1),
            "filtered_mask2":aus2200_Fc_mask.mask,
            "field3":aus2200_F,
            "mask3":(aus2200_F_mask.all_labels >= 1),
            "filtered_mask3":aus2200_F_mask.mask,
            "field4":aus2200_sbi,
            "mask4":(aus2200_sbi_mask.all_labels >= 1),
            "filtered_mask4":aus2200_sbi_mask.mask,
            "variance_interp":angle_ds["variance_interp"]
             }
             ).sel(lat=lat_slice,lon=lon_slice,time=time_slice).chunk({"time":1,"lat":-1,"lon":-1})#.persist()
    # plot_ds = xr.Dataset(
    #         {
    #         "field1":aus2200_Fc_smooth,
    #         "mask1":(aus2200_Fc_mask_smooth.all_labels >= 1),
    #         "filtered_mask1":aus2200_Fc_mask_smooth.mask,
    #         "variance_interp":angle_ds["variance_interp"]
    #          }
    #          ).sel(lat=lat_slice,lon=lon_slice,time=time_slice).chunk({"time":1,"lat":-1,"lon":-1}).persist()

    xmovie_animation_plotmasks_4fields(
        plot_ds,
        vmins=[0,-50,-50,0],
        vmaxs=[0.5,50,50,1],
        field_titles=["Fuzzy function","Coast F","F","sbi"])
    # xmovie_animation_plotmasks_1fields(
    #     plot_ds,
    #     vmins=[-50],
    #     vmaxs=[50],
    #     field_titles=["Fc_smoothed"],
    #     cmap="RdBu")
    
def compare_models_animation(lat_slice,lon_slice,time_slice,outname=None):

    chunks = {"time":50,"lat":-1,"lon":-1}

    #Load funcs AUS2200
    aus2200_F = xr.open_dataset(
            "/g/data/ng72/ab4502/sea_breeze_detection/aus2200/F_mjo-elnino_201601010000_201601312300.nc",
            chunks=chunks).F
    aus2200_Fc = xr.open_dataset(
            "/g/data/ng72/ab4502/sea_breeze_detection/aus2200/Fc_mjo-elnino_201601010000_201601312300.nc",
            chunks=chunks).Fc    
    aus2200_sbi = xr.open_dataset(
            "/g/data/ng72/ab4502/sea_breeze_detection/aus2200/sbi_mjo-elnino_201601010100_201601312300.nc",
            chunks=chunks).sbi
    aus2200_fuzzy = xr.open_dataset(
            "/g/data/ng72/ab4502/sea_breeze_detection/aus2200/fuzzy_mjo-elnino_201601010000_201601312300.nc",
            chunks=chunks)["__xarray_dataarray_variable__"]

    #Load funcs BARRA-C
    barra_c_F = xr.open_dataset(
            "/g/data/ng72/ab4502/sea_breeze_detection/barra_c/F_201601010000_201601312300.nc",
            chunks=chunks).F
    barra_c_Fc = xr.open_dataset(
            "/g/data/ng72/ab4502/sea_breeze_detection/barra_c/Fc_201601010000_201601312300.nc",
            chunks=chunks).Fc    
    barra_c_fuzzy = xr.open_dataset(
            "/g/data/ng72/ab4502/sea_breeze_detection/barra_c/fuzzy_mean_201601010000_201601312300.nc",
            chunks=chunks)["__xarray_dataarray_variable__"]    
    
    #Load funcs BARRA-R
    barra_r_F = xr.open_dataset(
            "/g/data/ng72/ab4502/sea_breeze_detection/barra_r/F_201601010000_201601312300.nc",
            chunks=chunks).F
    barra_r_Fc = xr.open_dataset(
            "/g/data/ng72/ab4502/sea_breeze_detection/barra_r/Fc_201601010000_201601312300.nc",
            chunks=chunks).Fc    
    barra_r_hourly = xr.open_dataset(
            "/g/data/ng72/ab4502/sea_breeze_detection/barra_r/F_hourly_201601010000_201601312300.nc",
            chunks=chunks)
    barra_r_fuzzy = sea_breeze_funcs.fuzzy_function_combine(
        barra_r_hourly.wind_change,
        barra_r_hourly.q_change,
        barra_r_hourly.t_change,
        combine_method="mean")        
    
    #Load funcs ERA5
    era5_F = xr.open_dataset(
            "/g/data/ng72/ab4502/sea_breeze_detection/era5/F_201601010000_201601312300.nc",
            chunks=chunks).F
    era5_Fc = xr.open_dataset(
            "/g/data/ng72/ab4502/sea_breeze_detection/era5/Fc_201601010000_201601312300.nc",
            chunks=chunks).Fc    
    era5_sbi = xr.open_dataset(
            "/g/data/ng72/ab4502/sea_breeze_detection/era5/sbi_201601010000_201601312300.nc",
            chunks=chunks).sbi    
    era5_hourly = xr.open_dataset(
            "/g/data/ng72/ab4502/sea_breeze_detection/era5/F_hourly_201601010000_201601312300.nc",
            chunks=chunks)
    era5_fuzzy = sea_breeze_funcs.fuzzy_function_combine(
        era5_hourly.wind_change,
        era5_hourly.q_change,
        era5_hourly.t_change,
        combine_method="mean")        

    era5_ds = xr.merge(
            [{"era5_field1":era5_fuzzy},
            {"era5_field2":era5_Fc},
            {"era5_field3":era5_F},
            {"era5_field4":era5_sbi}],
            compat="override"
             ).sel(lat=lat_slice,lon=lon_slice,time=time_slice).chunk({"time":1,"lat":-1,"lon":-1}).persist()
    barra_r_ds = xr.merge(
            [{"barra_r_field1":barra_r_fuzzy},
            {"barra_r_field2":barra_r_Fc},
            {"barra_r_field3":barra_r_F},
            {"barra_r_field4":barra_r_F*np.nan}],
            compat="override"
             ).sel(lat=lat_slice,lon=lon_slice,time=time_slice).chunk({"time":1,"lat":-1,"lon":-1}).persist()
    barra_c_ds = xr.merge(
            [{"barra_c_field1":barra_c_fuzzy},
            {"barra_c_field2":barra_c_Fc},
            {"barra_c_field3":barra_c_F},
            {"barra_c_field4":barra_c_F*np.nan}],
            compat="override"
             ).sel(lat=lat_slice,lon=lon_slice,time=time_slice).chunk({"time":1,"lat":-1,"lon":-1}).persist()
    aus2200_ds = xr.merge(
            [{"aus2200_field1":aus2200_fuzzy},
            {"aus2200_field2":aus2200_Fc},
            {"aus2200_field3":aus2200_F},
            {"aus2200_field4":aus2200_sbi}],
            compat="override"
             ).sel(lat=lat_slice,lon=lon_slice,time=time_slice).chunk({"time":1,"lat":-1,"lon":-1}).persist()
    dt = xr.DataTree.from_dict({
            "/":xr.Dataset({"time":era5_ds.time}),
            "era5":era5_ds,
            "barra_r":barra_r_ds,
            "barra_c":barra_c_ds,
            "aus2200":aus2200_ds
        })

    xmovie_animation_plotmasks_4models(
        dt,
        vmins=[
            [0,0,0,0],
            [-4,-12,-60,-200],
            [-4,-12,-85,-300],
            [0,0,0,0]],
        vmaxs=[
            [0.3,0.3,0.3,0.3],
            [4,12,60,200],
            [4,12,85,300],
            [0.35,1,1,0.6]],
        field_titles=["Fuzzy function","Coast F","F","sbi"],
        cmaps=["viridis","RdBu","RdBu","viridis"],
        outname=outname)    

if __name__ == "__main__":

    client = Client()

    #Lat/lon/time bounds
    
    #lat_slice, lon_slice = utils.get_perth_large_bounds()
    lat_slice, lon_slice = utils.get_darwin_large_bounds()
    time_slice = slice("2016-01-06 00:00","2016-01-12 23:00")

    # compare_models_animation(lat_slice,lon_slice,time_slice,outname="compare_models_perth_20160106_20160112")

    # lat_slice, lon_slice = utils.get_darwin_large_bounds()
    # compare_models_animation(lat_slice,lon_slice,time_slice,outname="compare_models_darwin_20160106_20160112")

    #barra_c_animation(lat_slice,lon_slice,time_slice)
    aus2200_animation(lat_slice,lon_slice,time_slice)


    # rid="70"
    # field="corrected_velocity"
    # z=500    
    # for t in pd.date_range("2016-01-01 00:00","2016-01-31 00:00",freq="1D"):
    #     print(t)
    #     times = [t, t+dt.timedelta(days=1)]
    #     radar_animation(rid,times,field,z)


    #Animate
    # data = [
    #         barrar_fuzzy,
    #         (barra_r_mask.all_labels >= 1).persist(),
    #         barra_r_mask.mask        
    #     ]    
    # animate_pcolormesh(
    #     data,
    #     rows=1,
    #     cols=3,
    #     figsize=(20,5),
    #     vmins=[0,0,0],
    #     vmaxs=[1,1,1],
    #     titles=["BARRA-R fuzzy function","BARRA-R binary mask","BARRA-R sea breezes"],
    #     fname="barra_r_fuzzy_sb_perth_20160101_20160131",
    #     cmaps=[None,"Blues","Blues"])