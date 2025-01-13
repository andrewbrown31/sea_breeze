import xarray as xr
from sys import stdout
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import matplotlib.animation as animation

def plot_pcolormesh(da,vmin=None,vmax=None,save=True,fname="temp",cmap="viridis"):
    plt.figure(figsize=(10,5))
    ax = plt.axes(projection=ccrs.PlateCarree())
    da.plot.pcolormesh(ax=ax,transform=ccrs.PlateCarree(),vmin=vmin,vmax=vmax)
    ax.coastlines()
    if save:
        plt.savefig("/scratch/gb02/ab4502/temp_figs/"+fname+".png",bbox_inches="tight",dpi=300)

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
    ani.save('/g/data/gb02/ab4502/figs/animations/' + fname + '.mp4', writer='ffmpeg', dpi=300)
    #plt.close()

if __name__ == "__main__":

    #Load data
    barra_r = xr.open_dataset("/g/data/gb02/ab4502/sea_breeze_detection/barra_r/Fc_201601010000_201601312300.nc")
    era5 = xr.open_dataset("/g/data/gb02/ab4502/sea_breeze_detection/era5/Fc_201601010000_201601312300.nc")
    data = [barra_r["Fc"].isel(time=slice(0,10)),
            era5["Fc"].isel(time=slice(0,10))]

    #Animate Fc
    animate_pcolormesh(
        data,
        rows=1,
        cols=2,
        figsize=(20,5),
        vmins=[-20,-20],
        vmaxs=[20,20],
        titles=["BARRA-R","ERA5"],
        fname="barra_r_era5_Fc_20160101_20160131",
        cmaps=["RdBu","RdBu"])