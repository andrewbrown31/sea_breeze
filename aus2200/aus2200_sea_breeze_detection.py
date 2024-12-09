from sea_breeze.load_model_data import load_aus2200_static, load_aus2200_variable, destagger_aus2200, \
                            round_times, interp_times
from sea_breeze.sea_breeze_funcs import kinematic_frontogenesis, coast_relative_frontogenesis, load_angle_ds,\
                                        calc_sbi, moisture_flux_gradient
from climtas.nci import GadiClient
from dask.distributed import progress
import xarray as xr
import pandas as pd

class Chunks:

    def __init__(self):
        self.time_chunk = -1
        self.latlon_chunk = -1
        self.lev_chunk = -1

    def set_chunks(self,time_size=-1,latlon_size=-1,lev_size=-1):
        self.time_chunk = time_size
        self.latlon_chunk = latlon_size
        self.lev_chunk = lev_size
        return self

    def __str__(self):
        return f"Chunk sizes\nTime: {self.time_chunk} \nLat-lon: {self.latlon_chunk} \nLevels: {self.lev_chunk}"

if __name__ == "__main__":

    #Initiate distributed dask client on the Gadi HPC
    client = GadiClient()

    #Set the domain bounds
    # lon_slice = slice(108,158.98)
    # lat_slice = slice(-45.7,-6.831799)  
    lat_slice=slice(-38,-30)
    lon_slice=slice(112,120)    
    dx = 0.022
    u_lon_slice=slice(lon_slice.start,lon_slice.stop+dx)
    v_lat_slice=slice(lat_slice.start,lat_slice.stop+dx) 

    #Set time slice and model name    
    t1="2016-01-01 00:00"
    t2="2016-02-01 00:00"
    model="aus2200"

    #Initialise the chunking object to keep track of different chunking
    chunks = Chunks() 

    #Load AUS2200 surface data and static info
    orog, lsm = load_aus2200_static("mjo-elnino",lon_slice,lat_slice)
    aus2200_vas = round_times(load_aus2200_variable(["vas"],t1,t2,"mjo-elnino",lon_slice,v_lat_slice,"10min",chunks="auto")[0], "10min")
    aus2200_uas = round_times(load_aus2200_variable(["uas"],t1,t2,"mjo-elnino",u_lon_slice,lat_slice,"10min",chunks="auto")[0], "10min")
    aus2200_hus = round_times(load_aus2200_variable(["huss"],t1,t2,"mjo-elnino",lon_slice,lat_slice,"10min",chunks="auto")[0], "10min")
    angle_ds = load_angle_ds("/g/data/gb02/ab4502/coastline_data/aus2200_v3.nc",lat_slice,lon_slice)
    
    #Just do hourly times
    aus2200_vas = aus2200_vas.sel(time=aus2200_vas.time.dt.minute==0)
    aus2200_uas = aus2200_uas.sel(time=aus2200_uas.time.dt.minute==0)
    aus2200_hus = aus2200_hus.sel(time=aus2200_hus.time.dt.minute==0)

    #Destagger surface winds by centering v wind in lat and u wind in lon
    aus2200_vas = (aus2200_vas.isel(lat=slice(0,-1)).assign_coords({"lat":lsm.lat}) +
                    aus2200_vas.isel(lat=slice(1,aus2200_vas.lat.shape[0])).assign_coords({"lat":lsm.lat})) / 2
    aus2200_uas = (aus2200_uas.isel(lon=slice(0,-1)).assign_coords({"lon":lsm.lon}) +
                    aus2200_uas.isel(lon=slice(1,aus2200_uas.lon.shape[0])).assign_coords({"lon":lsm.lon})) / 2

    #Calculate rate of change in moisture flux, chunking only in latlon
    chunks = chunks.set_chunks(time_size=-1,latlon_size=200) 
    F_dqu = moisture_flux_gradient(
        aus2200_hus.chunk({"time":-1,"lat":"auto","lon":"auto"}),
        aus2200_uas.chunk({"time":-1,"lat":"auto","lon":"auto"}),
        aus2200_vas.chunk({"time":-1,"lat":"auto","lon":"auto"}),
        angle_ds)
    
    #Calculate frontogenesis, chunking only in time
    chunks = chunks.set_chunks(time_size=1,latlon_size=-1) 
    F = kinematic_frontogenesis(
        aus2200_hus.chunk({"time":"auto","lat":-1,"lon":-1}),
        aus2200_uas.chunk({"time":"auto","lat":-1,"lon":-1}),
        aus2200_vas.chunk({"time":"auto","lat":-1,"lon":-1}))

    #Calculate coast-relative frontogensis, chunking only in time
    chunks = chunks.set_chunks(time_size=1,latlon_size=-1) 
    Fc = coast_relative_frontogenesis(
        aus2200_hus,
        aus2200_uas,
        aus2200_vas,
        angle_ds) 

    #Save output
    out_path = "/g/data/gb02/ab4502/sea_breeze_detection/"+model+"/"

    F_dqu_fname = "F_dqu_"+pd.to_datetime(t1).strftime("%Y%m%d%H%M")+"_"+\
                    (pd.to_datetime(t2).strftime("%Y%m%d%H%M"))+".nc"
    F_fname = "F_"+pd.to_datetime(t1).strftime("%Y%m%d%H%M")+"_"+\
                    (pd.to_datetime(t2).strftime("%Y%m%d%H%M"))+".nc"
    Fc_fname = "Fc_"+pd.to_datetime(t1).strftime("%Y%m%d%H%M")+"_"+\
                    (pd.to_datetime(t2).strftime("%Y%m%d%H%M"))+".nc"
    sbi_fname = "sbi_"+pd.to_datetime(t1).strftime("%Y%m%d%H%M")+"_"+\
                    (pd.to_datetime(t2).strftime("%Y%m%d%H%M"))+".nc"        