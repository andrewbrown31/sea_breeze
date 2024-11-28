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

    #Set the domain and time slice
    lon_slice = slice(108,159)
    lat_slice = slice(-45.7,-6.831799)    
    # lat_slice=slice(-38,-30)
    # lon_slice=slice(112,120)    
    t1="2016-01-21 00:00"
    t2="2016-01-22 00:00"
    model="aus2200"

    #Initialise the chunking object to keep track of different chunking
    chunks = Chunks()

    #Load AUS2200 model level data
    chunks = chunks.set_chunks(time_size=1,latlon_size=-1,lev_size={})
    # aus2200_ml = load_aus2200_variable(["ua","va"],t1,t2,"mjo-elnino",lon_slice,lat_slice,"1hr",hgt_slice=slice(0,5000),
    #                                    chunks= {"time":chunks.time_chunk,
    #                                             "lev":chunks.lev_chunk,
    #                                             "lat":chunks.latlon_chunk,
    #                                             "lon":chunks.latlon_chunk})
    aus2200_ua, aus2200_va = load_aus2200_variable(["ua","va"],t1,t2,"mjo-elnino",lon_slice,lat_slice,"1hr",hgt_slice=slice(0,5000),
                                       chunks= {"time":chunks.time_chunk,
                                                "lev":chunks.lev_chunk,
                                                "lat":chunks.latlon_chunk,
                                                "lon":chunks.latlon_chunk})
    aus2200_ua = aus2200_ua.chunk({"lev":1}).persist()
    aus2200_va = aus2200_va.chunk({"lev":1}).persist()

    #Load AUS2200 static data
    orog, lsm = load_aus2200_static("mjo-elnino",lon_slice,lat_slice)
    angle_ds = load_angle_ds("/g/data/gb02/ab4502/coastline_data/aus2200_v3.nc",lat_slice,lon_slice)    

    #Load AUS2200 surface data
    aus2200_sfc = round_times(
        load_aus2200_variable(["vas","uas","hus"],t1,t2,"mjo-elnino",lon_slice,lat_slice,"10min",chunks="auto"),
        "10min")

    #Chunk U and V data on time and height/lev, then destagger in space 
    chunks = chunks.set_chunks(time_size=1,latlon_size=-1,lev_size=1)
    aus2200_sfc["uas"] = aus2200_sfc["uas"].chunk({"time":chunks.time_chunk,"lat":chunks.latlon_chunk,"lon":chunks.latlon_chunk})
    aus2200_sfc["vas"] = aus2200_sfc["vas"].chunk({"time":chunks.time_chunk,"lat":chunks.latlon_chunk,"lon":chunks.latlon_chunk})
    aus2200_sfc["hus"] = aus2200_sfc["hus"].chunk({"time":chunks.time_chunk,"lat":chunks.latlon_chunk,"lon":chunks.latlon_chunk})
    # aus2200_ua = aus2200_ua.chunk(
    #     {"time":chunks.time_chunk,
    #      "lev":chunks.lev_chunk,
    #      "lat":chunks.latlon_chunk,
    #      "lon":chunks.latlon_chunk})
    # aus2200_va = aus2200_va.chunk(
    #     {"time":chunks.time_chunk,
    #      "lev":chunks.lev_chunk,
    #      "lat":chunks.latlon_chunk,
    #      "lon":chunks.latlon_chunk})
    #aus2200_sfc = destagger_aus2200(aus2200_sfc,["uas","vas"],interp_to="hus",lsm=None)
    aus2200_ua = aus2200_ua.interp({"lat":lsm.lat,"lon":lsm.lon},method="linear")
    aus2200_va = aus2200_va.interp({"lat":lsm.lat,"lon":lsm.lon},method="linear")
    #aus2200_wind = xr.Dataset({"u":aus2200_ml["ua"], "v":aus2200_ml["va"]}).persist()
    aus2200_wind = xr.Dataset({"u":aus2200_ua, "v":aus2200_va}).persist()

    #Load boundary layer heights, rechunk, and interpolate in time to hourly model level wind data
    chunks = chunks.set_chunks(time_size=-1,latlon_size=200)
    aus2200_zmla = load_aus2200_variable(["zmla"],t1,t2,"mjo-elnino",lon_slice,lat_slice,"1hr",chunks="auto")[0]
    aus2200_zmla = aus2200_zmla.chunk(
            {"time":chunks.time_chunk,"lat":chunks.latlon_chunk,"lon":chunks.latlon_chunk})
    # aus2200_zmla = interp_times(
    #     aus2200_zmla,
    #     aus2200_wind.time,
    #     lower_bound=0)
    # aus2200_zmla["zmla"] = aus2200_zmla["zmla"].persist()
    aus2200_zmla = aus2200_zmla.interp(coords={"time":aus2200_ua.time},method="linear",kwargs={"fill_value":"extrapolate"})
    aus2200_zmla = xr.where(aus2200_zmla < 0, 0, aus2200_zmla).persist()
    
    #Calculate rate of change in moisture flux, chunking only in latlon
    chunks = chunks.set_chunks(time_size=-1,latlon_size=200) 
    F_dqu = moisture_flux_gradient(
        aus2200_sfc["hus"].chunk({"time":chunks.time_chunk,"lat":chunks.latlon_chunk,"lon":chunks.latlon_chunk}),
        aus2200_sfc["uas"].chunk({"time":chunks.time_chunk,"lat":chunks.latlon_chunk,"lon":chunks.latlon_chunk}),
        aus2200_sfc["vas"].chunk({"time":chunks.time_chunk,"lat":chunks.latlon_chunk,"lon":chunks.latlon_chunk}),
        angle_ds.chunk({"lat":chunks.latlon_chunk,"lon":chunks.latlon_chunk}))
    
    #Calculate frontogenesis, chunking only in time
    chunks = chunks.set_chunks(time_size=1,latlon_size=-1) 
    F = kinematic_frontogenesis(
        aus2200_sfc["hus"].chunk({"time":chunks.time_chunk,"lat":chunks.latlon_chunk,"lon":chunks.latlon_chunk}),
        aus2200_sfc["uas"].chunk({"time":chunks.time_chunk,"lat":chunks.latlon_chunk,"lon":chunks.latlon_chunk}),
        aus2200_sfc["vas"].chunk({"time":chunks.time_chunk,"lat":chunks.latlon_chunk,"lon":chunks.latlon_chunk}))

    #Calculate coast-relative frontogensis, chunking only in time
    chunks = chunks.set_chunks(time_size=1,latlon_size=-1) 
    Fc = coast_relative_frontogenesis(
        aus2200_sfc["hus"].chunk({"time":chunks.time_chunk,"lat":chunks.latlon_chunk,"lon":chunks.latlon_chunk}),
        aus2200_sfc["uas"].chunk({"time":chunks.time_chunk,"lat":chunks.latlon_chunk,"lon":chunks.latlon_chunk}),
        aus2200_sfc["vas"].chunk({"time":chunks.time_chunk,"lat":chunks.latlon_chunk,"lon":chunks.latlon_chunk}),
        angle_ds.chunk({"lat":chunks.latlon_chunk,"lon":chunks.latlon_chunk}))

    #Rechunk again then compute SBI
    chunks = chunks.set_chunks(time_size=1,latlon_size=-1) 
    sbi = calc_sbi(
        aus2200_wind.rename({"lev":"height"}),
        angle_ds.chunk({"lat":chunks.latlon_chunk,"lon":chunks.latlon_chunk}),
        subtract_mean=True,
        height_mean=False,
        height_method="blh",
        blh_da=aus2200_zmla.chunk({"time":chunks.time_chunk,"lat":chunks.latlon_chunk,"lon":chunks.latlon_chunk}),
        alpha_height=0)
    # sbi = calc_sbi(
    #     aus2200_wind.rename({"lev":"height"}),
    #     angle_ds,
    #     subtract_mean=True,
    #     height_mean=False,
    #     height_method="blh",
    #     blh_da=aus2200_zmla,
    #     alpha_height=0)    

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
    
    print("Calculating SBI...")
    sbi.to_netcdf(out_path+sbi_fname)    
    print("Calculating moisture flux rate of change...")
    F_dqu.sel(time=F_dqu.time.dt.minute==0).to_netcdf(out_path+F_dqu_fname)
    print("Calculating frontogenesis...")
    F.sel(time=F.time.dt.minute==0).to_netcdf(out_path+F_fname)
    print("Calculating coast-relative frontogenesis...")
    Fc.sel(time=Fc.time.dt.minute==0).to_netcdf(out_path+Fc_fname)