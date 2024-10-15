import intake
import xarray as xr
import numpy as np
import pandas as pd
import datetime as dt

def interp_np(x, xp, fp):
    return np.interp(x, xp, fp)

def interp_model_level_to_z(z_da,var_da,mdl_dim,heights):

    '''
    Linearly interpolate from model level data to geopotential height levels

    Input
    z_da: xarray Dataarray of geopotential height (either AGL or above geoid)
    var_da: xarray Dataarray of variable to interpolate
    mdl_dim: name of the model level dimension (e.g. hybrid). NOTE that model levels must be decreasing (so height is increasing)
    heights: numpy array of height levels
    '''

    assert z_da[mdl_dim][0] > z_da[mdl_dim][-1], "Model levels should be decreasing"

    interp_da = xr.apply_ufunc(interp_np,
                heights,
                z_da,
                var_da,
                input_core_dims=[ ["height"], [mdl_dim], [mdl_dim]],
                output_core_dims=[["height"]],
                exclude_dims=set((mdl_dim,)),
                dask="parallelized",
                output_dtypes=var_da.dtype,
                vectorize=True)
    interp_da["height"] = heights
    
    return interp_da

def load_era5_ml(path,t1,t2,lat_slice,lon_slice,heights=np.arange(0,4100,100)):

    '''
    Load ERA5 model level data, as downloaded from the Google cloud and stored on gb02 (using era5_download_google.ipynb)
    t1: start time in %Y-%m-%d %H:%M"
    t1: end time in %Y-%m-%d %H:%M"
    lat_slice: a slice to restrict lat domain
    lon_slice: a slice to restrict lon domain    
    '''

    #Load data from disk
    f = xr.open_dataset(path,chunks={"latitude":5,"longitude":5}).rename({"longitude":"lon","latitude":"lat"})
    
    #Reverse model level data for interpolation, as well as the lat dimension, and slice
    f = f.isel(hybrid=slice(None,None,-1))
    f = f.isel(lat=slice(None,None,-1))
    f = f.sel(lat=lat_slice,lon=lon_slice)

    #Load static data
    topo,lsm = load_era5_static(lon_slice,lat_slice,t1,t2)

    #Adjust geopotential to height above surface, using topography data saved on gb02
    g = 9.80665    #https://confluence.ecmwf.int/display/CKB/ERA5%3A+compute+pressure+and+geopotential+on+model+levels%2C+geopotential+height+and+geometric+height#heading-Geopotentialheight
    #topo = xr.open_dataset("/g/data/gb02/ab4502/era5_static/raw_date-variable-static_2021_12_31_geopotential_static.nc")
    #topo = topo.squeeze().sel(longitude=f.longitude, latitude=f.latitude)["z"] / g
    topo = topo / g
    f["geopotential"] = (f["geopotential"] / g) - topo
    f = f.rename({"geopotential":"geopotential_hgt_agl"})

    #Convert to height levels
    interp_v = interp_model_level_to_z(f["geopotential_hgt_agl"],f["v_component_of_wind"],"hybrid",heights)
    interp_u = interp_model_level_to_z(f["geopotential_hgt_agl"],f["u_component_of_wind"],"hybrid",heights)
    interp_era5 = xr.Dataset({"u":interp_u, "v":interp_v})

    return interp_era5, lsm


def combine_winds(u,v,uname,vname,ws_name):

    '''
    From u and v wind component dataarrays, construct a wind speed dataarray
    '''

    wind_da = xr.Dataset({uname:u[uname],vname:v[vname]})
    wind_da[ws_name] = np.sqrt(wind_da[uname]**2 + wind_da[vname]**2)

    return wind_da

def get_intake_cat():

    '''
    Return the intake catalog for barra
    '''

    #See here: https://opus.nci.org.au/pages/viewpage.action?pageId=264241965
    data_catalog = intake.open_esm_datastore("/g/data/ob53/catalog/v2/esm/catalog.json")

    return data_catalog

def get_intake_cat_era5():

    '''
    Return the intake catalog for era5
    '''

    #See here: https://opus.nci.org.au/pages/viewpage.action?pageId=264241965
    data_catalog = intake.open_esm_datastore("/g/data/rt52/catalog/v2/esm/catalog.json")

    return data_catalog

def load_barra_wind_data(unames, vnames, ws_names, t1, t2, domain_id, freq, lat_slice, lon_slice):

    '''
    unames: list of u wind component variables to get (e.g. u100m). Must be a corresponding vname and ws_name
    vnames: list of v wind component variables to get (e.g. v100m). Must be a corresponding uname and ws_name
    ws_names: list of names for wind speed output (e.g. ws_100m). Must be a corresponding uname and vname
    t1: start time in %Y-%m-%d %H:%M"
    t1: end time in %Y-%m-%d %H:%M"
    domain_id: for barra, either AUS-04 or AUST-11
    freq: frequency string (e.g. 1hr)
    lat_slice: a slice to restrict lat domain
    lon_slice: a slice to restrict lon domain
    '''

    data_catalog = get_intake_cat()
    times = pd.date_range(pd.to_datetime(t1).replace(day=1),t2,freq="MS").strftime("%Y%m").astype(int).values
    wind_ds_list = []
    for uname,vname,wsname in zip(unames,vnames,ws_names):
        u_ds = data_catalog.search(variable_id=uname,
                            domain_id=domain_id,
                            freq=freq,
                            start_time=times).to_dask().sel(lon=lon_slice, lat=lat_slice, time=slice(t1,t2))
        v_ds = data_catalog.search(variable_id=vname,
                            domain_id=domain_id,
                            freq=freq,
                            start_time=times).to_dask().sel(lon=lon_slice, lat=lat_slice, time=slice(t1,t2))
        wind_ds_list.append(combine_winds(u_ds,v_ds,uname,vname,wsname)\
                        .chunk({"time":-1,"lat":20,"lon":20}))
        
    wind_ds = xr.merge(wind_ds_list,compat="override")
    return wind_ds

def load_barra_variable(vname, t1, t2, domain_id, freq, lat_slice, lon_slice):

    '''
    vname: name of barra variable
    t1: start time in %Y-%m-%d %H:%M"
    t1: end time in %Y-%m-%d %H:%M"
    domain_id: for barra, either AUS-04 or AUST-11
    freq: frequency string (e.g. 1h)
    lat_slice: a slice to restrict lat domain
    lon_slice: a slice to restrict lon domain
    '''

    data_catalog = get_intake_cat()
    times = pd.date_range(pd.to_datetime(t1).replace(day=1),t2,freq="MS").strftime("%Y%m").astype(int).values
    ds = data_catalog.search(variable_id=vname,
                            domain_id=domain_id,
                            freq=freq,
                            start_time=times).to_dask().sel(lon=lon_slice, lat=lat_slice, time=slice(t1,t2))
        
    return ds

def load_barra_static(domain_id,lon_slice,lat_slice):

    '''
    For a barra domain, load static variables
    domain_id: for barra, either AUS-04 or AUST-11
    lat_slice: a slice to restrict lat domain
    lon_slice: a slice to restrict lon domain
    '''

    data_catalog = get_intake_cat()
    orog = data_catalog.search(variable_id="orog",domain_id=domain_id).to_dask().sel(lon=lon_slice, lat=lat_slice)
    lsm = data_catalog.search(variable_id="sftlf",domain_id=domain_id).to_dask().sel(lon=lon_slice, lat=lat_slice)

    return orog.orog, (lsm.sftlf >= 0.5) * 1

def load_era5_static(lon_slice,lat_slice,t1,t2):

    '''
    For ERA5, load static variables using the first time step of the period.
    Also flip the latitude coord and convert -180-180 lons to 0-360 (for consistency with BARRA)
    lat_slice: a slice to restrict lat domain
    lon_slice: a slice to restrict lon domain
    t1: start time in %Y-%m-%d %H:%M"
    t1: end time in %Y-%m-%d %H:%M"
    '''

    data_catalog = get_intake_cat_era5()
    time_starts = pd.date_range(pd.to_datetime(t1).replace(day=1),t2,freq="MS").strftime("%Y%m%d").astype(int).values
    time_ends = [(t + dt.timedelta(days=32)).replace(day=1) - dt.timedelta(days=1) for t in pd.to_datetime(time_starts,format="%Y%m%d")]
    times = [str(t1) + "-" + t2.strftime("%Y%m%d") for t1,t2 in zip(time_starts,time_ends)]

    #times = pd.date_range(pd.to_datetime(t1).replace(day=1),t2,freq="MS").strftime("%Y%m").astype(int).values
    orog = data_catalog.search(variable="z",product="era5-reanalysis",time_range=times,levtype="sfc").to_dask()
    orog = orog.isel(latitude=slice(None,None,-1))
    orog["longitude"] = (orog.longitude % 360)
    orog = orog.sortby("longitude")    
    orog = orog.rename({"longitude":"lon","latitude":"lat"}).sel(lon=lon_slice, lat=lat_slice).isel(time=0)

    lsm = data_catalog.search(variable="lsm",product="era5-reanalysis",time_range=times).to_dask()
    lsm = lsm.isel(latitude=slice(None,None,-1))
    lsm["longitude"] = (lsm.longitude % 360)
    lsm = lsm.sortby("longitude")
    lsm = lsm.rename({"longitude":"lon","latitude":"lat"}).sel(lon=lon_slice, lat=lat_slice).isel(time=0)

    return orog.z, (lsm.lsm >= 0.5) * 1

def remove_era5_inland_lakes(lsm):

    '''
    From an ERA5 land sea mask dataarray (lsm), assign inland lakes in south Australia to land points
    '''

    return xr.where((lsm.lon>135) & (lsm.lon<142) & (lsm.lat>-32.25) & (lsm.lat<-25), 1, lsm).T