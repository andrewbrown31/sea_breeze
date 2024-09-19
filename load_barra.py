import intake
import xarray as xr
import numpy as np
import pandas as pd

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

def load_barra_wind_data(unames, vnames, ws_names, t1, t2, domain_id, freq, lat_slice, lon_slice):

    '''
    unames: list of u wind component variables to get (e.g. u100m). Must be a corresponding vname and ws_name
    vnames: list of v wind component variables to get (e.g. v100m). Must be a corresponding uname and ws_name
    ws_names: list of names for wind speed output (e.g. ws_100m). Must be a corresponding uname and vname
    t1: start time in %Y-%m-%d %H:%M"
    t1: end time in %Y-%m-%d %H:%M"
    domain_id: for barra, either AUS-04 or AUST-11
    freq: frequency string (e.g. 1h)
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