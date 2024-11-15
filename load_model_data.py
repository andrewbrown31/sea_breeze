import intake
import xarray as xr
import numpy as np
import pandas as pd
import datetime as dt
import metpy.calc as mpcalc
import metpy.units as units

def interp_np(x, xp, fp):
    return np.interp(x, xp, fp, right=np.nan)

def interp_model_level_to_z(z_da,var_da,mdl_dim,heights):

    '''
    Linearly interpolate from model level data to geopotential height levels.
    If the requested height is below the lowest model level, data from the lowest model level is returned.
    Note that for ERA5, the lowest model level is within the first few 10s of meters above the surface.
    If the requested height is above the highest model level, then NaNs are returned.

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

def load_era5_ml_and_interp(t1,t2,lat_slice,lon_slice,
                            upaths=None,vpaths=None,zpaths=None,
                            heights=np.arange(0,4600,100)):

    """
    ## Load in ERA5 data that was downladed from the Google cloud. That includes u and v wind components as well as geopotential height. Then, interpolate from model levels to height levels.
    I have tried to name the downloaded files systematically using the same monthly notation as in the rt52 project, however the paths can also be manually specified as lists (expecting one file for each variable)

    t1: start time in %Y-%m-%d %H:%M"
    t1: end time in %Y-%m-%d %H:%M"
    lat_slice: a slice to restrict lat domain
    lon_slice: a slice to restrict lon domain    
    upaths: an array of paths for u data. if none use time and look in gb02 dir
    vpaths: an array of paths for v data. if none use time and look in gb02 dir
    zpaths: an array of paths for z data. if none use time and look in gb02 dir    
    heights: to interpolate to (in metres)
    """

    #Load ERA5 model level data downloaded from ERA5
    if (upaths is None) | (vpaths is None) | (zpaths is None):
        time_starts = pd.date_range(
            pd.to_datetime(t1).replace(day=1),t2,freq="MS").strftime("%Y%m%d").astype(int).values
        time_ends = [(t + dt.timedelta(days=32)).replace(day=1) - dt.timedelta(days=1) 
                     for t in pd.to_datetime(time_starts,format="%Y%m%d")]
        upaths = ["/g/data/gb02/ab4502/era5_model_lvl/era5_mdl_u_" + 
                  str(t1) + "_" + t2.strftime("%Y%m%d") +".nc" for t1,t2 in zip(time_starts,time_ends)]
        vpaths = ["/g/data/gb02/ab4502/era5_model_lvl/era5_mdl_v_" + 
                        str(t1) + "_" + t2.strftime("%Y%m%d") +".nc" for t1,t2 in zip(time_starts,time_ends)]
        zpaths = ["/g/data/gb02/ab4502/era5_model_lvl/era5_mdl_z_" + 
                        str(t1) + "_" + t2.strftime("%Y%m%d") +".nc" for t1,t2 in zip(time_starts,time_ends)]                

    #Load the data
    u = xr.combine_by_coords([load_era5_ml(upath,t1,t2,lat_slice,lon_slice) for upath in upaths])
    v = xr.combine_by_coords([load_era5_ml(vpath,t1,t2,lat_slice,lon_slice) for vpath in vpaths])
    z = xr.combine_by_coords([load_era5_ml(zpath,t1,t2,lat_slice,lon_slice) for zpath in zpaths])
    topo,lsm,_= load_era5_static(lon_slice,lat_slice,t1,t2)
    f = xr.merge((u,v,z))

    #Adjust geopotential to height above surface, using topography data saved on gb02
    g = 9.80665    #https://confluence.ecmwf.int/display/CKB/ERA5%3A+compute+pressure+and+geopotential+on+model+levels%2C+geopotential+height+and+geometric+height#heading-Geopotentialheight
    topo = topo / g
    f["geopotential"] = (f["geopotential"] / g) - topo
    f = f.rename({"geopotential":"geopotential_hgt_agl"})

    #Convert to height levels
    interp_v = interp_model_level_to_z(f["geopotential_hgt_agl"],f["v_component_of_wind"],"hybrid",heights)
    interp_u = interp_model_level_to_z(f["geopotential_hgt_agl"],f["u_component_of_wind"],"hybrid",heights)
    interp_era5 = xr.Dataset({"u":interp_u, "v":interp_v})

    return interp_era5, lsm

def load_era5_variable(vnames,t1,t2,lon_slice,lat_slice,chunks="auto"):

    '''
    Load era5 data using the NCI intake catalog

    Input
    vname: list of names of era5 variables
    t1: start time in %Y-%m-%d %H:%M"
    t1: end time in %Y-%m-%d %H:%M"
    lat_slice: a slice to restrict lat domain
    lon_slice: a slice to restrict lon domain

    Output:
    xarray dataset
    '''

    #Set up times to search within catalog
    data_catalog = get_intake_cat_era5()
    time_starts = pd.date_range(pd.to_datetime(t1).replace(day=1),t2,freq="MS").strftime("%Y%m%d").astype(int).values
    time_ends = [(t + dt.timedelta(days=32)).replace(day=1) - dt.timedelta(days=1) for t in pd.to_datetime(time_starts,format="%Y%m%d")]
    times = [str(t1) + "-" + t2.strftime("%Y%m%d") for t1,t2 in zip(time_starts,time_ends)]

    #Load the data using intake
    out = dict.fromkeys(vnames)
    for vname in vnames:
        ds = data_catalog.search(variable=vname,
                                product="era5-reanalysis",
                                time_range=times).\
                                    to_dask(cdf_kwargs={"chunks":chunks}).\
                                        sel(time=slice(t1,t2))
        ds = ds.isel(latitude=slice(None,None,-1))
        ds["longitude"] = (ds.longitude % 360)
        ds = ds.sortby("longitude")    
        ds = ds.rename({"longitude":"lon","latitude":"lat"}).sel(lon=lon_slice, lat=lat_slice)
        out[vname] = ds
        
    return out

def load_era5_static(lon_slice,lat_slice,t1,t2):

    '''
    For ERA5, load static variables using the first time step of the period.
    Also flip the latitude coord and convert -180-180 lons to 0-360 (for consistency with BARRA)
    lat_slice: a slice to restrict lat domain
    lon_slice: a slice to restrict lon domain
    t1: start time in %Y-%m-%d %H:%M"
    t1: end time in %Y-%m-%d %H:%M"

    Returns orography, binary land sea mask, and binary lake conver mask
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

    cl = data_catalog.search(variable="cl",product="era5-reanalysis",time_range=times).to_dask()
    cl = cl.isel(latitude=slice(None,None,-1))
    cl["longitude"] = (cl.longitude % 360)
    cl = cl.sortby("longitude")
    cl = cl.rename({"longitude":"lon","latitude":"lat"}).sel(lon=lon_slice, lat=lat_slice).isel(time=0)

    return orog.z, (lsm.lsm >= 0.5) * 1, (cl.cl >= 0.5)

def remove_era5_inland_lakes(lsm,cl):

    '''
    Use the ERA5 lake cover mask (cl) to assign inland lakes as land points in the land sea mask (lsm)
    '''
    return xr.where(cl,1,lsm)
    #return xr.where((lsm.lon>135) & (lsm.lon<142) & (lsm.lat>-32.25) & (lsm.lat<-25), 1, lsm).T

def load_era5_ml(path,t1,t2,lat_slice,lon_slice,chunks={"time":"auto","hybrid":-1}):

    '''
    Load ERA5 model level data, as downloaded from the Google cloud and stored on gb02 (using era5_download_google.ipynb)
    t1: start time in %Y-%m-%d %H:%M"
    t1: end time in %Y-%m-%d %H:%M"
    lat_slice: a slice to restrict lat domain
    lon_slice: a slice to restrict lon domain    
    '''

    #Load data from disk
    f = xr.open_dataset(path,chunks=chunks)
    f = f.rename({"longitude":"lon","latitude":"lat"}).sel(time=slice(t1,t2))
    
    #Reverse model level data for interpolation, as well as the lat dimension, and slice
    f = f.isel(hybrid=slice(None,None,-1))
    f = f.isel(lat=slice(None,None,-1))
    f = f.sel(lat=lat_slice,lon=lon_slice)

    return f

def era5_sfc_moisture(era5_vars):

    """
    From a dict of ERA5 variables, calculate specific humidity and thetae
    Assumes era5_vars contains "sp", "2d", and "2t"
    """

    era5_vars["q"] = mpcalc.mixing_ratio_from_specific_humidity(
        mpcalc.specific_humidity_from_dewpoint(era5_vars["sp"]["sp"],era5_vars["2d"]["d2m"]))
    era5_vars["thetae"] = mpcalc.equivalent_potential_temperature(
        era5_vars["sp"]["sp"], era5_vars["2t"]["t2m"], era5_vars["2d"]["d2m"])
    
    return era5_vars

# def combine_winds(u,v,uname,vname,ws_name):

#     '''
#     From u and v wind component dataarrays, construct a wind speed dataarray
#     '''

#     wind_da = xr.Dataset({uname:u[uname],vname:v[vname]})
#     wind_da[ws_name] = np.sqrt(wind_da[uname]**2 + wind_da[vname]**2)

#     return wind_da

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

def load_barra_wind_data(unames, vnames, t1, t2, domain_id, freq, lat_slice, lon_slice, vert_coord, chunks="auto"):

    '''
    Load BARRA wind components using the intake catalog, and combine either by height (e.g. if combining u100m and u1000m) or pressure (e.g. if combining u1000 and u500).
    
    Input
    * unames: list of u wind component variables to get (e.g. u100m). Must be a corresponding vname and ws_name

    * vnames: list of v wind component variables to get (e.g. v100m). Must be a corresponding uname and ws_name

    * t1: start time in %Y-%m-%d %H:%M"

    * t2: end time in %Y-%m-%d %H:%M"

    * domain_id: for barra, either AUS-04 or AUST-11

    * freq: frequency string (e.g. 1hr)

    * lat_slice: a slice to restrict lat domain

    * lon_slice: a slice to restrict lon domain

    * vert_coord: coordinate to combine wind levels. either height or pressure for BARRA
    '''

    data_catalog = get_intake_cat()
    times = pd.date_range(pd.to_datetime(t1).replace(day=1),t2,freq="MS").strftime("%Y%m").astype(int).values
    #wind_ds_list = []
    u_ds = []
    v_ds = []
    # for uname,vname,wsname in zip(unames,vnames,ws_names):
    for uname,vname in zip(unames,vnames):

        u_ds.append(data_catalog.search(variable_id=uname,
                            domain_id=domain_id,
                            freq=freq,
                            start_time=times).to_dask(cdf_kwargs={"chunks":chunks}).sel(
                                lon=lon_slice, lat=lat_slice, time=slice(t1,t2)).rename({uname:"u"}))
        v_ds.append(data_catalog.search(variable_id=vname,
                                    domain_id=domain_id,
                                    freq=freq,
                                    start_time=times).to_dask(cdf_kwargs={"chunks":chunks}).sel(
                                        lon=lon_slice, lat=lat_slice, time=slice(t1,t2)).rename({vname:"v"}))        
        
    wind_ds = xr.Dataset({
        "u":xr.combine_nested(u_ds,concat_dim=vert_coord)["u"],
        "v":xr.combine_nested(v_ds,concat_dim=vert_coord)["v"]})
        

    #     u_ds = data_catalog.search(variable_id=uname,
    #                         domain_id=domain_id,
    #                         freq=freq,
    #                         start_time=times).to_dask().sel(lon=lon_slice, lat=lat_slice, time=slice(t1,t2))
    #     v_ds = data_catalog.search(variable_id=vname,
    #                         domain_id=domain_id,
    #                         freq=freq,
    #                         start_time=times).to_dask().sel(lon=lon_slice, lat=lat_slice, time=slice(t1,t2))
    #     wind_ds_list.append(combine_winds(u_ds,v_ds,uname,vname,wsname)\
    #                     .chunk({"time":-1,"lat":20,"lon":20}))
    #wind_ds = xr.merge(wind_ds_list,compat="override")
    
    return wind_ds

def load_barra_variable(vnames, t1, t2, domain_id, freq, lat_slice, lon_slice, chunks="auto"):

    '''
    vnames: list of names of barra variables
    t1: start time in %Y-%m-%d %H:%M"
    t2: end time in %Y-%m-%d %H:%M"
    domain_id: for barra, either AUS-04 or AUST-11
    freq: frequency string (e.g. 1h)
    lat_slice: a slice to restrict lat domain
    lon_slice: a slice to restrict lon domain
    chunks: dict describing the number of chunks. see xr.open_dataset
    '''

    data_catalog = get_intake_cat()
    times = pd.date_range(pd.to_datetime(t1).replace(day=1),t2,freq="MS").strftime("%Y%m").astype(int).values
    out = dict.fromkeys(vnames)
    for vname in vnames:
        ds = data_catalog.search(
            variable_id=vname,
            domain_id=domain_id,
            freq=freq,
            start_time=times)\
                .to_dask(cdf_kwargs={"chunks":chunks}).\
                    sel(lon=lon_slice, lat=lat_slice, time=slice(t1,t2))
        out[vname] = ds
        
    return out

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

def barra_sfc_moisture(barra_vars):

    """
    From a dict of BARRA variables, calculate specific humidity, dewpoint, and theta-e
    Assumes barra_vars contains "huss", "ps", and "tas"
    """

    barra_vars["q"] = mpcalc.mixing_ratio_from_specific_humidity(barra_vars["huss"]["huss"])
    barra_vars["dp"] = mpcalc.dewpoint_from_specific_humidity(barra_vars["ps"]["ps"], barra_vars["tas"]["tas"], barra_vars["huss"]["huss"])
    barra_vars["thetae"] = mpcalc.equivalent_potential_temperature(barra_vars["ps"]["ps"], barra_vars["tas"]["tas"], barra_vars["dp"])

    return barra_vars

def load_aus2200_static(exp_id,lon_slice,lat_slice):

    '''
    Load static fields for the mjo-enso AUS2200 experiment, stored on the ua8 project

    ## Input
    * exp_id: string describing the experiment. either 'mjo-elnino', 'mjo-lanina' or 'mjo-neutral'

    * lat_slice: a slice to restrict lat domain

    * lon_slice: a slice to restrict lon domain
    '''

    assert exp_id in ['mjo-elnino', 'mjo-lanina', 'mjo-neutral'], "exp_id must either be 'mjo-elnino', 'mjo-lanina' or 'mjo-neutral'"
    
    orog = xr.open_dataset("/g/data/ua8/AUS2200/"+exp_id+"/v1-0/fx/orog/orog_AUS2200_"+exp_id+"_fx.nc").\
            sel(lat=lat_slice,lon=lon_slice)
    lsm = xr.open_dataset("/g/data/ua8/AUS2200/"+exp_id+"/v1-0/fx/lmask/lmask_AUS2200_"+exp_id+"_fx.nc").\
            sel(lat=lat_slice,lon=lon_slice)

    return orog.orog, ((lsm.lmask==100)*1)

def load_aus2200_variable(vnames, t1, t2, exp_id, lon_slice, lat_slice, freq, hgt_slice=None, chunks="auto"):

    '''
    Load static fields for the mjo-enso AUS2200 experiment, stored on the ua8 project

    ## Input

    * vnames: list of names of aus2200 variables

    * t1: start time in %Y-%m-%d %H:%M"

    * t2: start time in %Y-%m-%d %H:%M"
    
    * exp_id: string describing the experiment. either 'mjo-elnino', 'mjo-lanina' or 'mjo-neutral'

    * lat_slice: a slice to restrict lat domain

    * lon_slice: a slice to restrict lon domain

    * freq: time frequency (string). either "10min", "1hr", "1hrPlev"

    * hgt_slice: a slice to restrict data in the vertical (in m)

    * chunks: dict describing the number of chunks. see xr.open_dataset
    '''

    assert exp_id in ['mjo-elnino', 'mjo-lanina', 'mjo-neutral'], "exp_id must either be 'mjo-elnino', 'mjo-lanina' or 'mjo-neutral'"
    assert freq in ["10min", "1hr", "1hrPlev"], "exp_id must either be '10min', '1hr', '1hrPlev'"

    out = dict.fromkeys(vnames)
    for vname in vnames:

        fnames = "/g/data/ua8/AUS2200/"+exp_id+"/v1-0/"+freq+"/"+vname+"/"+vname+"_AUS2200_"+exp_id+"_*.nc"
        ds = xr.open_mfdataset(fnames, chunks=chunks).sel(lat=lat_slice,lon=lon_slice,time=slice(t1,t2))
        if hgt_slice is not None:
            ds = ds.sel(lev=hgt_slice)
        out[vname] = ds

    return out

def destagger_aus2200(ds_dict,destag_list,interp_to=None,lsm=None):

    """
    
    From a dictionary of aus2200 datasets (output from load_aus2200_variable), destagger variables in destag_list by interpolating
    
    ## Input
    * ds_dict: a dictionary of aus2200 xarray datasets. output from load_aus2200_variable()

    * destag_list: list of variables to destagger

    * interp_to: variable for which to use spatial info to interp onto

    * lsm: land sea mask dataset to interp on to

    ## Output
    a dictionary of datasets with destaggered variables

    ## Example
    destagger_aus2200(ds_dict, ["uas","vas"], "hus")

    """

    for vars in destag_list:
        if interp_to is not None:
            ds_dict[vars] = ds_dict[vars].interp_like(ds_dict[interp_to],method="linear")
        elif lsm is not None:
            ds_dict[vars] = ds_dict[vars].interp_like(lsm,method="linear")
        else:
            raise Exception("Need to input either a variable to interp to, or a land sea mask, to get spatial info")
        
    return ds_dict